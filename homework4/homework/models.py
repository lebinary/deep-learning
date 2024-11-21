from pathlib import Path
import os
from typing import Tuple
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F


from homework.blocks import (
    ASPP,
    CrossAttentionBlock,
    MLPBlock,
    ResidualCNNBlock,
)
from homework.embeddings import TrackEmbedding

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class WaypointLoss(nn.Module):
    def __init__(self, longitudinal_weight=2.0, lateral_weight=1.0):
        super().__init__()
        self.longitudinal_weight = longitudinal_weight
        self.lateral_weight = lateral_weight

    def forward(self, pred, target, mask) -> torch.Tensor:
        """
        Args:
            pred, target: (B, 3, 2)
        """

        # L2 loss for smoother gradients
        error = (pred - target).abs()

        # Directional weighting
        directional_weight = torch.tensor(
            [self.longitudinal_weight, self.lateral_weight], device=pred.device
        )[None, None, :]

        masked_weighted_error = error * directional_weight * mask[..., None]
        return masked_weighted_error.mean()

class TrackLoss(nn.Module):
   def forward(self, pred, target) -> torch.Tensor:
        """
        Args:
            pred, target: (B, 10, 2)
        """

        # Split left/right boundaries
        n_points = pred.size(1) // 2
        left_pred, right_pred = pred[:, :n_points], pred[:, n_points:]
        left_target, right_target = target[:, :n_points], target[:, n_points:]
        
        # Simple L2 loss for each boundary
        boundary_loss = (left_pred - left_target).abs().mean() + \
                        (right_pred - right_target).abs().mean()
        
        return boundary_loss


""" AUTOENCODER
"""


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_size: int = 32,
        bottleneck_size: int = 16,
        dropout_rate: float = 0.2,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Calculate input features with additional geometric features
        input_features = self.n_track * 2 * 2

        # Input projection layer: (B, 40) -> (B, 256)
        in_dim, out_dim = input_features, hidden_size
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # ENCODER: (B, 64) -> (B, 16)
        in_dim, out_dim = out_dim, bottleneck_size
        self.encoder = MLPBlock(in_dim, out_dim, dropout_rate=dropout_rate)

        # DECODER: (B, 16) -> (B, 64)
        in_dim, out_dim = out_dim, hidden_size
        self.decoder = MLPBlock(in_dim, out_dim, dropout_rate=dropout_rate)

        # Separate prediction heads: (B, 64) -> (B, waypoints)
        in_dim, out_dim = out_dim, n_waypoints
        self.longitudinal_head = MLPBlock(in_dim, out_dim)
        self.lateral_head = MLPBlock(in_dim, out_dim)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Normalized based on the center line
        left_normalized, right_normalized = normalize_track_boundaries(
            track_left, track_right
        )

        # Flatten both track tensors and track direction, (B, 10, 2) -> (B, 20)
        left_flat = left_normalized.view(left_normalized.size(0), -1)
        right_flat = right_normalized.view(right_normalized.size(0), -1)

        # Concat flatten tracks, (B, 20) + (B, 20) -> (B, 60)
        x = torch.concat(
            [left_flat, right_flat],
            dim=1,
        )

        # Projection
        enc_input = self.projection(x)

        # Shared encoder - decoder
        bottleneck = self.encoder(enc_input)  # (B, 16)

        dec = self.decoder(bottleneck)  # (B, 64)
        dec += enc_input  # residual

        # Results
        y_longtitude = self.longitudinal_head(dec)
        y_latitude = self.lateral_head(dec)
        y = torch.stack([y_longtitude, y_latitude], dim=-1)

        return y

    def predict(
        self, track_left: torch.Tensor, track_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Used for inference, predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        pred_waypoints = self(track_left, track_right)

        return pred_waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Embeddings
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.keyval_embed = TrackEmbedding(d_model)

        # Attention
        in_dim, out_dim = d_model, d_model
        self.cross_attention = CrossAttentionBlock(in_dim, out_dim)

        # Prediction head
        in_dim, out_dim = out_dim, 2
        self.prediction_mlp = MLPBlock(in_dim, out_dim)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)

        # Normalized based on the center line
        left_normalized, right_normalized = normalize_track_boundaries(
            track_left, track_right
        )

        # Positional embedding for inputs
        left_embedded = self.keyval_embed(left_normalized)  # (B, 10, 64)
        right_embedded = self.keyval_embed(right_normalized)  # (B, 10, 64)

        # Set up inputs
        query = self.query_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, 3, 64)
        key_value = torch.cat([left_embedded, right_embedded], dim=1)  # (B, 20, 64)

        # Attention
        attn_output = self.cross_attention(query, key_value)  # (B, 3, 64)

        # Results
        y = self.prediction_mlp(attn_output)

        return y

    def predict(
        self, track_left: torch.Tensor, track_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Used for inference, predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        pred_waypoints = self(track_left, track_right)

        return pred_waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        n_track: int = 10,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.n_track = n_track

        self.register_buffer(
            "input_mean", torch.as_tensor(INPUT_MEAN), persistent=False
        )
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Input size = (3, H, W)

        # ENCODER:
        in_dim, out_dim = 3, 16
        self.encoder_block1 = ResidualCNNBlock(
            in_dim, out_dim, stride=2
        )  # -> (16, H/2, W/2)

        in_dim, out_dim = out_dim, 32
        self.encoder_block2 = ResidualCNNBlock(
            in_dim, out_dim, stride=2
        )  # -> (32, H/4, W/4)

        in_dim, out_dim = out_dim, 64
        self.encoder_block3 = ResidualCNNBlock(
            in_dim, out_dim, stride=2
        )  # -> (64, H/8, W/8)

        in_dim, out_dim = out_dim, 96
        self.encoder_block4 = ResidualCNNBlock(
            in_dim, out_dim, stride=2
        )  # -> (96, H/16, W/16)

        in_dim, out_dim = out_dim, 192
        self.encoder_block5 = ResidualCNNBlock(
            in_dim, out_dim, stride=2
        )  # -> (192, H/32, W/32)

        # BOTTLENECK: use ASPP for better feature extraction
        self.bottle_neck = ASPP(192, 192)

        # DECODER:
        in_dim, out_dim = out_dim, 96
        self.decoder_block1 = ResidualCNNBlock(
            in_dim, out_dim, stride=2, up_sampling=True
        )  #  -> (96, H/16, W/16)

        in_dim, out_dim = out_dim, 64
        self.decoder_block2 = ResidualCNNBlock(
            in_dim, out_dim, stride=2, up_sampling=True
        )  #  -> (64, H/8, W/8)

        in_dim, out_dim = out_dim, 32
        self.decoder_block3 = ResidualCNNBlock(
            in_dim, out_dim, stride=2, up_sampling=True
        )  #  -> (32, H/4, W/4)

        in_dim, out_dim = out_dim, 16
        self.decoder_block4 = ResidualCNNBlock(
            in_dim, out_dim, stride=2, up_sampling=True
        )  #  -> (16, H/2, W/2)

        in_dim, out_dim = out_dim, 16
        self.decoder_block5 = ResidualCNNBlock(
            in_dim, out_dim, stride=2, up_sampling=True
        )  #  -> (16, H, W)

        # FINAL LAYERS
        # predicting track
        in_dim, out_dim = out_dim, n_track * 4
        self.track_head = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # -> (B, 40, H, W)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),  # (B, 40)
        )

        # waypoints head
        in_dim, out_dim = out_dim, n_waypoints * 2
        self.waypoint_head = MLPBlock(in_dim, out_dim)  # (B, 6)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        batch_size = image.size(0)
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[
            None, :, None, None
        ]

        # Shared encoder - decoder
        enc1 = self.encoder_block1(x)  # (16, H/2, W/2)
        enc2 = self.encoder_block2(enc1)  # (32, H/4, W/4)
        enc3 = self.encoder_block3(enc2)  # (64, H//8, H/8)
        enc4 = self.encoder_block4(enc3)  # (96, H/16, H/16)
        enc5 = self.encoder_block5(enc4)  # (192, H/32, H/32)

        features = self.bottle_neck(enc5)  # features extraction

        dec1 = self.decoder_block1(features) + enc4  # (96, H/16, H/16)
        dec2 = self.decoder_block2(dec1) + enc3  # (64, H/8, H/8)
        dec3 = self.decoder_block3(dec2) + enc2  # (32, H/4, H/4)
        dec4 = self.decoder_block4(dec3) + enc1  # (16, H/2, H/2)
        dec5 = self.decoder_block5(dec4)  # (16, H, W)

        # Predict track
        y_track = self.track_head(dec5)
        if self.training:
            self.pred_track = y_track.view(batch_size, -1, 2)

        # Predict waypoints
        y_waypoints = self.waypoint_head(y_track).view(batch_size, -1, 2)

        return y_waypoints

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, predicts waypoints from the left and right boundaries of the track.

        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        pred_waypoints = self(image)

        return pred_waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def save_model(model: torch.nn.Module, identifier: str = None) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    # save record in model directory
    if identifier:
        # Create model directory if it doesn't exist
        model_dir = HOMEWORK_DIR / model_name
        os.makedirs(model_dir, exist_ok=True)

        unique_id = str(uuid.uuid4())[:8]
        output_path = model_dir / f"{model_name}_{identifier}_{unique_id}.th"
        torch.save(model.state_dict(), output_path)

    # overwrite one in main directory for grading
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def cal_track_center(
    track_left: torch.Tensor, track_right: torch.Tensor
) -> torch.Tensor:
    return (track_left + track_right) / 2


def normalize_track_boundaries(
    track_left: torch.Tensor, track_right: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes track boundaries around their center.

    Args:
        track_left: Left boundary points tensor of shape (B, N, 2)
        track_right: Right boundary points tensor of shape (B, N, 2)

    Returns:
        Tuple containing:
        - left_normalized: Normalized left boundary vectors (B, N, 2)
        - right_normalized: Normalized right boundary vectors (B, N, 2)
    """
    # Validate input shapes
    assert (
        track_left.shape == track_right.shape
    ), "Track boundaries must have same shape"
    assert track_left.size(-1) == 2, "Last dimension must be 2 for (x,y) coordinates"

    # Calculate center line
    track_center = cal_track_center(track_left, track_right)
    track_width = (track_right - track_left).norm(dim=-1, keepdim=True)

    # Normalize boundaries relative to center
    left_normalized = (track_left - track_center) / track_width
    right_normalized = (track_right - track_center) / track_width

    return left_normalized, right_normalized

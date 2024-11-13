from pathlib import Path
import os
import uuid
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class WaypointLoss(nn.Module):
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean squared error loss for waypoint prediction

        Args:
            pred: tensor (B, n_waypoints, 2) predicted future positions
            target: tensor (B, n_waypoints, 2) ground truth future positions
            mask: tensor (B, n_waypoints) boolean mask for valid waypoints

        Returns:
            tensor, scalar loss
        """
        # Calculate absolute error
        error = (pred - target).abs()  # (b, n, 2)
        error_masked = error * mask[..., None]  # (b, n, 2)

        return error_masked.sum()


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        num_residual_blocks: int = 3,
        num_endcoder_layers: int = 3,
        hidden_size: int = 512,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input projection layer: (B, 40) -> (B, 512)
        in_dim, out_dim = self.n_track * 2 * 2, hidden_size
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )

        # ResNet: (B, 512) -> (B, 512)
        in_dim = out_dim
        blocks = [
            ResidualBlock(hidden_size=in_dim, dropout_rate=dropout_rate)
            for _ in range(num_residual_blocks)
        ]
        self.resnet = nn.Sequential(*blocks)

        # Encoder to reducing the dimension: (B, 512) -> (B, 64)
        layers = []
        for _ in range(num_endcoder_layers):
            in_dim, out_dim = out_dim, max(32, out_dim // 2)
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.encoder = nn.Sequential(*layers)

        # Separate prediction heads: (B, 64) -> (B, waypoints)
        in_dim, out_dim = out_dim, n_waypoints
        self.longitudinal_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim)
        )

        self.lateral_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim)
        )

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
        track_center = (track_left + track_right) / 2
        track_width = (track_right - track_left).norm(dim=-1, keepdim=True)

        left_normalized = (track_left - track_center) / track_width
        right_normalized = (track_right - track_center) / track_width

        # Flatten both track tensors and track direction, (B, 10, 2) -> (B, 20)
        left_flat = left_normalized.view(left_normalized.size(0), -1)
        right_flat = right_normalized.view(right_normalized.size(0), -1)

        # Concat flatten tracks, (B, 20) + (B, 20) + (B, 20) -> (B, 60)
        x = torch.concat([left_flat, right_flat], dim=1)

        # Projection
        x = self.projection(x)

        # ResNet
        x = self.resnet(x)

        # Encoder
        x = self.encoder(x)

        # Longtitude
        y_longtitude = self.longitudinal_head(x)
        y_latitude = self.lateral_head(x)
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

        self.query_embed = nn.Embedding(n_waypoints, d_model)

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
        raise NotImplementedError


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer(
            "input_mean", torch.as_tensor(INPUT_MEAN), persistent=False
        )
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[
            None, :, None, None
        ]

        raise NotImplementedError


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

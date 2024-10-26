from pathlib import Path

import os
import uuid
import torch
import torch.nn as nn
import math

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_block: nn.Sequential, stride=1, up_sampling=False):
        super(ResidualCNNBlock, self).__init__()
        self.cnn_block = cnn_block
        
        if in_channels != out_channels or stride != 1:
            if up_sampling:
                self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.cnn_block(x)
        out += residual
        return out

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)
    
class RegressionLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Mean squared error loss for depth prediction

        Args:
            pred: tensor (b, 1, h, w) predicted depth values
            target: tensor (b, h, w) ground truth depth values

        Returns:
            tensor, scalar loss
        """
        return nn.functional.mse_loss(pred.squeeze(1), target)

class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        # Input size = (3, H, W)
        
        # Expand input
        in_cnn, out_cnn = in_channels, 64
        self.expand_block = nn.Conv2d(in_cnn, out_cnn, kernel_size=7, padding=3, stride=2) # -> (64, H/2, W/2)
        
        # Build CNN residual blocks
        in_cnn, out_cnn = out_cnn, 32
        first_cnn_block = ResidualCNNBlock(in_cnn, out_cnn, nn.Sequential(
            nn.Conv2d(in_cnn, out_cnn, kernel_size=3, padding=1), # -> (32, H/2, W/2)
            nn.BatchNorm2d(out_cnn),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ))
        
        in_cnn, out_cnn = out_cnn, 64
        second_cnn_block = ResidualCNNBlock(in_cnn, out_cnn, nn.Sequential(
            nn.Conv2d(in_cnn, out_cnn, kernel_size=3, padding=1), #  -> (64, H/2, W/2)
            nn.BatchNorm2d(out_cnn),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ))
        
        in_cnn, out_cnn = out_cnn, 128
        third_cnn_block = ResidualCNNBlock(in_cnn, out_cnn, nn.Sequential(
            nn.Conv2d(in_cnn, out_cnn, kernel_size=3, padding=1), #  -> (128, H/2, W/2)
            nn.BatchNorm2d(out_cnn),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ))
        
        self.cnn = nn.Sequential(
            first_cnn_block,
            second_cnn_block,
            third_cnn_block,
        )
        
        # Global residual connection, connect expanded block to CNN output
        self.cnn_shortcut = nn.Conv2d(64, 128, kernel_size=1)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # -> (128, 1, 1) -> flatten (1, 128)
        
        # Build classification blocks
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), # -> (1, 256)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # -> (1, num_classes) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Expand input
        expanded_z = self.expand_block(z)

        # Global residual connection
        cnn_residual = self.cnn_shortcut(expanded_z)
        
        # Main CNN path
        out = self.cnn(expanded_z)
        
        # Add global residual
        out += cnn_residual

        # Adaptive pooling and flattening
        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        
        # Classification
        logits = self.classifier(out)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        # Input size = (3, H, W)
        
        # ENCODER:
        in_dim, out_dim = in_channels, 16
        self.encoder_block1 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1), # -> (16, H/2, W/2)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2)
        
        in_dim, out_dim = out_dim, 32
        self.encoder_block2 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1), #  -> (32, H/4, W/4)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2)

        in_dim, out_dim = out_dim, 64
        self.encoder_block3 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1), #  -> (64, H/8, W/8)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2)
        
        in_dim, out_dim = out_dim, 128
        self.encoder_block4 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1), #  -> (128, H/16, W/16)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2)
        
        # BOTTLENECK: use ASPP for better feature extraction
        self.bottle_neck = ASPP(128, 128)
        
        # DECODER:
        in_dim, out_dim = out_dim, 64
        self.decoder_block4 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (64, H/8, W/8)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2, up_sampling=True)
        
        in_dim, out_dim = out_dim, 32
        self.decoder_block3 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (32, H/4, W/4)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2, up_sampling=True)
        
        in_dim, out_dim = out_dim, 16
        self.decoder_block2 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1), #  -> (16, H/2, W/2)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2, up_sampling=True)
        
        in_dim, out_dim = out_dim, 16
        self.decocder_block1 = ResidualCNNBlock(in_dim, out_dim, nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1), #  -> (16, H, W)
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        ), stride=2, up_sampling=True) 
        
        # FINAL LAYERS
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(out_dim, num_classes, kernel_size=3, padding=1), # -> (num_classes, H, W)
        )
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(out_dim, 1, kernel_size=3, padding=1), # -> (1, H, W)
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Shared encoder - decoder
        enc1 = self.encoder_block1(z)       # (16, H/2, W/2)
        enc2 = self.encoder_block2(enc1)    # (32, H/4, W/4)
        enc3 = self.encoder_block3(enc2)    # (64, H//8, H/8)
        enc4 = self.encoder_block4(enc3)    # (128, H/16, H/16)
        
        features = self.bottle_neck(enc4)   # features extraction
        
        dec4 = self.decoder_block4(features)# (64, H/8, H/8)
        dec4 = dec4 + enc3                  # residual
        dec3 = self.decoder_block3(dec4)    # (32, H/4, H/4)
        dec3 = dec3 + enc2                  # residual
        dec2 = self.decoder_block2(dec3)    # (16, H/2, H/2)
        dec2 = dec2 + enc1                  # residual
        dec1 = self.decocder_block1(dec2)   # (16, H, W)

        # Decoder branches
        logits = self.seg_decoder(dec1)
        raw_depth = self.depth_decoder(dec1).squeeze(1)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Atrous Spatial Pyramid Pooling
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)
        
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = nn.functional.interpolate(
            self.conv5(self.pool(x)),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        return self.project(out)

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
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
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()

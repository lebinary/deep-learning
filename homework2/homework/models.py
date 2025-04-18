"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


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


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        # Calculate the total number of input features
        input_features = 3 * h * w
        
        # Define the linear layer
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor, x is now (b, 3 * H * W)
        x = x.view(x.size(0), -1)
        
        # Apply the linear layer
        logits = self.linear(x)
        
        return logits

class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_size = 128
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        
        # Calculate the total number of input features
        input_features = 3 * h * w
        
        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor, x is now (b, 3 * H * W)
        x = x.view(x.size(0), -1)
        
        # Apply the MLP layer
        logits = self.mlp(x)
        
        return logits

class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_layers: int = 4,
        num_classes: int = 6,
        hidden_size: int = 128,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_layers: int, number of hidden layers
            hidden_size: int, size of hidden layers
            num_classes: int
        """
        super().__init__()
        # Calculate the total number of input features
        input_features = 3 * h * w

        # Build the layers
        layers = []
        in_features = input_features
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ])
            in_features = hidden_size
            
            # Pyramid structure: suggested by Claude to increase accuracy 
            hidden_size = max(16, hidden_size // 2)
        
        self.mlp_deep = nn.Sequential(*layers, nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor, x is now (b, 3 * H * W)
        x = x.view(x.size(0), -1)
        
        # Apply the Deep MLP layer
        logits = self.mlp_deep(x)
        
        return logits

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_layers: int = 4,
        num_classes: int = 6,
        hidden_size: int = 128
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        # Calculate the total number of input features
        input_features = 3 * h * w

        # Build the layers
        self.input_layer = nn.Linear(input_features, hidden_size)

        hidden_layers = []
        in_features = hidden_size
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ])
            in_features = hidden_size
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor, x is now (b, 3 * H * W)
        x = x.view(x.size(0), -1)
        
        # in: (b, 3 * H * W), out: (b, hidden_size)
        x = self.input_layer(x)
        
        # Global residual (we need to ensure dimensions match)
        global_res = x
        
        for i in range(0, len(self.hidden_layers) - 1, 3):
            local_res = x
            x = self.hidden_layers[i](x)   # Linear
            x = self.hidden_layers[i+1](x) # LayerNorm
            x = self.hidden_layers[i+2](x) # ReLU
            x = x + local_res  # Local residual connection
        
        # Global residual connection (if dimensions match)
        x = x + global_res
        
        # Final classification layer
        x = self.output_layer(x)
        
        return x


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r

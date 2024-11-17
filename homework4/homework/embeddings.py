import torch
import torch.nn as nn
import math


# Handle rotation for 2D points
class RoPEEmbedding(nn.Module):
    def __init__(self, base: int = 10000):
        super().__init__()
        self.base = base

    def _get_rotation(self, seq_len: int):
        position = torch.arange(seq_len)  # (10,)
        position = position.unsqueeze(1)  # (10, 1)

        k = torch.exp(torch.tensor(-math.log(self.base)))  # scalar
        k = k.unsqueeze(0)  # (1,)

        theta = position * k  # (10, 1)

        cos_rot = torch.cos(theta)  # (10, 1)
        sin_rot = torch.sin(theta)  # (10, 1)

        return cos_rot, sin_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, seq_len, 2) containing (x,y) pairs
        Returns:
            torch.Tensor: Output tensor of shape (B, seq_len, 2)
        """
        *leading_dims, seq_len, dim = x.shape
        assert dim == 2, "Last dimension must be 2 for (x,y) pairs"

        cos_rot, sin_rot = self._get_rotation(seq_len)  # (10, 1), (10, 1)
        cos_rot = cos_rot.to(x.device)
        sin_rot = sin_rot.to(x.device)

        # Extract x and y coordinates
        x_coord = x[..., 0]  # (B, seq_len)
        y_coord = x[..., 1]  # (B, seq_len)

        # Apply rotation
        x_out = x_coord * cos_rot.squeeze(-1) - y_coord * sin_rot.squeeze(
            -1
        )  # (B, seq_len)
        y_out = x_coord * sin_rot.squeeze(-1) + y_coord * cos_rot.squeeze(
            -1
        )  # (B, seq_len)

        # Combine back into pairs
        return torch.stack([x_out, y_out], dim=-1)  # (B, seq_len, 2)


class TrackEmbedding(nn.Module):
    def __init__(self, d_model: int = 64, base: int = 10000):
        super().__init__()
        self.rope = RoPEEmbedding(base=base)
        self.proj = nn.Linear(2, d_model)

    def forward(self, track_points):
        """
        Args:
            track_points: (B, seq_len=10, 2) track coordinates
        Returns:
            embedded: (B, seq_len=10, d_model=64)
        """
        # Apply RoPE to 2D coordinates
        rotated = self.rope(track_points)  # (B, 10, 2)

        # project to higher dimension
        embedded = self.proj(rotated)  # (B, 10, 64)

        return embedded

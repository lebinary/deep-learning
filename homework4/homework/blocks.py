import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, n_waypoints: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, n_waypoints),
        )

    def forward(self, x):
        return self.net(x)


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


class BottleneckResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int, dropout_rate: float):
        super().__init__()
        if not bottleneck_size:
            bottleneck_size = hidden_size // 2
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.LayerNorm(bottleneck_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.LayerNorm(bottleneck_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_size, hidden_size),
        )

    def forward(self, x):
        return x + self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.down(x)

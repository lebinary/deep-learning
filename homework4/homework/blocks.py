import torch
import torch.nn as nn
import math


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.mlp(x)


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


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, n_heads: int = 8, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_size = in_dim
        self.n_heads = n_heads
        self.head_dim = self.hidden_size // n_heads

        self.k = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.q = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.hidden_size, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key_value):
        batch_size = query.size(0)
        residual = query

        q = self.q(query)  # (B, 3, 64)
        k = self.k(key_value)  # (B, 10, 64)
        v = self.v(key_value)  # (B, 10, 64)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, 8, 3, 8)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, 8, 10, 8)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, 8, 10, 8)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, 8, 3, 10)
        attn = torch.softmax(scores, dim=-1)  # (B, 8, 3, 10)
        attn = self.attn_dropout(attn)  # (B, 8, 3, 10)

        # Apply attention and reshape back
        out = torch.matmul(attn, v)  # (B, 8, 3, 10) * (B, 8, 10, 8) -> (B, 8, 3, 8)
        out = out.transpose(1, 2).reshape(
            batch_size, -1, self.hidden_size
        )  # (B, 3, 64)

        # Output projection
        out = self.output_layer(out)  # (B, 3, 64)
        out = self.dropout(out)

        return residual + out  # (B, 3, 64)

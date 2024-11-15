import torch
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


class MLPBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, dropout_rate: float, with_skip: bool = False
    ):
        super().__init__()
        self.with_skip = with_skip
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        if self.with_skip:
            return x + self.mlp(x)
        return self.mlp(x)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.original_size = hidden_size
        self.hidden_size = (
            hidden_size if (hidden_size > n_heads * 2) else (hidden_size * n_heads * 2)
        )
        self.n_heads = n_heads
        self.head_dim = self.hidden_size // n_heads

        # Multi-head attention
        self.q = nn.Linear(self.original_size, self.hidden_size)
        self.k = nn.Linear(self.original_size, self.hidden_size)
        self.v = nn.Linear(self.original_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(self.hidden_size, self.original_size)
        self.layer_norm = nn.LayerNorm(self.original_size)

    def forward(self, x):
        # Input size: (B, 3, 8), num_heads = 4
        batch_size = x.size(0)

        # Split into weighted heads
        q = self.q(x)  # (B, 3, 8)
        k = self.k(x)  # (B, 3, 8)
        v = self.v(x)  # (B, 3, 8)

        # Transformation before apply softmax (B, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, 4, 3, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, 4, 3, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, 4, 3, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim).float()
        )  # (B, 4, 3, 3)
        attn = torch.softmax(scores, dim=-1)  # (B, 4, 3, 3)
        attn = self.attn_dropout(attn)

        # Combine heads
        out = torch.matmul(attn, v)  # (B, 4, 3, 2)
        out = out.transpose(1, 2)  # (B, 3, 4, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_size)  # (B, 3, 8)
        out = self.proj(out)  # (B, 3, 8)

        # Skip connection and layer norm
        return self.layer_norm(x + out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.original_size = hidden_size
        self.hidden_size = (
            hidden_size if (hidden_size > n_heads * 2) else (hidden_size * n_heads * 2)
        )
        self.n_heads = n_heads
        self.head_dim = self.hidden_size // n_heads

        self.q = nn.Linear(self.original_size, self.hidden_size)
        self.k = nn.Linear(self.original_size, self.hidden_size)
        self.v = nn.Linear(self.original_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(self.hidden_size, self.original_size)
        self.layer_norm = nn.LayerNorm(self.original_size)

    def forward(self, query, key_value):
        batch_size = query.size(0)

        q = self.q(query)
        k = self.k(key_value)
        v = self.v(key_value)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim).float()
        )
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        out = self.proj(out)

        return self.layer_norm(query + out)

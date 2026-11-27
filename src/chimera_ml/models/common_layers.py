import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for a single training sample."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout."""

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = -torch.log(torch.tensor(10000.0)) / dim
        div_term = torch.exp(torch.arange(0, dim, 2) * scale)
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length exceeds positional encoding max_len {self.encoding.shape[1]}.")
        return self.dropout(values + self.encoding[:, : values.shape[1]].to(values))


class TransformerCrossAttention(nn.Module):
    """Transformer cross-attention with masking and residual refinement."""

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        heads = min(8, dim)
        while dim % heads:
            heads -= 1
        self.position = PositionalEncoding(dim, dropout, max_len)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.feed_forward_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[1] == 0 or key.shape[1] == 0:
            return torch.zeros_like(query), query.new_zeros(query.shape[0], query.shape[1], key.shape[1])

        query = self.position(query)
        key = self.position(key)
        safe_key_valid = key_valid.clone()
        empty_rows = ~safe_key_valid.any(dim=1)
        if empty_rows.any():
            safe_key_valid[empty_rows, 0] = True

        safe_key = key.masked_fill(~key_valid.unsqueeze(-1), 0)
        attended, weights = self.attention(
            query,
            safe_key,
            safe_key,
            key_padding_mask=~safe_key_valid,
            need_weights=True,
        )
        result = self.norm(query + self.dropout(attended))
        result = self.feed_forward_norm(result + self.dropout(self.feed_forward(result)))
        result = result * query_valid.unsqueeze(-1).to(result.dtype)
        weights = weights * query_valid.unsqueeze(-1).to(weights.dtype)
        return result, weights


class StatPool(nn.Module):
    """Return masked population mean and standard deviation."""

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weight = valid.to(values.dtype).unsqueeze(-1)
        count = weight.sum(dim=1).clamp_min(1)
        mean = (values * weight).sum(dim=1) / count
        centered = (values - mean.unsqueeze(1)) * weight
        variance = centered.square().sum(dim=1) / count
        positive = variance > 0
        std = torch.sqrt(torch.where(positive, variance, torch.ones_like(variance)))
        return torch.cat((mean, std * positive.to(values.dtype)), dim=-1)

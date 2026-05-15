import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(F.relu(self.dropout(self.layer_1(x))))


class AddAndNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float | None = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)

        return self.layer_norm(x + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._build_pe(), persistent=True)

    def _build_pe(self) -> torch.Tensor:
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe.permute(1, 0, 2)

    def reset_parameters(self) -> None:
        self.pe.copy_(self._build_pe().to(device=self.pe.device, dtype=self.pe.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, input_dim: int, num_heads: int, dropout: float | None = 0.1, positional_encoding: bool = False
    ) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        # initialize layers
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout or 0.0)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.positional_encoding is not None:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        x, _ = self.self_attention(query=query, key=key, value=value)
        x = self.add_norm_after_attention(x, query)
        return self.add_norm_after_ff(self.feed_forward(x), x)


class PermuteLayer(nn.Module):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class StatPoolLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=self.dim), x.std(dim=self.dim, correction=0)], dim=-1)


class ClassificationHead(nn.Module):
    """ClassificationHead"""

    def __init__(self, input_size: int = 256, out_emo: int = 6, out_sen: int = 3) -> None:
        super().__init__()
        self.fc_emo = nn.Linear(input_size, out_emo)
        self.fc_sen = nn.Linear(input_size, out_sen)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_emo = self.fc_emo(x)
        x_sen = self.fc_sen(x)
        return {"emo": x_emo, "sen": x_sen}


class FeaturesDownsampler(nn.Module):
    def __init__(
        self, inp_t_size: int, inp_f_size: int, out_t_size: int = 30, out_f_size: int = 256, features_only: bool = True
    ) -> None:
        super().__init__()

        self.features_only = features_only

        f_modules = [PermuteLayer((0, 2, 1)), nn.Conv1d(inp_f_size, out_f_size, 1), nn.GELU(), PermuteLayer((0, 2, 1))]

        self.f_downsampler = nn.Sequential(*f_modules)

        t_modules = [
            nn.Conv1d(out_f_size, out_f_size, kernel_size=7),
            nn.GroupNorm(num_groups=out_f_size, num_channels=out_f_size, affine=True),
            nn.GELU(),
            nn.Conv1d(out_f_size, out_f_size, kernel_size=7),
            nn.GELU(),
        ]

        self.t_downsampler = nn.Sequential(*t_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f_downsampler(x)

        if not self.features_only:
            raise NotImplementedError("Time downsampler is not implemented")
            x = self.t_downsampler(x)

        return x


class PredictionsFusion(nn.Module):
    def __init__(self, out_emo: int = 7, out_sen: int = 3, fusion_type: str = "mean") -> None:
        super().__init__()
        self.out_emo = out_emo
        self.out_sen = out_sen
        self.fusion_type = fusion_type

        self.w_a = nn.Parameter(torch.empty(out_emo + out_sen, out_emo + out_sen))
        self.w_v = nn.Parameter(torch.empty(out_emo + out_sen, out_emo + out_sen))
        self.w_t = nn.Parameter(torch.empty(out_emo + out_sen, out_emo + out_sen))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_a)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_t)

    def forward(self, x: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if "mean" in self.fusion_type:
            res = torch.stack(x, dim=1).mean(1)
        elif "weighted" in self.fusion_type:
            res = torch.matmul(x[0], self.w_a) + torch.matmul(x[1], self.w_v) + torch.matmul(x[2], self.w_t) + x[3]
        else:
            raise NotImplementedError(f"Fusion {self.fusion_type} is not implemented")

        return res


class PredictionsUpsampler(nn.Module):
    def __init__(
        self,
        inp_t_size: int,
        inp_f_size: int,
        out_t_size: int = 30,
        out_f_size: int = 256,
        out_emo: int = 7,
        out_sen: int = 3,
        return_predictions: bool = False,
    ) -> None:
        super().__init__()

        self.return_predictions = return_predictions
        self.classifier = ClassificationHead(inp_f_size, out_emo=out_emo, out_sen=out_sen)

        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(out_emo + out_sen, out_f_size, out_t_size),
            nn.ReLU(),
            PermuteLayer((0, 2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = torch.mean(x, dim=1)
        x_predicts = self.classifier(features).values()

        x_emo_sen = torch.cat(list(x_predicts), dim=-1).unsqueeze(-1)  # bs, 10, 1
        if self.return_predictions:
            return self.upsampler(x_emo_sen), x_emo_sen.squeeze()

        return self.upsampler(x_emo_sen)

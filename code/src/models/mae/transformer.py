"""Pre-norm Transformer block shared by the BrainLM-EEG encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm → multi-head self-attention → residual, then
    LayerNorm → MLP (``mlp_ratio`` × expansion, GELU) → residual.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Pre-norm feed-forward
        x = x + self.mlp(self.norm2(x))
        return x

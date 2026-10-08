"""Classifier on one hidden layer of a frozen BrainLM-EEG encoder."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .brainlm_eeg import BrainLMEEG

logger = logging.getLogger(__name__)


def _build_mlp_head(in_dim: int, n_classes: int, dropout: float = 0.1) -> nn.Sequential:
    """3-layer MLP identical to BrainLMClassifier head."""
    hidden = in_dim * 2
    head = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return head


class LayerProbeClassifier(nn.Module):
    """Frozen BrainLMEEG encoder probed at one specific hidden layer."""

    def __init__(
        self,
        encoder: BrainLMEEG,
        layer_idx: int,
        n_classes: int,
        head_dropout: float = 0.1,
        pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_idx = layer_idx
        self.pool_mode = pool_mode

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = _build_mlp_head(encoder.encoder_dim, n_classes, head_dropout)

        layer_label = "final (encoder_norm)" if layer_idx == -1 else f"hidden[{layer_idx}]"
        total_head = sum(p.numel() for p in self.head.parameters())
        logger.info(
            f"LayerProbeClassifier | layer={layer_label} | pool={pool_mode} | "
            f"encoder FROZEN | head={encoder.encoder_dim}→{n_classes} | "
            f"trainable={total_head:,}"
        )

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, 1+n_vis, D). Returns (B, D)."""
        if self.pool_mode == "cls":
            return h[:, 0, :]
        return h[:, 1:, :].mean(dim=1)  # mean over patch tokens, exclude CLS

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.layer_idx == -1:
            latent, _, _ = self.encoder.encode(X, mask_ratio=0.0)
            features = self._pool(latent)
        else:
            _, _, _, hidden_states = self.encoder.encode(
                X, mask_ratio=0.0, output_hidden_states=True
            )
            h = hidden_states[self.layer_idx]   # (B, 1+n_vis, D)
            features = self._pool(h)
        return self.head(features)              # (B, n_classes)

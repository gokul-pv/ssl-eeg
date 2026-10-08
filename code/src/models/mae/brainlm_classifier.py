"""BrainLMClassifier — BrainLMEEG encoder with a 3-layer MLP classification head."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .brainlm_eeg import BrainLMEEG

logger = logging.getLogger(__name__)


class BrainLMClassifier(nn.Module):
    """BrainLMEEG encoder with a 3-layer MLP classification head."""

    def __init__(
        self,
        encoder: BrainLMEEG,
        n_classes: int,
        freeze_encoder: bool,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        d = encoder.encoder_dim          # 512
        hidden = d * 2                   # 1024

        # 3-layer MLP head with dropout (BrainLM paper: 10% dropout on head activations)
        # Dropout is applied regardless of freeze_encoder — even in linear probe the head
        # is being trained and benefits from regularisation on small clinical datasets.
        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, n_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info(
                f"BrainLMClassifier [linear probe]: encoder FROZEN | "
                f"CLS→MLP({d}→{hidden}→{hidden}→{n_classes}) dropout={head_dropout}"
            )
        else:
            logger.info(
                f"BrainLMClassifier [fine-tuning]: encoder UNFROZEN | "
                f"CLS→MLP({d}→{hidden}→{hidden}→{n_classes}) dropout={head_dropout}"
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.encoder.encode(X, mask_ratio=0.0)  # (B, 1+n_tokens, D)
        features = self.encoder.get_features(latent)            # (B, D) — CLS token
        return self.head(features)                              # (B, n_classes)

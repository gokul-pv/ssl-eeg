"""BrainLM-EEG — spatiotemporal Masked Autoencoder for EEG."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import _TransformerBlock
from .masking_strategies import (
    build_ch_to_region,
    channel_mask,
    temporal_mask,
    brain_region_mask,
    column_mask,
)

logger = logging.getLogger(__name__)


class BrainLMEEG(nn.Module):
    """BrainLM-EEG spatiotemporal masked autoencoder."""

    _VALID_MASK_STRATEGIES = ("random", "channel", "temporal", "brain_region", "column")

    def __init__(
        self,
        c_common: int,
        n_times: int = 3000,
        patch_size: int = 200,
        mask_ratio: float = 0.75,
        mask_strategy: str = "random",
        encoder_dim: int = 512,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 512,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_pix_loss: bool = False,
        common_ch_names: list[str] | None = None,
        **_ignored,
    ) -> None:
        super().__init__()

        if n_times % patch_size != 0:
            raise ValueError(
                f"n_times={n_times} must be divisible by patch_size={patch_size}."
            )
        if mask_strategy not in self._VALID_MASK_STRATEGIES:
            raise ValueError(
                f"mask_strategy='{mask_strategy}' is not valid. "
                f"Choose from {self._VALID_MASK_STRATEGIES}."
            )
        if mask_strategy == "brain_region" and common_ch_names is None:
            raise ValueError(
                "mask_strategy='brain_region' requires common_ch_names to be provided."
            )

        self.c_common = c_common
        self.n_times = n_times
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.norm_pix_loss = norm_pix_loss
        self.encoder_dim = encoder_dim

        # Brain-region masking: precompute channel → region ID mapping once at init
        self._ch_to_region: list[int] | None = None
        if mask_strategy == "brain_region":
            self._ch_to_region = build_ch_to_region(common_ch_names)  # type: ignore[arg-type]

        self.n_temporal = n_times // patch_size          # time patches per channel
        self.n_tokens = c_common * self.n_temporal       # total spatiotemporal tokens

        logger.info(
            f"BrainLMEEG | C={c_common} T={n_times} patch_size={patch_size} "
            f"n_temporal={self.n_temporal} n_tokens={self.n_tokens} "
            f"mask_ratio={mask_ratio} mask_strategy={mask_strategy} "
            f"encoder={encoder_depth}×{encoder_dim} decoder={decoder_depth}×{decoder_dim}"
        )

        # ── Patch projection ───────────────────────────────────────────
        # Each token = one channel × one time patch of length patch_size
        self.patch_embed = nn.Linear(patch_size, encoder_dim)

        # Token (ch_i, t_j) gets spatial[ch_i] + temporal[t_j].
        self.spatial_embed = nn.Embedding(c_common, encoder_dim)
        self.temporal_embed = nn.Embedding(self.n_temporal, encoder_dim)

        # ── CLS token (always visible, used as global summary) ─────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList([
            _TransformerBlock(encoder_dim, encoder_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # ── Encoder → decoder projection ──────────────────────────────
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # ── Mask token ────────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # ── Decoder positional embeddings (separate from encoder) ──────
        self.dec_spatial_embed = nn.Embedding(c_common, decoder_dim)
        self.dec_temporal_embed = nn.Embedding(self.n_temporal, decoder_dim)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList([
            _TransformerBlock(decoder_dim, decoder_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # ── Reconstruction head ────────────────────────────────────────
        # Reconstruct single-channel 1-second slice (patch_size values)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size, bias=True)

        # ── Attributes for pretrain_trainer.py compatibility ───────────
        self.encoder_key_prefixes = (
            "patch_embed",
            "spatial_embed",
            "temporal_embed",
            "cls_token",
            "encoder_blocks",
            "encoder_norm",
        )

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"BrainLMEEG parameters: {total:,}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Token index helpers ───────────────────────────────────────────────

    def _build_pos_ids(
        self, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build channel and time-patch index tensors for positional embeddings."""
        idx = torch.arange(n, device=device)
        chan_ids = idx // self.n_temporal
        temp_ids = idx % self.n_temporal
        return chan_ids, temp_ids

    # ── Patchify / unpatchify ─────────────────────────────────────────────

    def patchify(self, X: torch.Tensor) -> torch.Tensor:
        """Convert EEG signal to spatiotemporal patch tokens."""
        B, C, T = X.shape
        P = self.patch_size
        N_t = self.n_temporal
        patches = X.reshape(B, C, N_t, P)          # (B, C, N_t, P)
        return patches.reshape(B, C * N_t, P)       # (B, n_tokens, patch_size)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Reconstruct EEG signal from spatiotemporal patch tokens."""
        B, _, P = patches.shape
        C = self.c_common
        N_t = self.n_temporal
        x = patches.reshape(B, C, N_t, P)          # (B, C, N_t, P)
        return x.reshape(B, C, N_t * P)             # (B, C, T)

    # ── Random masking ────────────────────────────────────────────────────

    def _random_mask(
        self, tokens: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask spatiotemporal tokens (per-sample random permutation)."""
        B, N, D = tokens.shape
        n_mask = int(N * mask_ratio)
        n_visible = N - n_mask

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_visible = ids_shuffle[:, :n_visible]
        visible = torch.gather(
            tokens,
            dim=1,
            index=ids_visible.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
        mask[:, :n_visible] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible, mask, ids_restore

    # ── Encoder ───────────────────────────────────────────────────────────

    def encode(
        self,
        X: torch.Tensor,
        mask_ratio: float | None = None,
        output_hidden_states: bool = False,
    ) -> tuple:
        """Encoder forward pass."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B = X.shape[0]

        # 1. Spatiotemporal patchify
        patches = self.patchify(X)                              # (B, n_tokens, P)

        # 2. Project patches to embedding space
        tokens = self.patch_embed(patches)                      # (B, n_tokens, D)

        # 3. Add learnable spatial + temporal positional embeddings
        chan_ids, temp_ids = self._build_pos_ids(self.n_tokens, X.device)
        tokens = tokens + self.spatial_embed(chan_ids)          # (B, n_tokens, D)
        tokens = tokens + self.temporal_embed(temp_ids)         # (B, n_tokens, D)

        # 4. Prepend CLS token (always visible)
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)

        # 5. Masking (applied to non-CLS tokens only; strategy selected at init)
        if mask_ratio > 0.0:
            if self.mask_strategy == "channel":
                visible_st, mask, ids_restore = channel_mask(
                    tokens, self.n_temporal, self.c_common, mask_ratio
                )
            elif self.mask_strategy == "temporal":
                visible_st, mask, ids_restore = temporal_mask(
                    tokens, self.n_temporal, self.c_common, mask_ratio
                )
            elif self.mask_strategy == "brain_region":
                visible_st, mask, ids_restore = brain_region_mask(
                    tokens, self._ch_to_region, self.n_temporal, mask_ratio  # type: ignore[arg-type]
                )
            elif self.mask_strategy == "column":
                visible_st, mask, ids_restore = column_mask(
                    tokens, self.n_temporal, self.c_common, mask_ratio
                )
            else:  # "random" — default, preserves original behaviour
                visible_st, mask, ids_restore = self._random_mask(tokens, mask_ratio)
        else:
            visible_st = tokens
            mask = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=X.device)
            ids_restore = (
                torch.arange(self.n_tokens, device=X.device)
                .unsqueeze(0).expand(B, -1)
            )

        # CLS + visible spatiotemporal tokens → encoder input
        visible_tokens = torch.cat([cls, visible_st], dim=1)   # (B, 1+n_vis, D)

        # 6. Transformer encoder
        hidden_states: list[torch.Tensor] | None = (
            [visible_tokens] if output_hidden_states else None
        )
        for block in self.encoder_blocks:
            visible_tokens = block(visible_tokens)
            if hidden_states is not None:
                hidden_states.append(visible_tokens)

        latent = self.encoder_norm(visible_tokens)              # (B, 1+n_vis, D)

        if output_hidden_states:
            return latent, mask, ids_restore, hidden_states  # type: ignore[return-value]
        return latent, mask, ids_restore

    # ── Decoder ───────────────────────────────────────────────────────────

    def decode(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder forward pass."""
        B = latent.shape[0]
        N = ids_restore.shape[1]                                # n_tokens

        # 1. Project encoder → decoder dim
        x = self.enc_to_dec(latent)                             # (B, 1+n_vis, dec_dim)

        # 2. Split CLS and spatiotemporal tokens
        cls_lat = x[:, 0:1, :]                                  # (B, 1, dec_dim)
        vis_st = x[:, 1:, :]                                    # (B, n_vis, dec_dim)

        # 3. Append mask tokens to restore full sequence
        n_visible = vis_st.shape[1]
        n_mask = N - n_visible
        mask_tokens = self.mask_token.expand(B, n_mask, -1)    # (B, n_mask, dec_dim)
        x_full = torch.cat([vis_st, mask_tokens], dim=1)       # (B, N, dec_dim)

        # 4. Un-shuffle to restore original token order
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]),
        )

        # 5. Add decoder positional embeddings
        chan_ids, temp_ids = self._build_pos_ids(N, latent.device)
        x_full = x_full + self.dec_spatial_embed(chan_ids)
        x_full = x_full + self.dec_temporal_embed(temp_ids)

        # 6. Prepend CLS back (provides context but its output is discarded)
        x_full = torch.cat([cls_lat, x_full], dim=1)           # (B, 1+N, dec_dim)

        # 7. Transformer decoder
        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)

        # 8. Reconstruction head — drop CLS output
        pred = self.decoder_pred(x_full[:, 1:, :])             # (B, N, patch_size)
        return pred

    # ── Loss ─────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on masked spatiotemporal tokens only."""
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = F.mse_loss(pred, target, reduction="none")       # (B, N, patch_size)
        loss = loss.mean(dim=-1)                                # (B, N)

        n_masked = mask.sum()
        if n_masked == 0:
            return loss.mean()
        return (loss * mask.float()).sum() / n_masked

    # ── Full forward ──────────────────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full MAE forward pass (encode → decode → loss)."""
        target = self.patchify(X)                               # (B, n_tokens, P)
        latent, mask, ids_restore = self.encode(X, mask_ratio)
        pred = self.decode(latent, ids_restore)
        loss = self._compute_loss(target, pred, mask)
        return loss, pred, mask

    # ── Downstream interface ──────────────────────────────────────────────

    def get_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract CLS token as global recording representation."""
        return latent[:, 0, :]

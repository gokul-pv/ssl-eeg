"""BrainLM-SphereRoPE — spatiotemporal MAE with 2D Spherical Rotary Positional Encoding."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .brainlm_eeg import BrainLMEEG
from .sphere_rope import (
    get_channel_angles,
    precompute_rope_freqs,
    _RoPETransformerBlock,
)

logger = logging.getLogger(__name__)


class BrainLMSphereRoPE(BrainLMEEG):
    """BrainLM-EEG with 2D Spherical RoPE for spatial positional encoding."""

    def __init__(
        self,
        c_common: int,
        channel_names: list[str],
        n_times: int = 3000,
        patch_size: int = 200,
        mask_ratio: float = 0.75,
        encoder_dim: int = 512,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 512,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_pix_loss: bool = False,
        **_ignored,
    ) -> None:
        if len(channel_names) != c_common:
            raise ValueError(
                f"len(channel_names)={len(channel_names)} must equal c_common={c_common}"
            )

        # Build parent BrainLMEEG (creates spatial_embed, temporal_embed, etc.)
        super().__init__(
            c_common=c_common,
            n_times=n_times,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            norm_pix_loss=norm_pix_loss,
        )

        # ── Remove learnable spatial embeddings (replaced by RoPE) ─────────
        del self.spatial_embed
        del self.dec_spatial_embed

        # ── Channel angle buffer (fixed, loaded from MNE montage) ──────────
        channel_angles = get_channel_angles(channel_names)   # (C, 2)
        self.register_buffer("channel_angles", channel_angles)

        # ── RoPE frequency tables (registered as buffers for device moves) ──
        # RoPE is applied per attention head: each axis gets head_dim//2
        # dimensions → head_dim//4 rotation pairs.
        enc_freqs = precompute_rope_freqs(encoder_dim // encoder_heads // 2)   # (enc_head_dim//4,)
        dec_freqs = precompute_rope_freqs(decoder_dim // decoder_heads // 2)   # (dec_head_dim//4,)
        self.register_buffer("enc_rope_freqs", enc_freqs)
        self.register_buffer("dec_rope_freqs", dec_freqs)

        # ── Replace standard transformer blocks with RoPE-aware blocks ──────
        self.encoder_blocks = nn.ModuleList([
            _RoPETransformerBlock(encoder_dim, encoder_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            _RoPETransformerBlock(decoder_dim, decoder_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])

        # ── Update encoder_key_prefixes for checkpoint saving ───────────────
        # spatial_embed is gone; channel_angles + rope buffers are non-learnable
        # so they don't appear in named_parameters but ARE in state_dict.
        self.encoder_key_prefixes = (
            "patch_embed",
            "channel_angles",
            "enc_rope_freqs",
            "temporal_embed",
            "cls_token",
            "encoder_blocks",
            "encoder_norm",
        )

        # Re-run weight initialisation for the new RoPE blocks
        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"BrainLMSphereRoPE | C={c_common} T={n_times} "
            f"encoder={encoder_depth}×{encoder_dim} decoder={decoder_depth}×{decoder_dim} | "
            f"params={total:,} (no learnable spatial embed)"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_spatial_angles(
        self,
        chan_ids: torch.Tensor,
        prepend_cls: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-token azimuth and elevation tensors."""
        az = self.channel_angles[chan_ids, 0]   # (N,) or (B, N)
        el = self.channel_angles[chan_ids, 1]   # (N,) or (B, N)
        if prepend_cls:
            if chan_ids.dim() == 1:
                zeros = az.new_zeros(1)
                az = torch.cat([zeros, az])
                el = torch.cat([zeros, el])
            else:
                zeros = az.new_zeros(az.shape[0], 1)
                az = torch.cat([zeros, az], dim=1)
                el = torch.cat([zeros, el], dim=1)
        return az, el

    # ── Encoder ───────────────────────────────────────────────────────────

    def encode(
        self,
        X: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder forward pass with 2D Spherical RoPE."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B = X.shape[0]

        # 1. Spatiotemporal patchify
        patches = self.patchify(X)                              # (B, n_tokens, P)

        # 2. Project patches to embedding space
        tokens = self.patch_embed(patches)                      # (B, n_tokens, D)

        # 3. Add ONLY temporal positional embedding (spatial handled by RoPE)
        chan_ids, temp_ids = self._build_pos_ids(self.n_tokens, X.device)
        tokens = tokens + self.temporal_embed(temp_ids)         # (B, n_tokens, D)

        # 4. Prepend CLS token (always visible, gets zero-angle RoPE = identity)
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)

        # 5. Random masking (applied to non-CLS tokens only)
        if mask_ratio > 0.0:
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

        # 6. Per-sample angles of the visible tokens (masking differs per sample):
        #    argsort(ids_restore) recovers each sample's shuffle, whose first
        #    n_visible entries are the visible token indices.
        n_visible = visible_st.shape[1]
        ids_shuffle = torch.argsort(ids_restore, dim=1)          # (B, N)
        ids_tok_visible = ids_shuffle[:, :n_visible]              # (B, n_vis)
        vis_chan_ids = chan_ids[ids_tok_visible]                  # (B, n_vis) — per-sample
        az, el = self._get_spatial_angles(vis_chan_ids, prepend_cls=True)
        # az, el: (B, 1 + n_vis)

        # 7. Transformer encoder with RoPE
        for block in self.encoder_blocks:
            visible_tokens = block(visible_tokens, az, el, self.enc_rope_freqs)
        latent = self.encoder_norm(visible_tokens)              # (B, 1+n_vis, D)

        return latent, mask, ids_restore

    # ── Decoder ───────────────────────────────────────────────────────────

    def decode(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder forward pass with 2D Spherical RoPE."""
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

        # 5. Add ONLY temporal positional embeddings (spatial handled by RoPE)
        chan_ids, temp_ids = self._build_pos_ids(N, latent.device)
        x_full = x_full + self.dec_temporal_embed(temp_ids)

        # 6. Prepend CLS (gets zero-angle identity rotation)
        x_full = torch.cat([cls_lat, x_full], dim=1)           # (B, 1+N, dec_dim)

        # 7. Build spatial angles for the full restored sequence (CLS + all tokens)
        az, el = self._get_spatial_angles(chan_ids, prepend_cls=True)
        # az, el: (1 + N,)

        # 8. Transformer decoder with RoPE
        for block in self.decoder_blocks:
            x_full = block(x_full, az, el, self.dec_rope_freqs)
        x_full = self.decoder_norm(x_full)

        # 9. Reconstruction head — drop CLS output
        pred = self.decoder_pred(x_full[:, 1:, :])             # (B, N, patch_size)
        return pred

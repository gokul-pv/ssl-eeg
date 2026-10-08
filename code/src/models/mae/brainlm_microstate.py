"""BrainLM-Microstate — BrainLM-EEG with a self-contained EEG-microstate prior."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brainlm_eeg import BrainLMEEG

logger = logging.getLogger(__name__)

_MIN_CHANNEL_OVERLAP = 8

# Legacy -> modern 10-20 labels, as in src/datasets/base.py.
_CH_SYNONYMS: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def _align_channels(
    common_ch_names: list[str], map_channels: list[str]
) -> tuple[list[int], list[int]]:
    """Uppercase- and synonym-match `common_ch_names` (the model's own channel
    order) against `map_channels` (the canonical maps' fitted channel order).
    """
    def _norm(ch: str) -> str:
        ch_upper = ch.strip().upper()
        return _CH_SYNONYMS.get(ch_upper, ch_upper)

    common_norm = [_norm(c) for c in common_ch_names]
    map_norm = [_norm(c) for c in map_channels]
    map_pos = {ch: i for i, ch in enumerate(map_norm)}

    idx_in_common: list[int] = []
    idx_in_maps: list[int] = []
    for i, ch in enumerate(common_norm):
        if ch in map_pos:
            idx_in_common.append(i)
            idx_in_maps.append(map_pos[ch])
    return idx_in_common, idx_in_maps


class BrainLMMicrostate(BrainLMEEG):
    """BrainLM-EEG with a self-contained EEG-microstate prior — an auxiliary
    sub-bin classification loss, and nothing else (see module docstring).
    """

    # pretrain.py builds the loaders with return_unnormalized=True for this model.
    uses_unnormalized_input = True

    def __init__(
        self,
        c_common: int,
        common_ch_names: list[str],
        canonical_maps_path: str = "metadata/microstates/canonical_maps.npz",
        n_microstate_classes: int = 4,
        n_sub_bins: int = 8,
        aux_loss_weight: float = 0.5,
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
        **_ignored,
    ) -> None:
        if patch_size % n_sub_bins != 0:
            raise ValueError(f"patch_size={patch_size} must be divisible by n_sub_bins={n_sub_bins}")
        if not common_ch_names or len(common_ch_names) != c_common:
            raise ValueError(
                f"common_ch_names (len={len(common_ch_names) if common_ch_names else 0}) "
                f"must be provided and match c_common={c_common}."
            )

        super().__init__(
            c_common=c_common,
            n_times=n_times,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            common_ch_names=common_ch_names,
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

        self.n_microstate_classes = n_microstate_classes
        self.n_sub_bins = n_sub_bins
        self.aux_loss_weight = aux_loss_weight

        # ── Load + align frozen canonical maps ──────────────────────────
        npz = np.load(canonical_maps_path)
        maps_np = np.asarray(npz["maps"], dtype=np.float32)          # (k, C_fit)
        map_channels = [str(c) for c in npz["channels"]]
        # Maps must be fitted on the model's channels and the LEMON training split.
        fitted_split = str(npz["split"]) if "split" in npz.files else None
        if (fitted_split != "train"
                or {_CH_SYNONYMS.get(c.upper(), c.upper()) for c in map_channels}
                != {_CH_SYNONYMS.get(c.upper(), c.upper()) for c in common_ch_names}):
            raise ValueError(
                f"{canonical_maps_path} holds maps fitted on {len(map_channels)} channels "
                f"(split: {fitted_split}); BrainLM-EEG-Microstate needs maps fitted on the "
                f"model's {len(common_ch_names)} channels of the LEMON training partition. "
                "Refit them with scripts/microstates/fit_canonical_microstate_maps.py."
            )

        idx_in_common, idx_in_maps = _align_channels(common_ch_names, map_channels)
        if len(idx_in_common) < _MIN_CHANNEL_OVERLAP:
            raise ValueError(
                f"Only {len(idx_in_common)} channels overlap between "
                f"common_ch_names and the canonical maps' fitted channels "
                f"(need >= {_MIN_CHANNEL_OVERLAP}). Canonical maps: {map_channels}"
            )
        maps_aligned = maps_np[:, idx_in_maps]  # (k, n_overlap)
        overlap_ch_names = [common_ch_names[i] for i in idx_in_common]

        self.register_buffer(
            "microstate_maps", torch.tensor(maps_aligned, dtype=torch.float32)
        )  # (n_microstate_classes, n_overlap)
        self.register_buffer(
            "microstate_ch_idx", torch.tensor(idx_in_common, dtype=torch.long)
        )  # (n_overlap,)

        # ── Auxiliary sub-bin classification head ────────────────────────
        self.microstate_head = nn.Linear(decoder_dim, n_microstate_classes * n_sub_bins)

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"BrainLMMicrostate | C={c_common} T={n_times} "
            f"n_overlap={len(idx_in_common)}/{c_common} "
            f"mask_strategy={mask_strategy} "
            f"aux_loss_weight={aux_loss_weight} | params={total:,}"
        )
        logger.info(f"BrainLMMicrostate | overlapping channels: {overlap_ch_names}")

    # ── Live microstate-label computation (from X directly) ─────────────

    def _compute_microstate_labels(self, X: torch.Tensor) -> torch.Tensor:
        """Backfit the frozen canonical maps against `X` (the un-normalised
        window), returning `(B, n_temporal, n_sub_bins)` int64 labels — see
        module docstring for the correlation math.
        """
        B = X.shape[0]
        X_overlap = X[:, self.microstate_ch_idx, :]  # (B, n_overlap, T)
        n_overlap = X_overlap.shape[1]
        samples_per_subbin = self.patch_size // self.n_sub_bins

        X_bins = X_overlap.reshape(
            B, n_overlap, self.n_temporal, self.n_sub_bins, samples_per_subbin
        )
        topo = X_bins.mean(dim=-1)                     # (B, n_overlap, n_temporal, n_sub_bins)
        topo = topo.permute(0, 2, 3, 1)                 # (B, n_temporal, n_sub_bins, n_overlap)

        topo_centered = topo - topo.mean(dim=-1, keepdim=True)
        topo_norm = topo_centered / (topo_centered.norm(dim=-1, keepdim=True) + 1e-8)

        maps_centered = self.microstate_maps - self.microstate_maps.mean(dim=-1, keepdim=True)
        maps_norm = maps_centered / (maps_centered.norm(dim=-1, keepdim=True) + 1e-8)  # (K, n_overlap)

        corr = torch.einsum("btsc,kc->btsk", topo_norm, maps_norm)  # (B, n_temporal, n_sub_bins, K)
        labels = corr.abs().argmax(dim=-1)  # polarity-invariant
        return labels

    # ── Decoder ──────────────────────────────────────────────────────────

    def decode(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder forward pass. Returns both the reconstruction and the
        auxiliary sub-bin classification logits for every time patch (the
        latter compared against `microstate_subbin_ids` separately in
        `_compute_aux_loss`).
        """
        B = latent.shape[0]
        N = ids_restore.shape[1]

        x = self.enc_to_dec(latent)

        cls_lat = x[:, 0:1, :]
        vis_st = x[:, 1:, :]

        n_visible = vis_st.shape[1]
        n_mask = N - n_visible
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x_full = torch.cat([vis_st, mask_tokens], dim=1)

        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )

        chan_ids, temp_ids = self._build_pos_ids(N, latent.device)
        x_full = x_full + self.dec_spatial_embed(chan_ids)
        x_full = x_full + self.dec_temporal_embed(temp_ids)

        x_full = torch.cat([cls_lat, x_full], dim=1)

        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)

        dec_out = x_full[:, 1:, :]  # (B, N, dec_dim) — drop CLS
        pred = self.decoder_pred(dec_out)

        # Average the decoder tokens over channels for each time patch.
        pooled = dec_out.view(B, self.c_common, self.n_temporal, -1).mean(dim=1)
        aux_logits = self.microstate_head(pooled)
        aux_logits = aux_logits.view(B, self.n_temporal, self.n_sub_bins, self.n_microstate_classes)

        return pred, aux_logits

    # ── Auxiliary loss ───────────────────────────────────────────────────

    def _compute_aux_loss(
        self,
        aux_logits: torch.Tensor,
        microstate_subbin_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy over **every** sub-bin of every time patch —
        `n_temporal * n_sub_bins` terms per sample, independent of masking.
        """
        logits = aux_logits.reshape(-1, self.n_microstate_classes)
        targets = microstate_subbin_ids.reshape(-1)
        loss = F.cross_entropy(logits, targets)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == targets).float().mean()
        return loss, acc

    # ── Full forward ─────────────────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,
        mask_ratio: float | None = None,
        X_unnormalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full MAE forward pass, as `BrainLMEEG.forward` plus the auxiliary loss."""
        microstate_subbin_ids = None
        if self.aux_loss_weight > 0:
            if X_unnormalized is None:
                raise ValueError(
                    "BrainLMMicrostate needs X_unnormalized (the window before the "
                    "per-channel z-score) to compute its microstate labels; build the "
                    "pretraining dataset with return_unnormalized=True."
                )
            microstate_subbin_ids = self._compute_microstate_labels(X_unnormalized)

        target = self.patchify(X)
        latent, mask, ids_restore = self.encode(X, mask_ratio)
        pred, aux_logits = self.decode(latent, ids_restore)
        recon_loss = self._compute_loss(target, pred, mask)

        if self.aux_loss_weight > 0:
            aux_loss, aux_acc = self._compute_aux_loss(aux_logits, microstate_subbin_ids)
            loss = recon_loss + self.aux_loss_weight * aux_loss
            self.last_recon_loss = recon_loss.detach()
            self.last_aux_loss = aux_loss.detach()
            self.last_aux_acc = aux_acc
        else:
            loss = recon_loss
            self.last_recon_loss = recon_loss.detach()
            self.last_aux_loss = None
            self.last_aux_acc = None

        return loss, pred, mask

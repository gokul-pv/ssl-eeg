"""Spherical RoPE utilities for EEG electrode geometry."""

from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel coordinate lookup
# ---------------------------------------------------------------------------


def get_channel_angles(channel_names: list[str]) -> torch.Tensor:
    """Return spherical angles for a list of EEG channel names."""
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "MNE-Python is required for channel coordinate lookup. "
            "Install with: pip install mne"
        ) from e

    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]  # dict: name → (x, y, z) in metres

    # Normalise MNE channel names for lookup (MNE uses mixed case)
    mne_names_lower = {k.lower(): k for k in pos_dict}

    angles = []
    for name in channel_names:
        mne_key = mne_names_lower.get(name.lower())
        if mne_key is None:
            raise KeyError(
                f"Channel '{name}' not found in MNE standard_1020 montage. "
                f"Available channels: {sorted(pos_dict.keys())}"
            )
        xyz = pos_dict[mne_key]               # (3,) array in metres
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        # Project to unit sphere
        r = math.sqrt(x**2 + y**2 + z**2)
        if r < 1e-9:
            r = 1.0
        xu, yu, zu = x / r, y / r, z / r

        # Spherical angles in MNE head coords (+x=right ear, +y=nose, +z=vertex)
        azimuth = math.atan2(xu, yu)          # 0 at nose, +π/2 at right ear
        elevation = math.asin(max(-1.0, min(1.0, zu)))  # 0 at equator, +π/2 at vertex

        angles.append([azimuth, elevation])

    tensor = torch.tensor(angles, dtype=torch.float32)  # (C, 2)
    logger.info(
        f"SphereRoPE channel angles (azimuth, elevation) in radians:\n"
        + "\n".join(
            f"  {name:6s}: θ={a[0]:+.3f}  φ={a[1]:+.3f}"
            for name, a in zip(channel_names, tensor.tolist())
        )
    )
    return tensor


# ---------------------------------------------------------------------------
# RoPE frequency table
# ---------------------------------------------------------------------------


def precompute_rope_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency bands."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)    # (dim//2,)
    freqs = 1.0 / (base ** (i / dim))                   # (dim//2,)
    return freqs


# ---------------------------------------------------------------------------
# Core RoPE rotation
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]."""
    x1 = x[..., 0::2]   # even indices
    x2 = x[..., 1::2]   # odd indices
    # Interleave -x2, x1 back to original shape
    out = torch.stack([-x2, x1], dim=-1)
    return out.flatten(-2)


def apply_rope_1d(
    x: torch.Tensor,
    angles: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply 1D RoPE to a segment of `x` using scalar angles per token."""
    theta = angles.unsqueeze(-1) * freqs              # (..., S, dim//2)

    # Duplicate each frequency for cos/sin application to paired dims
    cos_theta = theta.cos().repeat_interleave(2, dim=-1)   # (..., S, dim)
    sin_theta = theta.sin().repeat_interleave(2, dim=-1)   # (..., S, dim)

    return x * cos_theta + _rotate_half(x) * sin_theta


def apply_rope_2d(
    x: torch.Tensor,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D factored spherical RoPE to `x`."""
    half = x.shape[-1] // 2
    x_az  = x[..., :half]   # first half → azimuth rotation
    x_el  = x[..., half:]   # second half → elevation rotation

    x_az_rot  = apply_rope_1d(x_az,  azimuth,   freqs)
    x_el_rot  = apply_rope_1d(x_el,  elevation, freqs)

    return torch.cat([x_az_rot, x_el_rot], dim=-1)


# ---------------------------------------------------------------------------
# RoPE-aware Transformer block
# ---------------------------------------------------------------------------


class _RoPETransformerBlock(nn.Module):
    """Pre-norm Transformer block with 2D Spherical RoPE on Q and K."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        assert (dim // n_heads) % 4 == 0, (
            f"head_dim={dim // n_heads} must be divisible by 4 for 2D factored RoPE "
            f"(each axis gets head_dim//2 dims, split into head_dim//4 rotation pairs)"
        )

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # ── Pre-norm attention ─────────────────────────────────────────────
        residual = x
        x_norm = self.norm1(x)

        # Linear projections → (B, S, D)
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        # Reshape to (B, H, S, Dh) for multi-head attention
        Q = Q.view(B, S, H, Dh).transpose(1, 2)   # (B, H, S, Dh)
        K = K.view(B, S, H, Dh).transpose(1, 2)
        V = V.view(B, S, H, Dh).transpose(1, 2)

        # 2D spherical RoPE within every head: the first Dh/2 dims of Q and K
        # rotate by azimuth, the last Dh/2 by elevation (freqs: (Dh//4,)).
        if azimuth.dim() == 2:                    # (B, S) → (B, 1, S): same angles for every head
            azimuth, elevation = azimuth.unsqueeze(1), elevation.unsqueeze(1)
        Q = apply_rope_2d(Q, azimuth, elevation, freqs)   # (B, H, S, Dh)
        K = apply_rope_2d(K, azimuth, elevation, freqs)

        # Scaled dot-product attention (FlashAttention-2 compatible)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )   # (B, H, S, Dh)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.out_proj(attn_out)

        # ── Pre-norm FFN ───────────────────────────────────────────────────
        x = x + self.mlp(self.norm2(x))
        return x

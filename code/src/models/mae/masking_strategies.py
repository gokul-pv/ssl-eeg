"""EEG-specific masking strategies for BrainLM-EEG ablation experiments."""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Anatomical region assignments (standard 10-20 nomenclature, uppercase)
# ---------------------------------------------------------------------------

_CHANNEL_REGION: dict[str, str] = {
    # Prefrontal / frontal
    "FP1": "frontal", "FP2": "frontal", "FPZ": "frontal",
    "AF3": "frontal", "AF4": "frontal", "AF7": "frontal", "AF8": "frontal", "AFZ": "frontal",
    "F1": "frontal",  "F2": "frontal",  "F3": "frontal",  "F4": "frontal",
    "F5": "frontal",  "F6": "frontal",  "F7": "frontal",  "F8": "frontal",  "FZ": "frontal",
    # Frontocentral
    "FC1": "frontocentral", "FC2": "frontocentral", "FC3": "frontocentral",
    "FC4": "frontocentral", "FC5": "frontocentral", "FC6": "frontocentral", "FCZ": "frontocentral",
    # Central (motor cortex)
    "C1": "central", "C2": "central", "C3": "central", "C4": "central",
    "C5": "central", "C6": "central", "CZ": "central",
    # Temporal
    "T7": "temporal", "T8": "temporal", "TP7": "temporal", "TP8": "temporal",
    # Centroparietal
    "CP1": "centroparietal", "CP2": "centroparietal", "CP3": "centroparietal",
    "CP4": "centroparietal", "CP5": "centroparietal", "CP6": "centroparietal", "CPZ": "centroparietal",
    # Parietal (including parieto-occipital)
    "P1": "parietal", "P2": "parietal", "P3": "parietal", "P4": "parietal",
    "P5": "parietal", "P6": "parietal", "P7": "parietal", "P8": "parietal", "PZ": "parietal",
    "PO3": "parietal", "PO4": "parietal", "PO7": "parietal", "PO8": "parietal", "POZ": "parietal",
    # Occipital (visual cortex)
    "O1": "occipital", "O2": "occipital", "OZ": "occipital",
}

_REGIONS = ["frontal", "frontocentral", "central", "temporal", "centroparietal", "parietal", "occipital", "other"]
_REGION_ID: dict[str, int] = {r: i for i, r in enumerate(_REGIONS)}


def build_ch_to_region(ch_names: list[str]) -> list[int]:
    """Map a list of channel names to integer region IDs."""
    result = []
    for ch in ch_names:
        region_name = _CHANNEL_REGION.get(ch.upper(), "other")
        result.append(_REGION_ID[region_name])
    return result


# ---------------------------------------------------------------------------
# Strategy 1: Channel-wise masking
# ---------------------------------------------------------------------------

def channel_mask(
    tokens: torch.Tensor,
    n_temporal: int,
    c_common: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask ALL time patches of randomly selected channels."""
    B, N, D = tokens.shape

    n_ch_mask = max(1, int(c_common * mask_ratio))
    n_ch_mask = min(n_ch_mask, c_common - 1)  # always keep ≥1 visible channel
    n_ch_visible = c_common - n_ch_mask
    n_visible = n_ch_visible * n_temporal

    # Random channel permutation (B, C)
    noise = torch.rand(B, c_common, device=tokens.device)
    ids_ch_shuffle = torch.argsort(noise, dim=1)          # (B, C) random permutation

    ids_ch_visible = ids_ch_shuffle[:, :n_ch_visible]     # (B, n_ch_visible)
    ids_ch_mask    = ids_ch_shuffle[:, n_ch_visible:]     # (B, n_ch_mask)

    # Convert channel indices → token indices (channel-major: ch c → tokens [c*n_temporal, (c+1)*n_temporal))
    t_offsets = torch.arange(n_temporal, device=tokens.device)   # (n_temporal,)
    # (B, n_ch_visible, n_temporal) → (B, n_visible)
    ids_tok_visible = (
        ids_ch_visible.unsqueeze(-1) * n_temporal + t_offsets.unsqueeze(0).unsqueeze(0)
    ).reshape(B, -1)
    # (B, n_ch_mask, n_temporal) → (B, n_mask)
    ids_tok_mask = (
        ids_ch_mask.unsqueeze(-1) * n_temporal + t_offsets.unsqueeze(0).unsqueeze(0)
    ).reshape(B, -1)

    return _ids_to_outputs(tokens, ids_tok_visible, ids_tok_mask, B, N, D)


# ---------------------------------------------------------------------------
# Strategy 2: Temporal (causal / future-prediction) masking
# ---------------------------------------------------------------------------

def temporal_mask(
    tokens: torch.Tensor,
    n_temporal: int,
    c_common: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask the LAST mask_ratio fraction of time patches across ALL channels."""
    B, N, D = tokens.shape

    n_t_mask = max(1, int(n_temporal * mask_ratio))
    n_t_mask = min(n_t_mask, n_temporal - 1)  # always keep ≥1 visible time step
    n_t_visible = n_temporal - n_t_mask
    n_visible = n_t_visible * c_common

    ch_idx = torch.arange(c_common, device=tokens.device)          # (C,)
    t_vis  = torch.arange(n_t_visible, device=tokens.device)        # (n_t_visible,)
    t_msk  = torch.arange(n_t_visible, n_temporal, device=tokens.device)  # (n_t_mask,)

    # (C, n_t_visible) → (n_visible,) then expand to (B, n_visible)
    ids_tok_visible = (
        ch_idx.unsqueeze(1) * n_temporal + t_vis.unsqueeze(0)
    ).reshape(-1).unsqueeze(0).expand(B, -1)

    # (C, n_t_mask) → (n_mask,) then expand to (B, n_mask)
    ids_tok_mask = (
        ch_idx.unsqueeze(1) * n_temporal + t_msk.unsqueeze(0)
    ).reshape(-1).unsqueeze(0).expand(B, -1)

    return _ids_to_outputs(tokens, ids_tok_visible, ids_tok_mask, B, N, D)


# ---------------------------------------------------------------------------
# Strategy 3: Brain-region block masking
# ---------------------------------------------------------------------------

def region_subsets_closest_under(
    ch_to_region: list[int],
    n_temporal: int,
    mask_ratio: float,
) -> list[tuple[int, ...]]:
    """All sets of whole regions whose token count is the largest total not
    exceeding ``int(N * mask_ratio)`` (N = channels × n_temporal).
    """
    from itertools import combinations

    n_tokens = len(ch_to_region) * n_temporal
    budget = int(n_tokens * mask_ratio)
    size = {r: ch_to_region.count(r) * n_temporal for r in sorted(set(ch_to_region))}
    regions = list(size)
    best, subsets = 0, []
    for k in range(1, len(regions) + 1):
        for combo in combinations(regions, k):
            total = sum(size[r] for r in combo)
            if total > budget:
                continue
            if total > best:
                best, subsets = total, [combo]
            elif total == best:
                subsets.append(combo)
    if best == 0:
        raise ValueError(f"no whole brain region fits under mask_ratio={mask_ratio}")
    return subsets


def brain_region_mask(
    tokens: torch.Tensor,
    ch_to_region: list[int],
    n_temporal: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask all tokens of randomly selected anatomical brain regions, whole
    regions only.
    """
    B, N, D = tokens.shape
    device = tokens.device

    subsets = region_subsets_closest_under(ch_to_region, n_temporal, mask_ratio)
    n_regions = max(ch_to_region) + 1
    in_subset = torch.zeros(len(subsets), n_regions, dtype=torch.bool, device=device)
    for i, combo in enumerate(subsets):
        in_subset[i, list(combo)] = True

    # Region of every token (channel-major), then the chosen set per sample.
    ch_idx_per_token = torch.arange(N, device=device) // n_temporal
    region_of_token = torch.tensor(ch_to_region, dtype=torch.long, device=device)[ch_idx_per_token]
    choice = torch.randint(len(subsets), (B,), device=device)
    masked = in_subset[choice][:, region_of_token]                          # (B, N) bool
    n_mask = int(masked[0].sum())

    # Visible tokens first (in random order), masked tokens after.
    score = torch.rand(B, N, device=device) + (~masked).float() * 2.0
    ids_shuffle = torch.argsort(score, dim=1, descending=True)              # (B, N)
    ids_tok_visible = ids_shuffle[:, :N - n_mask]
    ids_tok_mask = ids_shuffle[:, N - n_mask:]

    return _ids_to_outputs(tokens, ids_tok_visible, ids_tok_mask, B, N, D)


# ---------------------------------------------------------------------------
# Strategy 4: Column masking (temporal analogue of channel masking)
# ---------------------------------------------------------------------------

def column_mask(
    tokens: torch.Tensor,
    n_temporal: int,
    c_common: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask ALL channel patches at randomly selected time steps."""
    B, N, D = tokens.shape

    n_t_mask = max(1, int(n_temporal * mask_ratio))
    n_t_mask = min(n_t_mask, n_temporal - 1)  # always keep ≥1 visible time step
    n_t_visible = n_temporal - n_t_mask

    # Random time-step permutation per sample (B, n_temporal)
    noise = torch.rand(B, n_temporal, device=tokens.device)
    ids_t_shuffle = torch.argsort(noise, dim=1)

    ids_t_visible = ids_t_shuffle[:, :n_t_visible]   # (B, n_t_visible)
    ids_t_mask    = ids_t_shuffle[:, n_t_visible:]   # (B, n_t_mask)

    ch_offsets = torch.arange(c_common, device=tokens.device)  # (C,)

    # Visible token indices: (B, C, n_t_visible) → (B, n_visible)
    ids_tok_visible = (
        ch_offsets.unsqueeze(0).unsqueeze(-1) * n_temporal
        + ids_t_visible.unsqueeze(1)
    ).reshape(B, -1)

    # Masked token indices: (B, C, n_t_mask) → (B, n_mask)
    ids_tok_mask = (
        ch_offsets.unsqueeze(0).unsqueeze(-1) * n_temporal
        + ids_t_mask.unsqueeze(1)
    ).reshape(B, -1)

    return _ids_to_outputs(tokens, ids_tok_visible, ids_tok_mask, B, N, D)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _ids_to_outputs(
    tokens: torch.Tensor,
    ids_tok_visible: torch.Tensor,
    ids_tok_mask: torch.Tensor,
    B: int,
    N: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given explicit visible/masked token indices, produce the (visible, mask, ids_restore)
    triple expected by BrainLMEEG.encode() and .decode().
    """
    ids_shuffle = torch.cat([ids_tok_visible, ids_tok_mask], dim=1)       # (B, N)
    ids_restore = torch.argsort(ids_shuffle, dim=1)                        # (B, N)

    n_visible = ids_tok_visible.shape[1]

    visible = torch.gather(
        tokens,
        dim=1,
        index=ids_tok_visible.unsqueeze(-1).expand(-1, -1, D),
    )

    # mask[b, j] = True iff token j is masked
    # In shuffled space: first n_visible slots = False (visible), rest = True (masked)
    mask_shuffled = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
    mask_shuffled[:, :n_visible] = False
    mask = torch.gather(mask_shuffled, dim=1, index=ids_restore)

    return visible, mask, ids_restore

#!/usr/bin/env python
"""Pre-flight check for `BrainLMMicrostate` — what auxiliary accuracy is non-trivial?

Usage
-----
  python scripts/microstates/check_microstate_target_stats.py \\
      --config configs/train_ssl/microstate_pretrain.yaml \\
      --n-windows 2000 --seed 42

Requires `metadata/microstates/canonical_maps.npz` (produced by
`scripts/microstates/fit_canonical_microstate_maps.py`).
"""

from __future__ import annotations

import argparse
import bisect
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.datasets.pretrain import build_pretrain_dataset
from src.models import build_mae_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/train_ssl/microstate_pretrain.yaml",
        help="Pretrain config — the same one the real run will use.",
    )
    parser.add_argument("--n-windows", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sfreq", type=float, default=200.0,
        help="Sampling rate of the preprocessed data, used only to express dwell "
             "time and sub-bin width in ms. Neither the configs nor the dataset "
             "record it; 200 Hz matches the current preprocessing.",
    )
    return parser.parse_args()


def _window_recordings(multi_ds, idx: list[int]) -> list[tuple[str, int]]:
    """Map sampled global window indices to their source recording, as
    ``(dataset_id, recording_index)``.
    """
    out: list[tuple[str, int]] = []
    for i in idx:
        d = bisect.bisect_right(multi_ds.cumulative_sizes, i)
        ds = multi_ds.datasets[d]
        local = i - (multi_ds.cumulative_sizes[d - 1] if d > 0 else 0)
        real_idx = int(ds.window_indices[local])
        out.append((ds.dataset_id, ds._win_to_rec[real_idx]))
    return out


def _variance_decomposition(target: torch.Tensor, groups: list) -> tuple[float, int]:
    """Split total between-window variance into between-recording and
    within-recording parts. Returns ``(between_frac, n_groups)``.
    """
    uniq = {g: k for k, g in enumerate(dict.fromkeys(groups))}
    gid = torch.tensor([uniq[g] for g in groups])
    n_groups = len(uniq)

    grand = target.mean(dim=0, keepdim=True)
    total = ((target - grand) ** 2).sum(dim=0)

    within = torch.zeros_like(total)
    for k in range(n_groups):
        sel = target[gid == k]
        within += ((sel - sel.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)

    between_frac = float(1.0 - (within.sum() / total.sum().clamp_min(1e-12)))
    return between_frac, n_groups


def _persistence_accuracy(seq: torch.Tensor) -> float:
    """Accuracy of predicting each sub-bin's label as the previous sub-bin's,
    over the flattened per-window sequence (so patch boundaries are crossed,
    matching how the label sequence actually runs in time).
    """
    return float((seq[:, 1:] == seq[:, :-1]).float().mean())


def _mean_dwell_subbins(seq: torch.Tensor) -> float:
    """Mean run length in sub-bins (a run = consecutive identical labels)."""
    n_changes = (seq[:, 1:] != seq[:, :-1]).sum().item()
    n_runs = n_changes + seq.shape[0]          # one extra run per window
    return seq.numel() / max(n_runs, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    # Same merge pattern as pretrain.py — train config wins on conflict.
    if model_config_path := cfg.get("model_config"):
        cfg = {**load_config(model_config_path), **cfg}
    if cfg.get("model_name") != "BrainLMMicrostate":
        raise SystemExit(
            f"Config's model is {cfg.get('model_name')!r}, not BrainLMMicrostate — "
            "point --config at a microstate pretrain config."
        )

    logger.info("Building pretraining dataset...")
    multi_ds = build_pretrain_dataset({**cfg, "return_unnormalized": True})
    cfg["expected_n_times"] = multi_ds.n_times
    logger.info(f"{len(multi_ds):,} windows | C_common={multi_ds.c_common} | T={multi_ds.n_times}")

    # The model is used purely as the label calculator here — no training, no
    # forward pass through the transformer.
    model = build_mae_model(
        cfg, c_common=multi_ds.c_common, device=torch.device("cpu"),
        common_ch_names=multi_ds.common_ch_names,
    ).eval()

    n = min(args.n_windows, len(multi_ds))
    idx = torch.randperm(len(multi_ds))[:n].tolist()
    logger.info(f"Computing microstate labels over {n:,} sampled windows...")

    chunks = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            batch = torch.stack(
                [torch.as_tensor(multi_ds[i][1], dtype=torch.float32)   # window before z-scoring
                 for i in idx[start:start + args.batch_size]]
            )
            chunks.append(model._compute_microstate_labels(batch))   # (B, n_temporal, n_sub_bins)

    labels = torch.cat(chunks)                       # (N, n_temporal, n_sub_bins)
    seq = labels.reshape(labels.shape[0], -1)        # (N, n_temporal * n_sub_bins)
    K = model.n_microstate_classes
    subbin_ms = 1000.0 * model.patch_size / model.n_sub_bins / args.sfreq

    # ── Label statistics ──────────────────────────────────────────────────
    prior = torch.bincount(seq.reshape(-1), minlength=K).float()
    prior /= prior.sum()
    entropy_bits = float(-(prior * prior.clamp_min(1e-12).log2()).sum())
    majority = float(prior.max())
    persistence = _persistence_accuracy(seq)
    dwell = _mean_dwell_subbins(seq)

    logger.info("")
    logger.info("── Auxiliary target statistics ─────────────────────────────")
    logger.info(f"   sub-bin width          : {subbin_ms:.0f} ms "
                f"({model.n_temporal} patches × {model.n_sub_bins} sub-bins per window)")
    logger.info(f"   class prior            : {np.round(prior.numpy(), 4).tolist()}")
    logger.info(f"   label entropy          : {entropy_bits:.3f} bits (max {np.log2(K):.3f})")
    logger.info(f"   mean dwell time        : {dwell * subbin_ms:.0f} ms ({dwell:.2f} sub-bins)")
    logger.info("")
    logger.info("── Accuracy baselines (compare pretrain/epoch/*_aux_acc against these) ──")
    logger.info(f"   uniform chance         : {1.0 / K:.1%}")
    logger.info(f"   majority class         : {majority:.1%}")
    logger.info(f"   persistence (copy t-1) : {persistence:.1%}   <-- the one that matters")

    # ── Per-window class coverage: subject trait or fluctuation? ──────────
    coverage = torch.stack(
        [(seq == k).float().mean(dim=1) for k in range(K)], dim=1
    )                                                # (N, K)
    groups = _window_recordings(multi_ds, idx)
    between, n_rec = _variance_decomposition(coverage, groups)

    logger.info("")
    logger.info("── Per-window class coverage ───────────────────────────────")
    logger.info(f"   between-window std     : {np.round(coverage.std(dim=0).numpy(), 4).tolist()}")
    logger.info(f"   variance between recordings = {between:.1%} of total "
                f"(over {n_rec} recordings; the rest is within-recording)")

    if n_rec < 30:
        logger.warning(
            f"Only {n_rec} distinct recordings in the sample. Windows from one recording "
            "are near-duplicates for these statistics, so the effective sample size is the "
            "recording count, not --n-windows. Raise --n-windows until the recording count "
            "stops growing."
        )

    logger.info("")
    strongest = max(majority, persistence)
    logger.info(
        f"Bottom line: aux accuracy must clear {strongest:.1%} to be evidence of anything "
        "beyond label statistics. If it saturates near 100% within the first few epochs, "
        "the auxiliary term has stopped contributing gradient — report that rather than "
        "treating a high number as success."
    )
    if between < 0.25:
        logger.warning(
            f"Only {between:.0%} of coverage variance is between recordings — the target is "
            "mostly within-recording fluctuation, which is less likely to help subject-level "
            "clinical classification downstream."
        )


if __name__ == "__main__":
    main()

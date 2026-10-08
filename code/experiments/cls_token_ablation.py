#!/usr/bin/env python
"""CLS token contribution to reconstruction: decode with the CLS latent kept,
zeroed or replaced by Gaussian noise.

Usage
-----
  python experiments/cls_token_ablation.py \\
      --config configs/experiments/cls_token_ablation.yaml

  # Quick smoke test (CPU):
  python experiments/cls_token_ablation.py \\
      --config configs/experiments/cls_token_ablation.yaml \\
      --set gpu=-1 batch_size=4 num_workers=0

Evaluated on the LEMON/SRM/DVS pretraining validation participants; the paired
one-sided Wilcoxon signed-rank test is computed over per-batch mean losses.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from scipy import stats

def _find_code_root() -> Path:
    """Walk up from this file until we find the directory that contains 'src/'."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "src").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not find code root (no 'src/' directory found in any parent).")

sys.path.insert(0, str(_find_code_root()))

from src.config import apply_cli_overrides, load_config
from src.datasets import build_pretrain_dataset, build_pretrain_dataloader
from src.models import build_mae_model
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

_VALID_MODES = ("zeros", "gaussian")


# ---------------------------------------------------------------------------
# Core ablation logic
# ---------------------------------------------------------------------------


def _perturb_cls(latent: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a clone of latent with the CLS token (index 0) replaced."""
    latent_p = latent.clone()
    if mode == "zeros":
        latent_p[:, 0, :] = 0.0
    elif mode == "gaussian":
        latent_p[:, 0, :] = torch.randn_like(latent[:, 0, :])
    else:
        raise ValueError(f"Unknown perturbation mode: {mode!r}")
    return latent_p


def run_ablation(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    modes: list[str],
) -> dict[str, list[float]]:
    """Iterate over val batches, compute per-batch losses for baseline and each
    perturbation mode, and return all losses as lists (for paired statistics).
    """
    model.eval()
    results: dict[str, list[float]] = {"baseline": []}
    for mode in modes:
        results[mode] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            (X,) = batch
            X = X.to(device)

            target = model.patchify(X)

            # Encode once — same mask & ids_restore for all conditions
            latent, mask, ids_restore = model.encode(X)

            # ── Baseline ──────────────────────────────────────────────────
            pred = model.decode(latent, ids_restore)
            loss = model._compute_loss(target, pred, mask)
            results["baseline"].append(loss.item())

            # ── Perturbed conditions ───────────────────────────────────────
            for mode in modes:
                latent_p = _perturb_cls(latent, mode)
                pred_p = model.decode(latent_p, ids_restore)
                loss_p = model._compute_loss(target, pred_p, mask)
                results[mode].append(loss_p.item())

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  Processed {batch_idx + 1} batches...")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def compute_statistics(
    results: dict[str, list[float]],
    modes: list[str],
) -> list[dict]:
    """Compute summary statistics and Wilcoxon test for each perturbation mode
    vs. baseline.
    """
    baseline = np.array(results["baseline"])
    rows = []

    # Baseline row
    rows.append({
        "condition": "baseline",
        "mean": float(baseline.mean()),
        "std": float(baseline.std()),
        "delta_pct": None,
        "p_value": None,
        "statistic": None,
        "n_batches": len(baseline),
    })

    for mode in modes:
        perturbed = np.array(results[mode])
        delta_pct = (perturbed.mean() - baseline.mean()) / baseline.mean() * 100.0
        stat, p = stats.wilcoxon(perturbed, baseline, alternative="greater")
        rows.append({
            "condition": mode,
            "mean": float(perturbed.mean()),
            "std": float(perturbed.std()),
            "delta_pct": float(delta_pct),
            "p_value": float(p),
            "statistic": float(stat),
            "n_batches": len(perturbed),
        })

    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Condition':<15}  {'Mean Loss':>10}  {'± Std':>8}  "
        f"{'Δ vs baseline':>15}  {'p-value':>12}  {'n_batches':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        delta = f"+{r['delta_pct']:.2f}%" if r["delta_pct"] is not None else "—"
        pval = f"{r['p_value']:.4g}" if r["p_value"] is not None else "—"
        print(
            f"{r['condition']:<15}  {r['mean']:>10.4f}  {r['std']:>8.4f}  "
            f"{delta:>15}  {pval:>12}  {r['n_batches']:>9}"
        )
    print(sep + "\n")


def interpret(rows: list[dict]) -> None:
    baseline_mean = rows[0]["mean"]
    print("Interpretation")
    print("-" * 60)
    for r in rows[1:]:
        sig = r["p_value"] is not None and r["p_value"] < 0.05
        direction = "SIGNIFICANT" if sig else "NOT significant"
        print(
            f"  [{r['condition']:>8}]  Δ={r['delta_pct']:+.2f}%  "
            f"p={r['p_value']:.4g}  → {direction}"
        )
    print()
    print(
        "If Δ is large and significant: CLS token IS load-bearing for reconstruction.\n"
        "If Δ ≈ 0 and not significant: decoder does not rely on CLS token,\n"
        "  supporting the professor's concern that CLS learns weak representations."
    )
    print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLS token ablation experiment for BrainLMEEG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key, e.g. --set gpu=1 batch_size=32",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.set)   # model_config merged by load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seed_everything(int(cfg.get("seed", 42)))

    # ── Device ──────────────────────────────────────────────────────────────
    gpu = int(cfg.get("gpu", 0))
    if gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu}")
    logger.info(f"Device: {device}")

    # ── Checkpoint (load first to know which channels the model expects) ─────
    ckpt_path = cfg.get("checkpoint_path")
    if not ckpt_path:
        raise ValueError("checkpoint_path must be set in config.")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_ch_names: list[str] = ckpt["common_ch_names"]

    # ── Val dataset ─────────────────────────────────────────────────────────
    logger.info("Building val dataset...")
    val_ds = build_pretrain_dataset(cfg)
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset is empty. Check pretrain_datasets split values in config.")

    # Override channel intersection to match the checkpoint (val datasets may
    # have a wider intersection than what the model was trained on).
    if val_ds.common_ch_names != ckpt_ch_names:
        logger.info(
            f"Restricting val dataset channels from {val_ds.c_common} → {len(ckpt_ch_names)} "
            "to match checkpoint."
        )
        for sub_ds in val_ds.datasets:
            sub_ds.set_common_ch_names(ckpt_ch_names)
        val_ds.common_ch_names = ckpt_ch_names
        val_ds.c_common = len(ckpt_ch_names)

    val_loader = build_pretrain_dataloader(
        val_ds,
        {**cfg, "pretrain_balanced_sampling": False},
    )
    logger.info(
        f"Val dataset: {len(val_ds):,} windows | "
        f"C_common={val_ds.c_common} | T={val_ds.n_times}"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = build_mae_model(
        cfg,
        c_common=len(ckpt_ch_names),
        device=device,
        common_ch_names=ckpt_ch_names,
    )

    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    logger.info(f"Checkpoint loaded (epoch {ckpt.get('epoch', '?')}, "
                f"best_val_loss={ckpt.get('best_val_loss', '?')})")

    model.eval()

    # ── Validate model type ──────────────────────────────────────────────────
    if not hasattr(model, "cls_token"):
        raise TypeError(
            f"Model {type(model).__name__} has no CLS token. "
            "This experiment is only applicable to the BrainLM-EEG variants."
        )

    # ── Perturbation modes ────────────────────────────────────────────────────
    modes: list[str] = cfg.get("perturbation_modes", list(_VALID_MODES))
    for m in modes:
        if m not in _VALID_MODES:
            raise ValueError(f"Unknown perturbation mode {m!r}. Valid: {_VALID_MODES}")
    logger.info(f"Perturbation modes: {modes}")

    # ── Run ablation ─────────────────────────────────────────────────────────
    logger.info("Running ablation...")
    results = run_ablation(model, val_loader, device, modes)
    logger.info(f"Done. Processed {len(results['baseline'])} batches.")

    # ── Statistics & reporting ────────────────────────────────────────────────
    rows = compute_statistics(results, modes)
    print_table(rows)
    interpret(rows)

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = Path(cfg.get("output_dir", "outputs/experiments/cls_token_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    payload = {
        "checkpoint": str(ckpt_path),
        "n_batches": len(results["baseline"]),
        "n_val_windows": len(val_ds),
        "c_common": val_ds.c_common,
        "n_times": val_ds.n_times,
        "perturbation_modes": modes,
        "summary": rows,
        "per_batch_losses": {k: v for k, v in results.items()},
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    # ── Optional WandB ────────────────────────────────────────────────────────
    if cfg.get("use_wandb", False) and cfg.get("wandb_mode", "disabled") != "disabled":
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "eeg-ssl-pretrain"),
            name=f"cls_token_ablation/{ckpt_path.parent.name}",
            mode=cfg.get("wandb_mode", "offline"),
            config=cfg,
        )
        for r in rows:
            prefix = f"cls_ablation/{r['condition']}"
            wandb.log({f"{prefix}/mean_loss": r["mean"], f"{prefix}/std_loss": r["std"]})
            if r["delta_pct"] is not None:
                wandb.log({
                    f"{prefix}/delta_pct": r["delta_pct"],
                    f"{prefix}/p_value": r["p_value"],
                })
        wandb.finish()


if __name__ == "__main__":
    main()

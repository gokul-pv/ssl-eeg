#!/usr/bin/env python
"""Self-supervised pretraining of BrainLM-EEG, BrainLM-EEG-RoPE,
BrainLM-EEG-Microstate and the masking-ablation models.

Usage
-----
  python pretrain.py --config configs/pretrain/brainlm_eeg.yaml
  python pretrain.py --config configs/pretrain/brainlm_eeg_rope.yaml
  python pretrain.py --config configs/pretrain/brainlm_eeg_microstate.yaml
  python pretrain.py --config configs/ablation/masking_ratio/ratio_50.yaml

  # quick smoke test
  python pretrain.py --config configs/pretrain/brainlm_eeg.yaml \\
      --set train_epochs=2 batch_size=8 wandb_mode=disabled
"""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import apply_cli_overrides, load_config
from src.datasets import build_pretrain_dataloader, build_pretrain_dataset
from src.engine.pretrain_trainer import PretrainTrainer
from src.models import MAE_MODEL_REGISTRY, build_mae_model
from src.utils.runtime import finish_wandb, init_wandb, slug, tee_console
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BrainLM-EEG self-supervised pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Pretraining YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set train_epochs=5 wandb_mode=online",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Diagnostic linear probe (monitoring only)
# ---------------------------------------------------------------------------


def build_probe_loader_fn(cfg: dict, common_ch_names: list[str] | None = None):
    """Return a callable building ``(train_loader, val_loader, n_classes)`` for the
    periodic diagnostic probe (``linear_probe: true``, ``probe_dataset_config``),
    or None when the probe is disabled. The probe never influences optimisation,
    early stopping or checkpoint selection.
    """
    if not cfg.get("linear_probe", False):
        return None
    probe_cfg_path = cfg.get("probe_dataset_config")
    if not probe_cfg_path:
        logger.warning("linear_probe=true but probe_dataset_config is not set — probe skipped.")
        return None

    def _factory():
        from src.datasets import (build_dataloader, build_dataset, get_subject_split, load_windows_dataset,
                                  pretrain_reserve_for)

        probe_cfg = load_config(probe_cfg_path)
        probe_cfg.update({k: v for k, v in cfg.items()
                          if k in ("pretrain_seed", "split_save_dir", "batch_size", "num_workers",
                                   "persistent_workers")})
        probe_cfg["seed"] = int(cfg.get("pretrain_seed", 42))
        probe_cfg["batch_size"] = int(cfg.get("probe_batch_size", cfg.get("batch_size", 64)))
        if common_ch_names is not None:
            # Same channels as the pretrained model.
            probe_cfg["common_ch_names"] = common_ch_names

        windows_ds = load_windows_dataset(probe_cfg)
        # The pretraining reserve was trained on, so the probe excludes it (as
        # every downstream evaluation does).
        reserve = pretrain_reserve_for(windows_ds, probe_cfg)
        train_idx, val_idx, _ = get_subject_split(windows_ds, probe_cfg, exclude_sids=reserve)
        train_ds = build_dataset(windows_ds, train_idx, probe_cfg)
        val_ds = build_dataset(windows_ds, val_idx, probe_cfg)
        return (build_dataloader(train_ds, probe_cfg, split="train"),
                build_dataloader(val_ds, probe_cfg, split="val"),
                train_ds.n_classes)

    return _factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_val_config(cfg: dict) -> dict:
    """Pretraining-validation corpus: the held-out 10% ``val`` participants of the
    sources pretrained on their ``train`` split (DVS, SRM, LEMON). The clinical
    pretraining reserves are training data only and are not part of it.
    """
    val_cfg = deepcopy(cfg)
    val_cfg["pretrain_datasets"] = [
        {**entry, "split": "val"}
        for entry in cfg.get("pretrain_datasets", []) if entry.get("split") == "train"
    ]
    return val_cfg


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)

    model_slug = slug(cfg.get("model_name", "brainlmeeg"))
    run_save_dir = Path(cfg.get("save_dir", "outputs/checkpoints/pretrain")) / model_slug
    run_save_dir.mkdir(parents=True, exist_ok=True)
    cfg["save_dir"] = str(run_save_dir)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(cfg.get("wandb_run_name") or f"pretrain/{model_slug}/{ts}")
    cfg["wandb_run_name"] = run_name
    log_path = (Path(cfg.get("terminal_log_dir", "outputs/logs/pretrain")) / model_slug
                / f"{slug(run_name)}.log")

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = trainer = None
        try:
            logger.info(f"Terminal log: {log_path}")
            logger.info(f"Run name    : {run_name}")
            for k, v in cfg.items():
                logger.info(f"  {k:<30} = {v}")

            seed_everything(int(cfg.get("seed", 42)))
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device: {device}")

            # ── Training corpus ──────────────────────────────────────────────
            # BrainLM-EEG-Microstate takes its labels from the window before
            # the per-channel z-score, so the loaders also return that window.
            cfg["return_unnormalized"] = bool(
                getattr(MAE_MODEL_REGISTRY.get(cfg.get("model_name", "BrainLMEEG")),
                        "uses_unnormalized_input", False))
            multi_ds = build_pretrain_dataset(cfg)
            train_loader = build_pretrain_dataloader(multi_ds, cfg)
            cfg["expected_n_times"] = multi_ds.n_times
            logger.info(
                f"Pretraining dataset: {len(multi_ds):,} windows | "
                f"C_common={multi_ds.c_common} | T={multi_ds.n_times}"
            )
            logger.info(f"Common channels: {multi_ds.common_ch_names}")

            # ── Validation corpus ────────────────────────────────────────────
            val_multi_ds = build_pretrain_dataset(build_val_config(cfg),
                                                  common_ch_names=multi_ds.common_ch_names)
            val_loader = (
                build_pretrain_dataloader(
                    val_multi_ds,
                    {**cfg, "batch_size": cfg.get("batch_size", 64), "pretrain_balanced_sampling": False},
                )
                if len(val_multi_ds) > 0 else None
            )

            # ── Model ────────────────────────────────────────────────────────
            model = build_mae_model(cfg, c_common=multi_ds.c_common, device=device,
                                    common_ch_names=multi_ds.common_ch_names)

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-ssl-pretrain",
                wandb_dir=Path(cfg.get("wandb_dir", "outputs/wandb/pretrain")) / model_slug,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "model/total_params": sum(p.numel() for p in model.parameters()),
                    "model/trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                })

            trainer = PretrainTrainer(
                model=model,
                train_loader=train_loader,
                cfg=cfg,
                device=device,
                val_loader=val_loader,
                wandb_run=wandb_run,
                probe_loader_fn=build_probe_loader_fn(cfg, common_ch_names=multi_ds.common_ch_names),
                common_ch_names=multi_ds.common_ch_names,
            )
            best_val_loss = trainer.train()
            logger.info(f"Pretraining complete. Best val_loss={best_val_loss:.4f}")
        finally:
            finish_wandb(
                wandb_run, wandb_module, cfg, run_name=run_name, log_path=log_path, tee_file=tee_file,
                summary={"pretrain_best_encoder_ckpt": str(trainer.ckpt_encoder) if trainer else "n/a"},
            )


if __name__ == "__main__":
    main()

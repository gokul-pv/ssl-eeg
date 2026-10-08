#!/usr/bin/env python
"""BrainLM-EEG downstream adaptation — linear probing and finetuning.

Usage
-----
  # Five-fold CV, linear probe, AD vs HC
  python finetune.py --config configs/linear_probe/brainlm_eeg_lp_adftd.yaml

  # LOSO, finetuning, MDD
  python finetune.py --config configs/finetune/brainlm_eeg_ft_mdd_loso.yaml

  # All FEP participants (checkpoint for the FEP → SCZ evaluation)
  python finetune.py --config configs/finetune/brainlm_eeg_ft_fep_full.yaml

  # Point to a different pretrained encoder
  python finetune.py --config ... --set pretrained_encoder_path=/path/best_pretrain_encoder.pth

Protocols (``protocol: rolling | loso | full``) are described in
``src/engine/protocols.py``. Outputs:
  <save_dir>/<model>/<dataset_id>/<task>/<mode>/seed<seed>/...
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import apply_cli_overrides, load_config
from src.datasets import load_windows_dataset
from src.engine.protocols import ProtocolContext, resolve_protocol, run_protocol
from src.models import build_mae_classifier
from src.utils.runtime import finish_wandb, init_wandb, slug, tee_console
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BrainLM-EEG linear probing / finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Linear-probe or finetune YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set wandb_mode=disabled",
    )
    return parser.parse_args()


def _load_encoder_metadata(encoder_path: str, device: torch.device) -> list[str]:
    """Channel list stored with the pretrained encoder (defines the input channels)."""
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "common_ch_names" not in ckpt:
        raise ValueError(
            f"Checkpoint at '{encoder_path}' does not contain 'common_ch_names' "
            "(expected an encoder checkpoint written by pretrain.py)."
        )
    best = ckpt.get("best_val_loss")
    best_str = f"{best:.4f}" if isinstance(best, float) else str(best)
    logger.info(f"Pretrained encoder: epoch={ckpt.get('epoch', '?')} | best_val_loss={best_str}")
    logger.info(f"Common channels ({len(ckpt['common_ch_names'])}): {ckpt['common_ch_names']}")
    return list(ckpt["common_ch_names"])


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)

    mode = cfg.get("mode", "linear_probe")
    if mode not in ("linear_probe", "finetune"):
        raise ValueError(f"mode must be 'linear_probe' or 'finetune', got '{mode}'")
    protocol = resolve_protocol(cfg)

    dataset_id = cfg.get("dataset_id", "unknown")
    dataset_name = cfg.get("dataset_name", dataset_id)
    seed = int(cfg.get("seed", 42))
    model_slug = slug(cfg.get("model_name", "brainlmeeg"))
    task_slug = slug(cfg.get("classify_choice") or dataset_name)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = Path(cfg.get("terminal_log_dir", "outputs/logs/finetune")) / dataset_id / task_slug / mode
    log_path = log_dir / f"{model_slug}_{protocol}_seed{seed}_{ts}.log"
    run_name = f"{model_slug}/{dataset_name}/{task_slug}/{mode}/{protocol}/seed{seed}/{ts}"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            logger.info(f"Terminal log : {log_path}")
            logger.info(f"Mode={mode}  protocol={protocol}  dataset={dataset_id}  seed={seed}")
            for k, v in cfg.items():
                if not isinstance(v, (dict, list)):
                    logger.info(f"  {k:<30} = {v}")

            seed_everything(seed)
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device: {device}")

            encoder_path = cfg.get("pretrained_encoder_path")
            if not encoder_path:
                raise ValueError("cfg['pretrained_encoder_path'] must be set.")
            # The encoder's channel list defines the input channels (19 common channels).
            cfg["common_ch_names"] = _load_encoder_metadata(encoder_path, device)

            windows_ds = load_windows_dataset(cfg)

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-ssl-finetune",
                wandb_dir=Path(cfg.get("wandb_dir", "outputs/wandb/finetune")) / dataset_id,
            )

            def build(run_cfg: dict, train_ds):
                model = build_mae_classifier(run_cfg, n_classes=train_ds.n_classes, device=device)
                return model, None

            out_dir = (
                Path(cfg.get("save_dir", "outputs/checkpoints/finetune"))
                / model_slug / dataset_id / task_slug / mode / f"seed{seed}"
            )
            run_protocol(ProtocolContext(
                cfg=cfg, windows_ds=windows_ds, device=device, build_model=build,
                out_dir=out_dir, file_tag=f"{model_slug}_{mode}",
                vote_model=f"{model_slug}_{mode}",
                results_meta={"mode": mode, "model_slug": model_slug},
                wandb_run=wandb_run,
            ))
        finally:
            finish_wandb(wandb_run, wandb_module, cfg, run_name=run_name,
                         log_path=log_path, tee_file=tee_file)


if __name__ == "__main__":
    main()

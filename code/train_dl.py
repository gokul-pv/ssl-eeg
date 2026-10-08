#!/usr/bin/env python
"""Supervised deep-learning baselines — EEGNeX, EEGConformer, ATCNet.

Usage
-----
  python train_dl.py --config configs/train_dl/train_atcnet_adftd.yaml          # five-fold CV
  python train_dl.py --config configs/train_dl/train_eegnex_mdd_loso.yaml       # LOSO
  python train_dl.py --config configs/train_dl/train_eegconformer_fep_full.yaml # all FEP (→ SCZ)

  # quick smoke test
  python train_dl.py --config configs/train_dl/train_atcnet_adftd.yaml --set train_epochs=2 wandb_mode=disabled

Outputs: <save_dir>/<model>/<dataset_id>/<task>/seed<seed>/...
(see ``src/engine/protocols.py``). This entrypoint replaces the original
``train_cv.py`` (five-fold CV, LOSO) and ``train.py`` (full-data training).
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
from src.datasets.channels import infer_ch_names, make_data_info, resolve_ch_names
from src.engine.protocols import ProtocolContext, resolve_protocol, run_protocol
from src.models import build_model
from src.utils.runtime import finish_wandb, init_wandb, slug, tee_console
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised DL baselines (EEGNeX / EEGConformer / ATCNet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set wandb_mode=disabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)

    protocol = resolve_protocol(cfg)
    dataset_id = cfg.get("dataset_id", "unknown")
    dataset_name = cfg.get("dataset_name", dataset_id)
    seed = int(cfg.get("seed", 42))
    model_slug = slug(cfg.get("model_name", "model"))
    task_slug = slug(cfg.get("classify_choice") or dataset_name)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    strategy_tag = {"rolling": f"cv{cfg.get('n_folds', 5)}", "loso": "loso", "full": "full"}[protocol]

    run_name = str(cfg.get("wandb_run_name") or f"{model_slug}/{slug(dataset_name)}/{task_slug}/{strategy_tag}/{ts}")
    log_dir = Path(cfg.get("terminal_log_dir", Path("outputs") / "logs" / slug(dataset_id) / "train_dl"))
    log_path = log_dir / f"{slug(run_name)}.log"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            logger.info(f"Terminal log : {log_path}")
            logger.info(f"Model={cfg.get('model_name')}  protocol={protocol}  dataset={dataset_id}  seed={seed}")
            for k, v in cfg.items():
                if not isinstance(v, dict):
                    logger.info(f"  {k:<30} = {v}")

            seed_everything(seed)
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device: {device}")

            windows_ds = load_windows_dataset(cfg)
            fallback_ch_names = infer_ch_names(windows_ds)

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-baselines",
                wandb_dir=cfg.get("wandb_dir", "outputs/wandb"),
            )
            if wandb_run is not None:
                wandb_run.log({"data/total_windows": len(windows_ds),
                               "data/n_folds": int(cfg.get("n_folds", 5)),
                               "cv_strategy": protocol})

            def build(run_cfg: dict, train_ds):
                ch_names = resolve_ch_names(train_ds, fallback_ch_names)
                info = make_data_info(train_ds, run_cfg, None)   # these models take no ch_names
                return build_model(run_cfg, info, device), ch_names

            out_dir = (
                Path(cfg.get("save_dir", "outputs/checkpoints/train_dl"))
                / model_slug / dataset_id / task_slug / f"seed{seed}"
            )
            run_protocol(ProtocolContext(
                cfg=cfg, windows_ds=windows_ds, device=device, build_model=build,
                out_dir=out_dir, file_tag=model_slug, vote_model=model_slug,
                results_meta={"model_name": model_slug},
                wandb_run=wandb_run,
            ))
        finally:
            finish_wandb(wandb_run, wandb_module, cfg, run_name=run_name,
                         log_path=log_path, tee_file=tee_file)


if __name__ == "__main__":
    main()

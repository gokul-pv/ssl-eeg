#!/usr/bin/env python
"""Foundation-model baselines — linear probing of LaBraM, CBraMod and REVE.

Usage
-----
  python probe.py --config configs/probe/labram_lp_adftd.yaml           # five-fold CV
  python probe.py --config configs/probe/cbramod_lp_mdd_loso.yaml       # LOSO
  python probe.py --config configs/probe/reve_lp_fep_full.yaml          # all FEP (→ SCZ)

  # quick smoke test
  python probe.py --config configs/probe/labram_lp_adftd.yaml --set train_epochs=2 wandb_mode=disabled

Outputs: <save_dir>/<model>/<dataset_id>/<task>/linear_probe/seed<seed>/...
(see ``src/engine/protocols.py``).
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
        description="Foundation-model linear probe (LaBraM / CBraMod / REVE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Probe YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set wandb_mode=disabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    if not cfg.get("freeze_backbone", False):
        logger.warning("freeze_backbone is not true — probe.py is meant for linear probing.")

    mode = cfg.get("mode", "linear_probe")
    protocol = resolve_protocol(cfg)
    dataset_id = cfg.get("dataset_id", "unknown")
    dataset_name = cfg.get("dataset_name", dataset_id)
    seed = int(cfg.get("seed", 42))
    model_slug = slug(cfg.get("model_name", "foundation-model"))
    task_slug = slug(cfg.get("classify_choice") or dataset_name)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = Path(cfg.get("terminal_log_dir", "outputs/logs/probe")) / dataset_id / task_slug / mode
    log_path = log_dir / f"{model_slug}_{protocol}_seed{seed}_{ts}.log"
    run_name = f"{model_slug}/{dataset_name}/{task_slug}/{mode}/{protocol}/seed{seed}/{ts}"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            logger.info(f"Terminal log : {log_path}")
            logger.info(f"Model={cfg.get('model_name')}  protocol={protocol}  dataset={dataset_id}  seed={seed}")
            for k, v in cfg.items():
                if not isinstance(v, (dict, list)):
                    logger.info(f"  {k:<30} = {v}")

            seed_everything(seed)
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device: {device}")

            windows_ds = load_windows_dataset(cfg)
            fallback_ch_names = infer_ch_names(windows_ds)

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-ssl-finetune",
                wandb_dir=Path(cfg.get("wandb_dir", "outputs/wandb/probe")) / dataset_id,
            )

            def build(run_cfg: dict, train_ds):
                ch_names = resolve_ch_names(train_ds, fallback_ch_names)
                model = build_model(run_cfg, make_data_info(train_ds, run_cfg, ch_names), device)
                return model, ch_names

            out_dir = (
                Path(cfg.get("save_dir", "outputs/checkpoints/probe"))
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

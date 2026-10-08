#!/usr/bin/env python
"""Layer-wise probing of a frozen BrainLM-EEG encoder.

Usage
-----
  python experiments/layer_analysis.py --config configs/experiments/layer_analysis_adftd_ad.yaml

  # smoke test
  python experiments/layer_analysis.py --config configs/experiments/layer_analysis_adftd_ad.yaml \\
      --set train_epochs=2 wandb_mode=disabled
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import apply_cli_overrides, load_config
from src.datasets import (
    build_dataloader,
    build_dataset,
    downstream_subject_ids,
    get_rolling_cv_folds,
    load_windows_dataset,
)
from src.engine.ml_eval import aggregate_metrics, log_dl_rolling_cv_results
from src.engine.protocols import ProtocolContext, _persist_folds
from src.engine.trainer import Trainer
from src.models import build_mae_model
from src.models.mae.layer_probe_classifier import LayerProbeClassifier
from src.utils.metrics import prefix_metrics
from src.utils.runtime import finish_wandb, init_wandb, slug, tee_console
from src.utils.seed import seed_everything
from src.utils.subject_votes import append_vote_records

logger = logging.getLogger(__name__)
_SEP = "─" * 90

# Patch embedding, blocks 1-3, final (last block + norm).
DEFAULT_PROBES = [0, 1, 2, 3, -1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BrainLM-EEG layer-wise probing",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", required=True, help="Experiment YAML config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values")
    return p.parse_args()


def probe_label(probe_id: int) -> str:
    return f"layer_{probe_id}" if probe_id >= 0 else "final"


def load_pretrained_encoder(cfg: dict, device: torch.device):
    """Frozen BrainLM-EEG encoder and its channel list from an encoder checkpoint."""
    ckpt = torch.load(cfg["pretrained_encoder_path"], map_location=device, weights_only=False)
    common_ch_names = list(ckpt["common_ch_names"])
    encoder = build_mae_model(cfg, c_common=len(common_ch_names), device=device,
                              common_ch_names=common_ch_names)
    missing, unexpected = encoder.load_state_dict(ckpt["state_dict"], strict=False)
    non_decoder = [k for k in missing if not k.startswith(("enc_to_dec", "mask_token", "dec_", "decoder"))]
    if non_decoder or unexpected:
        raise RuntimeError(f"Encoder checkpoint mismatch: missing {non_decoder}, unexpected {unexpected}")
    logger.info(f"Encoder loaded (epoch={ckpt.get('epoch', '?')}, {len(common_ch_names)} channels)")
    return encoder, common_ch_names


def run_probe(probe_id, *, encoder, n_classes, windows_ds, folds, cfg, out_dir, device, votes_csv, wandb_run):
    """Five-fold CV of one probe; returns the per-fold metric dicts."""
    label = probe_label(probe_id)
    seed = int(cfg.get("seed", 42))
    fold_results, val_sam_l, val_sub_l, test_sam_l, test_sub_l = [], [], [], [], []

    for fold_i, (train_wins, val_wins, test_wins) in enumerate(folds):
        logger.info(_SEP)
        logger.info(f"Probe={label} | lr={cfg['learning_rate']} | fold {fold_i + 1}/{len(folds)}")
        run_cfg = deepcopy(cfg)
        run_cfg["save_dir"] = str(out_dir / label / f"run{fold_i + 1}")

        seed_everything(seed)
        model = LayerProbeClassifier(encoder=encoder, layer_idx=probe_id, n_classes=n_classes,
                                     pool_mode=cfg.get("pool_mode", "mean")).float().to(device)
        loaders = [build_dataloader(build_dataset(windows_ds, w, run_cfg), run_cfg, split=s)
                   for w, s in ((train_wins, "train"), (val_wins, "val"), (test_wins, "test"))]
        trainer = Trainer(model=model, train_loader=loaders[0], val_loader=loaders[1],
                          test_loader=loaders[2], cfg=run_cfg, device=device, wandb_run=None)
        trainer.train()

        _, val_sam, val_sub, val_raw = trainer.evaluate(loaders[1], return_raw=True)
        _, test_sam, test_sub, test_raw = trainer.evaluate(loaders[2], return_raw=True)
        val_sam_l.append(val_sam); test_sam_l.append(test_sam)
        if val_sub:
            val_sub_l.append(val_sub)
        if test_sub:
            test_sub_l.append(test_sub)
        for split, raw in (("val", val_raw), ("test", test_raw)):
            if raw.get("vote_records"):
                append_vote_records(votes_csv, raw["vote_records"], strategy="rolling",
                                    model=f"brainlmeeg_{label}", dataset=cfg.get("dataset_name"),
                                    task=cfg.get("classify_choice"), split=split, run=fold_i + 1)

        fold = {f"val_{k}": v for k, v in val_sam.items()}
        fold.update({f"test_{k}": v for k, v in test_sam.items()})
        fold.update({f"val_subj_{k}": v for k, v in (val_sub or {}).items()})
        fold.update({f"test_subj_{k}": v for k, v in (test_sub or {}).items()})
        fold_results.append(fold)

        if wandb_run is not None:
            payload = {f"{label}/rolling_run": fold_i + 1}
            payload.update(prefix_metrics(val_sam, f"{label}/rolling_run/val/sample"))
            payload.update(prefix_metrics(test_sam, f"{label}/rolling_run/test/sample"))
            if val_sub:
                payload.update(prefix_metrics(val_sub, f"{label}/rolling_run/val/subject"))
            if test_sub:
                payload.update(prefix_metrics(test_sub, f"{label}/rolling_run/test/subject"))
            wandb_run.log(payload)

    log_dl_rolling_cv_results(val_sam_l, val_sub_l, test_sam_l, test_sub_l,
                              aggregate_metrics(val_sam_l), aggregate_metrics(val_sub_l) if val_sub_l else None,
                              aggregate_metrics(test_sam_l), aggregate_metrics(test_sub_l) if test_sub_l else None,
                              n_runs=len(fold_results))
    return fold_results


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    dataset_id = cfg.get("dataset_id", "unknown")
    seed = int(cfg.get("seed", 42))
    task_slug = slug(cfg.get("classify_choice") or cfg.get("dataset_name", dataset_id))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"layer_analysis/brainlmeeg/{cfg.get('dataset_name', dataset_id)}/{task_slug}/seed{seed}/{ts}"
    log_path = Path(cfg.get("terminal_log_dir", "outputs/logs/layer_analysis")) / dataset_id / task_slug / f"brainlmeeg_seed{seed}_{ts}.log"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            seed_everything(seed)
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

            encoder, common_ch_names = load_pretrained_encoder(cfg, device)
            cfg["common_ch_names"] = common_ch_names
            probes = list(cfg.get("layers_to_probe") or DEFAULT_PROBES)

            windows_ds = load_windows_dataset(cfg)
            all_sids = downstream_subject_ids(windows_ds, cfg)
            folds, folds_sids = get_rolling_cv_folds(windows_ds, all_sids, cfg, return_sids=True)
            out_dir = Path(cfg.get("save_dir", "outputs/experiments/layer_analysis")) / dataset_id / task_slug
            out_dir.mkdir(parents=True, exist_ok=True)
            _persist_folds(ProtocolContext(cfg=cfg, windows_ds=windows_ds, device=device, build_model=None,
                                           out_dir=out_dir, file_tag="", vote_model=""), folds_sids, strategy=None)
            n_classes = build_dataset(windows_ds, folds[0][0], cfg).n_classes

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-ssl-layer-analysis",
                wandb_dir=Path(cfg.get("wandb_dir", "outputs/wandb/layer_analysis")) / dataset_id)

            votes_csv = out_dir / "layer_analysis_subject_votes.csv"
            votes_csv.unlink(missing_ok=True)
            summary: dict = {}
            for probe_id in probes:
                folds_out = run_probe(probe_id, encoder=encoder, n_classes=n_classes, windows_ds=windows_ds,
                                      folds=folds, cfg=cfg, out_dir=out_dir, device=device,
                                      votes_csv=votes_csv, wandb_run=wandb_run)
                agg = aggregate_metrics(folds_out)
                label = probe_label(probe_id)
                summary[label] = {k: {"mean": float(m), "std": float(s)} for k, (m, s) in agg.items()}
                summary[label]["learning_rate"] = float(cfg["learning_rate"])
                if wandb_run is not None:
                    payload = {f"{label}/best_lr/{k}_mean": m for k, (m, _) in agg.items()}
                    payload.update({f"{label}/best_lr/{k}_std": s for k, (_, s) in agg.items()})
                    payload[f"{label}/best_lr/lr"] = float(cfg["learning_rate"])
                    wandb_run.log(payload)
                    wandb_run.summary.update(payload)

            summary_path = out_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            logger.info(f"Summary saved → {summary_path}")
            for label, res in summary.items():
                ba = res.get("test_subj_BalancedAccuracy", {})
                ap = res.get("test_subj_AUPRC", {})
                logger.info(f"  {label:<10} test subject BAcc={ba.get('mean', float('nan')):.4f} "
                            f"AUPRC={ap.get('mean', float('nan')):.4f}")
        finally:
            finish_wandb(wandb_run, wandb_module, cfg, run_name=run_name, log_path=log_path, tee_file=tee_file)


if __name__ == "__main__":
    main()

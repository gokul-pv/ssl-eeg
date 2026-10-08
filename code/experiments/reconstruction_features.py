#!/usr/bin/env python
"""Classify BrainLM-EEG reconstructions vs. the original EEG (LDA and ATCNet).

Usage
-----
  python experiments/reconstruction_features.py --config configs/experiments/reconstruction_features_adftd_ad.yaml

Outputs (``<save_dir>``): results.json, reconstruction_subject_votes.csv,
run<N>/dl_{raw,rec}/best.pth. W&B summary keys:
``final/<ml_raw|ml_rec|dl_raw|dl_rec>/test/subject/<metric>_mean``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import apply_cli_overrides, load_config, load_yaml
from src.datasets import (
    build_dataloader,
    build_dataset,
    collect_numpy_split,
    downstream_subject_ids,
    get_rolling_cv_folds,
    get_windows_for_subjects,
    load_windows_dataset,
)
from src.engine.ml_eval import aggregate_metrics, evaluate_ml
from src.engine.protocols import ProtocolContext, _persist_folds
from src.engine.trainer import Trainer
from src.features import extract_features_from_windows
from src.models import build_mae_model, build_model
from src.models.traditional import build_ml_pipeline
from src.utils.metrics import prefix_metrics
from src.utils.runtime import agg_to_json, finish_wandb, init_wandb, slug, tee_console
from src.utils.seed import seed_everything
from src.utils.subject_votes import append_vote_records

logger = logging.getLogger(__name__)
SEP = "─" * 80
CONDITIONS = ["ml_raw", "ml_rec", "dl_raw", "dl_rec"]


class NumpyEEGDataset(Dataset):
    """(X [C,T], label, subject_id) items from numpy arrays — same format as SubjectSplitDataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray):
        self._X = torch.from_numpy(X.astype(np.float32))
        self._y = torch.from_numpy(y.astype(np.int64))
        self._sids = torch.from_numpy(subject_ids.astype(np.int64))

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int):
        return self._X[idx], self._y[idx], self._sids[idx]

    @property
    def n_channels(self) -> int:
        return int(self._X.shape[1])

    @property
    def n_timesteps(self) -> int:
        return int(self._X.shape[2])

    @property
    def n_classes(self) -> int:
        return int(self._y.max().item()) + 1


@torch.no_grad()
def reconstruct(mae_model, dataset, batch_size: int, mask_ratio: float, device) -> np.ndarray:
    """Encoder → decoder reconstruction of every window, in dataset order."""
    parts = []
    for X, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        latent, _, ids_restore = mae_model.encode(X.to(device), mask_ratio=mask_ratio)
        parts.append(mae_model.unpatchify(mae_model.decode(latent, ids_restore)).cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_ml(label, train, val, test, sfreq, cfg, votes, run):
    feats = [extract_features_from_windows(X, sfreq=sfreq, cfg=cfg) for X, _, _ in (train, val, test)]
    pipeline = build_ml_pipeline(cfg)
    pipeline.fit(feats[0], train[1])
    use_vote = bool(cfg.get("use_subject_vote", True))
    out = []
    for split, X_feat, (_, y, sid) in (("val", feats[1], val), ("test", feats[2], test)):
        sam, sub, details = evaluate_ml(pipeline, X_feat, y, sid, f"{label}/{split}", use_vote, return_details=True)
        if details is not None:
            votes(label, split, run, details["records"])
        out.extend([sam, sub or {}])
    return out


def run_dl(label, train, val, test, cfg, save_dir: Path, device, votes, run):
    run_cfg = {**deepcopy(cfg), "save_dir": str(save_dir)}
    loaders = [build_dataloader(NumpyEEGDataset(*arr), run_cfg, s)
               for arr, s in ((train, "train"), (val, "val"), (test, "test"))]
    ds = loaders[0].dataset
    info = {"n_chans": ds.n_channels, "n_classes": ds.n_classes, "n_times": ds.n_timesteps,
            "sfreq": float(run_cfg.get("sfreq", 200))}
    seed_everything(int(cfg.get("seed", 42)))
    trainer = Trainer(model=build_model(run_cfg, info, device), train_loader=loaders[0], val_loader=loaders[1],
                      test_loader=loaders[2], cfg=run_cfg, device=device, wandb_run=None)
    trainer.train()
    out = []
    for split, loader in (("val", loaders[1]), ("test", loaders[2])):
        _, sam, sub, raw = trainer.evaluate(loader, return_raw=True)
        if raw.get("vote_records"):
            votes(label, split, run, raw["vote_records"])
        out.extend([sam, sub or {}])
    return out


def parse_args():
    p = argparse.ArgumentParser(description="BrainLM-EEG reconstruction-as-features analysis",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", required=True, help="Experiment YAML config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    dataset_id = str(cfg.get("dataset_id", "dataset"))
    task_slug = slug(cfg.get("classify_choice") or cfg.get("dataset_name", "task"))
    seed = int(cfg.get("seed", 42))
    sfreq = float(cfg.get("sfreq", 200))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(cfg.get("wandb_run_name") or
                   f"brainlm-recon/{slug(cfg.get('dataset_name', dataset_id))}/{task_slug}/"
                   f"{slug(cfg.get('ml_model', 'lda'))}-{slug(cfg.get('model_name', 'atcnet'))}/{ts}")
    log_path = Path(cfg.get("terminal_log_dir", "outputs/logs/experiments/reconstruction")) / f"{slug(run_name)}.log"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            seed_everything(seed)
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

            # ── Pretrained encoder + decoder ────────────────────────────────
            ckpt = torch.load(cfg["pretrained_checkpoint"], map_location=device, weights_only=False)
            common_ch_names = list(ckpt["common_ch_names"])
            mae_model = build_mae_model(load_yaml(cfg.get("mae_model_config", "configs/model/brainlm_eeg.yaml")),
                                        c_common=len(common_ch_names), device=device,
                                        common_ch_names=common_ch_names)
            mae_model.load_state_dict(ckpt["state_dict"])
            mae_model.eval()
            cfg = {**cfg, "common_ch_names": common_ch_names}

            # ── Downstream pool, reconstructions and folds ──────────────────
            windows_ds = load_windows_dataset(cfg)
            all_sids = downstream_subject_ids(windows_ds, cfg)
            full_ds = build_dataset(windows_ds, get_windows_for_subjects(windows_ds, all_sids), cfg)
            mask_ratio = float(cfg.get("reconstruction_mask_ratio", 0.0))
            X_rec_all = reconstruct(mae_model, full_ds, int(cfg.get("reconstruction_batch_size", 256)),
                                    mask_ratio, device)
            pos = {int(w): i for i, w in enumerate(full_ds.window_indices)}
            del mae_model

            runs, runs_sids = get_rolling_cv_folds(windows_ds, all_sids, cfg, return_sids=True)
            out_dir = Path(cfg.get("save_dir", f"outputs/experiments/reconstruction/{dataset_id}/{task_slug}"))
            out_dir.mkdir(parents=True, exist_ok=True)
            _persist_folds(ProtocolContext(cfg=cfg, windows_ds=windows_ds, device=device, build_model=None,
                                           out_dir=out_dir, file_tag="", vote_model=""), runs_sids, strategy=None)

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-brainlm-reconstruction",
                wandb_dir=cfg.get("wandb_dir", "outputs/wandb/experiments/reconstruction"))

            votes_csv = out_dir / "reconstruction_subject_votes.csv"
            votes_csv.unlink(missing_ok=True)

            def votes(label, split, run, records):
                append_vote_records(votes_csv, records, strategy="rolling", model=f"{label}",
                                    dataset=cfg.get("dataset_name"), task=cfg.get("classify_choice"),
                                    split=split, run=run)

            per_run = {c: {"val_sample": [], "val_subject": [], "test_sample": [], "test_subject": []} for c in CONDITIONS}
            for run_i, (train_win, val_win, test_win) in enumerate(runs):
                logger.info(SEP)
                logger.info(f"ROLLING RUN {run_i + 1}/{len(runs)}")
                seed_everything(seed)
                raw, rec = {}, {}
                for split, wins in (("train", train_win), ("val", val_win), ("test", test_win)):
                    ds = build_dataset(windows_ds, wins, cfg)
                    raw[split] = collect_numpy_split(ds, desc=split)
                    rec[split] = (X_rec_all[[pos[int(w)] for w in ds.window_indices]],) + raw[split][1:]

                results = {
                    "ml_raw": run_ml("ml_raw", raw["train"], raw["val"], raw["test"], sfreq, cfg, votes, run_i + 1),
                    "ml_rec": run_ml("ml_rec", rec["train"], rec["val"], rec["test"], sfreq, cfg, votes, run_i + 1),
                    "dl_raw": run_dl("dl_raw", raw["train"], raw["val"], raw["test"], cfg,
                                     out_dir / f"run{run_i + 1}" / "dl_raw", device, votes, run_i + 1),
                    "dl_rec": run_dl("dl_rec", rec["train"], rec["val"], rec["test"], cfg,
                                     out_dir / f"run{run_i + 1}" / "dl_rec", device, votes, run_i + 1),
                }
                payload = {"rolling_run": run_i + 1}
                for cond, (vsam, vsub, tsam, tsub) in results.items():
                    for key, val in zip(("val_sample", "val_subject", "test_sample", "test_subject"),
                                        (vsam, vsub, tsam, tsub)):
                        per_run[cond][key].append(val)
                    payload.update(prefix_metrics(vsub, f"rolling_run/{cond}/val/subject"))
                    payload.update(prefix_metrics(tsub, f"rolling_run/{cond}/test/subject"))
                if wandb_run is not None:
                    wandb_run.log(payload)

            record = {"dataset": cfg.get("dataset_name", dataset_id), "task": cfg.get("classify_choice"),
                      "ml_model": cfg.get("ml_model", "lda"), "dl_model": slug(cfg.get("model_name", "atcnet")),
                      "n_runs": len(runs), "mask_ratio": mask_ratio, "conditions": {}}
            final = {}
            for cond, res in per_run.items():
                agg = {k: aggregate_metrics([m for m in res[k] if m]) for k in res}
                record["conditions"][cond] = {**{f"per_run_{k}": v for k, v in res.items()},
                                              **{f"aggregated_{k}": agg_to_json(v) for k, v in agg.items()}}
                for split in ("test", "val"):
                    for name, (mean, std) in agg[f"{split}_subject"].items():
                        final[f"final/{cond}/{split}/subject/{name}_mean"] = mean
                        final[f"final/{cond}/{split}/subject/{name}_std"] = std
                tsub = agg["test_subject"]
                logger.info(f"  {cond:<7} test subject BAcc={tsub.get('BalancedAccuracy', (np.nan,))[0]:.4f} "
                            f"AUPRC={tsub.get('AUPRC', (np.nan,))[0]:.4f}")
            results_path = out_dir / "results.json"
            results_path.write_text(json.dumps(record, indent=2))
            logger.info(f"Results saved → {results_path}")
            if wandb_run is not None:
                wandb_run.log(final)
                wandb_run.summary.update(final)
                wandb_run.summary["results_path"] = str(results_path)
        finally:
            finish_wandb(wandb_run, wandb_module, cfg, run_name=run_name, log_path=log_path, tee_file=tee_file)


if __name__ == "__main__":
    main()

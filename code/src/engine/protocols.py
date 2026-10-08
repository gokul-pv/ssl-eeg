"""Evaluation protocols shared by every deep-learning entrypoint
(``finetune.py``, ``probe.py``, ``train_dl.py``).
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.datasets import (
    _build_subject_to_windows,
    build_dataloader,
    build_dataset,
    downstream_subject_ids,
    get_loso_folds,
    get_rolling_cv_folds,
    save_rolling_cv_split,
)
from src.datasets.split_io import _cv_split_path
from src.engine.ml_eval import aggregate_metrics, log_dl_rolling_cv_results
from src.engine.trainer import Trainer
from src.utils.metrics import compute_subject_metrics, fmt_metrics, prefix_metrics
from src.utils.runtime import agg_to_json
from src.utils.seed import seed_everything
from src.utils.subject_votes import (
    append_vote_records,
    log_vote_report,
    vote_records_to_wandb_table,
    vote_summary_to_wandb,
)

logger = logging.getLogger(__name__)

PROTOCOLS = ("rolling", "loso", "full")

# build_model(run_cfg, train_ds) -> (model, ch_names passed to model.forward or None)
ModelBuilder = Callable[[dict, object], "tuple[nn.Module, list[str] | None]"]


def resolve_protocol(cfg: dict) -> str:
    """``protocol`` key, or the legacy ``train_full`` / ``cv_strategy`` keys."""
    protocol = cfg.get("protocol")
    if protocol is None:
        if bool(cfg.get("train_full", False)):
            protocol = "full"
        elif cfg.get("cv_strategy") == "loso":
            protocol = "loso"
        else:
            protocol = "rolling"
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got '{protocol}'.")
    return protocol


@dataclass
class ProtocolContext:
    """Everything a protocol runner needs; built by the entrypoint."""

    cfg: dict
    windows_ds: object
    device: torch.device
    build_model: ModelBuilder
    out_dir: Path            # per-seed output directory (checkpoints + results)
    file_tag: str            # prefix of result files, e.g. "brainlmeeg_linear_probe"
    vote_model: str          # "model" column of the subject-vote CSV
    results_meta: dict = field(default_factory=dict)   # extra fields of the JSON record
    wandb_run: object = None

    @property
    def seed(self) -> int:
        return int(self.cfg.get("seed", 42))

    @property
    def dataset_id(self) -> str:
        return str(self.cfg.get("dataset_id", "unknown"))

    @property
    def task(self):
        return self.cfg.get("classify_choice")


def run_protocol(ctx: ProtocolContext) -> None:
    protocol = resolve_protocol(ctx.cfg)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Protocol: {protocol} | outputs → {ctx.out_dir}")
    {"rolling": run_rolling, "loso": run_loso, "full": run_full}[protocol](ctx)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _persist_folds(ctx: ProtocolContext, runs_sids, strategy: str | None) -> None:
    """Save the participant folds, or — if a fold file already exists — check that
    it holds exactly the folds this run uses. (The original code skipped saving
    when a file existed, so a stale file could silently disagree with the folds
    actually trained on.)
    """
    out_dir = ctx.cfg.get("split_save_dir", "metadata")
    path = _cv_split_path(ctx.dataset_id, ctx.seed, out_dir, task=ctx.task, strategy=strategy)
    if path.exists():
        with open(path) as f:
            saved = json.load(f)["folds"]
        current = [
            {k: sorted(int(s) for s in sids) for k, sids in
             zip(("train_subject_ids", "val_subject_ids", "test_subject_ids"), run)}
            for run in runs_sids
        ]
        saved = [{k: fold[k] for k in current[0]} for fold in saved] if current else saved
        if saved != current:
            raise RuntimeError(
                f"Saved folds in {path} differ from the folds generated for this run. "
                "Delete the file to regenerate it, or check split_save_dir / seed / task."
            )
        logger.info(f"Folds match the saved split file {path}.")
        return
    save_rolling_cv_split(
        dataset_id=ctx.dataset_id, seed=ctx.seed, runs_sids=runs_sids,
        task=ctx.task, out_dir=out_dir, strategy=strategy,
    )


def _build_run(ctx: ProtocolContext, run_cfg: dict, train_win, val_win, test_win):
    train_ds = build_dataset(ctx.windows_ds, train_win, run_cfg)
    val_ds = build_dataset(ctx.windows_ds, val_win, run_cfg)
    test_ds = build_dataset(ctx.windows_ds, test_win, run_cfg)
    loaders = (
        build_dataloader(train_ds, run_cfg, split="train"),
        build_dataloader(val_ds, run_cfg, split="val"),
        build_dataloader(test_ds, run_cfg, split="test"),
    )
    logger.info(
        f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,} windows | "
        f"n_channels={train_ds.n_channels} n_timesteps={train_ds.n_timesteps} "
        f"n_classes={train_ds.n_classes}"
    )
    seed_everything(ctx.seed)                      # fresh, identically seeded model per run
    model, ch_names = ctx.build_model(run_cfg, train_ds)
    trainer = Trainer(
        model=model, train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2],
        cfg=run_cfg, device=ctx.device, wandb_run=None, ch_names=ch_names,
    )
    return train_ds, loaders, trainer


def _train(ctx: ProtocolContext, trainer: Trainer, run_label: str):
    try:
        return trainer.train()
    except Exception:
        if not bool(ctx.cfg.get("skip_failed_runs", False)):
            raise
        logger.error(f"{run_label} training failed — skipped (skip_failed_runs=true).", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Five-fold rolling cross-validation
# ---------------------------------------------------------------------------


def run_rolling(ctx: ProtocolContext) -> None:
    cfg, windows_ds, wandb_run = ctx.cfg, ctx.windows_ds, ctx.wandb_run
    all_sids = downstream_subject_ids(windows_ds, cfg)
    runs, runs_sids = get_rolling_cv_folds(windows_ds, all_sids, cfg, return_sids=True)
    _persist_folds(ctx, runs_sids, strategy=None)

    run_val_sample_metrics: list[dict] = []
    run_val_subject_metrics: list[dict] = []
    run_test_sample_metrics: list[dict] = []
    run_test_subject_metrics: list[dict] = []
    run_val_vote_summaries: list[dict] = []
    run_test_vote_summaries: list[dict] = []

    votes_csv = ctx.out_dir / f"{ctx.file_tag}_rolling_subject_votes.csv"
    votes_csv.unlink(missing_ok=True)
    vote_ctx = dict(strategy="rolling", model=ctx.vote_model,
                    dataset=cfg.get("dataset_name", ctx.dataset_id), task=ctx.task)

    for run_i, (train_win, val_win, test_win) in enumerate(runs):
        logger.info("=" * 70)
        logger.info(f"ROLLING RUN {run_i + 1}/{len(runs)}")
        logger.info("=" * 70)
        if not train_win or not val_win or not test_win:
            raise RuntimeError(f"Run {run_i + 1}: empty train/val/test window list.")

        run_cfg = deepcopy(cfg)
        run_cfg["save_dir"] = str(ctx.out_dir / f"run{run_i + 1}")
        _, (_, val_loader, test_loader), trainer = _build_run(ctx, run_cfg, train_win, val_win, test_win)
        best_metric = _train(ctx, trainer, f"Run {run_i + 1}")
        if best_metric is None:
            continue

        # Evaluate with the selected (best-validation) checkpoint
        _, val_sam, val_sub, val_raw = trainer.evaluate(val_loader, return_raw=True)
        _, test_sam, test_sub, test_raw = trainer.evaluate(test_loader, return_raw=True)
        run_val_sample_metrics.append(val_sam)
        if val_sub:
            run_val_subject_metrics.append(val_sub)
        run_test_sample_metrics.append(test_sam)
        if test_sub:
            run_test_subject_metrics.append(test_sub)

        run_votes = {"val": val_raw, "test": test_raw}
        for split, raw in run_votes.items():
            if not raw.get("vote_records"):
                continue
            log_vote_report(f"{split}/run{run_i + 1}", raw["vote_records"], raw["vote_summary"])
            append_vote_records(votes_csv, raw["vote_records"], split=split, run=run_i + 1, **vote_ctx)
        if val_raw.get("vote_summary"):
            run_val_vote_summaries.append(val_raw["vote_summary"])
        if test_raw.get("vote_summary"):
            run_test_vote_summaries.append(test_raw["vote_summary"])
        logger.info(f"  Run {run_i + 1} complete. best_metric={best_metric:.4f}")

        if wandb_run is not None:
            payload: dict = {"rolling_run": run_i + 1}
            payload.update(prefix_metrics(val_sam, "rolling_run/val/sample"))
            payload.update(prefix_metrics(test_sam, "rolling_run/test/sample"))
            if val_sub:
                payload.update(prefix_metrics(val_sub, "rolling_run/val/subject"))
            if test_sub:
                payload.update(prefix_metrics(test_sub, "rolling_run/test/subject"))
            for split, raw in run_votes.items():
                if not raw.get("vote_records"):
                    continue
                payload.update(vote_summary_to_wandb(raw["vote_summary"], f"rolling_run/{split}"))
                table = vote_records_to_wandb_table(raw["vote_records"])
                if table is not None:
                    payload[f"rolling_run/{split}/subject_votes"] = table
            wandb_run.log(payload)

    n_completed = len(run_val_sample_metrics)
    if n_completed == 0:
        logger.error("No runs completed — nothing to aggregate.")
        return

    agg_val_sample = aggregate_metrics(run_val_sample_metrics)
    agg_val_subject = aggregate_metrics(run_val_subject_metrics) if run_val_subject_metrics else None
    agg_test_sample = aggregate_metrics(run_test_sample_metrics)
    agg_test_subject = aggregate_metrics(run_test_subject_metrics) if run_test_subject_metrics else None
    agg_val_vote = aggregate_metrics(run_val_vote_summaries) if run_val_vote_summaries else None
    agg_test_vote = aggregate_metrics(run_test_vote_summaries) if run_test_vote_summaries else None

    log_dl_rolling_cv_results(
        run_val_sample_metrics, run_val_subject_metrics or [],
        run_test_sample_metrics, run_test_subject_metrics or [],
        agg_val_sample, agg_val_subject, agg_test_sample, agg_test_subject,
        n_runs=n_completed,
    )

    results_path = ctx.out_dir / f"{ctx.file_tag}_rolling_cv_results.json"
    record = {
        "n_runs": n_completed,
        "cv_strategy": "rolling",
        **ctx.results_meta,
        "dataset": cfg.get("dataset_name", ctx.dataset_id),
        "task": ctx.task,
        "seed": ctx.seed,
        "per_run_val_sample": run_val_sample_metrics,
        "per_run_val_subject": run_val_subject_metrics,
        "per_run_test_sample": run_test_sample_metrics,
        "per_run_test_subject": run_test_subject_metrics,
        "aggregated_val_sample": agg_to_json(agg_val_sample),
        "aggregated_val_subject": agg_to_json(agg_val_subject),
        "aggregated_test_sample": agg_to_json(agg_test_sample),
        "aggregated_test_subject": agg_to_json(agg_test_subject),
        "per_run_val_vote_summary": run_val_vote_summaries,
        "per_run_test_vote_summary": run_test_vote_summaries,
        "aggregated_val_vote": agg_to_json(agg_val_vote),
        "aggregated_test_vote": agg_to_json(agg_test_vote),
        "subject_votes_csv": str(votes_csv),
    }
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Rolling CV results saved → {results_path}")

    if agg_test_vote:
        print("  Test subject-vote decisiveness mean±std :")
        for name, (mean, std) in agg_test_vote.items():
            print(f"    {name:<28}: {mean:.4f} ± {std:.4f}")
        print("=" * 90)

    if wandb_run is not None:
        payload = {}
        for prefix, agg in (
            ("rolling_test/sample", agg_test_sample), ("rolling_test/subject", agg_test_subject),
            ("rolling_val/sample", agg_val_sample), ("rolling_val/subject", agg_val_subject),
            ("rolling_test/vote", agg_test_vote), ("rolling_val/vote", agg_val_vote),
        ):
            for name, (mean, std) in (agg or {}).items():
                payload[f"{prefix}/{name}_mean"] = mean
                payload[f"{prefix}/{name}_std"] = std
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        wandb_run.summary["subject_votes_csv"] = str(votes_csv)
        wandb_run.summary["rolling_cv_results_path"] = str(results_path)


# ---------------------------------------------------------------------------
# Leave-one-subject-out
# ---------------------------------------------------------------------------


def run_loso(ctx: ProtocolContext) -> None:
    """One run per participant; the participant-level predictions of all runs are
    pooled and scored once (per-run metrics beyond accuracy are undefined, as
    each test set is a single participant of a single class).
    """
    cfg, windows_ds, wandb_run = ctx.cfg, ctx.windows_ds, ctx.wandb_run
    all_sids = downstream_subject_ids(windows_ds, cfg)
    runs, runs_sids = get_loso_folds(windows_ds, all_sids, cfg, return_sids=True)
    _persist_folds(ctx, runs_sids, strategy="loso")

    pooled_preds, pooled_y, pooled_sids, pooled_probs = [], [], [], []
    n_classes_seen: int | None = None
    n_completed = 0

    for run_i, (train_win, val_win, test_win) in enumerate(runs):
        logger.info("=" * 70)
        logger.info(f"LOSO RUN {run_i + 1}/{len(runs)}")
        logger.info("=" * 70)
        if not train_win or not val_win or not test_win:
            raise RuntimeError(f"Run {run_i + 1}: empty train/val/test window list.")

        run_cfg = deepcopy(cfg)
        run_cfg["save_dir"] = str(ctx.out_dir / "loso" / f"run{run_i + 1}")
        train_ds, (_, _, test_loader), trainer = _build_run(ctx, run_cfg, train_win, val_win, test_win)
        best_metric = _train(ctx, trainer, f"Run {run_i + 1}")
        if best_metric is None:
            continue

        _, _, _, test_raw = trainer.evaluate(test_loader, return_raw=True)
        pooled_preds.append(test_raw["preds"])
        pooled_y.append(test_raw["y_true"])
        pooled_sids.append(test_raw["sids"])
        pooled_probs.append(test_raw["probs"])
        n_classes_seen = train_ds.n_classes
        n_completed += 1
        logger.info(f"  Run {run_i + 1} complete. best_metric={best_metric:.4f}")
        if wandb_run is not None:
            wandb_run.log({"loso_run": run_i + 1, "loso_run/best_metric": best_metric})

    if n_completed == 0:
        logger.error("No runs completed — nothing to aggregate.")
        return

    pooled_subject, vote_records, vote_summary = compute_subject_metrics(
        np.concatenate(pooled_preds), np.concatenate(pooled_y), np.concatenate(pooled_sids),
        num_classes=n_classes_seen, probs=np.concatenate(pooled_probs), return_details=True,
    )
    print()
    print("=" * 90)
    print(f"  LOSO RESULTS — Pooled Subject-Level (N={n_completed} subjects)")
    print("=" * 90)
    print(f"  {fmt_metrics(pooled_subject)}")
    print("=" * 90)
    log_vote_report("loso_test", vote_records, vote_summary, num_classes=n_classes_seen)

    votes_csv = ctx.out_dir / f"{ctx.file_tag}_loso_subject_votes.csv"
    votes_csv.unlink(missing_ok=True)
    append_vote_records(
        votes_csv, vote_records, strategy="loso", model=ctx.vote_model,
        dataset=cfg.get("dataset_name", ctx.dataset_id), task=ctx.task,
        split="pooled_test", run="pooled",
    )
    logger.info(f"LOSO per-subject votes saved → {votes_csv}")

    results_path = ctx.out_dir / f"{ctx.file_tag}_loso_cv_results.json"
    record = {
        "n_runs": n_completed,
        "cv_strategy": "loso",
        **ctx.results_meta,
        "dataset": cfg.get("dataset_name", ctx.dataset_id),
        "task": ctx.task,
        "seed": ctx.seed,
        "pooled_test_subject": pooled_subject,
        "pooled_test_vote_summary": vote_summary,
        "subject_votes_csv": str(votes_csv),
    }
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"LOSO CV results saved → {results_path}")

    if wandb_run is not None:
        payload = {f"loso_test/pooled_subject/{k}": v for k, v in pooled_subject.items()}
        payload.update(vote_summary_to_wandb(vote_summary, "loso_test/pooled_subject"))
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        table = vote_records_to_wandb_table(vote_records)
        if table is not None:
            wandb_run.log({"loso_test/pooled_subject/subject_votes": table})
        wandb_run.summary["loso_cv_results_path"] = str(results_path)
        wandb_run.summary["subject_votes_csv"] = str(votes_csv)


# ---------------------------------------------------------------------------
# Full downstream pool (FEP → SCZ transfer model)
# ---------------------------------------------------------------------------


def run_full(ctx: ProtocolContext) -> None:
    """Train one model on every participant of the downstream pool. Validation and
    test loaders are the training data itself, so early stopping monitors the
    in-sample loss (there is no held-out participant). The checkpoint is used for
    the external FEP → SCZ evaluation (``evaluate.py``).
    """
    cfg, windows_ds = ctx.cfg, ctx.windows_ds
    pool = set(downstream_subject_ids(windows_ds, cfg))
    sid_to_wins = _build_subject_to_windows(windows_ds)
    all_win_idx = [w for sid in sorted(sid_to_wins) if sid in pool for w in sid_to_wins[sid]]
    logger.info(f"Full-train: {len(all_win_idx)} windows from {len(pool)} downstream participants")

    full_cfg = deepcopy(cfg)
    full_cfg["save_dir"] = str(ctx.out_dir / "full")
    all_ds = build_dataset(windows_ds, all_win_idx, full_cfg)
    all_loader = build_dataloader(all_ds, full_cfg, split="train")
    logger.info(
        f"  windows={len(all_ds):,} n_channels={all_ds.n_channels} "
        f"n_timesteps={all_ds.n_timesteps} n_classes={all_ds.n_classes}"
    )

    seed_everything(ctx.seed)
    model, ch_names = ctx.build_model(full_cfg, all_ds)
    trainer = Trainer(
        model=model,
        train_loader=all_loader,
        val_loader=all_loader,    # in-sample by design (no held-out participants)
        test_loader=all_loader,
        cfg=full_cfg,
        device=ctx.device,
        wandb_run=ctx.wandb_run,
        ch_names=ch_names,
    )
    best_metric = trainer.train()
    logger.info(
        f"Full-train complete. Best {full_cfg.get('early_stopping_metric', 'val_loss')} = "
        f"{best_metric:.4f} | checkpoint → {trainer.ckpt_path}"
    )
    if ctx.wandb_run is not None:
        ctx.wandb_run.summary["full_train_best_metric"] = best_metric
        ctx.wandb_run.summary["full_train_checkpoint"] = str(trainer.ckpt_path)

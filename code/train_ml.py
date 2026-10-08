#!/usr/bin/env python
"""Classical machine-learning baselines — LDA, linear SVM, XGBoost on
handcrafted EEG features.

Usage
-----
  python train_ml.py --config configs/train_ml/baseline_lda_adftd.yaml        # five-fold CV
  python train_ml.py --config configs/train_ml/baseline_svm_mdd_loso.yaml     # LOSO
  python train_ml.py --config configs/train_ml/baseline_xgb_fep_full.yaml     # all FEP (→ SCZ)

Protocols (``protocol: rolling | loso | full``; legacy ``cv_strategy`` /
``train_full`` keys accepted):
  rolling : five-fold CV; the validation fold is reported but not used
            (classical pipelines have no early stopping).
  loso    : one test participant per run (the rotating validation participant
            is left out of training, as in the deep-learning LOSO protocol);
            pooled participant-level metrics.
  full    : one pipeline fitted on the whole downstream pool (FEP → SCZ).

Outputs: <save_dir>/<ml_model>/<dataset_id>/<task>/seed<seed>/...
Features are cached under <feature_cache_dir>/<dataset_id>/<task>/<key>/.
This entrypoint replaces the original ``baseline_ml_cv.py`` and ``baseline_ml.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import apply_cli_overrides, load_config
from src.datasets import (
    build_dataset,
    collect_numpy_split,
    downstream_subject_ids,
    get_loso_folds,
    get_rolling_cv_folds,
    load_windows_dataset,
)
from src.engine.ml_eval import (
    _probs_in_label_order,
    aggregate_metrics,
    evaluate_ml,
    log_rolling_cv_results,
)
from src.engine.protocols import ProtocolContext, _persist_folds, resolve_protocol
from src.features import extract_features_from_windows
from src.features.cache import feature_cache_key
from src.models.traditional import build_ml_pipeline
from src.utils.metrics import compute_subject_metrics, fmt_metrics, prefix_metrics
from src.utils.runtime import agg_to_json, finish_wandb, init_wandb, slug, tee_console
from src.utils.seed import seed_everything
from src.utils.subject_votes import (
    append_vote_records,
    log_vote_report,
    vote_records_to_wandb_table,
    vote_summary_to_wandb,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical ML baselines (LDA / SVM / XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Baseline YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set ml_model=svm wandb_mode=disabled",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def load_features(cfg: dict, windows_ds):
    """Features of every window of the task (all participants, including the
    pretraining reserve, which is removed later by the protocol), with their
    labels, participant IDs and a map from global window index → feature row.
    """
    all_ds = build_dataset(windows_ds, list(range(len(windows_ds))), cfg)
    sfreq = float(cfg.get("sfreq", 200))
    task_slug = slug(cfg.get("classify_choice") or cfg.get("dataset_name", "task"))
    cache_dir = (
        Path(cfg.get("feature_cache_dir", "outputs/features"))
        / str(cfg.get("dataset_id")) / task_slug / feature_cache_key(cfg)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "all_features.npz"

    if cache_path.exists():
        logger.info(f"Loading features from cache: {cache_path}")
        cached = np.load(cache_path)
        X_all, y_all, sid_all, win_idx = cached["X"], cached["y"], cached["sid"], cached["win"]
        if not np.array_equal(win_idx, np.asarray(all_ds.window_indices)):
            raise RuntimeError(f"Feature cache {cache_path} does not match the loaded windows.")
    else:
        X_raw, y_all, sid_all = collect_numpy_split(all_ds, desc="all")
        logger.info("Extracting features ...")
        X_all = extract_features_from_windows(X_raw, sfreq=sfreq, cfg=cfg)
        win_idx = np.asarray(all_ds.window_indices)
        np.savez(cache_path, X=X_all, y=y_all, sid=sid_all, win=win_idx)
        logger.info(f"Saved features to cache: {cache_path}")

    pos_map = {int(w): pos for pos, w in enumerate(win_idx)}
    info = {"n_windows": len(y_all), "n_classes": all_ds.n_classes, "n_features": X_all.shape[1]}
    logger.info(f"Features: {X_all.shape} | n_classes={info['n_classes']}")
    return X_all, y_all, sid_all, pos_map, info


def _positions(pos_map: dict[int, int], windows) -> np.ndarray:
    return np.array([pos_map[w] for w in windows if w in pos_map], dtype=np.int64)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def run_rolling(ctx: ProtocolContext, feats) -> None:
    cfg, wandb_run, ml_model = ctx.cfg, ctx.wandb_run, ctx.file_tag
    X_all, y_all, sid_all, pos_map, info = feats
    use_vote = bool(cfg.get("use_subject_vote", True))

    all_sids = downstream_subject_ids(ctx.windows_ds, cfg)
    runs, runs_sids = get_rolling_cv_folds(ctx.windows_ds, all_sids, cfg, return_sids=True)
    _persist_folds(ctx, runs_sids, strategy=None)

    val_sam_l, val_sub_l, test_sam_l, test_sub_l, val_vote_l, test_vote_l = [], [], [], [], [], []
    votes_csv = ctx.out_dir / f"{ml_model}_rolling_subject_votes.csv"
    votes_csv.unlink(missing_ok=True)

    for run_i, (train_win, val_win, test_win) in enumerate(runs):
        logger.info("=" * 70)
        logger.info(f"ROLLING RUN {run_i + 1}/{len(runs)}")
        logger.info("=" * 70)
        tr, va, te = (_positions(pos_map, w) for w in (train_win, val_win, test_win))
        if tr.size == 0 or va.size == 0 or te.size == 0:
            raise RuntimeError(f"Run {run_i + 1}: empty train/val/test after mapping.")
        logger.info(f"  train: {tr.size} windows  val: {va.size} windows  test: {te.size} windows")

        pipeline = build_ml_pipeline(cfg)
        pipeline.fit(X_all[tr], y_all[tr])
        tr_sam, tr_sub, tr_votes = evaluate_ml(pipeline, X_all[tr], y_all[tr], sid_all[tr],
                                               f"train/run{run_i + 1}", use_vote, return_details=True)
        val_sam, val_sub, val_votes = evaluate_ml(pipeline, X_all[va], y_all[va], sid_all[va],
                                                  f"val/run{run_i + 1}", use_vote, return_details=True)
        test_sam, test_sub, test_votes = evaluate_ml(pipeline, X_all[te], y_all[te], sid_all[te],
                                                     f"test/run{run_i + 1}", use_vote, return_details=True)
        val_sam_l.append(val_sam)
        test_sam_l.append(test_sam)
        if val_sub:
            val_sub_l.append(val_sub)
        if test_sub:
            test_sub_l.append(test_sub)

        vote_ctx = dict(strategy="rolling", model=ml_model, dataset=cfg.get("dataset_name", ctx.dataset_id),
                        task=ctx.task, run=run_i + 1)
        for split, details in (("train", tr_votes), ("val", val_votes), ("test", test_votes)):
            if details is not None:
                append_vote_records(votes_csv, details["records"], split=split, **vote_ctx)
        if val_votes is not None:
            val_vote_l.append(val_votes["summary"])
        if test_votes is not None:
            test_vote_l.append(test_votes["summary"])

        ckpt_path = ctx.out_dir / f"run{run_i + 1}" / f"{ml_model}_pipeline.pkl"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ckpt_path)
        logger.info(f"Run {run_i + 1} pipeline saved → {ckpt_path}")

        if wandb_run is not None:
            payload = {"rolling_run": run_i + 1}
            for split, sam, sub in (("train", tr_sam, tr_sub), ("val", val_sam, val_sub), ("test", test_sam, test_sub)):
                payload.update(prefix_metrics(sam, f"rolling_run/{split}/sample"))
                if sub:
                    payload.update(prefix_metrics(sub, f"rolling_run/{split}/subject"))
            for split, details in (("train", tr_votes), ("val", val_votes), ("test", test_votes)):
                if details is None:
                    continue
                payload.update(vote_summary_to_wandb(details["summary"], f"rolling_run/{split}"))
                table = vote_records_to_wandb_table(details["records"])
                if table is not None:
                    payload[f"rolling_run/{split}/subject_votes"] = table
            wandb_run.log(payload)

    agg = {
        "val_sample": aggregate_metrics(val_sam_l),
        "val_subject": aggregate_metrics(val_sub_l) if val_sub_l else None,
        "test_sample": aggregate_metrics(test_sam_l),
        "test_subject": aggregate_metrics(test_sub_l) if test_sub_l else None,
        "val_vote": aggregate_metrics(val_vote_l) if val_vote_l else None,
        "test_vote": aggregate_metrics(test_vote_l) if test_vote_l else None,
    }
    log_rolling_cv_results(val_sam_l, val_sub_l, test_sam_l, test_sub_l,
                           agg["val_sample"], agg["val_subject"], agg["test_sample"], agg["test_subject"],
                           n_runs=len(runs))

    results_path = ctx.out_dir / f"{ml_model}_rolling_cv_results.json"
    record = {
        "n_runs": len(runs), "cv_strategy": "rolling", "ml_model": ml_model,
        "dataset": cfg.get("dataset_name", ctx.dataset_id), "task": ctx.task, "seed": ctx.seed,
        "per_run_val_sample": val_sam_l, "per_run_val_subject": val_sub_l,
        "per_run_test_sample": test_sam_l, "per_run_test_subject": test_sub_l,
        **{f"aggregated_{k}": agg_to_json(v) for k, v in agg.items()},
        "per_run_val_vote_summary": val_vote_l, "per_run_test_vote_summary": test_vote_l,
        "subject_votes_csv": str(votes_csv),
    }
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Rolling CV results saved → {results_path}")

    if wandb_run is not None:
        payload = {}
        for prefix, a in (("rolling_test/sample", agg["test_sample"]), ("rolling_test/subject", agg["test_subject"]),
                          ("rolling_val/sample", agg["val_sample"]), ("rolling_val/subject", agg["val_subject"]),
                          ("rolling_test/vote", agg["test_vote"]), ("rolling_val/vote", agg["val_vote"])):
            for name, (mean, std) in (a or {}).items():
                payload[f"{prefix}/{name}_mean"] = mean
                payload[f"{prefix}/{name}_std"] = std
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        wandb_run.summary["rolling_cv_results_path"] = str(results_path)
        wandb_run.summary["subject_votes_csv"] = str(votes_csv)


def run_loso(ctx: ProtocolContext, feats) -> None:
    cfg, wandb_run, ml_model = ctx.cfg, ctx.wandb_run, ctx.file_tag
    X_all, y_all, sid_all, pos_map, info = feats
    n_classes = info["n_classes"]

    all_sids = downstream_subject_ids(ctx.windows_ds, cfg)
    runs, runs_sids = get_loso_folds(ctx.windows_ds, all_sids, cfg, return_sids=True)
    _persist_folds(ctx, runs_sids, strategy="loso")

    preds_l, y_l, sid_l, probs_l = [], [], [], []
    for run_i, (train_win, val_win, test_win) in enumerate(runs):
        logger.info(f"LOSO RUN {run_i + 1}/{len(runs)}")
        tr, te = _positions(pos_map, train_win), _positions(pos_map, test_win)
        if not val_win or tr.size == 0 or te.size == 0:
            raise RuntimeError(f"Run {run_i + 1}: empty train/val/test window list.")
        pipeline = build_ml_pipeline(cfg)
        pipeline.fit(X_all[tr], y_all[tr])
        preds = pipeline.predict(X_all[te])
        probs = _probs_in_label_order(pipeline, X_all[te], preds, getattr(pipeline, "classes_", None), n_classes)
        preds_l.append(preds)
        y_l.append(y_all[te])
        sid_l.append(sid_all[te])
        probs_l.append(probs)
        ckpt_path = ctx.out_dir / "loso" / f"run{run_i + 1}" / f"{ml_model}_pipeline.pkl"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ckpt_path)
        if wandb_run is not None:
            wandb_run.log({"loso_run": run_i + 1})

    pooled, vote_records, vote_summary = compute_subject_metrics(
        np.concatenate(preds_l), np.concatenate(y_l), np.concatenate(sid_l),
        num_classes=n_classes, probs=np.concatenate(probs_l), return_details=True,
    )
    print()
    print("=" * 90)
    print(f"  LOSO RESULTS — Pooled Subject-Level (N={len(preds_l)} subjects)")
    print("=" * 90)
    print(f"  {fmt_metrics(pooled)}")
    print("=" * 90)
    log_vote_report("loso_test", vote_records, vote_summary, num_classes=n_classes)

    votes_csv = ctx.out_dir / f"{ml_model}_loso_subject_votes.csv"
    votes_csv.unlink(missing_ok=True)
    append_vote_records(votes_csv, vote_records, strategy="loso", model=ml_model,
                        dataset=cfg.get("dataset_name", ctx.dataset_id), task=ctx.task,
                        split="pooled_test", run="pooled")
    results_path = ctx.out_dir / f"{ml_model}_loso_cv_results.json"
    record = {
        "n_runs": len(preds_l), "cv_strategy": "loso", "ml_model": ml_model,
        "dataset": cfg.get("dataset_name", ctx.dataset_id), "task": ctx.task, "seed": ctx.seed,
        "pooled_test_subject": pooled, "pooled_test_vote_summary": vote_summary,
        "subject_votes_csv": str(votes_csv),
    }
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"LOSO CV results saved → {results_path}")

    if wandb_run is not None:
        payload = {f"loso_test/pooled_subject/{k}": v for k, v in pooled.items()}
        payload.update(vote_summary_to_wandb(vote_summary, "loso_test/pooled_subject"))
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        table = vote_records_to_wandb_table(vote_records)
        if table is not None:
            wandb_run.log({"loso_test/pooled_subject/subject_votes": table})
        wandb_run.summary["loso_cv_results_path"] = str(results_path)
        wandb_run.summary["subject_votes_csv"] = str(votes_csv)


def run_full(ctx: ProtocolContext, feats) -> None:
    cfg, wandb_run, ml_model = ctx.cfg, ctx.wandb_run, ctx.file_tag
    X_all, y_all, sid_all, _, _ = feats
    pool = set(downstream_subject_ids(ctx.windows_ds, cfg))
    mask = np.array([int(s) in pool for s in sid_all], dtype=bool)
    X, y, sid = X_all[mask], y_all[mask], sid_all[mask]
    logger.info(f"Full-train: {len(y)} windows from {len(pool)} downstream participants")

    pipeline = build_ml_pipeline(cfg)
    pipeline.fit(X, y)
    use_vote = bool(cfg.get("use_subject_vote", True))
    tr_sam, tr_sub, tr_votes = evaluate_ml(pipeline, X, y, sid, "full_train", use_vote, return_details=True)
    logger.info(f"  In-sample | sample : {fmt_metrics(tr_sam)}")
    if tr_sub:
        logger.info(f"  In-sample | subject: {fmt_metrics(tr_sub)}")

    full_dir = ctx.out_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    if tr_votes is not None:
        votes_csv = full_dir / f"{ml_model}_full_train_subject_votes.csv"
        votes_csv.unlink(missing_ok=True)
        append_vote_records(votes_csv, tr_votes["records"], strategy="full_train", model=ml_model,
                            dataset=cfg.get("dataset_name", ""), task=ctx.task or "",
                            split="full_train", run="full")
    ckpt_path = full_dir / f"{ml_model}_full_pipeline.pkl"
    joblib.dump(pipeline, ckpt_path)
    logger.info(f"Full-train pipeline saved → {ckpt_path}")

    if wandb_run is not None:
        payload = dict(prefix_metrics(tr_sam, "train/sample"))
        if tr_sub:
            payload.update(prefix_metrics(tr_sub, "train/subject"))
        if tr_votes is not None:
            payload.update(vote_summary_to_wandb(tr_votes["summary"], "train"))
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        wandb_run.summary["checkpoint"] = str(ckpt_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)

    protocol = resolve_protocol(cfg)
    ml_model = str(cfg.get("ml_model", "lda")).lower()
    dataset_id = cfg.get("dataset_id", "unknown")
    dataset_name = cfg.get("dataset_name", dataset_id)
    seed = int(cfg.get("seed", 42))
    task_slug = slug(cfg.get("classify_choice") or dataset_name)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    strategy_tag = {"rolling": f"cv{cfg.get('n_folds', 5)}", "loso": "loso", "full": "full"}[protocol]
    run_name = str(cfg.get("wandb_run_name") or f"{ml_model}/{slug(dataset_name)}/{task_slug}/{strategy_tag}/{ts}")
    log_dir = Path(cfg.get("terminal_log_dir", Path("outputs") / "logs" / slug(dataset_id) / "train_ml"))
    log_path = log_dir / f"{slug(run_name)}.log"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            logger.info(f"Terminal log : {log_path}")
            logger.info(f"Model={ml_model}  protocol={protocol}  dataset={dataset_id}  seed={seed}")
            for k, v in cfg.items():
                if not isinstance(v, dict):
                    logger.info(f"  {k:<30} = {v}")
            seed_everything(seed)

            windows_ds = load_windows_dataset(cfg)
            feats = load_features(cfg, windows_ds)

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-baselines",
                wandb_dir=cfg.get("wandb_dir", "outputs/wandb"),
            )
            if wandb_run is not None:
                wandb_run.log({"data/total_windows": feats[4]["n_windows"],
                               "data/n_classes": feats[4]["n_classes"],
                               "data/n_features": feats[4]["n_features"]})

            out_dir = (
                Path(cfg.get("save_dir", "outputs/checkpoints/train_ml"))
                / ml_model / dataset_id / task_slug / f"seed{seed}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            ctx = ProtocolContext(
                cfg=cfg, windows_ds=windows_ds, device=None, build_model=None,
                out_dir=out_dir, file_tag=ml_model, vote_model=ml_model,
                results_meta={"ml_model": ml_model}, wandb_run=wandb_run,
            )
            {"rolling": run_rolling, "loso": run_loso, "full": run_full}[protocol](ctx, feats)
        finally:
            finish_wandb(wandb_run, wandb_module, cfg, run_name=run_name,
                         log_path=log_path, tee_file=tee_file)


if __name__ == "__main__":
    main()

"""Evaluation utilities for traditional ML baselines."""

from __future__ import annotations

import logging

import numpy as np

from ..utils.metrics import (
    compute_subject_metrics,
    fmt_metrics,
    metrics_from_scores,
    prefix_metrics,
)
from ..utils.subject_votes import log_vote_report

# Re-export for convenience
__all__ = [
    "aggregate_metrics",
    "evaluate_ml",
    "log_cv_results",
    "log_dl_rolling_cv_results",
    "log_final_results",
    "log_rolling_cv_results",
    "prefix_metrics",
    "fmt_metrics",
]

logger = logging.getLogger(__name__)

_SEP = "─" * 90


def evaluate_ml(
    pipeline,
    X_feat: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    split_name: str,
    use_subject_vote: bool = True,
    return_details: bool = False,
    num_classes: int | None = None,
):
    """Evaluate a fitted sklearn pipeline on one split."""
    preds = pipeline.predict(X_feat)
    fitted_classes = getattr(pipeline, "classes_", None)
    if num_classes is None:
        if fitted_classes is None:
            raise ValueError(
                f"evaluate_ml({split_name}): pipeline exposes no classes_, so num_classes "
                "must be passed explicitly by the caller."
            )
        num_classes = len(fitted_classes)

    probs = _probs_in_label_order(pipeline, X_feat, preds, fitted_classes, num_classes)
    sample_metrics = metrics_from_scores(probs, y_true, num_classes, preds=preds)

    subject_metrics = None
    vote_details = None
    if use_subject_vote:
        subject_metrics, vote_records, vote_summary = compute_subject_metrics(
            preds,
            y_true,
            subject_ids,
            num_classes=num_classes,
            probs=probs,
            return_details=True,
        )
        vote_details = {"records": vote_records, "summary": vote_summary}

    logger.info(_SEP)
    logger.info(f"  {split_name.upper()} — Sample  : {fmt_metrics(sample_metrics)}")
    if subject_metrics:
        logger.info(f"  {split_name.upper()} — Subject : {fmt_metrics(subject_metrics)}")
    if vote_details is not None:
        log_vote_report(
            split_name,
            vote_details["records"],
            vote_details["summary"],
            num_classes=num_classes,
        )

    if return_details:
        return sample_metrics, subject_metrics, vote_details
    return sample_metrics, subject_metrics


def _probs_in_label_order(
    pipeline,
    X_feat: np.ndarray,
    preds: np.ndarray,
    fitted_classes,
    num_classes: int,
) -> np.ndarray:
    """`predict_proba` output as an (N, num_classes) array whose column k is class k."""
    if not hasattr(pipeline, "predict_proba"):
        return np.eye(num_classes)[preds.astype(int)]

    raw = np.asarray(pipeline.predict_proba(X_feat), dtype=float)
    if fitted_classes is None:
        if raw.shape[1] != num_classes:
            logger.warning(
                f"predict_proba returned {raw.shape[1]} columns for a {num_classes}-class "
                "task and the pipeline exposes no classes_ to align them. "
                "Falling back to one-hot predictions for the ranking metrics."
            )
            return np.eye(num_classes)[preds.astype(int)]
        return raw

    probs = np.zeros((raw.shape[0], num_classes), dtype=float)
    for col, cls in enumerate(np.asarray(fitted_classes).astype(int)):
        probs[:, cls] = raw[:, col]
    return probs


def log_final_results(
    train_metrics: tuple,
    val_metrics: tuple,
    test_metrics: tuple,
) -> None:
    """Pretty-print final results table."""
    print()
    print("=" * 90)
    print("  FINAL RESULTS")
    print("=" * 90)
    for split_name, (sam, sub) in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        print(f"  Sample  | {split_name:<5} : {fmt_metrics(sam)}")
        if sub:
            print(f"  Subject | {split_name:<5} : {fmt_metrics(sub)}")
    print("=" * 90)


def aggregate_metrics(
    metrics_list: list[dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Aggregate per-fold metric dicts into (mean, std) pairs."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    result: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m.get(k, -1.0) != -1.0]
        if vals:
            result[k] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            result[k] = (-1.0, 0.0)
    return result


def log_cv_results(
    fold_val_sample_metrics: list[dict[str, float]],
    fold_val_subject_metrics: list[dict[str, float]] | None,
    agg_val_sample: dict[str, tuple[float, float]],
    agg_val_subject: dict[str, tuple[float, float]] | None,
    final_test_sample: dict[str, float],
    final_test_subject: dict[str, float] | None,
    n_folds: int,
) -> None:
    """Print CV validation summaries and final test results."""
    print()
    print("=" * 90)
    print(f"  CV RESULTS — Dev Set K-Fold (N={n_folds} folds)")
    print("=" * 90)
    for fold_i, fm in enumerate(fold_val_sample_metrics):
        print(f"  Val fold {fold_i + 1:<2} sample  : {fmt_metrics(fm)}")
        if fold_val_subject_metrics and fold_i < len(fold_val_subject_metrics):
            sm = fold_val_subject_metrics[fold_i]
            print(f"  Val fold {fold_i + 1:<2} subject : {fmt_metrics(sm)}")
    print(_SEP)
    print("  Val sample mean±std :")
    for name, (mean, std) in agg_val_sample.items():
        print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    if agg_val_subject:
        print("  Val subject mean±std :")
        for name, (mean, std) in agg_val_subject.items():
            print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    print("=" * 90)
    print("  FINAL TEST (retrained on full dev set)")
    print("=" * 90)
    print(f"  Sample  | test : {fmt_metrics(final_test_sample)}")
    if final_test_subject:
        print(f"  Subject | test : {fmt_metrics(final_test_subject)}")
    print("=" * 90)


def log_rolling_cv_results(
    run_val_sample_metrics: list[dict[str, float]],
    run_val_subject_metrics: list[dict[str, float]] | None,
    run_test_sample_metrics: list[dict[str, float]],
    run_test_subject_metrics: list[dict[str, float]] | None,
    agg_val_sample: dict[str, tuple[float, float]],
    agg_val_subject: dict[str, tuple[float, float]] | None,
    agg_test_sample: dict[str, tuple[float, float]],
    agg_test_subject: dict[str, tuple[float, float]] | None,
    n_runs: int,
) -> None:
    """Print per-run val and test metrics for rolling CV, plus aggregates."""
    print()
    print("=" * 90)
    print(f"  ROLLING CV RESULTS — {n_runs} runs (train/val/test rotated per run)")
    print("=" * 90)
    for run_i, vm in enumerate(run_val_sample_metrics):
        print(f"  Val  run {run_i + 1:<2} sample  : {fmt_metrics(vm)}")
        if run_val_subject_metrics and run_i < len(run_val_subject_metrics):
            print(f"  Val  run {run_i + 1:<2} subject : {fmt_metrics(run_val_subject_metrics[run_i])}")
        tm = run_test_sample_metrics[run_i] if run_i < len(run_test_sample_metrics) else {}
        print(f"  Test run {run_i + 1:<2} sample  : {fmt_metrics(tm)}")
        if run_test_subject_metrics and run_i < len(run_test_subject_metrics):
            print(f"  Test run {run_i + 1:<2} subject : {fmt_metrics(run_test_subject_metrics[run_i])}")
    print(_SEP)
    print("  Val  sample mean±std :")
    for name, (mean, std) in agg_val_sample.items():
        print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    if agg_val_subject:
        print("  Val  subject mean±std :")
        for name, (mean, std) in agg_val_subject.items():
            print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    print(_SEP)
    print("  Test sample mean±std :")
    for name, (mean, std) in agg_test_sample.items():
        print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    if agg_test_subject:
        print("  Test subject mean±std :")
        for name, (mean, std) in agg_test_subject.items():
            print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    print("=" * 90)


def log_dl_rolling_cv_results(
    run_val_sample_metrics: list[dict[str, float]],
    run_val_subject_metrics: list[dict[str, float]] | None,
    run_test_sample_metrics: list[dict[str, float]],
    run_test_subject_metrics: list[dict[str, float]] | None,
    agg_val_sample: dict[str, tuple[float, float]],
    agg_val_subject: dict[str, tuple[float, float]] | None,
    agg_test_sample: dict[str, tuple[float, float]],
    agg_test_subject: dict[str, tuple[float, float]] | None,
    n_runs: int,
) -> None:
    """Pretty-print rolling CV results for DL models."""
    print()
    print("=" * 90)
    print(f"  ROLLING CV RESULTS (DL) — {n_runs} runs (train/val/test rotated per run)")
    print("=" * 90)
    for run_i, vm in enumerate(run_val_sample_metrics):
        print(f"  Val  run {run_i + 1:<2} sample  : {fmt_metrics(vm)}")
        if run_val_subject_metrics and run_i < len(run_val_subject_metrics):
            print(f"  Val  run {run_i + 1:<2} subject : {fmt_metrics(run_val_subject_metrics[run_i])}")
        tm = run_test_sample_metrics[run_i] if run_i < len(run_test_sample_metrics) else {}
        print(f"  Test run {run_i + 1:<2} sample  : {fmt_metrics(tm)}")
        if run_test_subject_metrics and run_i < len(run_test_subject_metrics):
            print(f"  Test run {run_i + 1:<2} subject : {fmt_metrics(run_test_subject_metrics[run_i])}")
    print(_SEP)
    print("  Val  sample mean±std :")
    for name, (mean, std) in agg_val_sample.items():
        print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    if agg_val_subject:
        print("  Val  subject mean±std :")
        for name, (mean, std) in agg_val_subject.items():
            print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    print(_SEP)
    print("  Test sample mean±std :")
    for name, (mean, std) in agg_test_sample.items():
        print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    if agg_test_subject:
        print("  Test subject mean±std :")
        for name, (mean, std) in agg_test_subject.items():
            print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    print("=" * 90)

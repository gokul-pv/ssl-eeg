"""Classification metrics: sample-level and subject-level."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-averaged specificity across classes."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return float(np.mean(specificities))


def _sentinel_if_undefined(value: float) -> float:
    """Map an undefined score onto the -1.0 sentinel the aggregation understands."""
    return value if np.isfinite(value) else -1.0


def _safe_roc_auc(y_true_onehot, probs) -> float:
    try:
        return _sentinel_if_undefined(float(roc_auc_score(y_true_onehot, probs, multi_class="ovr")))
    except ValueError:
        return -1.0


def _safe_auprc_macro(y_true_onehot, probs) -> float:
    """Average precision macro-averaged over all classes (each treated as positive)."""
    try:
        return _sentinel_if_undefined(
            float(average_precision_score(y_true_onehot, probs, average="macro"))
        )
    except ValueError:
        return -1.0


def _safe_auprc_positive(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Average precision for the positive class only."""
    if num_classes != 2:
        return _safe_auprc_macro(np.eye(num_classes)[y_true.astype(int)], probs)
    try:
        return _sentinel_if_undefined(
            float(average_precision_score((y_true == 1).astype(int), probs[:, 1]))
        )
    except ValueError:
        return -1.0


# The metric keys every level (sample / subject) and every family (DL / ML) emits.
METRIC_KEYS = (
    "Accuracy",
    "BalancedAccuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUROC",
    "AUPRC",
    "AUPRC_macro",
)

_UNDEFINED_WITHOUT_TWO_CLASSES = tuple(k for k in METRIC_KEYS if k != "Accuracy")


def metrics_from_scores(
    scores: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
    preds: np.ndarray | None = None,
) -> dict[str, float]:
    """The single definition of the metric set, shared by every entry point."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)
    preds = np.argmax(scores, axis=1) if preds is None else np.asarray(preds)

    metrics: dict[str, float] = {"Accuracy": float(accuracy_score(y_true, preds))}

    if len(np.unique(y_true)) < 2:
        metrics.update({k: -1.0 for k in _UNDEFINED_WITHOUT_TWO_CLASSES})
        return metrics

    onehot = np.eye(num_classes)[y_true.astype(int)]
    metrics["BalancedAccuracy"] = float(balanced_accuracy_score(y_true, preds))
    metrics["Precision"] = float(precision_score(y_true, preds, average="macro", zero_division=0))
    metrics["Recall"] = float(recall_score(y_true, preds, average="macro", zero_division=0))
    metrics["Specificity"] = _multiclass_specificity(y_true, preds, num_classes)
    metrics["F1"] = float(f1_score(y_true, preds, average="macro", zero_division=0))
    metrics["AUROC"] = _safe_roc_auc(onehot, scores)
    metrics["AUPRC"] = _safe_auprc_positive(y_true, scores, num_classes)
    metrics["AUPRC_macro"] = _safe_auprc_macro(onehot, scores)

    return metrics


# ---------------------------------------------------------------------------
# Sample-level metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    pred_logits: torch.Tensor,
    true_labels: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute classification metrics from logits and ground-truth labels."""
    probs = torch.softmax(pred_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    y_true = true_labels.cpu().numpy()

    metrics = metrics_from_scores(
        probs, y_true, num_classes=pred_logits.shape[1], preds=preds
    )
    return metrics, preds


# ---------------------------------------------------------------------------
# Subject-level majority-vote metrics
# ---------------------------------------------------------------------------


def compute_subject_vote_details(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    subject_ids: np.ndarray,
    num_classes: int,
    probs: np.ndarray | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """Break down the per-subject majority vote into its components."""
    unique_subjects = np.unique(subject_ids)
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    subject_ids = np.asarray(subject_ids)

    records: list[dict] = []

    for sid in unique_subjects:
        idx = np.where(subject_ids == sid)[0]
        counts = np.bincount(predictions[idx].astype(int), minlength=num_classes)
        n_win = int(counts.sum())

        mean_probs = probs[idx].mean(axis=0) if probs is not None else None

        max_count = counts.max()
        winners = np.flatnonzero(counts == max_count)
        tie = bool(winners.size > 1)
        if tie and mean_probs is not None:
            pred_label = int(winners[int(np.argmax(mean_probs[winners]))])
        else:
            pred_label = int(winners[0])

        true_label = int(true_labels[idx][0])
        sorted_counts = np.sort(counts)[::-1]
        runner_up = int(sorted_counts[1]) if num_classes > 1 else 0

        frac = counts / n_win if n_win > 0 else np.zeros_like(counts, dtype=float)
        nz = frac[frac > 0]
        entropy = abs(float(-(nz * np.log(nz)).sum())) if nz.size else 0.0
        norm_entropy = entropy / np.log(num_classes) if num_classes > 1 else 0.0

        record: dict = {
            "subject_id": int(sid),
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": bool(pred_label == true_label),
            "n_windows": n_win,
            "win_frac": float(max_count / n_win) if n_win > 0 else 0.0,
            "true_frac": float(counts[true_label] / n_win) if n_win > 0 else 0.0,
            "vote_margin": float((max_count - runner_up) / n_win) if n_win > 0 else 0.0,
            "vote_entropy": float(norm_entropy),
            "tie": tie,
        }
        for k in range(num_classes):
            record[f"votes_c{k}"] = int(counts[k])
        if mean_probs is not None:
            for k in range(num_classes):
                record[f"prob_c{k}"] = float(mean_probs[k])
            record["soft_pred"] = int(np.argmax(mean_probs))
        else:
            record["soft_pred"] = pred_label
        record["soft_agrees"] = bool(record["soft_pred"] == pred_label)
        records.append(record)

    return records, summarize_subject_votes(records)


def summarize_subject_votes(records: list[dict]) -> dict[str, float]:
    """Reduce per-subject vote records to scalar decisiveness statistics."""
    if not records:
        return {"n_subjects": 0}

    win = np.array([r["win_frac"] for r in records])
    correct = np.array([r["correct"] for r in records])
    trues = np.array([r["true_label"] for r in records])
    soft = np.array([r["soft_pred"] for r in records])

    def _mean(values) -> float:
        return float(np.mean(values)) if len(values) else -1.0

    summary: dict[str, float] = {
        "n_subjects": len(records),
        "mean_win_frac": float(win.mean()),
        "median_win_frac": float(np.median(win)),
        "min_win_frac": float(win.min()),
        "mean_win_frac_correct": _mean(win[correct]),
        "mean_win_frac_incorrect": _mean(win[~correct]),
        "mean_true_frac": float(np.mean([r["true_frac"] for r in records])),
        "mean_vote_margin": float(np.mean([r["vote_margin"] for r in records])),
        "mean_vote_entropy": float(np.mean([r["vote_entropy"] for r in records])),
        "frac_subjects_below_60pct": float(np.mean(win < 0.60)),
        "frac_subjects_below_70pct": float(np.mean(win < 0.70)),
        "n_ties": int(sum(r["tie"] for r in records)),
        "hard_soft_agreement": float(np.mean([r["soft_agrees"] for r in records])),
        "softvote_accuracy": float(accuracy_score(trues, soft)),
    }
    summary["softvote_balanced_accuracy"] = (
        float(balanced_accuracy_score(trues, soft)) if len(np.unique(trues)) >= 2 else -1.0
    )
    return summary


def compute_subject_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    subject_ids: np.ndarray,
    num_classes: int,
    probs: np.ndarray | None = None,
    return_details: bool = False,
):
    """Aggregate window-level predictions per subject via majority vote,
    then compute classification metrics at the subject level.
    """
    records, summary = compute_subject_vote_details(
        predictions, true_labels, subject_ids, num_classes, probs=probs
    )
    metrics = subject_metrics_from_records(records, num_classes)
    return (metrics, records, summary) if return_details else metrics


def subject_metrics_from_records(
    records: list[dict],
    num_classes: int,
) -> dict[str, float]:
    """Subject-level metrics from per-subject vote records."""
    subj_preds = np.array([r["pred_label"] for r in records])
    subj_trues = np.array([r["true_label"] for r in records])

    has_probs = bool(records) and "prob_c0" in records[0]
    if has_probs:
        score_matrix = np.array(
            [[r[f"prob_c{k}"] for k in range(num_classes)] for r in records], dtype=float
        )
    else:
        score_matrix = np.eye(num_classes)[subj_preds.astype(int)]

    return metrics_from_scores(
        score_matrix, subj_trues, num_classes=num_classes, preds=subj_preds
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}: {v:.4f}" if v != -1.0 else f"{k}: N/A")
    return " | ".join(parts)


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Return a new dict with all keys prefixed, lowercased, e.g. 'val/sample/f1'."""
    return {f"{prefix}/{k.lower()}": v for k, v in metrics.items()}


def fmt_vote_summary(summary: dict[str, float]) -> str:
    """One-line rendering of the vote decisiveness statistics."""
    n = summary.get("n_subjects", 0)
    if not n:
        return "Vote decisiveness: no subjects"

    def _pct(key: str) -> str:
        v = summary.get(key, -1.0)
        return "N/A" if v == -1.0 else f"{v:.3f}"

    return (
        f"mean win%={_pct('mean_win_frac')} "
        f"(correct {_pct('mean_win_frac_correct')} / wrong {_pct('mean_win_frac_incorrect')}) | "
        f"min={_pct('min_win_frac')} | margin={_pct('mean_vote_margin')} | "
        f"entropy={_pct('mean_vote_entropy')} | "
        f"<60%: {round(summary.get('frac_subjects_below_60pct', 0.0) * n)}/{n} | "
        f"ties: {summary.get('n_ties', 0)} | "
        f"hard==soft: {round(summary.get('hard_soft_agreement', 0.0) * n)}/{n} | "
        f"softvote acc={_pct('softvote_accuracy')}"
    )


def fmt_vote_table(records: list[dict], num_classes: int) -> str:
    """Multi-line per-subject vote table: one row per subject."""
    if not records:
        return "  (no subjects)"

    has_probs = "prob_c0" in records[0]
    votes_w = max(len(f"votes[{','.join(str(k) for k in range(num_classes))}]"), 6 * num_classes + 2)
    header = (
        f"  {'sid':>6} {'true':>4} {'pred':>4} {'ok':>2} {'n_win':>6} "
        f"{'votes[' + ','.join(str(k) for k in range(num_classes)) + ']':<{votes_w}} "
        f"{'win%':>6} {'true%':>6} {'margin':>6} {'entr':>5}"
    )
    if has_probs:
        header += f" {'probs':<{max(7 * num_classes + 2, 7)}} {'soft':>4}"
    lines = [header, "  " + "─" * (len(header) - 2)]

    for r in records:
        votes = "[" + ",".join(f"{r[f'votes_c{k}']:>5}" for k in range(num_classes)) + "]"
        line = (
            f"  {r['subject_id']:>6} {r['true_label']:>4} {r['pred_label']:>4} "
            f"{'Y' if r['correct'] else 'N':>2} {r['n_windows']:>6} {votes:<{votes_w}} "
            f"{r['win_frac']:>6.3f} {r['true_frac']:>6.3f} {r['vote_margin']:>6.3f} "
            f"{r['vote_entropy']:>5.2f}"
        )
        if has_probs:
            probs = "[" + ",".join(f"{r[f'prob_c{k}']:>6.3f}" for k in range(num_classes)) + "]"
            line += f" {probs:<{max(7 * num_classes + 2, 7)}} {r['soft_pred']:>4}"
        if r["tie"]:
            line += "  *tie"
        elif r["win_frac"] < 0.60:
            line += "  *near-tie"
        if not r["soft_agrees"]:
            line += "  *soft-differs"
        lines.append(line)

    return "\n".join(lines)

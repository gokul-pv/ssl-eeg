#!/usr/bin/env python
"""Cross-dataset evaluation: models trained on FEP are applied, unchanged, to every participant of the
independent SCZ dataset.

Usage
-----
  python evaluate.py --config configs/eval/eval_atcnet_fep_to_scz.yaml --rolling
  python evaluate.py --config configs/eval/eval_atcnet_fep_to_scz.yaml

  # explicit checkpoint (directory for --rolling, file otherwise)
  python evaluate.py --config configs/eval/eval_lda_fep_to_scz.yaml --rolling \\
      --checkpoint outputs/checkpoints/train_ml/lda/ds003944/fep/seed42

The config sets ``checkpoint_path`` (single) and ``rolling_checkpoint_dir``
(rolling); ``--checkpoint`` overrides whichever applies.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import apply_cli_overrides, cfg_to_flat_dict, load_config
from src.datasets import build_dataloader, build_dataset, collect_numpy_split, load_windows_dataset
from src.datasets.channels import infer_ch_names, make_data_info, resolve_ch_names
from src.engine.ml_eval import aggregate_metrics, evaluate_ml
from src.features import extract_features_from_windows
from src.features.cache import feature_cache_key
from src.models import build_mae_classifier, build_model
from src.utils.metrics import (
    compute_metrics,
    compute_subject_metrics,
    compute_subject_vote_details,
    fmt_metrics,
    prefix_metrics,
)
from src.utils.runtime import agg_to_json, finish_wandb, init_wandb, slug, tee_console
from src.utils.subject_votes import (
    append_vote_records,
    log_vote_report,
    vote_records_to_wandb_table,
    vote_summary_to_wandb,
)

logger = logging.getLogger(__name__)
_SEP = "─" * 90


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _model_slug(cfg: dict) -> str:
    return slug(cfg.get("model_name", cfg.get("ml_model", "model")))


def _vote_model_tag(cfg: dict) -> str:
    """``<model>_<mode>`` (as written by finetune.py / probe.py), or ``<model>``."""
    mode = cfg.get("mode")
    return f"{_model_slug(cfg)}_{mode}" if mode else _model_slug(cfg)


def _src_tgt(cfg: dict) -> tuple[str, str]:
    return (slug(str(cfg.get("source_dataset_id", "src"))),
            slug(str(cfg.get("dataset_id", cfg.get("dataset_name", "tgt")))))


def _votes_filename(cfg: dict, kind: str) -> str:
    """Per-subject vote file; ``kind`` is "rolling" or "full" (both share results_dir)."""
    src, tgt = _src_tgt(cfg)
    return f"{_vote_model_tag(cfg)}_{kind}_{src}_to_{tgt}_subject_votes.csv"


def _results_dir(cfg: dict) -> Path:
    _, tgt = _src_tgt(cfg)
    path = Path(cfg.get("results_dir", f"outputs/cv_results/{tgt}/eval"))
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def _save_confusion_matrix(y_true, y_pred, class_names, save_path: Path, vote_records=None) -> None:
    """Sample- and (if available) subject-level confusion matrices as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cms = [confusion_matrix(y_true, y_pred)]
    titles = ["Sample-level Confusion Matrix"]
    if vote_records is not None:
        cms.append(confusion_matrix([r["true_label"] for r in vote_records],
                                    [r["pred_label"] for r in vote_records]))
        titles.append("Subject-level Confusion Matrix")

    fig, axes = plt.subplots(1, len(cms), figsize=(6 * len(cms), 5))
    axes = np.atleast_1d(axes)
    for ax, data, title in zip(axes, cms, titles):
        im = ax.imshow(data, interpolation="nearest", cmap="Pastel1", alpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13, pad=12)
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_ylabel("True label", fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=11)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
    fig.suptitle(f"Cross-dataset evaluation\nClasses: {class_names} | n={len(y_true)} windows",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_dl(model: nn.Module, loader, device, use_subject_vote: bool, ch_names=None):
    """Inference over ``loader``; returns loss, metrics, predictions and vote details."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    try:
        supports_ch_names = "ch_names" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        supports_ch_names = False

    losses, logits_l, labels_l, sids_l = [], [], [], []
    for X, labels, sids in loader:
        X, labels = X.float().to(device), labels.long().to(device)
        logits = model(X, ch_names=ch_names) if (supports_ch_names and ch_names is not None) else model(X)
        losses.append(criterion(logits, labels).item())
        logits_l.append(logits.cpu())
        labels_l.append(labels.cpu())
        sids_l.append(sids)

    logits, labels, sids = torch.cat(logits_l), torch.cat(labels_l), torch.cat(sids_l).numpy()
    sample_metrics, preds = compute_metrics(logits, labels)
    subject_metrics = vote_details = None
    if use_subject_vote:
        probs = torch.softmax(logits, dim=1).numpy()
        subject_metrics, records, summary = compute_subject_metrics(
            preds, labels.numpy(), sids, num_classes=logits.shape[1], probs=probs, return_details=True)
        vote_details = {"records": records, "summary": summary}
    return (float(np.mean(losses)) if losses else 0.0, sample_metrics, subject_metrics,
            preds, labels.numpy(), sids, vote_details)


def _load_dl_model(cfg: dict, ckpt_path: Path, data_info: dict, device):
    if cfg.get("model_type") == "mae_classifier":
        model = build_mae_classifier(cfg, n_classes=data_info["n_classes"], device=device)
    else:
        model = build_model(cfg, data_info, device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    return model


def _eval_features(cfg: dict, eval_ds, sfreq: float):
    """Features of the target dataset (cached; key covers data + feature settings)."""
    _, tgt = _src_tgt(cfg)
    cache_dir = (Path(cfg.get("feature_cache_dir", "outputs/features")) / tgt
                 / slug(cfg.get("classify_choice", "eval")) / feature_cache_key(cfg))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "eval_features.npz"
    if cache_path.exists():
        logger.info(f"Loading eval features from cache: {cache_path}")
        cached = np.load(cache_path)
        return cached["X"], cached["y"], cached["sid"]
    X_raw, y, sid = collect_numpy_split(eval_ds, desc="eval")
    X = extract_features_from_windows(X_raw, sfreq=sfreq, cfg=cfg)
    np.savez(cache_path, X=X, y=y, sid=sid)
    logger.info(f"Features cached → {cache_path}")
    return X, y, sid


def _discover_rolling_checkpoints(runs_dir: Path, ml_model: str | None) -> list[Path]:
    pattern = f"run*/{ml_model}_pipeline.pkl" if ml_model else "run*/best.pth"
    files = sorted(runs_dir.glob(pattern), key=lambda p: int(re.search(r"run(\d+)", p.parent.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No {pattern} checkpoints under {runs_dir}; run the five-fold CV first.")
    return files


# ---------------------------------------------------------------------------
# Rolling evaluation (five CV models, aggregated)
# ---------------------------------------------------------------------------


def run_rolling_eval(cfg, runs_dir: Path, eval_ds, eval_loader, data_info, device, ch_names, wandb_run):
    ml_model = cfg.get("ml_model")
    ckpt_files = _discover_rolling_checkpoints(runs_dir, ml_model)
    use_vote = bool(cfg.get("use_subject_vote", True))
    logger.info(f"Rolling eval ({'ML' if ml_model else 'DL'}): {len(ckpt_files)} checkpoints under {runs_dir}")

    feats = _eval_features(cfg, eval_ds, data_info["sfreq"]) if ml_model else None
    src, tgt = _src_tgt(cfg)
    results_dir = _results_dir(cfg)
    votes_csv = results_dir / _votes_filename(cfg, kind="rolling")
    votes_csv.unlink(missing_ok=True)
    vote_ctx = dict(strategy="rolling_eval", model=_vote_model_tag(cfg), dataset=f"{src}_to_{tgt}",
                    task=cfg.get("classify_choice", "unknown"), split="eval")

    sample_l, subject_l, vote_l = [], [], []
    for run_num, ckpt_path in enumerate(ckpt_files, start=1):
        logger.info(f"ROLLING EVAL RUN {run_num}/{len(ckpt_files)}: {ckpt_path}")
        if ml_model:
            import joblib
            sample_m, subject_m, vote_details = evaluate_ml(
                joblib.load(str(ckpt_path)), *feats, f"eval/run{run_num}", use_vote, return_details=True)
        else:
            model = _load_dl_model(cfg, ckpt_path, data_info, device)
            _, sample_m, subject_m, _, _, _, vote_details = evaluate_dl(
                model, eval_loader, device, use_vote, ch_names=ch_names)
            if vote_details is not None:
                log_vote_report(f"eval/run{run_num}", vote_details["records"], vote_details["summary"])
        sample_l.append(sample_m)
        if subject_m:
            subject_l.append(subject_m)
        logger.info(f"  Run {run_num} | Sample  : {fmt_metrics(sample_m)}")
        if subject_m:
            logger.info(f"  Run {run_num} | Subject : {fmt_metrics(subject_m)}")
        if vote_details is not None:
            append_vote_records(votes_csv, vote_details["records"], run=run_num, **vote_ctx)
            vote_l.append(vote_details["summary"])
        if wandb_run is not None:
            payload = {"rolling_run": run_num}
            payload.update(prefix_metrics(sample_m, "rolling_run/eval/sample"))
            if subject_m:
                payload.update(prefix_metrics(subject_m, "rolling_run/eval/subject"))
            if vote_details is not None:
                payload.update(vote_summary_to_wandb(vote_details["summary"], "rolling_run/eval"))
                table = vote_records_to_wandb_table(vote_details["records"])
                if table is not None:
                    payload["rolling_run/eval/subject_votes"] = table
            wandb_run.log(payload)

    agg_sample = aggregate_metrics(sample_l)
    agg_subject = aggregate_metrics(subject_l) if subject_l else None
    agg_vote = aggregate_metrics(vote_l) if vote_l else None
    print()
    print("=" * 90)
    print(f"  ROLLING EVAL RESULTS — {len(sample_l)} runs | {src} → {tgt} ({len(eval_ds)} windows)")
    print("=" * 90)
    for name, (mean, std) in (agg_subject or agg_sample).items():
        print(f"    {name:<14}: {mean:.4f} ± {std:.4f}")
    print("=" * 90)

    # File name carries the mode, so LP and FT evaluations of one backbone do
    # not overwrite each other (they did in the original code).
    results_path = results_dir / f"{_vote_model_tag(cfg)}_{src}_to_{tgt}_rolling_eval.json"
    record = {
        "n_runs": len(sample_l), "mode": "rolling_eval",
        "model_name": cfg.get("model_name", cfg.get("ml_model", "model")),
        "adaptation_mode": cfg.get("mode"),
        "source_dataset": cfg.get("source_dataset_id", "unknown"), "target_dataset": cfg.get("dataset_id"),
        "task": cfg.get("classify_choice", "unknown"), "checkpoints": [str(p) for p in ckpt_files],
        "per_run_sample": sample_l, "per_run_subject": subject_l,
        "aggregated_sample": agg_to_json(agg_sample), "aggregated_subject": agg_to_json(agg_subject),
        "per_run_vote_summary": vote_l, "aggregated_vote": agg_to_json(agg_vote),
        "subject_votes_csv": str(votes_csv),
    }
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Rolling eval results saved → {results_path}")

    if wandb_run is not None:
        payload = {}
        for prefix, agg in (("rolling_eval/sample", agg_sample), ("rolling_eval/subject", agg_subject),
                            ("rolling_eval/vote", agg_vote)):
            for name, (mean, std) in (agg or {}).items():
                payload[f"{prefix}/{name}_mean"] = mean
                payload[f"{prefix}/{name}_std"] = std
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        wandb_run.summary["rolling_eval_results_path"] = str(results_path)
        wandb_run.summary["subject_votes_csv"] = str(votes_csv)


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation (model trained on the full source pool)
# ---------------------------------------------------------------------------


def run_single_eval(cfg, ckpt_path: Path, eval_ds, eval_loader, data_info, device, ch_names,
                    wandb_run, wandb_module, run_name):
    use_vote = bool(cfg.get("use_subject_vote", True))
    ml_model = cfg.get("ml_model")
    if ml_model:
        import joblib
        pipeline = joblib.load(str(ckpt_path))
        X, y_true, sids = _eval_features(cfg, eval_ds, data_info["sfreq"])
        sample_m, subject_m, vote_details = evaluate_ml(pipeline, X, y_true, sids, "eval", use_vote,
                                                        return_details=True)
        mean_loss, preds = 0.0, pipeline.predict(X)
    else:
        model = _load_dl_model(cfg, ckpt_path, data_info, device)
        mean_loss, sample_m, subject_m, preds, y_true, sids, vote_details = evaluate_dl(
            model, eval_loader, device, use_vote, ch_names=ch_names)
        if vote_details is not None:
            log_vote_report("eval", vote_details["records"], vote_details["summary"])

    print()
    print("=" * 90)
    print(f"  EVALUATION RESULTS | {cfg.get('source_dataset_id', 'unknown')} → {cfg.get('dataset_id')} "
          f"({len(eval_ds)} windows) | checkpoint: {ckpt_path}")
    print(_SEP)
    print(f"  Sample-level  : {fmt_metrics(sample_m)}")
    if subject_m:
        print(f"  Subject-level : {fmt_metrics(subject_m)}")
    print("=" * 90)

    class_names = cfg.get("class_names", ["HC", "Psychosis"])
    cm_path = Path(cfg.get("figures_dir", "outputs/figures")) / slug(run_name) / "confusion_matrix.png"
    _save_confusion_matrix(y_true, preds, class_names, cm_path,
                           vote_records=vote_details["records"] if vote_details else None)

    votes_csv = None
    if vote_details is not None:
        src, tgt = _src_tgt(cfg)
        votes_csv = _results_dir(cfg) / _votes_filename(cfg, kind="full")
        votes_csv.unlink(missing_ok=True)
        append_vote_records(votes_csv, vote_details["records"], strategy="eval", model=_vote_model_tag(cfg),
                            dataset=f"{src}_to_{tgt}", task=cfg.get("classify_choice", "unknown"),
                            split="eval", run=ckpt_path.name)
        logger.info(f"Per-subject votes saved → {votes_csv}")

    if wandb_run is not None:
        payload = {"eval/loss": mean_loss}
        payload.update(prefix_metrics(sample_m, "eval/sample"))
        if subject_m:
            payload.update(prefix_metrics(subject_m, "eval/subject"))
        if vote_details is not None:
            payload.update(vote_summary_to_wandb(vote_details["summary"], "eval"))
            table = vote_records_to_wandb_table(vote_details["records"])
            if table is not None:
                payload["eval/subject_votes"] = table
        wandb_run.log(payload)
        wandb_run.summary["eval/loss"] = mean_loss
        wandb_run.summary.update(prefix_metrics(sample_m, "eval/sample"))
        if subject_m:
            wandb_run.summary.update(prefix_metrics(subject_m, "eval/subject"))
        if vote_details is not None:
            wandb_run.summary.update(vote_summary_to_wandb(vote_details["summary"], "eval"))
            wandb_run.summary["subject_votes_csv"] = str(votes_csv)
        wandb_run.summary["checkpoint"] = str(ckpt_path)
        wandb_run.summary["source_dataset"] = cfg.get("source_dataset_id", "unknown")
        wandb_run.summary["target_dataset"] = cfg.get("dataset_id")
        wandb_run.log({"eval/confusion_matrix": wandb_module.Image(str(cm_path))})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation (FEP → SCZ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Evaluation YAML config")
    parser.add_argument("--checkpoint", default=None, metavar="PATH",
                        help="Checkpoint file, or with --rolling the directory holding run<N>/")
    parser.add_argument("--rolling", action="store_true",
                        help="Evaluate the five five-fold-CV models and aggregate (mean ± std)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override any config value, e.g. --set wandb_mode=disabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args.set)

    key = "rolling_checkpoint_dir" if args.rolling else "checkpoint_path"
    if args.checkpoint:
        cfg[key] = args.checkpoint
    ckpt_path = Path(cfg.get(key, ""))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{key} not found: {ckpt_path} (pass --checkpoint or set it in the config).")
    if args.rolling != ckpt_path.is_dir():
        raise ValueError(f"--rolling expects a directory, single mode a file; got {ckpt_path}.")

    src, tgt = _src_tgt(cfg)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(cfg.get("wandb_run_name")
                   or f"{'eval/rolling' if args.rolling else 'eval'}/{_vote_model_tag(cfg)}/{src}-to-{tgt}/{ts}")
    log_path = Path(cfg.get("terminal_log_dir", Path("outputs") / "logs" / tgt / "eval")) / f"{slug(run_name)}.log"

    with tee_console(log_path) as tee_file:
        wandb_run = wandb_module = None
        try:
            logger.info(f"Terminal log: {log_path}")
            for k, v in cfg.items():
                if not isinstance(v, dict):
                    logger.info(f"  {k:<30} = {v}")
            gpu = int(cfg.get("gpu", 0))
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

            if cfg.get("model_type") == "mae_classifier":
                # BrainLM-EEG: the encoder's channel list defines the input channels.
                enc = torch.load(cfg["pretrained_encoder_path"], map_location="cpu", weights_only=False)
                cfg["common_ch_names"] = enc["common_ch_names"]

            windows_ds = load_windows_dataset(cfg)
            # Whole target dataset = one evaluation set, through the dataset registry.
            eval_ds = build_dataset(windows_ds, list(range(len(windows_ds))), cfg)
            eval_loader = build_dataloader(eval_ds, cfg, split="test")
            ch_names = resolve_ch_names(eval_ds, infer_ch_names(windows_ds))
            data_info = make_data_info(eval_ds, cfg, ch_names)
            logger.info(
                f"Target {cfg.get('dataset_id')}: {len(eval_ds)} windows | n_channels={data_info['n_chans']} "
                f"n_times={data_info['n_times']} n_classes={data_info['n_classes']} | channels: {ch_names}"
            )

            wandb_run, wandb_module = init_wandb(
                cfg, run_name=run_name, default_project="eeg-cross-eval",
                wandb_dir=cfg.get("wandb_dir", "outputs/wandb"),
                config={**cfg_to_flat_dict(cfg), "checkpoint": str(ckpt_path),
                        "source_dataset_id": cfg.get("source_dataset_id", "unknown"),
                        "target_dataset_id": cfg.get("dataset_id"), "eval_windows": len(eval_ds)},
            )
            if args.rolling:
                run_rolling_eval(cfg, ckpt_path, eval_ds, eval_loader, data_info, device, ch_names, wandb_run)
            else:
                run_single_eval(cfg, ckpt_path, eval_ds, eval_loader, data_info, device, ch_names,
                                wandb_run, wandb_module, run_name)
            logger.info("✓ Evaluation complete.")
        finally:
            finish_wandb(wandb_run, wandb_module, cfg, run_name=run_name, log_path=log_path, tee_file=tee_file)


if __name__ == "__main__":
    main()

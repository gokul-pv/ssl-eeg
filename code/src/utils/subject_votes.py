"""Persistence and reporting for per-subject majority-vote breakdowns."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .metrics import fmt_vote_summary, fmt_vote_table

logger = logging.getLogger(__name__)

__all__ = [
    "append_vote_records",
    "log_vote_report",
    "vote_records_to_wandb_table",
    "vote_summary_to_wandb",
]

# Context columns come first so the CSV reads left-to-right: who/what, then the vote.
_CONTEXT_FIRST = ["strategy", "model", "dataset", "task", "split", "run"]


def append_vote_records(csv_path, records: list[dict], **context) -> None:
    """Append per-subject vote records to a long-format CSV, one row per subject."""
    if not records:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = {k: v for k, v in context.items() if v is not None}
    ordered_ctx = [k for k in _CONTEXT_FIRST if k in ctx]
    ordered_ctx += [k for k in ctx if k not in _CONTEXT_FIRST]
    fieldnames = ordered_ctx + list(records[0].keys())

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow({**ctx, **record})


def vote_summary_to_wandb(summary: dict[str, float], prefix: str) -> dict[str, float]:
    """Namespace the vote summary for WandB, e.g. 'test/vote/mean_win_frac'."""
    if not summary:
        return {}
    return {f"{prefix}/vote/{k}": v for k, v in summary.items()}


def vote_records_to_wandb_table(records: list[dict]):
    """Build a `wandb.Table` from vote records, or None if unavailable."""
    if not records:
        return None
    try:
        import wandb
    except ImportError:  # pragma: no cover - wandb is optional at runtime
        return None

    columns = list(records[0].keys())
    table = wandb.Table(columns=columns)
    for record in records:
        table.add_data(*[record.get(c) for c in columns])
    return table


def log_vote_report(
    split_name: str,
    records: list[dict],
    summary: dict[str, float],
    num_classes: int | None = None,
    table: bool | None = None,
) -> None:
    """Log the vote breakdown for one split."""
    if not records:
        return

    if table is None:
        table = not split_name.lower().lstrip("_/ ").startswith("train")

    logger.info(f"  {split_name.upper()} — Votes   : {fmt_vote_summary(summary)}")
    if table:
        if num_classes is None:
            num_classes = sum(1 for k in records[0] if k.startswith("votes_c"))
        logger.info(f"  {split_name.upper()} — Per-subject vote breakdown:")
        for line in fmt_vote_table(records, num_classes).splitlines():
            logger.info(line)

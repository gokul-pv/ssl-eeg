"""Run-time plumbing shared by every entrypoint."""

from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import cfg_to_flat_dict

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Console / file logging
# ---------------------------------------------------------------------------


class TeeStream:
    """Write to multiple text streams (console + file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


def setup_logging(console_stream, log_path: Path) -> None:
    """Send the root logger to ``console_stream`` and append to ``log_path``."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    for handler in (
        logging.StreamHandler(console_stream),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        root.addHandler(handler)


@contextmanager
def tee_console(log_path: Path) -> Iterator:
    """Mirror stdout/stderr into ``log_path`` for the duration of the block."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    tee_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(original_stdout, tee_file)
    sys.stderr = TeeStream(original_stderr, tee_file)
    setup_logging(original_stdout, log_path)
    try:
        yield tee_file
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        tee_file.flush()
        tee_file.close()


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def slug(value) -> str:
    """Lower-case, non-alphanumeric runs → ``-``; empty result → ``"na"``."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "na"


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------


def wandb_enabled(cfg: dict) -> bool:
    return bool(cfg.get("use_wandb", True)) and cfg.get("wandb_mode", "offline") != "disabled"


def resolve_wandb_data_dir(cfg: dict, wandb_dir: str | Path) -> Path:
    """Writable W&B data dir used for artifact staging / cache."""
    explicit = cfg.get("wandb_data_dir")
    data_dir = Path(explicit) if explicit else Path(wandb_dir) / "_wandb_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def init_wandb(
    cfg: dict,
    *,
    run_name: str,
    default_project: str,
    wandb_dir: str | Path,
    config: dict | None = None,
    **init_kwargs,
):
    """Start a W&B run, or return ``(None, None)`` when W&B is disabled."""
    if not wandb_enabled(cfg):
        return None, None

    import wandb

    wandb_dir = Path(wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_wandb_data_dir(cfg, wandb_dir)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("WANDB_CACHE_DIR", str(data_dir / "cache"))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(wandb_dir / "artifacts"))

    run = wandb.init(
        project=cfg.get("wandb_project", default_project),
        entity=cfg.get("wandb_entity"),
        name=run_name,
        mode=cfg.get("wandb_mode", "offline"),
        dir=str(wandb_dir),
        config=cfg_to_flat_dict(cfg) if config is None else config,
        **init_kwargs,
    )
    logger.info(f"WandB run: {run.name}  mode={cfg.get('wandb_mode', 'offline')}")
    return run, wandb


def finish_wandb(
    run,
    wandb_module,
    cfg: dict,
    *,
    run_name: str,
    log_path: Path,
    tee_file=None,
    summary: dict | None = None,
) -> None:
    """Write summary fields, optionally upload the terminal log, finish the run."""
    if run is None:
        return
    run.summary["run_name"] = run_name
    run.summary["terminal_log_file"] = str(log_path)
    for key, value in (summary or {}).items():
        run.summary[key] = value
    if bool(cfg.get("wandb_upload_terminal_log_artifact", True)) and wandb_module is not None:
        if tee_file is not None:
            tee_file.flush()
        artifact = wandb_module.Artifact(name=f"terminal-log-{slug(run_name)}", type="logs")
        artifact.add_file(str(log_path), name=Path(log_path).name)
        run.log_artifact(artifact)
        logger.info("Uploaded terminal log as WandB artifact.")
    run.finish()


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def agg_to_json(agg: dict | None) -> dict:
    """``{metric: (mean, std)}`` → ``{metric: {"mean": .., "std": ..}}``."""
    if not agg:
        return {}
    return {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in agg.items()}

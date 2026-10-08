#!/usr/bin/env python
"""EEG Preprocessing Entry Point.

Usage
-----
  # Process all subjects:
  python preprocess.py --config configs/preprocess/fep.yaml

  # Process a single subject (for SLURM array jobs):
  python preprocess.py --config configs/preprocess/fep.yaml --subject sub-001

  # Same, by position in the sorted subject list (SLURM_ARRAY_TASK_ID):
  python preprocess.py --config configs/preprocess/fep.yaml --subject-index 0
  python preprocess.py --config configs/preprocess/fep.yaml --list-subjects

  # Override config values:
  python preprocess.py --config configs/preprocess/fep.yaml --set raw_data_path=/path/to/ds003944 n_jobs=8

This script:
  1. Loads the specified YAML config
  2. Applies any --set CLI overrides
  3. Optionally restricts processing to a single subject (--subject)
  4. Builds and runs the dataset-specific preprocessor
  5. Saves the braindecode WindowsDataset to data/<dataset_id>/<save_folder>/
     (or <save_folder>/<subject>/ in single-subject mode) plus a
     preprocess_meta_<folder|subject>.json one level above it
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make src importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import apply_cli_overrides, load_config
from src.preprocess import build_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

METADATA_DIR = Path(__file__).resolve().parent / "metadata"


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to preprocessing YAML config (e.g. configs/preprocess/adftd.yaml)",
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values, e.g. --set seg_len=200 sfreq=100",
    )
    parser.add_argument(
        "--subject", default=None, metavar="SUBJECT_ID",
        help=(
            "Process only this subject (e.g. 'sub-001'). "
            "Intended for SLURM array jobs where each task handles one subject. "
            "Output is saved to <save_folder>/<subject_id>/ instead of <save_folder>/."
        ),
    )
    parser.add_argument(
        "--subject-index", type=int, default=None, metavar="N",
        help=(
            "Process the N-th subject (0-based) of the sorted subject list in "
            "metadata/<dataset_id>/master_subject_table_*.csv, e.g. "
            "--subject-index $SLURM_ARRAY_TASK_ID."
        ),
    )
    parser.add_argument(
        "--list-subjects", action="store_true",
        help="Print the sorted subject list used by --subject-index and exit.",
    )
    return parser.parse_args()


def subject_list(dataset_id: str) -> list[str]:
    """Sorted unique ``subject_id`` values of the dataset's master subject table."""
    tables = sorted((METADATA_DIR / dataset_id).glob("master_subject_table_*.csv"))
    if len(tables) != 1:
        raise FileNotFoundError(
            f"expected one master_subject_table_*.csv in {METADATA_DIR / dataset_id}, found {len(tables)}"
        )
    return sorted(pd.read_csv(tables[0], usecols=["subject_id"], dtype=str)["subject_id"].dropna().unique())


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.set)

    if args.list_subjects or args.subject_index is not None:
        subjects = subject_list(cfg["dataset_id"])
        if args.list_subjects:
            print("\n".join(f"{i}\t{sub}" for i, sub in enumerate(subjects)))
            return
        if not 0 <= args.subject_index < len(subjects):
            raise SystemExit(f"--subject-index {args.subject_index} out of range "
                             f"(0-{len(subjects) - 1} for {cfg['dataset_id']})")
        if args.subject and args.subject != subjects[args.subject_index]:
            raise SystemExit("--subject and --subject-index disagree")
        args.subject = subjects[args.subject_index]

    # Single-subject mode: inject subject_id and adjust save_folder
    if args.subject:
        cfg["subject_id"] = args.subject
        cfg["save_folder"] = f"{cfg['save_folder']}/{args.subject}"
        logger.info(f"Single-subject mode: {args.subject}")

    logger.info("=" * 70)
    logger.info("Preprocessing Configuration")
    logger.info("=" * 70)
    for k, v in cfg.items():
        logger.info(f"  {k:<30} = {v}")
    logger.info("=" * 70)

    preprocessor = build_preprocessor(cfg)
    preprocessor.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Fit canonical EEG microstate maps (k=4, A/B/C/D) — ONE-TIME script.

Usage
-----
  python scripts/microstates/fit_canonical_microstate_maps.py \\
      --dataset-config configs/dataset/lemon.yaml --split train \\
      --n-windows 2000 --seed 42 \\
      --out metadata/microstates/canonical_maps.npz

After running, IMPORTANT MANUAL STEP: inspect the fitted maps with
`scripts/microstates/plot_canonical_microstate_maps.py` (saves a topomap PNG) against a
published reference (Michel & Koenig 2018, NeuroImage, Fig. 2) to determine
the A/B/C/D reordering permutation, then re-run THIS script with
`--reorder i,j,k,l` to bake that permutation into the saved maps:

  python scripts/microstates/plot_canonical_microstate_maps.py
  # inspect metadata/microstates/canonical_maps.png, note the permutation
  python scripts/microstates/fit_canonical_microstate_maps.py --reorder 2,0,3,1  # example

Until that manual step is done, the saved maps are in raw (arbitrary)
cluster-fit order — a loud warning is logged to make this impossible to miss.

Performance: pass `--n-jobs` to control parallelism (defaults to
$SLURM_CPUS_PER_TASK if set inside a SLURM job, else all local CPU cores).
This parallelizes both the braindecode window-loading step and the
per-window array-stacking step (via joblib), and is forwarded to
`ModKMeans.fit(..., n_jobs=...)` for its multiple random-initialization
restarts.

Uses pycrostates' public API (`extract_gfp_peaks`, `ModKMeans.fit`,
`.cluster_centers_`, `.GEV_`; tested with 0.6.1 in tests/test_pipelines.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.datasets.base import load_windows_dataset
from src.datasets.pretrain import PretrainDataset, _load_pretrain_split

# The model's 19 channels, in its input order.
MODEL_CHANNELS = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
                  "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit canonical EEG microstate maps on a LEMON window subsample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-config", default="configs/dataset/lemon.yaml",
        help="Path to the LEMON dataset YAML (same one build_pretrain_dataset uses).",
    )
    parser.add_argument(
        "--split", default="train", choices=["train"],
        help="LEMON partition to fit on (the pretraining training partition).",
    )
    parser.add_argument("--pretrain-seed", type=int, default=42,
                        help="Seed of the saved pretraining split (metadata/LEMON/split_seed<seed>.json).")
    parser.add_argument("--split-save-dir", default="metadata")
    parser.add_argument("--channels", default=",".join(MODEL_CHANNELS),
                        help="Comma-separated channels, in the model's order (default: the 19 model channels).")
    parser.add_argument(
        "--n-windows", type=int, default=2000,
        help="Cap on the number of (randomly sampled) windows used for fitting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help=(
            "Parallelism for window loading/stacking and ModKMeans' random-"
            "restart fitting. Defaults to $SLURM_CPUS_PER_TASK if set, else "
            "all local CPU cores (os.cpu_count())."
        ),
    )
    parser.add_argument("--out", default="metadata/microstates/canonical_maps.npz")
    parser.add_argument("--out-meta", default="metadata/microstates/canonical_maps_meta.json")
    parser.add_argument(
        "--reorder", default=None,
        help=(
            "Comma-separated permutation of cluster indices into A,B,C,D order "
            "(e.g. '2,0,3,1'), determined via one-time visual inspection. "
            "If omitted, identity order is used and a loud warning is logged."
        ),
    )
    return parser.parse_args()


def _build_mne_epochs(
    pretrain_ds: PretrainDataset, ch_names: list[str], sfreq: float, n_windows: int, seed: int, n_jobs: int
):
    """Wrap a random subsample of the (un-normalised, channel-selected) training
    windows into an average-referenced `mne.EpochsArray` with a standard_1020
    montage attached, ready for `pycrostates.preprocessing.extract_gfp_peaks`.
    """
    import numpy as np
    import mne
    from joblib import Parallel, delayed

    n_total = len(pretrain_ds)
    idx = list(range(n_total))
    if n_windows < n_total:
        rng = random.Random(seed)
        idx = sorted(rng.sample(idx, n_windows))

    logger.info(
        f"Building MNE Epochs from {len(idx)}/{n_total} LEMON training windows (n_jobs={n_jobs})..."
    )
    arrays = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(lambda i: pretrain_ds[i][0].numpy().astype(np.float64))(i) for i in idx
    )
    data = np.stack(arrays, axis=0)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, match_case=False, on_missing="raise", verbose=False)
    # Mean-centre every topography across the 19 channels (= average reference),
    # as the label definition does.
    epochs.set_eeg_reference("average", projection=False, verbose=False)
    return epochs


def _resolve_n_jobs(requested: int | None) -> int:
    """Same convention as pretrain.py: SLURM_CPUS_PER_TASK if set, else all local cores."""
    if requested is not None:
        return max(1, requested)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)


def main() -> None:
    args = parse_args()
    n_jobs = _resolve_n_jobs(args.n_jobs)
    logger.info(f"Using n_jobs={n_jobs}")

    ds_cfg = load_config(args.dataset_config)
    ds_cfg["n_jobs"] = n_jobs  # parallelizes braindecode's shard loading
    windows_ds = load_windows_dataset(ds_cfg)
    sfreq = float(windows_ds.datasets[0].raw.info["sfreq"])

    # LEMON training partition, as pretraining loads it.
    entry = {"dataset_id": ds_cfg["dataset_id"], "split": args.split, "dataset_config": args.dataset_config}
    train_idx = _load_pretrain_split(
        windows_ds, entry, {"pretrain_seed": args.pretrain_seed, "split_save_dir": args.split_save_dir})
    ch_names = [c.strip().upper() for c in args.channels.split(",")]
    pretrain_ds = PretrainDataset(windows_ds, train_idx, normalize=False, dataset_id=ds_cfg["dataset_id"],
                                  expected_n_times=len(windows_ds[0][0][0]))
    pretrain_ds.set_common_ch_names(ch_names)          # raises if a recording lacks a channel

    epochs = _build_mne_epochs(pretrain_ds, ch_names, sfreq, args.n_windows, args.seed, n_jobs)

    from pycrostates.cluster import ModKMeans
    from pycrostates.preprocessing import extract_gfp_peaks
    import numpy as np

    logger.info("Extracting GFP peaks...")
    gfp_peaks = extract_gfp_peaks(epochs)

    logger.info(
        f"Fitting ModKMeans(n_clusters={args.n_clusters}, random_state={args.seed}, "
        f"n_jobs={n_jobs})..."
    )
    cluster_model = ModKMeans(n_clusters=args.n_clusters, random_state=args.seed)
    cluster_model.fit(gfp_peaks, n_jobs=n_jobs)

    order = [int(x) for x in args.reorder.split(",")] if args.reorder else None
    if order is not None:
        cluster_model.reorder_clusters(order=order)
        cluster_model.rename_clusters(new_names=list("ABCD"[: args.n_clusters]))
    else:
        logger.warning(
            "No --reorder given — saved maps are in RAW (arbitrary) cluster-fit "
            "order, NOT yet the A/B/C/D neuroscience convention. Inspect the "
            "maps and re-run with --reorder before using them for real training."
        )

    gev = float(np.sum(cluster_model.GEV_)) if getattr(cluster_model, "GEV_", None) is not None else float("nan")
    maps = np.asarray(cluster_model.cluster_centers_, dtype=np.float32)  # (n_clusters, C)
    channels = list(ch_names)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, maps=maps, channels=np.array(channels), split=np.array(args.split))
    logger.info(f"Saved canonical microstate maps → {out_path} (shape={maps.shape})")

    manifest_hash = hashlib.sha256(
        json.dumps(
            {"n_windows": args.n_windows, "n_clusters": args.n_clusters, "seed": args.seed,
             "split": args.split, "channels": channels},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    meta = {
        "dataset_config": str(args.dataset_config),
        "split": args.split,
        "pretrain_seed": args.pretrain_seed,
        "n_train_windows": len(pretrain_ds),
        "signal": "un-normalised (µV), average-referenced over the listed channels",
        "n_windows": args.n_windows,
        "n_clusters": args.n_clusters,
        "seed": args.seed,
        "channels": channels,
        "gev": gev,
        "reorder": order,
        "reordered": order is not None,
        "manifest_hash": manifest_hash,
    }
    out_meta_path = Path(args.out_meta)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved canonical map metadata → {out_meta_path}")
    if order is None:
        logger.info(
            "Next step: python scripts/microstates/plot_canonical_microstate_maps.py "
            f"--maps {out_path} --meta {out_meta_path}"
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()

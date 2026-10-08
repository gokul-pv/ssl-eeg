#!/usr/bin/env python
"""Plot canonical EEG microstate maps for visual inspection — the concrete
"how do I inspect these" step referenced by
`scripts/microstates/fit_canonical_microstate_maps.py`.

Usage
-----
  python scripts/microstates/plot_canonical_microstate_maps.py
  python scripts/microstates/plot_canonical_microstate_maps.py \\
      --maps metadata/microstates/canonical_maps.npz \\
      --meta metadata/microstates/canonical_maps_meta.json \\
      --out metadata/microstates/canonical_maps.png

After inspecting the PNG, note which cluster index (0-indexed, left to
right in the plot) visually matches each canonical class, then re-run the
fitting script with that permutation baked in:
  python scripts/microstates/fit_canonical_microstate_maps.py --reorder i,j,k,l
where i,j,k,l are the cluster indices that should become A,B,C,D
respectively (e.g. `--reorder 2,0,3,1` means cluster 2 -> A, cluster 0 -> B,
cluster 3 -> C, cluster 1 -> D).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot canonical EEG microstate maps for visual A/B/C/D inspection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--maps", default="metadata/microstates/canonical_maps.npz")
    parser.add_argument("--meta", default="metadata/microstates/canonical_maps_meta.json")
    parser.add_argument("--out", default="metadata/microstates/canonical_maps.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import mne
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: works over SSH / SLURM with no display
    import matplotlib.pyplot as plt

    npz = np.load(args.maps)
    maps = np.asarray(npz["maps"], dtype=np.float64)  # (k, C)
    channels = [str(c) for c in npz["channels"]]
    k = maps.shape[0]

    labels = [f"Cluster {i}" for i in range(k)]
    meta_path = Path(args.meta)
    reordered = False
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        reordered = bool(meta.get("reordered", False))
        if reordered:
            labels = list("ABCD"[:k])
        gev = meta.get("gev")
        print(
            f"Loaded metadata: n_windows={meta.get('n_windows')} seed={meta.get('seed')} "
            f"gev={gev if gev is None else f'{gev:.4f}'} reordered={reordered}"
        )
    else:
        print(f"No metadata file found at {meta_path} — labeling clusters generically.")

    # Build a montage-equipped container so mne.viz.plot_topomap has electrode
    # positions. Info objects don't carry montage directly — wrap the k maps
    # as an EvokedArray (one "timepoint" per map) and set_montage on that.
    info = mne.create_info(ch_names=channels, sfreq=1.0, ch_types="eeg")
    evoked = mne.EvokedArray(maps.T, info, verbose=False)  # (C, k)
    montage = mne.channels.make_standard_montage("standard_1020")
    evoked.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.6))
    if k == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            evoked.data[:, i], evoked.info, axes=ax, show=False, contours=6, cmap="RdBu_r"
        )
        ax.set_title(labels[i], fontsize=14)

    fig.suptitle(
        "Canonical microstate maps — compare spatial PATTERN (polarity/sign is "
        "arbitrary) against Michel & Koenig (2018) NeuroImage Fig. 2",
        fontsize=10,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out_path}")

    if not reordered:
        print()
        print("Reference (Michel & Koenig 2018, Fig. 2 — canonical A/B/C/D):")
        print("  A: right-frontal to left-posterior gradient")
        print("  B: left-frontal to right-posterior gradient")
        print("  C: fronto-occipital / anterior-posterior midline gradient")
        print("  D: fronto-central maximum, parietal minimum (or vice versa)")
        print()
        print("Next: note which cluster index above matches each letter, then run:")
        print("  python scripts/microstates/fit_canonical_microstate_maps.py --reorder i,j,k,l")
        print("(i,j,k,l = the cluster indices that should become A,B,C,D)")


if __name__ == "__main__":
    main()

# Self-Supervised Learning of Latent EEG Representations for Mental Health Related Analysis

Code and manuscript of an MSc thesis in Artificial Intelligence at the Technical University
of Applied Sciences Würzburg-Schweinfurt (THWS).

The repository implements **BrainLM-EEG**, a masked-autoencoder EEG model adapted from
BrainLM, and two variants of it:

- **BrainLM-EEG-RoPE**, with spherical rotary positional encoding;
- **BrainLM-EEG-Microstate**, with auxiliary microstate prediction.

It also contains every baseline and evaluation protocol used in the thesis.

| Path | Contents |
|---|---|
| `code/` | Preprocessing, models, training and evaluation entrypoints, experiments, configs, tests |
| `manuscript/` | LaTeX sources and the submitted PDF |

## Installation

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). On Linux, PyTorch is
installed from the CUDA 12.6 wheel index.

```bash
cd code
uv sync                 # creates code/.venv from uv.lock
uv run pytest           # about 7 minutes on CPU; uses synthetic data only
```

For SLURM clusters, `code/singularity/` holds container definitions and
`code/scripts/slurm/` holds one job script per entrypoint. Cluster-specific
settings (container image, data mounts) live in `code/configs/cluster/cluster.env`.

## Data

Download EEG data from the original sources and set `raw_data_path` in `code/configs/preprocess/<dataset>.yaml`.

| Name | Role | Source |
|---|---|---|
| DVS | pretraining | [OpenNeuro ds005385](https://openneuro.org/datasets/ds005385) |
| SRM | pretraining | [OpenNeuro ds003775](https://openneuro.org/datasets/ds003775) |
| LEMON | pretraining | [MPI Leipzig Mind-Brain-Body](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) (Babayan et al., 2019) |
| ADFTD | AD vs HC, FTD vs HC | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) |
| FEP | FEP vs HC | [OpenNeuro ds003944](https://openneuro.org/datasets/ds003944) |
| MDD | MDD vs HC | [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478) |
| SCZ | external test set | Rácz et al., 2025 ([doi:10.1038/s41537-025-00568-3](https://doi.org/10.1038/s41537-025-00568-3)) |

`code/metadata/` contains everything needed to reproduce the participant
partitions:
- the per-dataset participant tables;
- the 90/10 pretraining splits;
- the 20% pretraining reserves of ADFTD, FEP and MDD;
- the five-fold cross-validation folds;
- the microstate maps.

The code reads data from `raw_data_path`.

## Quick start

From `code/`:

```bash
python preprocess.py --config configs/preprocess/adftd.yaml        # raw EEG → 15 s windows
python pretrain.py   --config configs/pretrain/brainlm_eeg.yaml      # BrainLM-EEG
python finetune.py   --config configs/linear_probe/brainlm_eeg_lp_adftd.yaml
```

Every entrypoint takes `--config` and `--set KEY=VALUE` overrides. The evaluation
protocol is part of the config: `protocol: rolling` (five-fold CV), `loso` or
`full` (all FEP participants, for the FEP → SCZ transfer).

| Entrypoint | Purpose |
|---|---|
| `preprocess.py` | EEGPrep, 0.5–45 Hz band-pass, notch, ICA/ICLabel, 15 s windows at 200 Hz |
| `pretrain.py` | BrainLM-EEG variants and the masking ablations |
| `finetune.py` | BrainLM-EEG linear probe (`configs/linear_probe/`) and finetuning (`configs/finetune/`) |
| `probe.py` | LaBraM, CBraMod, REVE with a frozen backbone |
| `train_dl.py` | EEGNeX, EEGConformer, ATCNet |
| `train_ml.py` | LDA, SVM, XGBoost on handcrafted features |
| `evaluate.py` | FEP → SCZ evaluation of any of the above |
| `experiments/` | Layer-wise probing, CLS-token test, reconstruction-as-features |

Results are logged to Weights & Biases (offline by default) and to per-participant
vote files under `code/outputs/`. [REPRODUCE.md](REPRODUCE.md) gives the full
sequence of runs.

## Citation

> Gokul Perumbayil Vijayakrishnan. *Self-Supervised Learning of Latent EEG
> Representations for Mental Health Related Analysis.* Master thesis, Technical
> University of Applied Sciences Würzburg-Schweinfurt (THWS), 2026.


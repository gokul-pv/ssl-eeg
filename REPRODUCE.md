# Reproducing the results

All commands run from `code/` with the environment from `uv sync` (prefix them
with `uv run`, or activate `code/.venv`). On a SLURM cluster, each step has a job
script in `scripts/slurm/`, submitted with
`bash scripts/slurm/submit.sh [sbatch options] scripts/slurm/<job>.sh <args>`.
Set the cluster paths in `configs/cluster/cluster.env` first.

Runs log to Weights & Biases in offline mode by default (`wandb_mode` in the
configs). Set `wandb_entity` and use `bash scripts/slurm/wandb_sync.sh`, or
`wandb sync`, to upload the runs afterwards.

## 1. Preprocessing

Set `raw_data_path` (and `excel_path` for SCZ) in `configs/preprocess/<dataset>.yaml`,
then run each dataset:

```bash
for ds in lemon srm dvs adftd fep mdd scz; do
    python preprocess.py --config configs/preprocess/$ds.yaml
done
```

On a cluster, use one array task per subject. The array sizes are listed in
`scripts/slurm/preprocess.sh`; `--list-subjects` prints the subject order.

```bash
bash scripts/slurm/submit.sh --array=0-87%16 scripts/slurm/preprocess.sh adftd
```

Windows are written to `data/<dataset_id>/<save_folder>/`. Participant tables,
splits and folds are read from `metadata/`.

## 2. Microstate maps (BrainLM-EEG-Microstate only)

The maps are fitted on the LEMON training split, using the model's 19 channels.
The cluster order of k-means is arbitrary, so the classes are matched to the
conventional A–D maps by inspection:

```bash
python scripts/microstates/fit_canonical_microstate_maps.py
python scripts/microstates/plot_canonical_microstate_maps.py   # inspect metadata/microstates/canonical_maps.png
python scripts/microstates/fit_canonical_microstate_maps.py --reorder i,j,k,l
```

The `--reorder` values are the cluster indices that correspond to A, B, C and D.
The fit is deterministic (`--seed 42`), so the second call reproduces the same
clusters.

## 3. Pretraining

| Run | Configs | SLURM |
|---|---|---|
| BrainLM-EEG, -RoPE, -Microstate | `configs/pretrain/brainlm_eeg{,_rope,_microstate}.yaml` | `pretrain.sh <config>` |
| Masking ablation (8 models) | `configs/ablation/masking_{ratio,strategy}/*.yaml` | `--array=0-7 ablation_pretrain.sh` |

```bash
python pretrain.py --config configs/pretrain/brainlm_eeg.yaml
```

Encoders are written to `outputs/checkpoints/pretrain/<model>/best_pretrain_encoder.pth`,
and the downstream configs point there.

## 4. Downstream evaluation (five-fold CV, LOSO, all-FEP)

Each config defines one task and one protocol. Run every config in each folder:

| Models | Entrypoint | Configs | SLURM |
|---|---|---|---|
| BrainLM-EEG variants, linear probe | `finetune.py` | `configs/linear_probe/*.yaml` | `finetune.sh` |
| BrainLM-EEG variants, finetuning | `finetune.py` | `configs/finetune/*.yaml` | `finetune.sh` |
| LaBraM, CBraMod, REVE | `probe.py` | `configs/probe/*.yaml` | `probe.sh` |
| EEGNeX, EEGConformer, ATCNet | `train_dl.py` | `configs/train_dl/*.yaml` | `train_dl.sh` |
| LDA, SVM, XGBoost | `train_ml.py` | `configs/train_ml/*.yaml` | `train_ml.sh` |

```bash
for c in configs/linear_probe/*.yaml configs/finetune/*.yaml; do python finetune.py --config $c; done
for c in configs/probe/*.yaml;    do python probe.py    --config $c; done
for c in configs/train_dl/*.yaml; do python train_dl.py --config $c; done
for c in configs/train_ml/*.yaml; do python train_ml.py --config $c; done
```

The foundation models download their weights from Hugging Face. REVE is gated, so
it needs `HF_TOKEN`.

## 5. FEP → SCZ transfer

This step needs the `*_fep_cross.yaml` (five-fold) and `*_fep_full.yaml` (all FEP)
runs from step 4.

```bash
for c in configs/eval/*.yaml; do
    python evaluate.py --config $c --rolling     # the five cross-validation models
    python evaluate.py --config $c               # the model trained on all FEP participants
done
```

## 6. Masking-ablation probes

This step needs the 8 ablation encoders from step 3. It runs a linear probe for
AD, FTD and FEP with every encoder:

```bash
bash scripts/slurm/submit.sh --array=0-23 scripts/slurm/ablation_finetune.sh
```

Without SLURM, run `finetune.py` with each `configs/ablation/finetune/*.yaml` and set
`ablation_name`, `pretrained_encoder_path`, `save_dir`, `wandb_dir` and
`terminal_log_dir` per ablation, as `scripts/slurm/ablation_finetune.sh` does.

## 7. Analysis experiments (BrainLM-EEG encoder)

```bash
for c in configs/experiments/layer_analysis_*.yaml;          do python experiments/layer_analysis.py --config $c; done
for c in configs/experiments/reconstruction_features_*.yaml; do python experiments/reconstruction_features.py --config $c; done
python experiments/cls_token_ablation.py --config configs/experiments/cls_token_ablation.yaml
```

Outputs go to `outputs/experiments/`. The figures read `cls_token_ablation/results.json`
from there.

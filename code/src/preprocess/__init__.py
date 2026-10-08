"""Preprocessor registry."""

from .adftd import ADFTD_Preprocessor
from .fep import FEP_Preprocessor
from .lemon import LEMON_Preprocessor
from .mdd import MDD_Preprocessor
from .scz import SCZ_Preprocessor
from .srm import SRM_Preprocessor
from .dvs import DVS_Preprocessor

PREPROCESSOR_REGISTRY: dict = {
    "ds004504": ADFTD_Preprocessor,
    "ds003944": FEP_Preprocessor,
    "ds003478": MDD_Preprocessor,
    # ds003775: SRM resting-state EEG, BioSemi ActiveTwo, 64ch, healthy controls only
    # Used for self-supervised pretraining (all subjects label=0)
    "ds003775": SRM_Preprocessor,
    # ds005385: Dortmund Vital Study, BrainVision BrainAmp DC, 64ch, healthy controls only
    # 608 subjects × 4 conditions × up to 2 sessions — used for pretraining
    "ds005385": DVS_Preprocessor,
    # LEMON: MPI Leipzig Mind-Brain-Body, BrainVision, 62ch, 228 healthy controls
    # Resting-state eyes-open, RSEEG/ layout; used for self-supervised pretraining
    "LEMON": LEMON_Preprocessor,
    # scz: Schizophrenia vs HC resting-state EEG (Zenodo 14808296, Racz et al. 2025)
    # Neuroscan SynAmps format (.dat/.dap/.rs3), flat directory, metadata in Excel
    "scz": SCZ_Preprocessor,
}


def build_preprocessor(cfg: dict):
    """Instantiate the preprocessor for the given dataset_id."""
    dataset_id = cfg.get("dataset_id")
    if dataset_id not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. "
            f"Available: {list(PREPROCESSOR_REGISTRY.keys())}"
        )
    return PREPROCESSOR_REGISTRY[dataset_id](cfg)

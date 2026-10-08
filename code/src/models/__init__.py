"""Model registry and factories."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .braindecode_models import build_braindecode_model
from .mae.brainlm_classifier import BrainLMClassifier
from .mae.brainlm_eeg import BrainLMEEG
from .mae.brainlm_microstate import BrainLMMicrostate
from .mae.brainlm_sphere_rope import BrainLMSphereRoPE
from .mae.layer_probe_classifier import LayerProbeClassifier

logger = logging.getLogger(__name__)


# ── Self-supervised model registry ────────────────────────────────────────
MAE_MODEL_REGISTRY: dict[str, type] = {
    "BrainLMEEG": BrainLMEEG,                 # BrainLM-EEG
    "BrainLMSphereRoPE": BrainLMSphereRoPE,   # BrainLM-EEG-RoPE
    "BrainLMMicrostate": BrainLMMicrostate,   # BrainLM-EEG-Microstate
}

# Keys that are absent from an encoder-only checkpoint by design.
_DECODER_PREFIXES = (
    "enc_to_dec", "mask_token",
    "dec_spatial_embed", "dec_temporal_embed",
    "dec_rope_freqs",                          # BrainLMSphereRoPE
    "microstate_head", "microstate_maps", "microstate_ch_idx",  # BrainLMMicrostate
    "decoder_blocks", "decoder_norm", "decoder_pred",
)


# ── Public API ────────────────────────────────────────────────────────────


def build_model(cfg: dict, data_info: dict, device: torch.device) -> nn.Module:
    """Build a braindecode model and move it to ``device``."""
    model_type = cfg.get("model_type", "braindecode")
    if model_type != "braindecode":
        raise ValueError(f"Unknown model_type='{model_type}'. Only 'braindecode' is supported.")

    model = build_braindecode_model(cfg, data_info).float().to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: total={total:,}  trainable={trainable:,}")
    return model


def build_mae_model(
    cfg: dict,
    c_common: int,
    device: torch.device,
    common_ch_names: list[str] | None = None,
) -> nn.Module:
    """Build a BrainLM-EEG variant and move it to ``device``."""
    model_name = cfg.get("model_name", "BrainLMEEG")
    if model_name not in MAE_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MAE model_name='{model_name}'. "
            f"Available: {list(MAE_MODEL_REGISTRY.keys())}"
        )
    model_cls = MAE_MODEL_REGISTRY[model_name]
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    model_kwargs["c_common"] = c_common
    if model_name == "BrainLMSphereRoPE":
        if not common_ch_names:
            raise ValueError(
                "BrainLMSphereRoPE requires common_ch_names. "
                "Pass common_ch_names=multi_ds.common_ch_names to build_mae_model()."
            )
        model_kwargs["channel_names"] = common_ch_names
    model_kwargs.setdefault("n_times", int(cfg.get("n_times", cfg.get("expected_n_times", 3000))))
    model_kwargs.setdefault("patch_size", int(cfg.get("patch_size", 200)))
    model_kwargs.setdefault("mask_ratio", float(cfg.get("mask_ratio", 0.75)))
    model_kwargs.setdefault("encoder_dim", int(cfg.get("encoder_dim", 512)))
    model_kwargs.setdefault("encoder_depth", int(cfg.get("encoder_depth", 4)))
    model_kwargs.setdefault("encoder_heads", int(cfg.get("encoder_heads", 4)))
    model_kwargs.setdefault("decoder_dim", int(cfg.get("decoder_dim", 512)))
    model_kwargs.setdefault("decoder_depth", int(cfg.get("decoder_depth", 2)))
    model_kwargs.setdefault("decoder_heads", int(cfg.get("decoder_heads", 4)))
    model_kwargs.setdefault("mlp_ratio", float(cfg.get("mlp_ratio", 4.0)))
    model_kwargs.setdefault("norm_pix_loss", bool(cfg.get("norm_pix_loss", False)))
    model_kwargs.setdefault("dropout", float(cfg.get("dropout", 0.0)))
    model_kwargs.setdefault("mask_strategy", str(cfg.get("mask_strategy", "random")))
    if common_ch_names is not None:
        model_kwargs.setdefault("common_ch_names", common_ch_names)
    if model_name == "BrainLMMicrostate":
        model_kwargs.setdefault(
            "canonical_maps_path",
            str(cfg.get("canonical_maps_path", "metadata/microstates/canonical_maps.npz")),
        )
        model_kwargs.setdefault("n_microstate_classes", int(cfg.get("n_microstate_classes", 4)))
        model_kwargs.setdefault("n_sub_bins", int(cfg.get("n_sub_bins", 8)))
        model_kwargs.setdefault("aux_loss_weight", float(cfg.get("aux_loss_weight", 0.5)))

    logger.info(
        f"Building MAE model: {model_name} | "
        f"c_common={c_common} n_times={model_kwargs['n_times']} "
        f"patch_size={model_kwargs['patch_size']} "
        f"mask_ratio={model_kwargs['mask_ratio']}"
    )
    model = model_cls(**model_kwargs).float().to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MAE parameters: total={total:,}  trainable={trainable:,}")
    return model


def build_mae_classifier(
    cfg: dict,
    n_classes: int,
    device: torch.device,
) -> BrainLMClassifier:
    """Pretrained BrainLM-EEG encoder + 3-layer MLP head on the CLS token."""
    encoder_path = cfg.get("pretrained_encoder_path")
    if not encoder_path:
        raise ValueError("cfg['pretrained_encoder_path'] is required for build_mae_classifier.")

    mode = cfg.get("mode", "linear_probe")
    if mode not in ("linear_probe", "finetune"):
        raise ValueError(f"mode must be 'linear_probe' or 'finetune', got '{mode}'.")
    freeze_encoder = mode == "linear_probe"

    common_ch_names = cfg.get("common_ch_names", [])
    c_common = len(common_ch_names)
    if c_common == 0:
        raise ValueError(
            "cfg['common_ch_names'] is empty. Load it from the checkpoint before calling "
            "build_mae_classifier: ckpt = torch.load(path); cfg['common_ch_names'] = ckpt['common_ch_names']"
        )

    # Reconstruct the pretrained architecture, then load the encoder weights.
    encoder = build_mae_model(cfg, c_common=c_common, device=device, common_ch_names=common_ch_names)
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    logger.info(
        f"Loaded encoder from {encoder_path} | "
        f"missing={len(missing)} unexpected={len(unexpected)} keys"
    )
    non_decoder_missing = [k for k in missing if not k.startswith(_DECODER_PREFIXES)]
    if non_decoder_missing or unexpected:
        raise RuntimeError(
            f"Encoder checkpoint {encoder_path} does not match the configured "
            f"{cfg.get('model_name')} architecture: missing encoder keys "
            f"{non_decoder_missing}, unexpected keys {list(unexpected)}."
        )

    model = BrainLMClassifier(
        encoder=encoder, n_classes=n_classes, freeze_encoder=freeze_encoder
    ).float().to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"BrainLMClassifier | mode={mode} | total={total:,} trainable={trainable:,} | "
        f"c_common={c_common} n_classes={n_classes}"
    )
    return model


__all__ = [
    "build_model",
    "build_mae_model",
    "build_mae_classifier",
    "MAE_MODEL_REGISTRY",
    "BrainLMEEG",
    "BrainLMClassifier",
    "BrainLMSphereRoPE",
    "BrainLMMicrostate",
    "LayerProbeClassifier",
]

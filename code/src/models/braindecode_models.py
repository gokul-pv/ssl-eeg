"""Braindecode model factory."""

from __future__ import annotations

import logging
import os

import torch.nn as nn

logger = logging.getLogger(__name__)

_reve_cache_patched = False


def _patch_reve_position_bank_cache(cache_dir: str) -> None:
    """Redirect REVE's electrode-position-bank cache to a writable directory."""
    global _reve_cache_patched
    if _reve_cache_patched:
        return
    from braindecode.models.reve import RevePositionBank

    os.makedirs(cache_dir, exist_ok=True)
    url_default, timeout_default, _ = RevePositionBank.__init__.__defaults__
    RevePositionBank.__init__.__defaults__ = (url_default, timeout_default, cache_dir)
    _reve_cache_patched = True
    logger.info(f"REVE position-bank cache redirected to: {cache_dir}")


def _canonicalize_reve_ch_names(ch_names: list[str]) -> list[str]:
    """Rewrite *ch_names* to the exact spellings REVE's position bank uses."""
    from braindecode.models.reve import RevePositionBank

    bank_keys = RevePositionBank().mapping
    case_insensitive: dict[str, str] = {}
    for key in bank_keys:
        # First spelling wins; pairs are coordinate-identical so the choice is moot.
        case_insensitive.setdefault(key.upper(), key)

    resolved: list[str] = []
    recased: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in ch_names:
        if name in bank_keys:
            resolved.append(name)
            continue
        canonical = case_insensitive.get(name.strip().upper())
        if canonical is None:
            missing.append(name)
            continue
        resolved.append(canonical)
        recased.append((name, canonical))

    if missing:
        raise ValueError(
            f"REVE position bank has no entry for {len(missing)} of {len(ch_names)} "
            f"channels: {missing}. REVE silently drops unresolvable channels, which "
            f"desynchronises its positional embedding from the input and fails later "
            f"as a tensor-size mismatch. Remove these channels from the dataset "
            f"config's common_ch_names (or exclude them in preprocessing)."
        )
    if recased:
        logger.info(
            f"REVE: re-cased {len(recased)} channel name(s) to the position bank's "
            f"spelling (e.g. {recased[0][0]} → {recased[0][1]})."
        )
    return resolved


def _load_hf_state_dict(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> dict:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id=repo_id, revision=revision, token=token)
    candidates = ["model.safetensors", "pytorch_model.bin"]
    filename = next((name for name in candidates if name in files), None)
    if filename is None:
        raise FileNotFoundError(
            f"No supported checkpoint file found in repo '{repo_id}'. "
            f"Expected one of: {candidates}"
        )

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )

    if filename.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    import torch

    return torch.load(path, map_location="cpu")


def _load_compatible_pretrained_weights(
    model: nn.Module,
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> None:
    state = _load_hf_state_dict(repo_id=repo_id, revision=revision, token=token)
    model_state = model.state_dict()

    compatible = {
        k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape
    }
    skipped = len(state) - len(compatible)

    model.load_state_dict(compatible, strict=False)
    logger.info(
        f"Loaded {len(compatible)} compatible pretrained tensors from {repo_id}; "
        f"skipped {skipped} shape-mismatched/unexpected tensors."
    )


def _freeze_backbone_for_probe(model: nn.Module) -> None:
    """Freeze every parameter except the model's classification head."""
    head = getattr(model, "final_layer", None)
    if head is None:
        raise ValueError(
            "freeze_backbone=True requires the model to expose a 'final_layer' "
            f"head, but {type(model).__name__} has none."
        )

    for p in model.parameters():
        p.requires_grad_(False)
    head_param_ids = set()
    for p in head.parameters():
        p.requires_grad_(True)
        head_param_ids.add(id(p))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if id(p) in head_param_ids
    )
    logger.info(
        f"Linear probe [freeze_backbone]: backbone FROZEN | "
        f"trainable head params={trainable:,} / total={total:,}"
    )


def build_braindecode_model(cfg: dict, data_info: dict) -> nn.Module:
    """Instantiate a braindecode model by name."""
    import braindecode.models as bd_models

    model_name = cfg.get("model_name")
    if model_name is None:
        raise ValueError("Config must specify 'model_name' for braindecode models.")

    model_cls = getattr(bd_models, model_name, None)
    if model_cls is None:
        available = [m for m in dir(bd_models) if not m.startswith("_")]
        raise ValueError(
            f"braindecode model '{model_name}' not found. "
            f"Available: {available}"
        )

    model_kwargs = dict(cfg.get("model_kwargs", {}))
    pretrained_repo_id = cfg.get("pretrained_repo_id")

    # For regular construction, infer standard EEG args from dataset.
    # For from_pretrained, do not force signal-shape args because the saved
    # HF config may include chs_info that conflicts with n_chans overrides.
    if not pretrained_repo_id:
        model_kwargs.setdefault("n_chans", data_info["n_chans"])
        model_kwargs.setdefault("n_outputs", data_info["n_classes"])
        model_kwargs.setdefault("n_times", data_info["n_times"])
        if "sfreq" in data_info:
            model_kwargs.setdefault("sfreq", data_info["sfreq"])

    logger.info(
        f"Building braindecode model: {model_name} | "
        f"n_chans={data_info['n_chans']} n_outputs={data_info['n_classes']} "
        f"n_times={data_info['n_times']}"
    )

    # ── Model-specific kwarg normalisation ────────────────────────────────
    # ATCNet: tcn_activation expects an nn.Module class, not a string.
    if model_name == "ATCNet" and "tcn_activation" in model_kwargs:
        import torch.nn as _nn
        _activation_map = {
            "elu":   _nn.ELU,
            "relu":  _nn.ReLU,
            "gelu":  _nn.GELU,
            "selu":  _nn.SELU,
            "tanh":  _nn.Tanh,
            "silu":  _nn.SiLU,
            "mish":  _nn.Mish,
        }
        act_val = model_kwargs["tcn_activation"]
        if isinstance(act_val, str):
            act_cls = _activation_map.get(act_val.lower())
            if act_cls is None:
                raise ValueError(
                    f"Unknown tcn_activation string '{act_val}'. "
                    f"Valid options: {list(_activation_map)}"
                )
            model_kwargs["tcn_activation"] = act_cls
            logger.info(f"  ATCNet: converted tcn_activation='{act_val}' → {act_cls}")

    # ATCNet: accept legacy 'fuse' key and convert to 'concat' bool.
    if model_name == "ATCNet" and "fuse" in model_kwargs:
        fuse_val = model_kwargs.pop("fuse")
        model_kwargs.setdefault("concat", fuse_val == "concat")
        logger.info(
            f"  ATCNet: converted legacy fuse='{fuse_val}' → concat={model_kwargs['concat']}"
        )

    # ── Foundation-model-specific construction kwargs ─────────────────────
    ch_names = data_info.get("ch_names")
    if model_name == "REVE":
        _patch_reve_position_bank_cache(
            cfg.get("reve_position_cache_dir", "outputs/hf_cache/reve_positions")
        )
        # REVE resolves 3D electrode positions from channel names at
        # construction (populating self.default_pos); forward() then works with
        # pos=None. Without chs_info it raises "No positions provided".
        if not ch_names:
            raise ValueError(
                "REVE requires data_info['ch_names'] to resolve electrode "
                "positions. Ensure the entry point passes channel names in "
                "data_info['ch_names']."
            )
        model_kwargs.setdefault(
            "chs_info", [{"ch_name": c} for c in _canonicalize_reve_ch_names(ch_names)]
        )
    if model_name == "CBraMod":
        # CBraMod builds a concrete Flatten+Linear head only when n_chans and
        # n_times are known at construction; otherwise it uses a LazyLinear head
        # whose params cannot be frozen for a linear probe until first forward.
        model_kwargs.setdefault("n_chans", data_info["n_chans"])
        model_kwargs.setdefault("n_times", data_info["n_times"])

    logger.info(f"Extra model_kwargs: {model_kwargs}")

    if pretrained_repo_id:
        if not hasattr(model_cls, "from_pretrained"):
            raise ValueError(
                f"Model '{model_name}' does not support from_pretrained()."
            )

        pretrained_kwargs = dict(cfg.get("pretrained_kwargs", {}))
        if cfg.get("pretrained_revision") and "revision" not in pretrained_kwargs:
            pretrained_kwargs["revision"] = cfg["pretrained_revision"]
        if cfg.get("pretrained_token") and "token" not in pretrained_kwargs:
            pretrained_kwargs["token"] = cfg["pretrained_token"]

        # Only pass safe overrides into from_pretrained.
        # n_outputs is commonly changed for downstream fine-tuning; shape-related
        # signal args are left to the saved HF config to avoid chs_info mismatch.
        if "n_outputs" not in pretrained_kwargs:
            pretrained_kwargs["n_outputs"] = model_kwargs.get(
                "n_outputs", data_info["n_classes"]
            )
        if "n_times" not in pretrained_kwargs and "n_times" in data_info:
            pretrained_kwargs["n_times"] = data_info["n_times"]
        # Apply user-provided model kwargs except shape-defining ones.
        for key, value in model_kwargs.items():
            if key in {"n_chans", "n_times", "sfreq", "input_window_seconds", "chs_info"}:
                continue
            pretrained_kwargs.setdefault(key, value)

        # Re-admit the shape args the two channel/position-sensitive models
        # genuinely need at construction (excluded above for the general case).
        if model_name == "REVE" and "chs_info" in model_kwargs:
            pretrained_kwargs.setdefault("chs_info", model_kwargs["chs_info"])
        if model_name == "CBraMod":
            pretrained_kwargs.setdefault("n_chans", model_kwargs["n_chans"])

        logger.info(
            f"Loading pretrained weights from Hugging Face Hub: {pretrained_repo_id}"
        )
        try:
            model = model_cls.from_pretrained(pretrained_repo_id, **pretrained_kwargs)
        except RuntimeError as exc:
            # A partial load leaves every shape-mismatched tensor randomly
            # initialised behind a frozen "pretrained" backbone, so it is an
            # error unless explicitly allowed.
            if not bool(cfg.get("allow_partial_load", False)):
                raise RuntimeError(
                    f"from_pretrained('{pretrained_repo_id}') failed: {exc}\n"
                    "Set allow_partial_load: true to load only the shape-compatible "
                    "tensors instead (the remaining weights stay randomly initialised)."
                ) from exc
            logger.warning(
                "from_pretrained failed with shape mismatch; allow_partial_load=true, "
                f"falling back to compatible tensor loading. Details: {exc}"
            )
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.setdefault("n_chans", data_info["n_chans"])
            fallback_kwargs.setdefault("n_outputs", data_info["n_classes"])
            fallback_kwargs.setdefault("n_times", data_info["n_times"])
            if "sfreq" in data_info:
                fallback_kwargs.setdefault("sfreq", data_info["sfreq"])
            model = model_cls(**fallback_kwargs)
            _load_compatible_pretrained_weights(
                model,
                repo_id=pretrained_repo_id,
                revision=pretrained_kwargs.get("revision"),
                token=pretrained_kwargs.get("token"),
            )
    else:
        model = model_cls(**model_kwargs)

    # ── Linear-probe: freeze the backbone, keep only the head trainable ────
    if cfg.get("freeze_backbone", False):
        _ensure_head_materialized(model, data_info)
        _freeze_backbone_for_probe(model)

    return model


def _ensure_head_materialized(model: nn.Module, data_info: dict) -> None:
    """Run one dummy forward if the model has any lazy (uninitialised) params."""
    import torch
    from torch.nn.parameter import UninitializedParameter

    if not any(isinstance(p, UninitializedParameter) for p in model.parameters()):
        return

    logger.info("Lazy head detected — running a dummy forward to materialise it.")
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, int(data_info["n_chans"]), int(data_info["n_times"]))
    with torch.no_grad():
        model(dummy)
    model.train(was_training)

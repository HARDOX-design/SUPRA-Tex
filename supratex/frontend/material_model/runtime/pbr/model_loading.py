# Adapted for SUPRA-Tex distribution (2026): packaged local inference runtime.
# Upstream FireRed portions remain under Apache-2.0; see THIRD_PARTY_NOTICES.md.
# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""Strict checkpoint loading for the FireRed material inference model."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
JOINT_MODULE_PREFIX = "modality_joint_blocks."


def _require_exact_loading_info(loading_info, source: str) -> None:
    problems = {
        name: list(loading_info.get(name, []))
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    if problems:
        summary = ", ".join(f"{name}={len(values)}" for name, values in problems.items())
        raise RuntimeError(f"{source} transformer load is not exact: {summary}")

def _load_base_transformer(args, weight_dtype, *, use_layer3d_rope: bool = False):
    from .qwen_pbr_factorized import (
        PRODUCTION_JOINT_HEADS,
        PRODUCTION_JOINT_MODALITIES,
        PRODUCTION_JOINT_RANK,
        QwenImagePBRFactorizedTransformer2DModel,
    )

    loaded = QwenImagePBRFactorizedTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        joint_sites=(),
        joint_modalities=PRODUCTION_JOINT_MODALITIES,
        joint_rank=PRODUCTION_JOINT_RANK,
        joint_heads=PRODUCTION_JOINT_HEADS,
        use_layer3d_rope=use_layer3d_rope,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("custom transformer loader did not return loading information")
    transformer, loading_info = loaded
    _require_exact_loading_info(loading_info, "base")
    if tuple(transformer.joint_sites):
        raise RuntimeError("base transformer must load without joint modules")
    return transformer

def _resolve_full_transformer_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "diffusion_pytorch_model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"full transformer checkpoint not found: {checkpoint}")
    return checkpoint

def _load_checkpoint_state(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"transformer checkpoint must contain a state dict: {path}")
    return state

def _load_exact_full_transformer(
    transformer,
    checkpoint_path: str | Path,
    *,
    allow_joint_sidecar: bool = False,
) -> None:
    path = _resolve_full_transformer_checkpoint(checkpoint_path)
    state = _load_checkpoint_state(path)
    if allow_joint_sidecar:
        state = {
            name: value
            for name, value in state.items()
            if not name.startswith(JOINT_MODULE_PREFIX)
        }
    result = transformer.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "full transformer checkpoint load is not exact: "
            f"missing_keys={len(result.missing_keys)}, "
            f"unexpected_keys={len(result.unexpected_keys)}"
        )
    logger.info( "Loaded exact full transformer checkpoint: %s", path)

def _resolve_joint_adapter_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "joint_adapter.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"joint adapter checkpoint not found: {checkpoint}")
    return checkpoint

def _load_exact_joint_adapter(transformer, checkpoint_path: str | Path) -> None:
    path = _resolve_joint_adapter_checkpoint(checkpoint_path)
    state = _load_checkpoint_state(path)
    expected = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if name.startswith(JOINT_MODULE_PREFIX)
    }
    actual_keys = set(state)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            "joint adapter checkpoint is not exact: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    mismatched = [
        name
        for name, parameter in expected.items()
        if not isinstance(state[name], torch.Tensor)
        or tuple(state[name].shape) != tuple(parameter.shape)
    ]
    if mismatched:
        raise RuntimeError(f"joint adapter checkpoint shape mismatch: {mismatched[:3]}")
    result = transformer.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "joint adapter checkpoint has unexpected keys after load: "
            f"{list(result.unexpected_keys)[:3]}"
        )
    logger.info( "Loaded exact joint adapter checkpoint: %s", path)

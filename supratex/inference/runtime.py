"""Inference-only attention selection and CPU-offload policy.

Training keeps its explicit attention configuration. This module does not
select a device or import CUDA extensions until configuration is requested.
"""
from __future__ import annotations

import importlib
import os
import sys

ATTENTION_CHOICES = ("auto", "flash_attn_3", "flash_attn", "sdpa", "xformers")
_EXTENSIONS = {
    "flash_attn_3": "flash_attn_interface",
    "flash_attn": "flash_attn",
    "flash_attn_4": "flash_attn.cute",
    "xformers": "xformers.ops",
}


def _cuda_info(device: str) -> dict | None:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    selected = torch.device(device)
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {"name": properties.name, "capability": (properties.major, properties.minor),
            "total_memory": properties.total_memory}


def _extension_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False
    return True


def _resolve_attention(requested: str, info: dict | None) -> str:
    major = info["capability"][0] if info is not None else None
    if requested == "auto":
        if major == 9 and _extension_available("flash_attn_interface"):
            return "flash_attn_3"
        if major in (8, 9) and _extension_available("flash_attn"):
            return "flash_attn"
        return "sdpa"
    if requested not in (*_EXTENSIONS, "sdpa"):
        raise ValueError(f"Unsupported inference attention backend: {requested!r}")
    if requested == "flash_attn_3" and major is not None and major != 9:
        raise ValueError("FlashAttention-3 requires a Hopper GPU; use --attention auto, "
                         "flash_attn, or sdpa on A100/RTX GPUs, and clear incompatible "
                         "ATTN_BACKEND / SPARSE_ATTN_BACKEND overrides.")
    if requested == "flash_attn" and major is not None and major < 8:
        raise ValueError("FlashAttention-2 requires an Ampere or newer supported GPU.")
    if requested in _EXTENSIONS and not _extension_available(_EXTENSIONS[requested]):
        raise RuntimeError(f"Requested attention backend {requested!r} cannot be imported; "
                           "install its matching extension or select --attention sdpa.")
    return requested


def _read_low_vram(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError("SUPRATEX_LOW_VRAM must be 1 or 0")


def configure_inference_runtime(*, attention: str = "auto", low_vram: bool | None = None,
                                device: str = "cuda", material_backend: str | None = None) -> dict:
    """Resolve defaults, preserve explicit overrides, and configure subprocesses.

    An explicit CLI attention selection overrides both backend environment
    variables. Auto respects them; with no override it probes hardware and
    importable extensions. CPU offload defaults on for GPUs up to 24 GiB.
    """
    if attention not in ATTENTION_CHOICES:
        raise ValueError(f"Unsupported --attention value: {attention!r}")
    info = _cuda_info(device)
    dense_request = attention if attention != "auto" else os.environ.get("ATTN_BACKEND", "auto")
    dense = _resolve_attention(dense_request, info)
    sparse_request = (attention if attention != "auto" else
                      os.environ.get("SPARSE_ATTN_BACKEND", dense))
    sparse = _resolve_attention(sparse_request, info)
    if low_vram is None:
        configured = os.environ.get("SUPRATEX_LOW_VRAM")
        low_vram = (_read_low_vram(configured) if configured is not None else
                    info is not None and info["total_memory"] <= 24 * 1024**3)
    if low_vram and material_backend == "firered":
        raise ValueError("Low-VRAM image-to-mesh inference currently supports the Qwen "
                         "material backend. Use --material-backend qwen-image; the full "
                         "FireRed material model has no CPU-offload path here.")

    # Set only after validation, so an invalid request cannot partially change
    # the environment inherited by geometry or material subprocesses.
    os.environ.update(ATTN_BACKEND=dense, SPARSE_ATTN_BACKEND=sparse,
                      SUPRATEX_LOW_VRAM="1" if low_vram else "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for package in ("supratex", "pixal3d"):
        module = sys.modules.get(f"{package}.modules.attention.config")
        if module is not None:
            module.set_backend(dense)
        module = sys.modules.get(f"{package}.modules.sparse.config")
        if module is not None:
            module.set_attn_backend(sparse)
    result = {"attention": dense, "sparse_attention": sparse,
              "low_vram": bool(low_vram), "device": info}
    print(f"[Supra-Tex inference] attention={dense}; sparse_attention={sparse}; "
          f"low_vram={bool(low_vram)}", flush=True)
    return result

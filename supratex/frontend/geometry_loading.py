"""Strict local checkpoint loading without initializing discarded geometry weights."""

import json
from pathlib import Path

import torch


_MODEL_NAMES = {
    "SparseStructureFlowModel", "SparseStructureDecoder",
    "ElasticSLatFlowModel", "FlexiDualGridVaeDecoder",
}


def _restore_position_frequencies(model):
    """Recreate small native tensors omitted from state_dict, on the CPU."""
    for module in model.modules():
        frequencies = vars(module).get("freqs")
        if not isinstance(frequencies, torch.Tensor) or not frequencies.is_meta:
            continue
        cls = type(module)
        with torch.device("cpu"):
            if (cls.__name__ in {"RotaryPositionEmbedder", "SparseRotaryPositionEmbedder"}
                    and cls.__module__.startswith("pixal3d.modules.")
                    and cls.__module__.endswith(".attention.rope")):
                original = cls(module.head_dim, dim=module.dim, rope_freq=module.rope_freq)
            elif (cls.__name__ == "AbsolutePositionEmbedder"
                    and cls.__module__ == "pixal3d.modules.transformer.blocks"):
                original = cls(module.channels, in_channels=module.in_channels)
            else:
                raise ValueError(f"Unsupported geometry meta frequency tensor: {cls.__module__}.{cls.__name__}")
        module.freqs = original.freqs


RUNTIME_SUFFIX = ".runtime.safetensors"


def _meta_model(path: Path):
    from pixal3d import models

    config = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if config["name"] not in _MODEL_NAMES:
        raise ValueError(f"Unsupported native geometry model: {config['name']}")
    with torch.device("meta"):
        return getattr(models, config["name"])(**config["args"])


def _source_identity(path: Path) -> dict[str, str]:
    stat = Path(str(path) + ".safetensors").resolve().stat()
    return {"source_size": str(stat.st_size), "source_mtime_ns": str(stat.st_mtime_ns)}


def _runtime_state(path: Path, expected: dict) -> dict:
    """Read checkpoint tensors and cast them to the meta shell's dtypes."""
    from safetensors.torch import load_file

    state = load_file(str(path) + ".safetensors", device="cpu")
    missing, unexpected = expected.keys() - state.keys(), state.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"Geometry checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, target in expected.items():
        value = state[name]
        if value.shape != target.shape:
            raise ValueError(f"Geometry checkpoint shape mismatch for {name}: {tuple(value.shape)} != {tuple(target.shape)}")
        state[name] = value.to(dtype=target.dtype)
    return state


def write_runtime_weights(path: Path) -> Path | None:
    """Save the cast state beside the checkpoint when that copy is smaller.

    Native flow checkpoints store float32 for bfloat16 blocks, so the cast copy
    halves the bytes read per load and loads bit-identical tensors. Decoders
    that store float16 for float32 layers would grow, so they are skipped (None).
    """
    from safetensors.torch import save_file

    path = Path(path)
    expected = _meta_model(path).state_dict()
    runtime_bytes = sum(target.numel() * target.element_size() for target in expected.values())
    # Only worth a second file when it is clearly smaller than the original.
    if runtime_bytes > .9 * Path(str(path) + ".safetensors").resolve().stat().st_size:
        return None
    target = Path(str(path) + RUNTIME_SUFFIX)
    temporary = target.with_name(target.name + ".tmp")
    save_file({name: tensor.contiguous() for name, tensor in _runtime_state(path, expected).items()},
              str(temporary), metadata=_source_identity(path))
    temporary.replace(target)
    return target


def _runtime_copy(path: Path) -> Path | None:
    """Use a runtime-dtype copy only if it was made from the current checkpoint."""
    from safetensors import safe_open

    runtime = Path(str(path) + RUNTIME_SUFFIX)
    if not runtime.is_file():
        return None
    with safe_open(str(runtime), framework="pt", device="cpu") as handle:
        if handle.metadata() != _source_identity(path):
            print(f"Ignoring stale runtime geometry weights: {runtime}", flush=True)
            return None
    return runtime


def load_native_geometry_model(path: Path):
    """Load a reviewed local Pixal3D network with its constructor's dtype ABI.

    Native flow checkpoints store float32 even for bfloat16 transformer blocks;
    decoder checkpoints can instead store float16 for float32 boundary layers.
    Cast to each meta-shell tensor's dtype before assignment, exactly as native
    copy-based load_state_dict does. A matching runtime-dtype copy written by
    scripts/convert_geometry_weights.py is read instead when present.
    Unknown/missing weights fail before return.
    """
    path = Path(path)
    model = _meta_model(path)
    expected = model.state_dict()
    runtime = _runtime_copy(path)
    state = _runtime_state(Path(str(runtime).removesuffix(".safetensors")) if runtime else path, expected)
    model.load_state_dict(state, strict=True, assign=True)
    _restore_position_frequencies(model)
    for name, module in model.named_modules():
        tensors = list(module._parameters.items()) + list(module._buffers.items()) + list(vars(module).items())
        for key, value in tensors:
            if isinstance(value, torch.Tensor) and value.is_meta:
                raise ValueError(f"Unmaterialized geometry tensor: {name}.{key}")
    return model.eval()

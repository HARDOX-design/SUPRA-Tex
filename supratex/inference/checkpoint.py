"""CPU loading and header-only validation for completion checkpoints."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in completion checkpoint JSON: {key!r}")
        result[key] = value
    return result


def _index_shards(path: Path) -> dict[Path, set[str]]:
    try:
        index = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"invalid completion checkpoint index JSON: {path}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("completion checkpoint index requires a nonempty weight_map object")
    shards: dict[Path, set[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("completion checkpoint index tensor names must be nonempty strings")
        if (not isinstance(filename, str) or not filename or "\\" in filename or "\x00" in filename
                or PurePosixPath(filename).is_absolute() or PureWindowsPath(filename).drive
                or any(part in ("", ".", "..") for part in filename.split("/"))
                or not filename.endswith(".safetensors")):
            raise ValueError(f"completion checkpoint shard must be a relative .safetensors filename without traversal: {filename!r}")
        # Hub snapshots legitimately use file symlinks into the cache's blobs
        # directory. Validate the lexical reference, not that symlink target.
        shard = path.parent / filename
        if not shard.is_file():
            raise FileNotFoundError(f"missing completion checkpoint shard: {shard}")
        shards.setdefault(shard, set()).add(key)
    return shards


def _safe_shards(path: Path) -> dict[Path, set[str]]:
    """Check every indexed key against shard headers without reading tensors."""
    from safetensors import SafetensorError, safe_open

    indexed = path.name.endswith(".safetensors.index.json")
    expected = _index_shards(path) if indexed else {path: set()}
    actual: dict[Path, set[str]] = {}
    seen: set[str] = set()
    for shard in expected:
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                # The safe loader validates header size/offsets but accepts
                # duplicate JSON names. Reject those before trusting its keys.
                with shard.open("rb") as source:
                    header_size = int.from_bytes(source.read(8), "little")
                    json.loads(source.read(header_size), object_pairs_hook=_unique_object)
                keys = set(handle.keys())
        except SafetensorError as error:
            raise ValueError(f"invalid completion safetensors checkpoint: {shard}: {error}") from error
        duplicate = seen & keys
        if duplicate:
            raise ValueError(f"duplicate tensor keys across completion checkpoint shards: {sorted(duplicate)}")
        seen.update(keys)
        actual[shard] = keys
    if indexed:
        for shard, keys in actual.items():
            missing, unexpected = expected[shard] - keys, keys - expected[shard]
            if missing or unexpected:
                raise ValueError(f"completion checkpoint weight_map mismatch for {shard.name}: "
                                 f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    return actual


def validate_completion_checkpoint(checkpoint: str | Path) -> Path:
    """Validate existence and safetensors headers; legacy pickle is not loaded."""
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    if path.name.endswith((".safetensors", ".safetensors.index.json")):
        _safe_shards(path)
    elif path.suffix not in (".pt", ".pth"):
        raise ValueError("completion checkpoint must be .pt, .pth, .safetensors, or .safetensors.index.json")
    return path


def load_completion_state_dict(checkpoint: str | Path, *, torch_module: Any = None) -> dict[str, Any]:
    """Load CPU tensors from safe files or a legacy weights-only checkpoint.

    Safetensors maps the files on CPU; the caller retains ownership of model
    device placement. All index/header validation finishes before tensor reads.
    """
    path = Path(checkpoint).expanduser()
    if path.name.endswith((".safetensors", ".safetensors.index.json")):
        from safetensors import safe_open
        shards = _safe_shards(path)
        state = {}
        for shard, keys in shards.items():
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                state.update((key, handle.get_tensor(key)) for key in sorted(keys))
    else:
        if path.suffix not in (".pt", ".pth"):
            raise ValueError("unsupported completion checkpoint format")
        if torch_module is None:
            import torch as torch_module
        state = torch_module.load(path, map_location="cpu", weights_only=True)
        state = state.get("model", state) if isinstance(state, dict) else state
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a model state dictionary")
    result = {}
    for key, tensor in state.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint model state keys must be strings")
        name = key.removeprefix("module.")
        if name in result:
            raise ValueError(f"duplicate completion tensor key after removing module prefix: {name!r}")
        result[name] = tensor
    return result

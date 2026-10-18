"""Record material artifact identity without loading model tensors."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any


@lru_cache(maxsize=64)
def _content_hash(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    current = Path(path).stat()
    if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (size, mtime_ns, ctime_ns):
        raise RuntimeError(f"material artifact changed while hashing: {path}")
    return value.hexdigest()


def artifact_identity(signature: dict[str, Any]) -> dict[str, Any]:
    """Hash adaptation weights and small contracts; bind large base shards by stat.

    This detects ordinary file replacement between the two predictions. Base
    shard metadata is explicitly recorded as metadata, not a content checksum.
    """
    result: dict[str, Any] = {"method": "adaptation SHA256; base shard size/mtime/ctime; small contract SHA256"}
    for key in ("base", "model", "weights", "spec", "checkpoint_manifest"):
        selected = signature.get(key)
        if selected is None:
            continue
        root = Path(selected)
        if not root.exists():
            raise FileNotFoundError(f"material provenance requires resolved local artifacts: {root}")
        files = [root] if root.is_file() else sorted(
            path for path in root.rglob("*") if path.is_file()
            and path.suffix in {".json", ".safetensors", ".pt", ".bin"})
        records = {}
        for path in files:
            stat = path.stat()
            record = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
            adaptation = key in {"model", "weights"} and path.suffix in {".safetensors", ".pt", ".bin"}
            if adaptation or stat.st_size <= 1024 * 1024:
                record["sha256"] = _content_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            records[path.name if root.is_file() else path.relative_to(root).as_posix()] = record
        result[key] = records
    return result

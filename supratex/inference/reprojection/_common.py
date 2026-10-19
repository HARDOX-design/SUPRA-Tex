"""File bindings and measured acceptance checks shared by atlas operators."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

OUTSIDE_COLOR_MAX_ABS = 5 / 255.0
REPLAY_MAX_ABS = 1e-5
# Validation render phases, as sub-pixel camera offsets: a 4x4 grid plus three
# off-grid points. Both operators render every candidate at all nineteen.
GRID_PHASES = tuple((x / 4, y / 4) for x in range(4) for y in range(4))
OFFGRID_PHASES = ((.125, .375), (.375, .625), (.875, .125))
IDENTITY_GATES = (
    "materials_exact", "mesh_structure_exact", "node_structure_exact",
    "nonbase_buffer_views_exact",
)


def outside_color_passes(value: float) -> bool:
    return math.isfinite(value) and 0 <= value <= OUTSIDE_COLOR_MAX_ABS


def write_json(path: Path, value: Any) -> None:
    """Publish one new JSON artifact without overwriting an earlier result."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def bind_file(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def check_binding(record: Mapping[str, Any], path: Path | None = None) -> Path:
    bound = Path(record["path"])
    if path is not None and bound.resolve() != path.resolve():
        raise ValueError(f"artifact path binding changed: {path}")
    if dict(record) != bind_file(bound):
        raise ValueError(f"artifact content binding changed: {bound}")
    return bound


def load_renderer(path: Path):
    """Only the renderer shipped in this package may establish these gates."""
    from . import _renderer
    if path.resolve() != Path(_renderer.__file__).resolve():
        raise ValueError("reprojection requires the bundled SupraTex renderer")
    return _renderer


def validate_candidate(directory: Path, *, continuous: bool) -> dict[str, Any]:
    """Read every phase receipt and validate numerical gates, not just status."""
    receipt = read_json(directory / "RECEIPT.json")
    if read_json(directory / "RESULT.json") != receipt:
        raise ValueError("candidate result and receipt disagree")
    if receipt.get("status") != "valid":
        raise ValueError(f"candidate was not accepted: {directory}")
    gates = receipt.get("gates", {})
    if gates.get("accepted") is not True or gates.get("all_19_phase_renders") is not True:
        raise ValueError("candidate lacks complete acceptance gates")
    if not all(gates.get("nonbase_identity", {}).get(key) is True for key in IDENTITY_GATES):
        raise ValueError("candidate lacks non-base preservation gates")
    if continuous:
        if not all(gates.get(key) is True for key in ("UV_covered_exact", "project_active_exact", "blocked_exact", "alpha_exact")):
            raise ValueError("continuous candidate lacks atlas preservation gates")
    elif receipt.get("atlas", {}).get("alpha_exact") is not True:
        raise ValueError("projection candidate did not preserve alpha")
    candidate = directory / "candidate" / "mesh.glb"
    if Path(receipt.get("candidate_glb", "")).resolve() != candidate.resolve() or not candidate.is_file():
        raise ValueError("candidate mesh binding is missing or incorrect")
    for split, expected in (("grid", GRID_PHASES), ("offgrid", OFFGRID_PHASES)):
        rows = receipt.get("phase_results", {}).get(split, [])
        if len(rows) != len(expected):
            raise ValueError(f"incomplete {split} phase coverage")
        for phase, row in zip(expected, rows):
            if row.get("phase") != list(phase) or row.get("passes") is not True:
                raise ValueError("phase coordinates or acceptance changed")
            replay = float(row["quantized_atlas_replay_max"])
            if not math.isfinite(replay) or not 0 <= replay <= REPLAY_MAX_ABS:
                raise ValueError("renderer replay gate failed")
            if row.get("outside_source_hardtrust_max_abs_threshold") != OUTSIDE_COLOR_MAX_ABS:
                raise ValueError("outside-source color threshold changed")
            if not outside_color_passes(float(row["outside_source_hardtrust_delta_max_abs"])):
                raise ValueError("outside-source color gate failed")
            if row.get("mask_exact") is not True or row.get("mra_exact") is not True:
                raise ValueError("phase mask or material preservation gate is missing")
            folder = directory / "renders" / split / f"x{phase[0]:.3f}_y{phase[1]:.3f}"
            if read_json(folder / "PHASE_RECEIPT.json") != row:
                raise ValueError("phase detail differs from candidate receipt")
            evidence = row.get("evidence", {})
            names = ("repair", "project") if continuous else ("candidate", "baseline")
            if set(evidence) != set(names):
                raise ValueError("phase lacks its two rendered evidence arrays")
            for name in names:
                check_binding(evidence[name], folder / f"{name}.npz")
    return receipt

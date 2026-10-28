"""Atlas reprojection with measured candidate acceptance gates.

Each reprojected view writes observed albedo into visible atlas texels, then
repairs one-texel UV gutters from the observed albedo gradient. Candidates
are re-rendered at nineteen validation phases (a 4x4 sub-pixel grid and three
off-grid offsets) and must pass every gate. Dual-view texturing
stitches front and back with exact front priority and binary preservation; the
stitched result is not itself subjected to another nineteen render phases.
Single-view texturing can reproject its reference view alone.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import _glb
from ._common import (
    bind_file, check_binding, read_json, validate_candidate, write_json,
)
from ._transform import (
    atlas, back_to_front, camera_spec, front_protection, stitch,
    transform_glb, transformed_payload, validate_stitch,
)

__all__ = ["run_dual_view", "run_front_view", "validate_dual_view"]
PROFILE = "supratex.dual_view.reproject_continuous"


def _prepare_inputs(source: Path, destination: Path, camera: dict[str, Any], keep: Any = None) -> dict[str, Any]:
    """Align observation alpha and RGB once, preserving native albedo/MRA maps."""
    from PIL import Image

    names = ("condition_rgb.png", "condition_rgba.png", "albedo.png", "mra.png")
    bindings = {name: bind_file(source / name) for name in names}
    if (source / "camera.json").is_file():
        bindings["camera.json"] = bind_file(source / "camera.json")
    destination.mkdir(parents=True)
    with Image.open(source / "condition_rgb.png") as rgb, Image.open(source / "condition_rgba.png") as rgba:
        if rgb.size != rgba.size:
            raise ValueError("condition RGB and RGBA dimensions must agree")
        resized = rgb.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        alpha = rgba.convert("RGBA").getchannel("A").point(lambda value: 255 if value >= 254 else 0)
        alpha = alpha.resize((1024, 1024), Image.Resampling.NEAREST)
        if keep is not None:
            import numpy as np
            # Apply at the actual projection grid. A 1024 -> small source ->
            # 1024 round trip could otherwise resurrect thin rejected strips.
            alpha = Image.fromarray(np.where(keep, np.asarray(alpha), 0).astype(np.uint8))
        resized.save(destination / "condition_rgb.png")
        aligned = resized.copy()
        aligned.putalpha(alpha)
        aligned.save(destination / "condition_rgba.png")
    for name in ("albedo.png", "mra.png"):
        with Image.open(source / name) as image:
            if image.size != (1024, 1024):
                raise ValueError(f"{name} must have native dimensions 1024 by 1024")
            image.convert("RGB").save(destination / name)
    write_json(destination / "camera.json", camera)
    return bindings


def _command(module: str, arguments: list[str], log: Path) -> None:
    command = [sys.executable, "-m", f"supratex.inference.reprojection.{module}", *arguments]
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        # Share the caller's process group so cancellation reaches GPU workers.
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            result = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if result:
        raise RuntimeError(f"reprojection worker failed with exit code {result}; see {log}")


def _run_side(baseline: Path, source: Path, output: Path, camera: dict[str, Any], keep: Any = None) -> Path:
    """Publish one view only after the projection and continuous candidates pass."""
    output.mkdir(parents=True)
    baseline_binding = bind_file(baseline)
    code = {path.name: bind_file(path) for path in Path(__file__).parent.glob("*.py")}
    inputs = output / "inputs_1024"
    source_bindings = (_prepare_inputs(source, inputs, camera) if keep is None else
                       _prepare_inputs(source, inputs, camera, keep=keep))
    input_bindings = {path.name: bind_file(path) for path in inputs.iterdir() if path.is_file()}
    renderer = Path(__file__).with_name("_renderer.py").resolve()
    project, continuous = output / "project", output / "continuous"
    _command("_project", [
        "--baseline-glb", str(baseline),
        "--source-albedo", str(inputs / "albedo.png"),
        "--source-rgba", str(inputs / "condition_rgba.png"),
        "--camera", str(inputs / "camera.json"), "--renderer", str(renderer),
        "--output", str(project),
    ], output / "project.log")
    validate_candidate(project, continuous=False)
    manifest = output / "continuous_inputs.json"
    write_json(manifest, [{"label": "observation", "baseline_glb": str(baseline),
                          "work": str(inputs), "project_output": str(project)}])
    _command("_continuous", [
        "--manifest", str(manifest), "--case", "observation",
        "--renderer", str(renderer), "--output", str(continuous),
    ], output / "continuous.log")
    validate_candidate(continuous, continuous=True)
    for name, record in source_bindings.items():
        check_binding(record, source / name)
    check_binding(baseline_binding, baseline)
    for name, record in code.items():
        check_binding(record, Path(__file__).with_name(name))
    final = output / "mesh.glb"
    shutil.copyfile(continuous / "candidate" / "mesh.glb", final)
    artifacts = {str(path.relative_to(output)): bind_file(path) for path in (
        project / "PLAN.json", project / "RECEIPT.json", project / "RESULT.json",
        project / "COMMON_DOMAIN.npz", project / "candidate" / "mesh.glb",
        continuous / "PLAN.json", continuous / "RECEIPT.json", continuous / "RESULT.json",
        continuous / "GUTTER_DIAGNOSTICS.npz", continuous / "candidate" / "mesh.glb",
        manifest, final,
    )}
    receipt = {
        "profile": PROFILE, "status": "valid", "baseline": baseline_binding,
        "source_inputs": source_bindings, "aligned_inputs": input_bindings,
        "code": code, "artifacts": artifacts, "mesh": str(final),
        "gates": {"accepted": True, "project_accepted_all_19": True, "continuous_accepted_all_19": True},
    }
    write_json(output / "receipt.json", receipt)
    _validate_side(output)
    return final


def _validate_side(output: Path) -> dict[str, Any]:
    import numpy as np

    receipt = read_json(output / "receipt.json")
    if receipt.get("profile") != PROFILE or receipt.get("status") != "valid":
        raise ValueError("invalid reprojection receipt profile or status")
    if not all(receipt.get("gates", {}).get(key) is True for key in ("accepted", "project_accepted_all_19", "continuous_accepted_all_19")):
        raise ValueError("reprojection side is missing mandatory gates")
    baseline = check_binding(receipt["baseline"])
    required_inputs = {"condition_rgb.png", "condition_rgba.png", "albedo.png", "mra.png"}
    if not required_inputs.issubset(receipt.get("source_inputs", {})):
        raise ValueError("reprojection side has incomplete source bindings")
    if set(receipt.get("aligned_inputs", {})) != required_inputs | {"camera.json"}:
        raise ValueError("reprojection side has incomplete aligned input bindings")
    for record in receipt["source_inputs"].values():
        check_binding(record)
    for name, record in receipt["aligned_inputs"].items():
        check_binding(record, output / "inputs_1024" / name)
    expected_code = {path.name for path in Path(__file__).parent.glob("*.py")}
    if set(receipt.get("code", {})) != expected_code:
        raise ValueError("reprojection code bindings are incomplete")
    for name, record in receipt["code"].items():
        check_binding(record, Path(__file__).with_name(name))
    expected_artifacts = {
        "project/PLAN.json", "project/RECEIPT.json", "project/RESULT.json", "project/COMMON_DOMAIN.npz",
        "project/candidate/mesh.glb",
        "continuous/PLAN.json", "continuous/RECEIPT.json", "continuous/RESULT.json",
        "continuous/GUTTER_DIAGNOSTICS.npz", "continuous/candidate/mesh.glb", "continuous_inputs.json", "mesh.glb",
    }
    if set(receipt.get("artifacts", {})) != expected_artifacts:
        raise ValueError("reprojection artifact bindings are incomplete")
    for name, record in receipt["artifacts"].items():
        check_binding(record, output / name)
    if Path(receipt.get("mesh", "")).resolve() != (output / "mesh.glb").resolve():
        raise ValueError("reprojection output mesh path changed")
    for name, continuous in (("project", False), ("continuous", True)):
        validate_candidate(output / name, continuous=continuous)
    project_plan = read_json(output / "project" / "PLAN.json")
    expected_plan = {
        "baseline_glb": baseline,
        "source_albedo": output / "inputs_1024" / "albedo.png",
        "source_rgba": output / "inputs_1024" / "condition_rgba.png",
        "camera": output / "inputs_1024" / "camera.json",
    }
    for key, path in expected_plan.items():
        if Path(project_plan.get(key, "")).resolve() != path.resolve():
            raise ValueError(f"projection plan has an incorrect {key} binding")
    base_doc, base_binary, base_view, _, _, base_pixels = atlas(baseline)
    for name in ("project/candidate/mesh.glb", "continuous/candidate/mesh.glb", "mesh.glb"):
        candidate = output / name
        _glb._nonbase_identity_check(base_doc, base_binary, candidate, base_view)
        pixels = atlas(candidate)[-1]
        if pixels.shape != base_pixels.shape or (pixels.shape[-1] == 4 and not np.array_equal(pixels[..., 3], base_pixels[..., 3])):
            raise ValueError("candidate altered atlas dimensions or alpha")
    if bind_file(output / "mesh.glb")["sha256"] != bind_file(output / "continuous/candidate/mesh.glb")["sha256"]:
        raise ValueError("side output differs from its accepted continuous candidate")
    return receipt


def run_front_view(native_glb: Path, input_dir: Path, output_dir: Path, keep: Any = None) -> Path:
    """Reproject the single reference view onto a single-view result.

    This is the front half of dual-view texturing with the canonical reference
    camera from ``camera.json``. Inputs are staged under ``output_dir`` so their
    bindings stay valid; a missing ``condition_rgb.png`` is derived from the
    RGBA observation with black premultiplication, as the frontend writes it.
    Optional ``keep`` excludes source pixels identified as image/geometry depth
    mismatches. It changes only staged projection trust, not source RGB, source
    files, or the final mesh's opacity. Other observation-admission filters are
    not passed through this mask.
    """
    import numpy as np
    from PIL import Image

    native, source = (Path(path).resolve(strict=True) for path in (native_glb, input_dir))
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"front reprojection requires a new destination: {output}")
    camera = read_json(source / "camera.json")
    if keep is not None:
        keep = np.asarray(keep)
        if keep.shape != (1024, 1024) or keep.dtype != np.bool_:
            raise ValueError("reprojection keep must be a boolean 1024 by 1024 source-image mask")
    try:
        camera = {"camera_angle_x": float(camera["camera_angle_x"]), "distance": float(camera["distance"]),
                  "mesh_scale": float(camera.get("mesh_scale", 1.0))}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("front reprojection requires a canonical camera with camera_angle_x and distance") from error
    staged = output / "source"
    staged.mkdir(parents=True)
    for name in ("condition_rgba.png", "albedo.png", "mra.png", "camera.json"):
        shutil.copyfile(source / name, staged / name)
    if (source / "condition_rgb.png").is_file():
        shutil.copyfile(source / "condition_rgb.png", staged / "condition_rgb.png")
    else:
        with Image.open(source / "condition_rgba.png") as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.
        Image.fromarray((np.clip(rgba[..., :3] * rgba[..., 3:4], 0, 1) * 255).astype(np.uint8)).save(staged / "condition_rgb.png")
    if keep is not None:
        return _run_side(native, staged, output / "front", camera, keep=keep)
    return _run_side(native, staged, output / "front", camera)


def run_dual_view(native_glb: Path, front_input_dir: Path, back_input_dir: Path,
                  cameras: dict[str, Any], output_dir: Path) -> Path:
    """Run both measured reprojections and publish the stitched final GLB.

    The caller owns geometry, generated-image, and material-model provenance.
    This function binds the provided model-independent maps, camera pack, native
    mesh, reprojection code, candidate gates, and final atlas in its own receipt.
    """
    native, front, back = (Path(path).resolve(strict=True) for path in (native_glb, front_input_dir, back_input_dir))
    output = Path(output_dir).resolve()
    raw = cameras.get("raw", cameras)
    front_camera, back_camera = camera_spec(raw, "front"), camera_spec(raw, "back")
    for path in (output / "post", output / "post_back"):
        if path.exists():
            raise FileExistsError(f"dual-view reprojection requires a new destination: {path}")
    output.mkdir(parents=True, exist_ok=True)
    final_dir = output / "post_back"
    final_dir.mkdir()
    try:
        write_json(final_dir / "cameras.json", raw)
        front_mesh = _run_side(native, front, output / "post", front_camera)
        rotation, translation, distance = back_to_front(raw["back"])
        rotated = final_dir / "rotated_baseline.glb"
        transform_glb(front_mesh, rotated, rotation, translation)
        canonical_back = {
            "extrinsic": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, distance], [0, 0, 0, 1]],
            "projection_intrinsic": back_camera["projection_intrinsic"], "distance": distance,
            "camera_angle_x": back_camera["camera_angle_x"], "mesh_scale": 1.0,
        }
        back_mesh = _run_side(rotated, back, final_dir / "reproject", canonical_back)
        protected = front_protection(output / "post" / "project" / "COMMON_DOMAIN.npz")
        final = final_dir / "mesh.glb"
        stitch_gates = stitch(front_mesh, back_mesh, protected, final)
        _validate_side(output / "post")
        _validate_side(final_dir / "reproject")
        receipt = {
            "profile": PROFILE, "status": "valid", "view": "dual", "mesh": str(final),
            "gates": {"accepted": True, "front_candidates_accepted_all_19": True,
                      "back_candidates_accepted_all_19": True, **stitch_gates},
            "front_post_receipt": bind_file(output / "post" / "receipt.json"),
            "back_post_receipt": bind_file(final_dir / "reproject" / "receipt.json"),
            "camera_pack": bind_file(final_dir / "cameras.json"), "final_mesh": bind_file(final),
            "front_protection": bind_file(output / "post" / "project" / "COMMON_DOMAIN.npz"),
            "back_transform": {"rotation": rotation.tolist(), "translation": translation.tolist(), "distance": distance},
            "validation_scope": "Both view candidates passed 19-phase gates; the stitched atlas has exact front priority and preserved non-base content.",
        }
        write_json(final_dir / "receipt.json", receipt)
        validate_dual_view(output, raise_on_invalid=True)
        return final
    except BaseException as error:
        failure = final_dir / "FAILED.json"
        if not failure.exists():
            write_json(failure, {"profile": PROFILE, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def validate_dual_view(output_dir: Path, *, raise_on_invalid: bool = False) -> dict[str, Any]:
    """Re-read the complete postprocess receipt chain and content bindings."""
    try:
        output = Path(output_dir).resolve()
        receipt = read_json(output / "post_back" / "receipt.json")
        if (output / "post_back" / "FAILED.json").exists():
            raise ValueError("dual-view reprojection has a recorded failure")
        if receipt.get("profile") != PROFILE or receipt.get("status") != "valid" or receipt.get("view") != "dual":
            raise ValueError("invalid dual-view reprojection identity or status")
        gates = receipt.get("gates", {})
        required = ("accepted", "front_candidates_accepted_all_19", "back_candidates_accepted_all_19",
                    "front_priority_exact", "alpha_exact", "position_exact", "normal_exact", "uv_exact")
        if not all(gates.get(key) is True for key in required) or gates.get("final_19_phase_rendered") is not False:
            raise ValueError("dual-view reprojection gates or validation scope are incomplete")
        check_binding(receipt["front_post_receipt"], output / "post" / "receipt.json")
        check_binding(receipt["back_post_receipt"], output / "post_back" / "reproject" / "receipt.json")
        check_binding(receipt["front_protection"], output / "post" / "project" / "COMMON_DOMAIN.npz")
        camera_path = check_binding(receipt["camera_pack"], output / "post_back" / "cameras.json")
        final = check_binding(receipt["final_mesh"], output / "post_back" / "mesh.glb")
        if Path(receipt.get("mesh", "")).resolve() != final:
            raise ValueError("dual-view final mesh path is inconsistent")
        cameras = read_json(camera_path)
        front_camera = camera_spec(cameras, "front")
        back_camera = camera_spec(cameras, "back")
        rotation, translation, distance = back_to_front(cameras["back"])
        expected_transform = {"rotation": rotation.tolist(), "translation": translation.tolist(), "distance": distance}
        if receipt.get("back_transform") != expected_transform:
            raise ValueError("back transform differs from the camera pack")
        if read_json(output / "post" / "inputs_1024" / "camera.json") != front_camera:
            raise ValueError("front postprocess did not use the bound front camera")
        expected_back = {
            "extrinsic": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, distance], [0, 0, 0, 1]],
            "projection_intrinsic": back_camera["projection_intrinsic"], "distance": distance,
            "camera_angle_x": back_camera["camera_angle_x"], "mesh_scale": 1.0,
        }
        if read_json(output / "post_back" / "reproject" / "inputs_1024" / "camera.json") != expected_back:
            raise ValueError("back postprocess did not use the canonicalized back camera")
        expected_document, expected_binary = transformed_payload(output / "post" / "mesh.glb", rotation, translation)
        actual_document, actual_binary = _glb._read_glb(output / "post_back" / "rotated_baseline.glb")
        if actual_document != expected_document or actual_binary != expected_binary:
            raise ValueError("temporary back mesh differs from the required rigid transform")
        _validate_side(output / "post")
        _validate_side(output / "post_back" / "reproject")
        protected = front_protection(output / "post" / "project" / "COMMON_DOMAIN.npz")
        actual = validate_stitch(output / "post" / "mesh.glb", output / "post_back" / "reproject" / "mesh.glb", final, protected)
        if any(gates.get(key) != value for key, value in actual.items()):
            raise ValueError("stitched preservation evidence changed")
        return {"valid": True, "reason": "Both candidate chains and the stitched final mesh are current", "receipt": receipt}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if raise_on_invalid:
            raise
        return {"valid": False, "reason": str(error)}

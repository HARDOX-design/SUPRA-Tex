"""Render, bind, and validate a flat SupraTex back-view condition pack.

Inputs are the current Stage 1-2 files: geometry.pt, camera.json,
geometry_receipt.json, and condition_rgba.png. No legacy COMPLETE.json or
experiment-bank directory layout is required or fabricated.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .camera import (ADAPTER_ID, BORDER_PX, DRIFT_PENALTY_PER_RAD, EXTRINSIC_ATOL,
                     NORMAL_SIGN_MIN, PACK_SCHEMA, PITCH_RANGE_DEG, PROTOCOL_ID,
                     RESOLUTION, SCALE_PENALTY_PER_LOG, SCALE_RANGE, YAW_RANGE_DEG,
                     back_eye, look_at_extrinsic, parse_front_camera,
                     touches_border, validate_normal_signs)

INPUT_FILES = {"geometry": "geometry.pt", "camera": "camera.json",
               "geometry_receipt": "geometry_receipt.json", "condition_rgba": "condition_rgba.png"}
PACK_FILES = ("front_rgb.png", "back_normal.png", "back_mask.png", "cameras.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def invalid(value):
        raise ValueError(f"non-finite JSON value {value} in {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _inputs(input_dir: Path) -> dict[str, Any]:
    import torch
    input_dir = Path(input_dir).resolve()
    paths = {name: input_dir / filename for name, filename in INPUT_FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"completed SupraTex geometry input is missing {path}")
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    camera, receipt = _json(paths["camera"]), _json(paths["geometry_receipt"])
    parsed = parse_front_camera(camera)
    if receipt.get("schema") != "supratex.geometry.v2":
        raise ValueError("geometry_receipt.json must be a current supratex.geometry.v2 receipt")
    payload = torch.load(paths["geometry"], map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("geometry.pt must contain a tensor dictionary")
    vertices, faces = payload.get("verts"), payload.get("faces")
    if not torch.is_tensor(vertices) or not torch.is_tensor(faces):
        raise ValueError("geometry.pt requires verts and faces tensors")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not torch.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.dtype not in (torch.int32, torch.int64)
            or int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("geometry.pt contains invalid vertices or triangle indices")
    if receipt.get("faces") != len(faces) or receipt.get("vertices") != len(vertices):
        raise ValueError("geometry receipt vertex/face counts differ from geometry.pt")
    scale = receipt.get("optimized_mesh_scale")
    if scale is None or not math.isclose(float(scale), parsed["source_mesh_scale"], rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("geometry receipt optimized_mesh_scale differs from the current camera")
    final_camera = receipt.get("final_generation_camera")
    if not isinstance(final_camera, dict):
        raise ValueError("geometry receipt is missing final_generation_camera")
    for key, expected in (("distance", parsed["source_distance"]), ("camera_angle_x", parsed["fov"]),
                          ("mesh_scale", parsed["source_mesh_scale"])):
        if key not in final_camera or not math.isclose(float(final_camera[key]), expected, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"geometry receipt final_generation_camera.{key} differs from camera.json")
    resolution = payload.get("resolution", receipt.get("resolution"))
    if resolution is not None:
        if isinstance(resolution, bool) or int(resolution) != resolution or int(resolution) <= 0:
            raise ValueError("geometry resolution must be a positive integer")
        resolution = int(resolution)
        if receipt.get("resolution", resolution) != resolution:
            raise ValueError("geometry and receipt resolutions differ")
    with Image.open(paths["condition_rgba"]) as image:
        if image.mode != "RGBA" or image.width != image.height:
            raise ValueError("condition_rgba.png must be a square RGBA image")
        rgba = image.copy()
    return {"paths": paths, "hashes": source_hashes, "camera": camera, "parsed_camera": parsed,
            "receipt": receipt, "payload": payload, "rgba": rgba,
            "geometry_resolution": resolution, "faces": len(faces), "vertices": len(vertices)}


def _pose(eye, target, extrinsic, intrinsic):
    pixel = np.asarray(intrinsic, dtype=np.float64).copy()
    pixel[:2] *= RESOLUTION
    return {"convention": "OpenCV world-to-camera, +X right +Y down +Z forward",
            "eye": np.asarray(eye).astype(float).tolist(), "look_at": list(target), "up": [0.0, 1.0, 0.0],
            "resolution": [RESOLUTION, RESOLUTION],
            "extrinsics_w2c": np.asarray(extrinsic).astype(float).tolist(),
            "intrinsics_normalized": np.asarray(intrinsic).astype(float).tolist(),
            "intrinsics_pixel": pixel.tolist(),
            "project": "q=w2c@[X,Y,Z,1]; u=fx*q.x/q.z+cx; v=fy*q.y/q.z+cy; pixel origin top-left"}


def _matrix(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix of shape {shape}")
    return result


def _equal(value, expected, name, tolerance=EXTRINSIC_ATOL):
    value, expected = np.asarray(value, dtype=float), np.asarray(expected, dtype=float)
    if value.shape != expected.shape or not np.isfinite(value).all() or not np.allclose(value, expected, atol=tolerance, rtol=0.0):
        raise ValueError(f"{name} does not match the bound v4 camera")


def validate_pack(pack_dir: Path, input_dir: Path | None = None) -> dict[str, Any]:
    """Validate images, current camera math, geometry provenance, and v4 gates.

    Passing input_dir supports relocated copies of an input when every source
    digest matches. Existing original paths are also checked if still present.
    Geometry is loaded on CPU; validation never starts a renderer or a model.
    """
    pack_dir = Path(pack_dir)
    if (pack_dir / "FAILED.json").exists():
        raise ValueError(f"back-view pack has a failure receipt: {pack_dir / 'FAILED.json'}")
    for filename in PACK_FILES:
        if not (pack_dir / filename).is_file():
            raise FileNotFoundError(f"back-view pack is missing {filename}")
    cameras = _json(pack_dir / "cameras.json")
    if cameras.get("protocol") != PROTOCOL_ID or cameras.get("schema") != PACK_SCHEMA:
        raise ValueError("back-view pack must use camera v4 and back_normal_pack.v3")
    if cameras.get("adapter") != ADAPTER_ID:
        raise ValueError("back-view pack must have a current SupraTex camera binding")
    if cameras.get("normal_space") != "camera_x_right_y_up_z_toward_camera":
        raise ValueError("back-view normals must use image-right, image-up, camera-facing axes")
    images = {}
    for name, mode in (("front_rgb.png", "RGBA"), ("back_normal.png", "RGB"), ("back_mask.png", "L")):
        with Image.open(pack_dir / name) as image:
            if image.size != (RESOLUTION, RESOLUTION) or image.mode != mode:
                raise ValueError(f"{name} must be {RESOLUTION}x{RESOLUTION} {mode}")
            images[name] = np.asarray(image).copy()
        if cameras.get("image_sha256", {}).get(name) != sha256(pack_dir / name):
            raise ValueError(f"{name} digest differs from the rendered pack")
    mask_u8 = images["back_mask.png"]
    if not np.isin(mask_u8, [0, 255]).all() or not (mask_u8 >= 128).any():
        raise ValueError("back_mask.png must contain a nonempty binary white-on-black silhouette")
    if (pack_dir / "back_rgba.png").is_file():
        with Image.open(pack_dir / "back_rgba.png") as image:
            if image.mode != "RGBA" or image.size != (RESOLUTION, RESOLUTION):
                raise ValueError("back_rgba.png must be 1024x1024 RGBA")
            if not np.array_equal(np.asarray(image)[..., 3] >= 128, mask_u8 >= 128):
                raise ValueError("back_rgba alpha does not exactly match back_mask")
    sources, hashes = cameras.get("sources", {}), cameras.get("source_sha256", {})
    if not all(name in sources and name in hashes for name in INPUT_FILES):
        raise ValueError("pack is missing current geometry/camera/receipt/condition provenance")
    for name in INPUT_FILES:
        if not Path(sources[name]).is_absolute():
            raise ValueError(f"sources.{name} must be an absolute path")
        original = Path(sources[name])
        if original.is_file() and sha256(original) != hashes[name]:
            raise ValueError(f"original source {name} changed after pack rendering")
    if input_dir is None:
        # Resolve each path independently: current handoffs may contain symlinks.
        missing = [name for name in INPUT_FILES if not Path(sources[name]).is_file()]
        if missing:
            raise FileNotFoundError(f"source files are unavailable ({', '.join(missing)}); supply input_dir for a relocated handoff")
        input_dir = Path(cameras.get("input_dir", Path(sources["camera"]).parent))
    inputs = _inputs(Path(input_dir))
    if inputs["hashes"] != hashes:
        raise ValueError("input geometry/camera/receipt/condition digests differ from the pack")
    expected_front = np.asarray(inputs["rgba"].resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS))
    if not np.array_equal(images["front_rgb.png"], expected_front):
        raise ValueError("front_rgb.png differs from the bound condition RGBA resized with LANCZOS")
    if cameras.get("faces_raw") != inputs["faces"] or cameras.get("geometry_resolution") != inputs["geometry_resolution"]:
        raise ValueError("pack geometry counts/resolution differ from the current input")
    faces, simplify = cameras.get("faces"), cameras.get("simplify")
    if (not isinstance(faces, int) or isinstance(faces, bool) or not 0 < faces <= inputs["faces"]
            or not isinstance(simplify, int) or isinstance(simplify, bool) or simplify < 0
            or (simplify == 0 and faces != inputs["faces"])
            or (simplify > 0 and faces > simplify)):
        raise ValueError("rendered face count/simplification receipt is inconsistent")
    if cameras.get("official_geometry_camera") != inputs["camera"]:
        raise ValueError("pack source camera differs from camera.json")
    parsed = inputs["parsed_camera"]
    for field in ("distance", "source_distance", "source_mesh_scale", "translation_dual", "front_eye",
                  "front_look_at", "back_look_at", "exact_back_eye"):
        _equal(cameras.get(field), parsed[field], field)
    if cameras.get("mesh_scale") != 1.0 or cameras.get("off_axis") != parsed["off_axis"]:
        raise ValueError("pack camera scale/off-axis flags disagree with its source camera")
    if cameras.get("camera_contract") != parsed["camera_contract"] or cameras.get("distance_policy") != parsed["distance_policy"]:
        raise ValueError("pack camera contract/distance policy is invalid")
    _equal(cameras.get("camera_angle_x"), parsed["fov"], "camera_angle_x")
    dyaw, pitch, scale = (float(cameras[key]) for key in ("back_dyaw_deg", "back_pitch_deg", "back_scale"))
    if not (-YAW_RANGE_DEG <= dyaw <= YAW_RANGE_DEG and -PITCH_RANGE_DEG <= pitch <= PITCH_RANGE_DEG
            and SCALE_RANGE[0] - 1e-6 <= scale <= SCALE_RANGE[1] + 1e-6):
        raise ValueError("back camera parameters exceed the v4 search space")
    expected_back = back_eye(parsed["distance"], dyaw, pitch, scale, parsed["back_look_at"])
    _equal(cameras.get("back_eye"), expected_back, "back_eye")
    _equal(cameras.get("back_yaw"), math.pi + math.radians(dyaw), "back_yaw")
    _equal(cameras.get("back_pitch"), math.radians(pitch), "back_pitch")
    for side, eye, target, expected_extrinsic in (
        ("front", parsed["front_eye"], parsed["front_look_at"], parsed["extrinsic"]),
        ("back", expected_back, parsed["back_look_at"], look_at_extrinsic(expected_back, parsed["back_look_at"])),
    ):
        view = cameras.get(side, {})
        if (view.get("resolution") != [RESOLUTION, RESOLUTION]
                or view.get("convention") != "OpenCV world-to-camera, +X right +Y down +Z forward"):
            raise ValueError(f"{side} view has an invalid resolution or convention")
        _equal(view.get("eye"), eye, f"{side}.eye")
        _equal(view.get("look_at"), target, f"{side}.look_at")
        _equal(_matrix(view.get("extrinsics_w2c"), (4, 4), f"{side}.w2c"), expected_extrinsic, f"{side}.w2c")
        intrinsic = _matrix(view.get("intrinsics_normalized"), (3, 3), f"{side}.intrinsic")
        _equal(intrinsic, parsed["intrinsic"], f"{side}.intrinsic")
        pixel = intrinsic.copy()
        pixel[:2] *= RESOLUTION
        _equal(view.get("intrinsics_pixel"), pixel, f"{side}.pixel_intrinsic")
    selection = cameras.get("selection", {})
    if selection.get("objective") != "max 1-|back_occupancy-front_occupancy|; no front/back silhouette IoU":
        raise ValueError("pack does not record the v4 occupancy objective")
    if any(not isinstance(selection.get(name), bool) for name in ("front_touches_border", "back_touches_border")):
        raise ValueError("pack must record both front and back border checks")
    actual_border = bool(touches_border(mask_u8 >= 128))
    if selection.get("back_touches_border") != actual_border:
        raise ValueError("back border receipt differs from back_mask")
    if not selection.get("front_touches_border") and actual_border:
        raise ValueError("back silhouette touches the border while the front silhouette does not")
    actual_occupancy = float(np.mean(mask_u8 >= 128))
    _equal(selection.get("back_occupancy"), actual_occupancy, "back_occupancy", tolerance=1e-9)
    for field in ("front_occupancy", "exact_back_occupancy", "back_occupancy"):
        value = float(selection.get(field, float("nan")))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"selection.{field} must be in (0,1]")
    _equal(selection.get("occupancy_abs_diff"), abs(actual_occupancy - selection["front_occupancy"]),
           "occupancy_abs_diff", tolerance=1e-9)
    if cameras.get("validation", {}).get("normal_sign_min") != NORMAL_SIGN_MIN:
        raise ValueError("pack must record SupraTex's explicit normal-sign acceptance threshold")
    validate_normal_signs(cameras.get("normal_sign_check", {}))
    return cameras


def _assemble(inputs, rendered, directory: Path) -> dict[str, Any]:
    parsed, search = inputs["parsed_camera"], rendered["search"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "viz").mkdir(exist_ok=True)
    rgba = inputs["rgba"].resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    rgba.save(directory / "front_rgb.png")
    Image.fromarray(rendered["back_normal"], "RGB").save(directory / "back_normal.png")
    back_mask = np.asarray(rendered["back_mask"], dtype=bool)
    Image.fromarray(back_mask.astype(np.uint8) * 255, "L").save(directory / "back_mask.png")
    Image.fromarray(rendered["front_normal"], "RGB").save(directory / "viz" / "front_normal.png")
    exact = np.asarray(rendered["exact_back_silhouette"], dtype=bool)
    overlay = np.zeros((*back_mask.shape, 3), dtype=np.uint8)
    overlay[exact & ~back_mask], overlay[back_mask & ~exact], overlay[exact & back_mask] = (220, 60, 60), (60, 200, 90), (230, 220, 90)
    Image.fromarray(overlay, "RGB").save(directory / "viz" / "overlay.png")
    alpha, front_mask = np.asarray(rgba)[..., 3] >= 128, np.asarray(rendered["front_silhouette"], dtype=bool)
    front_alignment = float((alpha & front_mask).sum() / max(1, (alpha | front_mask).sum()))
    front_occupancy, back_occupancy = float(front_mask.mean()), float(back_mask.mean())
    cameras = {
        "schema": PACK_SCHEMA, "protocol": PROTOCOL_ID, "adapter": ADAPTER_ID,
        "projection": "perspective", "normal_space": "camera_x_right_y_up_z_toward_camera",
        "encoding": "RGB=(n+1)/2*255, background black, face normals",
        **{name: parsed[name] for name in ("distance", "mesh_scale", "source_distance", "source_mesh_scale",
                                         "distance_policy", "camera_contract", "off_axis", "translation_dual",
                                         "front_eye", "front_look_at", "back_look_at", "exact_back_eye")},
        "camera_angle_x": parsed["fov"], "effective_distance": parsed["distance"],
        "geometry_resolution": inputs["geometry_resolution"],
        "geometry_resolution_source": "geometry_or_receipt" if inputs["geometry_resolution"] is not None else "not_recorded",
        "faces": int(rendered["faces"]), "faces_raw": inputs["faces"], "simplify": rendered["simplify"],
        "back_eye": np.asarray(rendered["back_eye"]).astype(float).tolist(),
        "back_yaw": math.pi + math.radians(search["dyaw_deg"]), "back_pitch": math.radians(search["pitch_deg"]),
        "back_scale": search["scale"], "back_dyaw_deg": search["dyaw_deg"], "back_pitch_deg": search["pitch_deg"],
        "front": _pose(parsed["front_eye"], parsed["front_look_at"], rendered["front_extrinsic"], rendered["intrinsic"]),
        "back": _pose(rendered["back_eye"], parsed["back_look_at"], rendered["back_extrinsic"], rendered["intrinsic"]),
        "selection": {
            "objective": "max 1-|back_occupancy-front_occupancy|; no front/back silhouette IoU",
            "tie_breaker": f"-{DRIFT_PENALTY_PER_RAD}/rad of |dyaw|+|pitch|, -{SCALE_PENALTY_PER_LOG}*|ln scale|; stable grid order",
            "border_rule": f"back may not touch the {BORDER_PX}px frame border unless front does",
            "search_space": {"dyaw_deg": [-YAW_RANGE_DEG, YAW_RANGE_DEG],
                             "pitch_deg": [-PITCH_RANGE_DEG, PITCH_RANGE_DEG], "scale": list(SCALE_RANGE)},
            "front_occupancy": front_occupancy, "back_occupancy": back_occupancy,
            "exact_back_occupancy": float(exact.mean()), "occupancy_abs_diff": abs(back_occupancy - front_occupancy),
            "front_touches_border": bool(touches_border(front_mask)), "back_touches_border": bool(touches_border(back_mask)),
            **search,
        },
        "front_alignment_iou_vs_condition_alpha": front_alignment,
        "normal_sign_check": rendered["normal_sign_check"],
        "validation": {"normal_sign_min": NORMAL_SIGN_MIN,
                       "normal_sign_policy": "SupraTex's numeric interpretation of the v4 qualitative close-to-1 requirement"},
        "official_geometry_camera": inputs["camera"],
        "input_dir": str(inputs["paths"]["camera"].parent),
        "sources": {name: str(path.resolve()) for name, path in inputs["paths"].items()},
        "source_sha256": inputs["hashes"],
        "image_sha256": {name: sha256(directory / name) for name in PACK_FILES if name.endswith(".png")},
    }
    _write_json(directory / "cameras.json", cameras)
    return cameras


def _report(directory: Path, cameras):
    selection = cameras["selection"]
    warnings = []
    if cameras["front_alignment_iou_vs_condition_alpha"] < 0.9:
        warnings.append("Front geometry alignment with condition alpha is below 0.9.")
    if selection["occupancy_abs_diff"] > 0.05:
        warnings.append("Front/back occupancy differs by more than five percentage points.")
    _write_json(directory / "MANIFEST.json", {"schema": PACK_SCHEMA, "protocol": PROTOCOL_ID,
                "adapter": ADAPTER_ID, "totals": {"cases": 1, "ready": 1, "failed": 0},
                "cases": [{"path": ".", "off_axis": cameras["off_axis"], "warnings": warnings,
                           "occupancy_abs_diff": selection["occupancy_abs_diff"], "source_sha256": cameras["source_sha256"]}]})
    figures = "".join(f'<figure><img src="{path}" width="256"><figcaption>{title}</figcaption></figure>'
                      for path, title in (("front_rgb.png", "Front RGB"), ("viz/front_normal.png", "Front normal"),
                                          ("back_normal.png", "Back normal"), ("viz/overlay.png", "Exact back / searched back")))
    warning_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>SupraTex back-view pack</title>'
        '<style>body{font-family:sans-serif;margin:24px}section{display:flex;flex-wrap:wrap}figure{margin:8px}.warnings{color:#a65a00}</style>'
        f'<h1>SupraTex back-view pack</h1><p>{PROTOCOL_ID}</p><section>{figures}</section>'
        '<p>Overlay: red = exact 180-degree back only; green = searched back only; yellow = both.</p>'
        f'<p>Occupancy difference: {selection["occupancy_abs_diff"]:.6f}. Normal-sign minimum: {NORMAL_SIGN_MIN}.</p>'
        f'<ul class="warnings">{warning_html}</ul></html>', encoding="utf-8")


def render_pack(input_dir: Path, output_dir: Path, *, device="cuda") -> Path:
    """Render one current geometry handoff, validate it, and publish the pack.

    Existing valid packs are reused only after their input digests are checked.
    Rendering failures leave FAILED.json; successful publication removes it.
    Reuse validation never modifies an existing completed pack.
    An OOM may retry once at 1.5M faces without changing the bound geometry file.
    """
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("render output must be separate from the Stage 1-2 input directory")
    if (output_dir / "cameras.json").is_file() and not (output_dir / "FAILED.json").exists():
        validate_pack(output_dir, input_dir)
        return output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-render-", dir=output_dir.parent))
    try:
        inputs = _inputs(input_dir)
        from .render import render_geometry
        retry_simplified = False
        try:
            rendered = render_geometry(inputs["payload"], inputs["parsed_camera"], device=device)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or inputs["faces"] <= 1_500_000:
                raise
            retry_simplified = True
        # Leave the exception scope first, releasing its traceback and GPU tensors.
        if retry_simplified:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            rendered = render_geometry(inputs["payload"], inputs["parsed_camera"], device=device, simplify=1_500_000)
        cameras = _assemble(inputs, rendered, staging)
        validate_pack(staging, input_dir)
        _report(staging, cameras)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep the destination unreadable as a complete pack until every file is published.
        _write_json(output_dir / "FAILED.json", {"protocol": PROTOCOL_ID, "status": "publishing"})
        for path in staging.iterdir():
            if path.name == "cameras.json":
                continue
            target = output_dir / path.name
            if path.is_dir() and target.exists():
                shutil.rmtree(target)
            os.replace(path, target)
        os.replace(staging / "cameras.json", output_dir / "cameras.json")
        (output_dir / "FAILED.json").unlink()
        return output_dir
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "FAILED.json", {"protocol": PROTOCOL_ID, "adapter": ADAPTER_ID,
                    "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

"""Two-view material observations for the current Supra-Tex completion model."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageFilter

from . import native_observation as observation

PROTOCOL = "supratex.dual_view"
CAMERA_PROTOCOL = "supratex.back_view_camera.v4"
RESOLUTION = 1024
SILHOUETTE_MIN_IOU = .85
PROJECTION_MIN_INSIDE = .95


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        value = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_material_receipt(root: Path) -> dict[str, Any]:
    from supratex.backview.material_identity import artifact_identity

    receipt = json.loads((root / "material_receipt.json").read_text(encoding="utf-8"))
    signature = receipt.get("model_signature", {})
    if (receipt.get("status") != "complete" or receipt.get("backend") not in {"qwen-image", "firered"}
            or signature.get("backend") != receipt["backend"] or not signature.get("base")
            or not (signature.get("model") or signature.get("weights"))):
        raise ValueError("dual-view materials require a complete backend and model receipt")
    if signature.get("artifacts") != artifact_identity(signature):
        raise ValueError("material model artifacts changed or lack current identity records")
    for name in ("condition_rgba.png", "condition_rgb.png", "albedo.png", "mra.png"):
        if receipt.get("files", {}).get(name) != digest(root / name):
            raise ValueError(f"material receipt does not bind the current {name}: {root}")
    for name in ("albedo.png", "mra.png"):
        with Image.open(root / name) as image:
            if image.mode != "RGB" or image.size != (RESOLUTION, RESOLUTION):
                raise ValueError(f"{name} must be RGB 1024x1024")
    return receipt


def view_camera(cameras: Mapping[str, Any], name: str, *, ray_precision: bool = False) -> dict[str, Any]:
    """Translate a v4 view into the completion camera contract."""
    if cameras.get("protocol") != CAMERA_PROTOCOL:
        raise ValueError("dual-view inference accepts only the back-view camera v4 protocol")
    view = cameras[name]
    extrinsic = np.asarray(view["extrinsics_w2c"], dtype=np.float64)
    intrinsic = np.asarray(view["intrinsics_normalized"], dtype=np.float64)
    fov = float(cameras["camera_angle_x"] if "camera_angle_x" in cameras
                else cameras["official_geometry_camera"]["camera_angle_x"])
    # The native canonical ray path computes focal length from FOV in float64.
    # Preserve its operation order rather than round-tripping a float32 JSON K.
    if ray_precision:
        focal = .5 / math.tan(fov / 2)
        intrinsic = np.array([[focal, 0, .5], [0, focal, .5], [0, 0, 1]], dtype=np.float64)
        if name == "front":
            source = cameras.get("official_geometry_camera", {})
            if "extrinsic" in source:
                extrinsic = np.asarray(source["extrinsic"], dtype=np.float64)
                intrinsic = np.asarray(source.get("projection_intrinsic", source.get("intrinsic")), dtype=np.float64)
            elif "distance" in source:
                # Rendering stores float32 matrices. Replay the scalar DDA with
                # its original float64 distance to preserve voxel ownership.
                extrinsic = np.diag([1., -1., -1., 1.])
                extrinsic[2, 3] = float(source["distance"]) * float(source.get("mesh_scale", 1.))
    eye, look_at = np.asarray(view["eye"]), np.asarray(view["look_at"])
    return {"extrinsic": extrinsic, "intrinsic": intrinsic, "camera_angle_x": fov,
            "distance": float(np.linalg.norm(eye - look_at)), "mesh_scale": 1.0}


def load_back_case(input_dir: str | Path, back_view_dir: str | Path) -> dict[str, Any]:
    """Check the original generated alpha and all input bindings before GPU work."""
    from supratex.backview.pack import validate_pack

    front, back = Path(input_dir).resolve(), Path(back_view_dir).resolve()
    cameras = validate_pack(back, input_dir=front)
    materials = [validate_material_receipt(root) for root in (front, back)]
    if materials[0]["model_signature"] != materials[1]["model_signature"]:
        raise ValueError("front and back must use the same material backend, weights and configuration")
    with Image.open(back / "back_rgba.png") as image:
        if image.mode != "RGBA" or image.size != (RESOLUTION, RESOLUTION):
            raise ValueError("generated back must be RGBA 1024x1024")
        original = np.asarray(image).copy()
    with Image.open(back / "back_mask.png") as image:
        mask = np.asarray(image.convert("L")) >= 128
    if not np.array_equal(original[..., 3] >= 128, mask):
        raise ValueError("original generated back alpha does not match the rendered back mask")
    generation = json.loads((back / "generation_receipt.json").read_text(encoding="utf-8"))
    if (generation.get("status") != "valid" or generation.get("worker_model") != "gpt-6-astra"
            or generation.get("worker_reasoning_effort") != "ultra"
            or generation.get("generation_tool") != "image_gen.imagegen"
            or generation.get("back_rgba_sha256") != digest(back / "back_rgba.png")
            or generation.get("cameras_sha256") != digest(back / "cameras.json")):
        raise ValueError("back-view generation requires a valid Astra/image_gen receipt bound to these files")
    from supratex.backview.harness import validate_result as validate_generation
    current_generation = validate_generation(Path(generation["run_root"]), case_id=generation["case_id"])
    if (current_generation.get("valid") is not True
            or current_generation.get("back_rgba_sha256") != generation["back_rgba_sha256"]
            or current_generation.get("cameras_sha256") != generation["cameras_sha256"]):
        raise ValueError("the original generation evidence is missing, changed, or no longer accepted")
    prepare = json.loads((back / "prepare.json").read_text(encoding="utf-8"))
    if (prepare.get("status") != "complete" or prepare.get("source_sha256") != digest(back / "back_rgba.png")
            or prepare.get("condition_rgba_sha256") != digest(back / "condition_rgba.png")
            or prepare.get("condition_rgb_sha256") != digest(back / "condition_rgb.png")):
        raise ValueError("back preprocessing receipt is missing or stale")
    with Image.open(back / "condition_rgba.png") as image:
        rgba = image.copy()
    if rgba.mode != "RGBA" or rgba.size != (RESOLUTION, RESOLUTION):
        raise ValueError("prepared back condition must be RGBA 1024x1024")
    with Image.open(back / "albedo.png") as image:
        albedo = image.copy()
    with Image.open(back / "mra.png") as image:
        mra = image.copy()
    return {"rgba": rgba, "albedo": albedo, "mra": mra, "mask": mask,
            "cameras": cameras, "root": back, "material": materials[1], "generation": generation}


def trusted_mask(rgba: Image.Image) -> torch.Tensor:
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    mask = Image.fromarray(np.where(alpha >= 254, 255, 0).astype(np.uint8))
    mask = mask.resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST).filter(ImageFilter.MinFilter(5))
    return torch.from_numpy(np.asarray(mask).copy() == 255)


def sample_map(coords: torch.Tensor, source: Image.Image, camera: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    import torch.nn.functional as F

    points = (coords.float() + .5) / RESOLUTION - .5
    extrinsic = torch.as_tensor(camera["extrinsic"], dtype=torch.float32, device=coords.device)
    intrinsic = torch.as_tensor(camera["intrinsic"], dtype=torch.float32, device=coords.device)
    projected = (torch.cat((points, torch.ones_like(points[:, :1])), dim=1) @ extrinsic.T)[:, :3]
    uvw = projected @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-8)
    grid = (uv * 2 - 1).view(1, 1, -1, 2)
    array = np.asarray(source.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS), dtype=np.float32).copy() / 255.
    pixels = torch.from_numpy(array).permute(2, 0, 1).to(coords.device)[None]
    values = F.grid_sample(pixels, grid, mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T.clamp(0, 1)
    return values, torch.cat((uv, projected[:, 2:3]), dim=1)


def fuse_observations(front_visible: torch.Tensor, front_rgb: torch.Tensor, front_mra: torch.Tensor,
                      back_visible: torch.Tensor, back_rgb: torch.Tensor, back_mra: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Union support and preserve the front observation on every overlap row."""
    if front_visible.shape != back_visible.shape or front_visible.dtype != torch.bool or back_visible.dtype != torch.bool:
        raise ValueError("view visibility masks must be equally sized boolean vectors")
    if any(value.shape != (front_visible.numel(), 3) for value in (front_rgb, front_mra, back_rgb, back_mra)):
        raise ValueError("view material observations must be matching Nx3 arrays")
    return (front_visible | back_visible, torch.where(front_visible[:, None], front_rgb, back_rgb),
            torch.where(front_visible[:, None], front_mra, back_mra))


def encode_two_views(surface: Any, shape: Any, front: Mapping[str, Any], back: Mapping[str, Any],
                     encoder: Any, normalization: Mapping[str, Any], runtime: Mapping[str, Any],
                     device: str) -> tuple[Any, dict[str, Any]]:
    """Validate both views, fuse their observed values, and encode exactly once."""
    coords = surface.coords[:, 1:].long()
    bits, ordered, order = observation.build_occupancy_index(coords)
    front_trusted = trusted_mask(front["rgba"])
    production_owners = observation.pixel_first_occupied(dict(front["camera"]), bits, front_trusted)
    front_visible = observation.owner_rows(coords, production_owners, ordered, order).to(device)
    front_camera = view_camera(back["cameras"], "front", ray_precision=True)
    back_camera = view_camera(back["cameras"], "back", ray_precision=True)
    replay = observation.pixel_first_occupied_cv(front_camera, bits, front_trusted)
    if not torch.equal(observation.owner_rows(coords, replay, ordered, order).to(device), front_visible):
        raise RuntimeError("v4 front-camera replay differs from the production visibility")
    hits = observation.pixel_hits_cv(back_camera, bits, np.ones((RESOLUTION, RESOLUTION), dtype=bool))
    mask = back["mask"]
    iou = float(np.count_nonzero(hits & mask) / max(1, np.count_nonzero(hits | mask)))
    if iou < SILHOUETTE_MIN_IOU:
        raise RuntimeError(f"back DDA silhouette IoU {iou:.6f} is below {SILHOUETTE_MIN_IOU}")
    back_owners = observation.pixel_first_occupied_cv(back_camera, bits, trusted_mask(back["rgba"]))
    back_visible = observation.owner_rows(coords, back_owners, ordered, order).to(device)
    # Sampling uses the renderer's stored matrices; only DDA rays require the
    # float64 FOV reconstruction for exact first-cell replay.
    front_rgb, _ = sample_map(coords, front["albedo"], view_camera(back["cameras"], "front"))
    front_mra, _ = sample_map(coords, front["mra"], view_camera(back["cameras"], "front"))
    back_rgb, uvz = sample_map(coords, back["albedo"], view_camera(back["cameras"], "back"))
    back_mra, _ = sample_map(coords, back["mra"], view_camera(back["cameras"], "back"))
    selected = uvz[back_visible]
    if not selected.numel() or not bool(torch.isfinite(selected).all()) or not bool((selected[:, 2] > 0).all()):
        raise RuntimeError("back-visible voxels must be finite and in front of the back camera")
    xy = (selected[:, :2] * RESOLUTION).floor().long().cpu().numpy()
    in_frame = ((xy >= 0) & (xy < RESOLUTION)).all(axis=1)
    in_mask = np.zeros(len(xy), dtype=bool)
    in_mask[in_frame] = mask[xy[in_frame, 1], xy[in_frame, 0]]
    inside = float(in_mask.mean())
    if inside < PROJECTION_MIN_INSIDE:
        raise RuntimeError(f"back visible-voxel mask coverage {inside:.6f} is below {PROJECTION_MIN_INSIDE}")
    visible, rgb, mra = fuse_observations(front_visible, front_rgb, front_mra, back_visible, back_rgb, back_mra)
    stats = {"front_replay_exact": True, "back_silhouette_iou_vs_mask": iou,
             "back_visible_inside_mask": inside, "back_positive_depth": True,
             "overlap_policy": "front wins", "surface_rows": len(coords),
             "front_rows": int(front_visible.sum()), "back_rows": int(back_visible.sum()),
             "overlap_rows": int((front_visible & back_visible).sum()),
             "back_added_rows": int((back_visible & ~front_visible).sum()), "union_rows": int(visible.sum()),
             "partial_encodings": 1}
    with torch.no_grad():
        encoded = observation.encode_partial(encoder, coords, visible, rgb, mra, shape.coords[:, 1:],
                                             dict(normalization), runtime["sparse"], device)
    return encoded, stats


def validate_result(work_dir: str | Path) -> dict[str, Any]:
    """Re-read the full current receipt chain, including maps and final atlas gates."""
    from .reprojection import validate_dual_view

    work = Path(work_dir).expanduser().resolve()
    receipt = json.loads((work / "receipt.json").read_text(encoding="utf-8"))
    texture = json.loads((work / "receipt_texture.json").read_text(encoding="utf-8"))
    if (receipt.get("protocol") != PROTOCOL or receipt.get("camera_protocol") != CAMERA_PROTOCOL
            or receipt.get("status") != "valid" or receipt.get("view") != "dual"
            or receipt.get("stage") != "complete" or receipt.get("stitched_19_phase_checked") is not False):
        raise ValueError("the dual-view completion receipt is not valid")
    if (texture.get("protocol") != PROTOCOL or texture.get("status") != "success"
            or texture.get("view") != "dual" or texture.get("sampling_calls") != 1):
        raise ValueError("the texture receipt does not record one successful dual sample")
    for name in ("front_input", "back_input", "checkpoint", "model_config", "seed", "steps", "sampler"):
        if receipt.get(name) != texture.get(name):
            raise ValueError(f"texture and final receipts disagree on {name}")
    stats = texture.get("dual_stats", {})
    if (stats != receipt.get("dual_stats") or stats.get("front_replay_exact") is not True
            or stats.get("back_positive_depth") is not True or stats.get("overlap_policy") != "front wins"
            or stats.get("partial_encodings") != 1):
        raise ValueError("dual observation gates are incomplete or inconsistent")
    for key, threshold in (("back_silhouette_iou_vs_mask", SILHOUETTE_MIN_IOU),
                           ("back_visible_inside_mask", PROJECTION_MIN_INSIDE)):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not threshold <= value <= 1:
            raise ValueError(f"dual observation gate failed: {key}")
    front, back = Path(receipt["front_input"]), Path(receipt["back_input"])
    load_back_case(front, back)
    bindings = {"geometry_sha256": front / "geometry.pt", "cameras_sha256": back / "cameras.json",
                "front_material_receipt_sha256": front / "material_receipt.json",
                "back_material_receipt_sha256": back / "material_receipt.json"}
    for key, path in bindings.items():
        if receipt.get(key) != digest(path) or texture.get(key) != receipt[key]:
            raise ValueError(f"input binding changed after completion: {key}")
    native = work / "preds/mesh.glb"
    if Path(texture.get("mesh", "")).resolve() != native or texture.get("mesh_sha256") != digest(native):
        raise ValueError("the native texture mesh does not match its receipt")
    validate_dual_view(work, raise_on_invalid=True)
    final, published = work / "post_back/mesh.glb", Path(receipt["output"])
    if (Path(receipt.get("mesh", "")).resolve() != final
            or receipt.get("mesh_sha256") != digest(final)
            or receipt.get("output_sha256") != digest(published)
            or receipt["mesh_sha256"] != receipt["output_sha256"]):
        raise ValueError("the stitched mesh and published copy do not match their receipts")
    return {"valid": True, "protocol": PROTOCOL, "camera_protocol": CAMERA_PROTOCOL,
            "output": str(published), "receipt": str(work / "receipt.json"),
            "candidate_19_phase_checked": True, "stitched_19_phase_checked": False}

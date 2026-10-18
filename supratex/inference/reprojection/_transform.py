"""Temporary back-camera transforms and front-priority atlas stitching."""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import _glb

CAMERA_PROTOCOL = "supratex.back_view_camera.v4"
CAMERA_SCHEMA = "supratex.back_normal_pack.v3"


def camera_spec(cameras: dict[str, Any], name: str) -> dict[str, Any]:
    """Validate a current camera pack and return a full native-frame camera."""
    if cameras.get("protocol") != CAMERA_PROTOCOL or cameras.get("schema") != CAMERA_SCHEMA:
        raise ValueError("reprojection requires a current camera v4 pack")
    view = cameras[name]
    if not str(view.get("convention", "")).startswith("OpenCV world-to-camera"):
        raise ValueError("camera must use OpenCV world-to-camera coordinates")
    if list(view.get("resolution", [])) != [1024, 1024]:
        raise ValueError("reprojection camera must have a 1024 square image")
    extrinsic = np.asarray(view["extrinsics_w2c"], dtype=np.float64)
    intrinsic = np.asarray(view["intrinsics_normalized"], dtype=np.float64)
    eye = np.asarray(view["eye"], dtype=np.float64)
    look_at = np.asarray(view["look_at"], dtype=np.float64)
    if extrinsic.shape != (4, 4) or intrinsic.shape != (3, 3) or eye.shape != (3,) or look_at.shape != (3,):
        raise ValueError("invalid camera array dimensions")
    if not all(np.isfinite(x).all() for x in (extrinsic, intrinsic, eye, look_at)):
        raise ValueError("camera arrays must be finite")
    rotation = extrinsic[:3, :3]
    if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("camera must contain a proper rigid transform")
    distance = float(np.linalg.norm(eye - look_at))
    if distance <= 0 or not np.allclose(-rotation.T @ extrinsic[:3, 3], eye, atol=1e-5):
        raise ValueError("camera eye binding is inconsistent")
    if not np.allclose(rotation @ (look_at - eye) / distance, [0, 0, 1], atol=1e-5):
        raise ValueError("camera optical direction differs from its look-at point")
    focal = float(intrinsic[0, 0])
    if focal <= 0 or not np.allclose(intrinsic, [[focal, 0, .5], [0, focal, .5], [0, 0, 1]], atol=1e-5):
        raise ValueError("camera requires square centered perspective intrinsics")
    return {"extrinsic": extrinsic.tolist(), "projection_intrinsic": intrinsic.tolist(),
            "distance": distance, "mesh_scale": 1.0, "camera_angle_x": 2 * math.atan(.5 / focal)}


def back_to_front(view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """Map an off-axis back camera onto a canonical +Z camera in GLB space."""
    extrinsic = np.asarray(view["extrinsics_w2c"], dtype=np.float64)
    eye, target = np.asarray(view["eye"], dtype=np.float64), np.asarray(view["look_at"], dtype=np.float64)
    distance = float(np.linalg.norm(eye - target))
    front = np.diag([1., -1., -1.])
    rotation = front.T @ extrinsic[:3, :3]
    translation = front.T @ (extrinsic[:3, 3] - [0., 0., distance])
    if not np.allclose(rotation @ eye + translation, [0, 0, distance], atol=1e-5) or not np.allclose(rotation @ target + translation, 0, atol=1e-5):
        raise ValueError("back-camera transform did not recover the canonical pose")
    sign = np.diag([-1., 1., -1.])
    return sign @ rotation @ sign, sign @ translation, distance


def attribute_bytes(path: Path, name: str) -> bytes:
    document, binary = _glb._read_glb(path)
    _glb._assert_fixed_scene(document)
    primitive = document["meshes"][0]["primitives"][0]
    accessor = document["accessors"][primitive["attributes"][name]]
    view = document["bufferViews"][accessor["bufferView"]]
    width = {"VEC2": 2, "VEC3": 3}.get(accessor.get("type"))
    if accessor.get("componentType") != 5126 or width is None or accessor.get("sparse"):
        raise ValueError(f"{name} must be a dense float32 vector attribute")
    stride = width * 4
    if int(view.get("byteStride", stride)) != stride:
        raise ValueError(f"{name} must be stored without interleaving")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    end = start + int(accessor["count"]) * stride
    if start < 0 or end > len(binary) or end > int(view.get("byteOffset", 0)) + int(view["byteLength"]):
        raise ValueError(f"{name} attribute exceeds its binary bounds")
    return binary[start:end]


def transformed_payload(source: Path, rotation: np.ndarray, translation: np.ndarray):
    """Return a temporary transform without changing other document content."""
    document, binary = _glb._read_glb(source)
    _glb._assert_fixed_scene(document)
    edited = bytearray(binary)
    primitive = document["meshes"][0]["primitives"][0]
    for name in ("POSITION", "NORMAL"):
        raw = attribute_bytes(source, name)
        accessor = document["accessors"][primitive["attributes"][name]]
        view = document["bufferViews"][accessor["bufferView"]]
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        values = np.frombuffer(raw, dtype="<f4").reshape(-1, 3).astype(np.float64) @ rotation.T
        if name == "POSITION":
            values += translation
            accessor["min"], accessor["max"] = values.min(0).tolist(), values.max(0).tolist()
        edited[start:start + len(raw)] = values.astype("<f4").tobytes()
    return document, bytes(edited)


def transform_glb(source: Path, destination: Path, rotation: np.ndarray, translation: np.ndarray) -> None:
    """Change only temporary POSITION/NORMAL data and POSITION bounds."""
    document, binary = transformed_payload(source, rotation, translation)
    _glb._write_glb(destination, document, binary)


def atlas(path: Path):
    document, binary = _glb._read_glb(path)
    _glb._assert_fixed_scene(document)
    image_index, view_index = _glb._base_color_image_ref(document)
    image, _, offset, length = _glb._decode_base_image(document, binary, image_index, view_index)
    return document, binary, view_index, offset, length, np.asarray(image).copy()


def front_protection(domain: Path) -> np.ndarray:
    with np.load(domain, allow_pickle=False) as data:
        active, blocked = data["active_texel_ids"], data["blocked_texel_ids"]
    for values in (active, blocked):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= 4096 ** 2) or len(np.unique(values)) != len(values):
            raise ValueError("invalid atlas protection domain")
    if not np.isin(blocked, active).all():
        raise ValueError("blocked atlas texels must belong to the active domain")
    protected = np.setdiff1d(active, blocked)
    if not protected.size:
        raise ValueError("front postprocess has no protected trusted atlas texels")
    return protected


def validate_stitch(front: Path, back: Path, final: Path, protected: np.ndarray) -> dict[str, Any]:
    fdoc, fbin, fview, _, _, fpixels = atlas(front)
    _, _, _, _, _, bpixels = atlas(back)
    _, _, _, _, _, pixels = atlas(final)
    if fpixels.shape != bpixels.shape or pixels.shape != fpixels.shape:
        raise ValueError("front, back, and final atlas shapes differ")
    expected = bpixels.copy()
    expected.reshape(-1, expected.shape[-1])[protected, :3] = fpixels.reshape(-1, fpixels.shape[-1])[protected, :3]
    if expected.shape[-1] == 4:
        expected[..., 3] = fpixels[..., 3]
    if not np.array_equal(pixels, expected):
        raise ValueError("final atlas does not follow exact front-priority stitching")
    for name in ("POSITION", "NORMAL", "TEXCOORD_0"):
        if attribute_bytes(front, name) != attribute_bytes(final, name):
            raise ValueError(f"final mesh changed {name}")
    identity = _glb._nonbase_identity_check(fdoc, fbin, final, fview)
    return {"front_priority_exact": True, "alpha_exact": True, "position_exact": True,
            "normal_exact": True, "uv_exact": True, "nonbase_identity": identity,
            "protected_texels": int(protected.size), "final_19_phase_rendered": False}


def stitch(front: Path, back: Path, protected: np.ndarray, destination: Path) -> dict[str, Any]:
    document, binary, view, offset, length, front_pixels = atlas(front)
    _, _, _, _, _, back_pixels = atlas(back)
    if front_pixels.shape != back_pixels.shape:
        raise ValueError("front and back atlas shapes differ")
    merged = back_pixels.copy()
    merged.reshape(-1, merged.shape[-1])[protected, :3] = front_pixels.reshape(-1, front_pixels.shape[-1])[protected, :3]
    if merged.shape[-1] == 4:
        merged[..., 3] = front_pixels[..., 3]
    payload = io.BytesIO()
    Image.fromarray(merged).save(payload, format="PNG")
    output_document = copy.deepcopy(document)
    output_binary = _glb._replace_image_payload(output_document, binary, view, offset, length, payload.getvalue())
    _glb._write_glb(destination, output_document, output_binary)
    return validate_stitch(front, back, destination, protected)

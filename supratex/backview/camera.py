"""Back-camera v4 math, adapted to the current SupraTex camera handoff.

The geometry is never rescaled here. For scalar SupraTex cameras the physical
distance is ``distance * mesh_scale``, exactly as in the texture pipeline.
Explicit OpenCV matrices already describe physical coordinates and are not
scaled again. The resulting v4 camera always has an effective mesh scale of 1.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

PROTOCOL_ID = "supratex.back_view_camera.v4"
PACK_SCHEMA = "supratex.back_normal_pack.v3"
ADAPTER_ID = "supratex.back_view_camera.v4"
CANONICAL_CONTRACT = "canonical_distance_fov_v1"
FULL_CV_CONTRACT = "full_cv_fixed_negative_z_from_translation_dual_v1"
RESOLUTION = 1024
NEAR, FAR = 0.05, 20.0
YAW_RANGE_DEG, PITCH_RANGE_DEG = 15.0, 10.0
SCALE_RANGE = (0.8, 1.6)
COARSE_RES, FINE_RES = 256, 512
DRIFT_PENALTY_PER_RAD, SCALE_PENALTY_PER_LOG = 0.05, 0.01
BORDER_PX, INFEASIBLE = 2, 10.0
EXTRINSIC_ATOL = 1e-5
# This repository makes the protocol's qualitative "close to 1" check explicit.
NORMAL_SIGN_MIN = 0.9


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def intrinsics_from_fov(fov: float) -> np.ndarray:
    """Normalized square-image intrinsics, without a GPU dependency."""
    fov = _finite(fov, "camera_angle_x")
    if not 0.0 < fov < math.pi:
        raise ValueError("camera_angle_x must be between 0 and pi")
    focal = 0.5 / math.tan(fov / 2.0)
    return np.array([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]])


def front_extrinsic(tx: float, ty: float, distance: float) -> np.ndarray:
    """Delivery-v4 OpenCV w2c: q = [x-tx, -(y-ty), distance-z]."""
    return np.array([[1.0, 0.0, 0.0, -tx], [0.0, -1.0, 0.0, ty],
                     [0.0, 0.0, -1.0, distance], [0.0, 0.0, 0.0, 1.0]])


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite {shape} matrix") from error
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return matrix


def parse_front_camera(camera: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the production camera and return its physical v4 equivalent.

    A translation-only off-axis input is rejected: SupraTex's existing scalar
    camera path does not consume translation_dual. Explicit matrices are needed
    for such a camera so back rendering and front texturing use the same rays.
    """
    source_distance = _finite(camera.get("distance"), "distance")
    mesh_scale = _finite(camera.get("mesh_scale", 1.0), "mesh_scale")
    if source_distance <= 0.0 or mesh_scale <= 0.0:
        raise ValueError("distance and mesh_scale must be positive")
    fov = _finite(camera.get("camera_angle_x"), "camera_angle_x")
    intrinsic = intrinsics_from_fov(fov)
    explicit = "extrinsic" in camera
    has_intrinsic = "projection_intrinsic" in camera or "intrinsic" in camera
    if explicit != has_intrinsic:
        raise ValueError("explicit cameras require both extrinsic and intrinsic matrices")
    translation = camera.get("translation_dual")
    if translation is not None:
        translation = np.asarray(translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation_dual must contain three finite values")
    if explicit:
        extrinsic = _matrix(camera["extrinsic"], (4, 4), "extrinsic")
        tx, ty, distance = -extrinsic[0, 3], extrinsic[1, 3], extrinsic[2, 3]
        if distance <= 0.0 or not math.isclose(distance, source_distance, abs_tol=1e-6, rel_tol=0.0):
            raise ValueError("explicit Delivery-v4 camera depth must equal distance")
        expected = front_extrinsic(tx, ty, distance)
        if not np.allclose(extrinsic, expected, atol=EXTRINSIC_ATOL, rtol=0.0):
            raise ValueError("extrinsic is not a Delivery-v4 fixed-negative-Z camera")
        if translation is not None and not np.allclose(translation, [tx, ty, distance], atol=1e-6, rtol=0.0):
            raise ValueError("translation_dual differs from the production extrinsic")
        source_intrinsic = _matrix(camera.get("projection_intrinsic", camera.get("intrinsic")),
                                   (3, 3), "intrinsic")
        if not np.allclose(source_intrinsic, intrinsic, atol=EXTRINSIC_ATOL, rtol=0.0):
            raise ValueError("explicit intrinsic does not match camera_angle_x")
        intrinsic = source_intrinsic
        distance_policy = "explicit_matrices_are_physical"
    else:
        if translation is not None and not np.allclose(translation, [0.0, 0.0, source_distance], atol=1e-6, rtol=0.0):
            raise ValueError("off-axis translation requires production extrinsic and intrinsic matrices")
        tx, ty, distance = 0.0, 0.0, source_distance * mesh_scale
        if not math.isfinite(distance):
            raise ValueError("effective camera distance must be finite")
        extrinsic = front_extrinsic(tx, ty, distance)
        distance_policy = "distance_times_mesh_scale"
    off_axis = abs(tx) > 1e-12 or abs(ty) > 1e-12
    contract = FULL_CV_CONTRACT if off_axis else CANONICAL_CONTRACT
    declared = camera.get("camera_contract")
    if declared is not None and declared not in {CANONICAL_CONTRACT, FULL_CV_CONTRACT}:
        raise ValueError(f"unsupported camera_contract: {declared!r}")
    if off_axis and declared == CANONICAL_CONTRACT:
        raise ValueError("off-axis camera cannot declare a canonical camera_contract")
    return {
        "distance": float(distance), "fov": fov, "mesh_scale": 1.0,
        "source_distance": source_distance, "source_mesh_scale": mesh_scale,
        "distance_policy": distance_policy, "explicit": explicit,
        "camera_contract": contract, "off_axis": bool(off_axis),
        "translation_dual": [float(tx), float(ty), float(distance)],
        "front_eye": [float(tx), float(ty), float(distance)],
        "front_look_at": [float(tx), float(ty), 0.0],
        "back_look_at": [float(-tx), float(ty), 0.0],
        "exact_back_eye": [float(-tx), float(ty), float(-distance)],
        "extrinsic": extrinsic, "intrinsic": intrinsic,
    }


def back_eye(distance: float, dyaw_deg: float, pitch_deg: float, scale: float,
             look_at=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Position around the 180-degree world-Y back view, with an offset target."""
    yaw, pitch = math.pi + math.radians(dyaw_deg), math.radians(pitch_deg)
    offset = distance * scale * np.array([math.sin(yaw) * math.cos(pitch),
                                         math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    return np.asarray(look_at, dtype=np.float64) + offset


def look_at_extrinsic(eye, target) -> np.ndarray:
    """OpenCV look-at matrix for validation and CPU camera tests."""
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if not math.isfinite(float(norm)) or norm < 1e-12:
        raise ValueError("camera eye and look-at must be distinct finite positions")
    forward /= norm
    right = np.cross(forward, [0.0, 1.0, 0.0])
    if np.linalg.norm(right) < 1e-12:
        raise ValueError("camera direction is parallel to its up vector")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, -rotation @ eye
    return result


def touches_border(mask: np.ndarray, px: int = BORDER_PX) -> np.ndarray:
    """Whether any foreground occupies the fixed-width frame border."""
    mask = np.asarray(mask, dtype=bool)
    return (mask[..., :px, :].any(axis=(-2, -1)) | mask[..., -px:, :].any(axis=(-2, -1))
            | mask[..., :, :px].any(axis=(-2, -1)) | mask[..., :, -px:].any(axis=(-2, -1)))


def occupancy_score(front_occupancy, back_occupancy, dyaw_deg, pitch_deg, scale,
                    back_touches_border, front_touches_border):
    """The v4 objective compares occupied area, never silhouette overlap."""
    drift = np.deg2rad(np.abs(dyaw_deg) + np.abs(pitch_deg))
    return (1.0 - np.abs(np.asarray(back_occupancy) - front_occupancy)
            - DRIFT_PENALTY_PER_RAD * drift - SCALE_PENALTY_PER_LOG * np.abs(np.log(scale))
            - INFEASIBLE * np.asarray(back_touches_border) * np.logical_not(front_touches_border))


def normal_sign_check(normals: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    """Measure the normal sign on each scanline's left/right/top boundary."""
    mask, normals = np.asarray(mask) > 0.5, np.asarray(normals)
    if normals.shape != (3, *mask.shape):
        raise ValueError("normal and mask shapes do not agree")
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    left = [normals[0, row, np.argmax(mask[row])] for row in rows]
    right = [normals[0, row, mask.shape[1] - 1 - np.argmax(mask[row, ::-1])] for row in rows]
    top = [normals[1, np.argmax(mask[:, col]), col] for col in cols]
    return {"left_edge_nx_neg": float(np.mean(np.asarray(left) < 0)) if left else None,
            "right_edge_nx_pos": float(np.mean(np.asarray(right) > 0)) if right else None,
            "top_edge_ny_pos": float(np.mean(np.asarray(top) > 0)) if top else None}


def validate_normal_signs(checks: Mapping[str, Any]) -> None:
    for side in ("front", "back"):
        for field in ("left_edge_nx_neg", "right_edge_nx_pos", "top_edge_ny_pos"):
            value = _finite(checks.get(side, {}).get(field), f"{side}.{field}")
            if not NORMAL_SIGN_MIN <= value <= 1.0:
                raise ValueError(f"{side}.{field}={value:.4f} is below normal-sign gate {NORMAL_SIGN_MIN}")

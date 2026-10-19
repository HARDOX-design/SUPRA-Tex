"""Optional removal of floating geometry that the reference image does not show.

Generated geometry can contain disconnected pieces outside the object in the
image. A piece is removed when it is not the largest connected component and
most of its faces project outside the image foreground, which is dilated to
tolerate ordinary misalignment. Parts hidden behind the foreground, such as
the back of the object, still project inside it and are kept, as are parts of
any size that the image shows. Extra geometry connected to the object is not
affected.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

FOREGROUND_ALPHA = 128
# Misalignment tolerance as a fraction of the image width (16 px at 1024).
MARGIN = 16 / 1024
MIN_INSIDE = 0.5
# Vertices closer than this are welded, so UV-seam duplicates stay connected.
WELD_STEP = 1e-6
KEY_BITS = 21


def _weld(vertices: torch.Tensor) -> torch.Tensor:
    """Map each vertex to the index of its quantized position."""
    low = vertices.amin(0)
    step = max(WELD_STEP, float((vertices.amax(0) - low).max()) / (2 ** KEY_BITS - 1))
    q = torch.round((vertices.double() - low) / step).long()
    key = (q[:, 0] << (2 * KEY_BITS)) | (q[:, 1] << KEY_BITS) | q[:, 2]
    return torch.unique(key, return_inverse=True)[1]


def _components(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Label faces by connected component with parallel union-find.

    Each round hooks every root to the smallest root among its edges, then
    compresses paths by pointer jumping; components halve per round.
    """
    welded = _weld(vertices)[faces]
    count = int(welded.max()) + 1
    first = torch.cat((welded[:, 0], welded[:, 1]))
    second = torch.cat((welded[:, 1], welded[:, 2]))
    parent = torch.arange(count, device=faces.device)
    while True:
        a, b = parent[first], parent[second]
        differ = a != b
        if not bool(differ.any()):
            break
        low, high = torch.minimum(a[differ], b[differ]), torch.maximum(a[differ], b[differ])
        parent.scatter_reduce_(0, high, low, reduce="amin")
        while True:
            jumped = parent[parent]
            if torch.equal(jumped, parent):
                break
            parent = jumped
        first, second = first[differ], second[differ]
    face_labels = torch.unique(parent[welded[:, 0]], return_inverse=True)[1]
    return face_labels, torch.bincount(face_labels)


def _project(points: torch.Tensor, camera: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized image coordinates (y down) and camera depth, matching completion."""
    if "extrinsic" in camera:
        extrinsic = torch.as_tensor(camera["extrinsic"], dtype=points.dtype, device=points.device)
        intrinsic = torch.as_tensor(camera["intrinsic"], dtype=points.dtype, device=points.device)
        local = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        uvw = local @ intrinsic.T
        return uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-8), local[:, 2]
    distance = float(camera["distance"]) * float(camera.get("mesh_scale", 1.0))
    focal = 0.5 / math.tan(float(camera["camera_angle_x"]) / 2)
    depth = distance - points[:, 2]
    safe = depth.clamp(min=1e-8)
    return torch.stack((points[:, 0] / safe * focal + 0.5, -points[:, 1] / safe * focal + 0.5), dim=1), depth


def _foreground(alpha: np.ndarray) -> np.ndarray:
    """Foreground mask dilated by the misalignment margin."""
    from scipy.ndimage import distance_transform_edt

    radius = max(1, round(MARGIN * alpha.shape[1]))
    return distance_transform_edt(alpha < FOREGROUND_ALPHA) <= radius


def remove_floaters(vertices: torch.Tensor, faces: torch.Tensor, camera: Mapping[str, Any],
                    alpha: np.ndarray, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Drop disconnected components that project mostly outside the image foreground."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source_device = vertices.device
    vertices, faces = vertices.to(device), faces.to(device).long()
    face_labels, counts = _components(vertices, faces)
    report = {"components": int(len(counts)), "removed_components": 0, "removed_faces": 0, "faces": int(len(faces))}
    floating = None
    if len(counts) > 1:
        height, width = alpha.shape
        foreground = torch.from_numpy(_foreground(alpha)).to(device)
        uv, depth = _project(vertices[faces].mean(dim=1), camera)
        x = torch.floor(uv[:, 0] * width).long()
        y = torch.floor(uv[:, 1] * height).long()
        valid = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        inside = torch.zeros_like(valid)
        inside[valid] = foreground[y[valid], x[valid]]
        inside_count = torch.bincount(face_labels, weights=inside.double(), minlength=len(counts))
        floating = inside_count < MIN_INSIDE * counts
        floating[torch.argmax(counts)] = False
    if floating is None or not bool(floating.any()):
        return vertices.to(source_device), faces.to(source_device), report
    kept_faces = faces[~floating[face_labels]]
    used, remap = torch.unique(kept_faces, return_inverse=True)
    report.update(removed_components=int(floating.sum()), removed_faces=int(counts[floating].sum()))
    return vertices[used].to(source_device), remap.reshape(kept_faces.shape).to(source_device), report

"""Visible-surface material observations for Supra-Tex inference.

This module owns the public pixel-first DDA visibility contract.  It has no
dataset, scheduler, or training dependency.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

RESOLUTION = 1024
LATENT_RESOLUTION = 64


def build_occupancy_index(coords: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exact packed occupancy bitset and sorted row map."""
    values = coords.detach().cpu().numpy().astype(np.int64, copy=False)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("surface coordinates must have shape [N,3]")
    if values.size and (values.min() < 0 or values.max() >= RESOLUTION):
        raise ValueError("surface coordinates are outside the material volume")
    keys = (values[:, 0] * RESOLUTION + values[:, 1]) * RESOLUTION + values[:, 2]
    order = np.argsort(keys)
    ordered = keys[order]
    if ordered.size > 1 and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("surface coordinates contain duplicate cells")
    bits = np.zeros((RESOLUTION ** 3 + 7) // 8, dtype=np.uint8)
    if keys.size:
        np.bitwise_or.at(bits, keys >> 3, np.left_shift(np.uint8(1), (keys & 7).astype(np.uint8)))
    return bits, ordered, order.astype(np.int64, copy=False)


_DDA_BATCH_KERNEL: Any | None = None

def _dda_batch_kernel() -> Any:
    """Lazily compile the CPU DDA kernel so mesh-depth mode needs no Numba."""
    global _DDA_BATCH_KERNEL
    if _DDA_BATCH_KERNEL is not None:
        return _DDA_BATCH_KERNEL
    try:
        import numba
    except ImportError as error:
        raise RuntimeError("pixel_dda visibility requires numba") from error

    @numba.njit(parallel=True, cache=True)
    def kernel(
        ray_origin: np.ndarray,
        ray_directions: np.ndarray,
        occupancy_bits: np.ndarray,
        grid_min: float,
        grid_max: float,
        resolution: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        ray_count = ray_directions.shape[0]
        owners = np.full((ray_count, 3), -1, dtype=np.int32)
        reasons = np.full(ray_count, 3, dtype=np.uint8)
        # Per-ray count of multi-axis (diagonal) steps.  Written only at
        # ray_index, so prange stays race-free; summed by the caller.
        ties = np.zeros(ray_count, dtype=np.int64)
        cell_size = (grid_max - grid_min) / resolution
        slab_epsilon = 1e-14
        tie_epsilon = 1e-12

        for ray_index in numba.prange(ray_count):
            direction = ray_directions[ray_index]
            valid_ray = True
            for axis in range(3):
                if not np.isfinite(ray_origin[axis]) or not np.isfinite(direction[axis]):
                    valid_ray = False
            if not valid_ray:
                continue
            if (
                abs(direction[0]) < slab_epsilon
                and abs(direction[1]) < slab_epsilon
                and abs(direction[2]) < slab_epsilon
            ):
                continue

            t_entry = -np.inf
            t_exit = np.inf
            intersects = True
            for axis in range(3):
                component = direction[axis]
                if abs(component) < slab_epsilon:
                    if ray_origin[axis] < grid_min or ray_origin[axis] > grid_max:
                        intersects = False
                        break
                else:
                    t1 = (grid_min - ray_origin[axis]) / component
                    t2 = (grid_max - ray_origin[axis]) / component
                    if t1 > t2:
                        temporary = t1
                        t1 = t2
                        t2 = temporary
                    if t1 > t_entry:
                        t_entry = t1
                    if t2 < t_exit:
                        t_exit = t2
            if not intersects or t_entry > t_exit or t_exit < 0.0:
                reasons[ray_index] = 2
                continue

            t_start = max(t_entry, 0.0)
            # Enter the half-open grid interior deterministically.  This avoids
            # floor(1024) for a ray entering exactly through the +AABB face.
            entry_epsilon = max(1e-12, abs(t_start) * 1e-12)
            point = ray_origin + (t_start + entry_epsilon) * direction
            current = np.empty(3, dtype=np.int32)
            step = np.zeros(3, dtype=np.int32)
            t_delta = np.full(3, np.inf, dtype=np.float64)
            t_max = np.full(3, np.inf, dtype=np.float64)
            for axis in range(3):
                cell = int(math.floor((point[axis] - grid_min) / cell_size))
                if cell < 0:
                    cell = 0
                elif cell >= resolution:
                    cell = resolution - 1
                current[axis] = cell
                component = direction[axis]
                if component > slab_epsilon:
                    step[axis] = 1
                    boundary = grid_min + (cell + 1) * cell_size
                    t_max[axis] = (boundary - ray_origin[axis]) / component
                    t_delta[axis] = cell_size / component
                elif component < -slab_epsilon:
                    step[axis] = -1
                    boundary = grid_min + cell * cell_size
                    t_max[axis] = (boundary - ray_origin[axis]) / component
                    t_delta[axis] = -cell_size / component

            max_steps = 3 * resolution + 3
            for _ in range(max_steps):
                key = (
                    (np.int64(current[0]) * resolution + current[1]) * resolution
                    + current[2]
                )
                byte_value = occupancy_bits[key >> 3]
                if byte_value & (1 << (key & 7)):
                    owners[ray_index, 0] = current[0]
                    owners[ray_index, 1] = current[1]
                    owners[ray_index, 2] = current[2]
                    reasons[ray_index] = 0
                    break

                next_t = min(t_max[0], t_max[1], t_max[2])
                if not np.isfinite(next_t) or next_t > t_exit + tie_epsilon:
                    reasons[ray_index] = 1
                    break
                tolerance = tie_epsilon * max(1.0, abs(next_t))
                in_bounds = True
                # Step all axes crossed at the same ray parameter.  This is the
                # half-open tie rule for edge/corner hits and is deterministic.
                crossed = 0
                for axis in range(3):
                    if abs(t_max[axis] - next_t) <= tolerance:
                        crossed += 1
                        current[axis] += step[axis]
                        t_max[axis] += t_delta[axis]
                        if current[axis] < 0 or current[axis] >= resolution:
                            in_bounds = False
                # A multi-axis step is a diagonal move, so it can skip cells that
                # share only an edge or corner.  The rule is frozen and fixture
                # verified; count the events so the magnitude is known over a
                # full corpus (~2e12 rays) instead of assumed negligible.
                if crossed > 1:
                    ties[ray_index] += 1
                if not in_bounds:
                    reasons[ray_index] = 1
                    break

        return owners, reasons, ties

    _DDA_BATCH_KERNEL = kernel
    return _DDA_BATCH_KERNEL


# Optional admission rules, off by default. On well-aligned demo inputs they
# removed 3-10% of trusted pixels without a visible change, so the released
# behavior keeps only the depth-discontinuity rule.
SILHOUETTE_MARGIN = 0
GRAZING_COS = 0.0


def _admitted_pixels(depth: np.ndarray, missed: np.ndarray, focal: tuple[float, float], depth_edge_rtol: float,
                     silhouette_margin: int = SILHOUETTE_MARGIN, grazing_cos: float = GRAZING_COS,
                     stats: dict | None = None) -> np.ndarray:
    """Return pixels whose first-hit depth is a reliable observation.

    ``depth`` is the first-hit view depth (NaN without a trusted hit) and
    ``missed`` marks trusted rays that left the volume, i.e. the image
    foreground outside the geometry silhouette. A local depth discontinuity
    always rejects a pixel. Two optional rules can also reject pixels near the
    geometry silhouette (``silhouette_margin`` pixels), where image/geometry
    misalignment lands colors on the wrong surface, and at grazing incidence
    (cosine below ``grazing_cos``, from the screen-space depth slope), where one
    pixel stretches over a large surface area.
    """
    import torch.nn.functional as F

    masked = torch.from_numpy(depth)
    local_max = F.max_pool2d(masked.nan_to_num(-torch.inf)[None, None], 3, 1, 1)[0, 0]
    local_min = -F.max_pool2d((-masked).nan_to_num(-torch.inf)[None, None], 3, 1, 1)[0, 0]
    relative = ((local_max - local_min) / masked).nan_to_num(nan=0., posinf=0., neginf=0.)
    edge = relative.numpy() > depth_edge_rtol
    near_silhouette = np.zeros_like(edge)
    if silhouette_margin > 0 and missed.any():
        size = 2 * silhouette_margin + 1
        near_silhouette = F.max_pool2d(torch.from_numpy(missed).float()[None, None], size, 1, silhouette_margin)[0, 0].numpy() > 0
    grazing = np.zeros_like(edge)
    if grazing_cos > 0:
        # Central differences in pixels; a missing neighbour leaves NaN and is
        # not rejected here. Dividing by the pixel footprint (depth / focal)
        # turns the depth change into the surface slope along each image axis.
        dz_dx = np.full_like(depth, np.nan)
        dz_dy = np.full_like(depth, np.nan)
        dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) * .5
        dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) * .5
        with np.errstate(invalid="ignore", divide="ignore"):
            slope2 = (dz_dx * np.float32(focal[0]) / depth) ** 2 + (dz_dy * np.float32(focal[1]) / depth) ** 2
            grazing = slope2 > np.float32(1 / grazing_cos ** 2 - 1)
    hit = ~np.isnan(depth)
    admitted = hit & ~edge & ~near_silhouette & ~grazing
    if stats is not None:
        stats.update(trusted_rays=int(hit.sum() + missed.sum()), first_hits=int(hit.sum()),
                     rejected_depth_edge=int((hit & edge).sum()),
                     rejected_silhouette=int((hit & ~edge & near_silhouette).sum()),
                     rejected_grazing=int((hit & ~edge & ~near_silhouette & grazing).sum()),
                     admitted=int(admitted.sum()))
    return admitted


def pixel_first_occupied(camera: dict[str, Any], occupancy: np.ndarray, trusted: torch.Tensor, depth_edge_rtol: float = 0.04,
                         stats: dict | None = None) -> torch.Tensor:
    """Map trusted camera pixels to their first occupied volume cell."""
    if "extrinsic" in camera:
        return pixel_first_occupied_cv(camera, occupancy, trusted, depth_edge_rtol, stats=stats)
    if trusted.shape != (RESOLUTION, RESOLUTION):
        raise ValueError("trusted alpha mask must be 1024 by 1024")
    if "camera_angle_x" not in camera or "distance" not in camera:
        raise ValueError("camera must supply camera_angle_x and distance")
    ys, xs = torch.nonzero(trusted, as_tuple=True)
    x, y = xs.cpu().numpy(), ys.cpu().numpy()
    focal = 16.0 / math.tan(float(camera["camera_angle_x"]) / 2.0) * RESOLUTION / 32.0
    directions = np.column_stack(((x + .5 - RESOLUTION / 2) / focal, -(y + .5 - RESOLUTION / 2) / focal, -np.ones_like(x))).astype(np.float64)
    origin = np.array([0.0, 0.0, float(camera["distance"]) * float(camera.get("mesh_scale", 1.0))], dtype=np.float64)
    owners, reasons, _ties = _dda_batch_kernel()(origin, directions, occupancy, -0.5, 0.5, RESOLUTION)
    hit_mask = reasons == 0
    hit = owners[hit_mask]
    if not hit.size:
        raise RuntimeError("no trusted pixels intersect the material volume")
    # Admission runs per pixel, before owner cells are deduplicated; applying
    # it after unique-owner reduction would be wrong.
    depth = np.full((RESOLUTION, RESOLUTION), np.nan, dtype=np.float32)
    hit_depth = origin[2] - ((hit[:, 2].astype(np.float64) + .5) / RESOLUTION - .5)
    depth[y[hit_mask], x[hit_mask]] = hit_depth.astype(np.float32)
    missed = np.zeros((RESOLUTION, RESOLUTION), dtype=bool)
    missed[y[~hit_mask], x[~hit_mask]] = True
    admitted = _admitted_pixels(depth, missed, (focal, focal), depth_edge_rtol, stats=stats)
    retained = hit_mask.copy()
    retained[hit_mask] &= admitted[y[hit_mask], x[hit_mask]]
    hit = owners[retained]
    if not hit.size:
        raise RuntimeError("observation admission removed every first-hit observation")
    keys = np.unique((hit[:, 0].astype(np.int64) * RESOLUTION + hit[:, 1]) * RESOLUTION + hit[:, 2])
    return torch.from_numpy(np.stack((keys // (RESOLUTION * RESOLUTION), (keys // RESOLUTION) % RESOLUTION, keys % RESOLUTION), axis=1).astype(np.int32))


def camera_rays(camera: dict[str, Any], ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pixel-center rays from a complete OpenCV world-to-camera pose."""
    extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(camera.get("projection_intrinsic", camera.get("intrinsic")), dtype=np.float64)
    if extrinsic.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("DDA camera requires 4x4 extrinsic and normalized 3x3 intrinsic")
    if not np.isfinite(extrinsic).all() or not np.isfinite(intrinsic).all():
        raise ValueError("DDA camera matrices must be finite")
    rotation = extrinsic[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
            or not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("DDA camera must contain a rigid world-to-camera transform")
    if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[[0, 1, 2, 2], [1, 0, 0, 1]], 0, atol=1e-10)
            or intrinsic[2, 2] != 1):
        raise ValueError("DDA requires positive focal lengths and an unskewed normalized intrinsic")
    eye = np.ascontiguousarray(-rotation.T @ extrinsic[:3, 3])
    x = (xs.astype(np.float64) + .5 - intrinsic[0, 2] * RESOLUTION) / (intrinsic[0, 0] * RESOLUTION)
    y = (ys.astype(np.float64) + .5 - intrinsic[1, 2] * RESOLUTION) / (intrinsic[1, 1] * RESOLUTION)
    directions = np.ascontiguousarray(np.stack((x, y, np.ones_like(x)), axis=1) @ rotation)
    return eye, directions, rotation


def pixel_hits_cv(camera: dict[str, Any], occupancy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the raw first-hit silhouette, without trusted-mask depth filtering."""
    if mask.shape != (RESOLUTION, RESOLUTION) or mask.dtype != np.bool_:
        raise ValueError("DDA pixel mask must be a boolean observation-resolution image")
    ys, xs = np.nonzero(mask)
    eye, directions, _ = camera_rays(camera, ys, xs)
    _, reasons, _ = _dda_batch_kernel()(eye, directions, occupancy, -.5, .5, RESOLUTION)
    hits = np.zeros_like(mask)
    hits[ys[reasons == 0], xs[reasons == 0]] = True
    return hits


def pixel_first_occupied_cv(camera: dict[str, Any], occupancy: np.ndarray, trusted: torch.Tensor,
                            depth_edge_rtol: float = .04, stats: dict | None = None) -> torch.Tensor:
    """Apply the production first-hit admission rules to a full CV camera."""
    if trusted.shape != (RESOLUTION, RESOLUTION) or trusted.dtype != torch.bool:
        raise ValueError("trusted alpha must be a boolean observation-resolution image")
    if not math.isfinite(depth_edge_rtol) or depth_edge_rtol < 0:
        raise ValueError("depth-edge tolerance must be finite and nonnegative")
    ys, xs = np.nonzero(trusted.detach().cpu().numpy())
    if not ys.size:
        raise RuntimeError("trusted alpha gate leaves no camera rays")
    eye, directions, rotation = camera_rays(camera, ys, xs)
    owners, reasons, _ = _dda_batch_kernel()(eye, directions, occupancy, -.5, .5, RESOLUTION)
    hits = reasons == 0
    if not hits.any():
        raise RuntimeError("no trusted pixels intersect the material volume")
    centers = (owners[hits].astype(np.float64) + .5) / RESOLUTION - .5
    hit_depth = ((centers - eye) @ rotation.T)[:, 2]
    if not np.isfinite(hit_depth).all() or not (hit_depth > 0).all():
        raise RuntimeError("first-hit owners must lie in front of the camera")
    depth = np.full((RESOLUTION, RESOLUTION), np.nan, dtype=np.float32)
    depth[ys[hits], xs[hits]] = hit_depth.astype(np.float32)
    missed = np.zeros((RESOLUTION, RESOLUTION), dtype=bool)
    missed[ys[~hits], xs[~hits]] = True
    intrinsic = np.asarray(camera.get("projection_intrinsic", camera.get("intrinsic")), dtype=np.float64)
    focal = (float(intrinsic[0, 0]) * RESOLUTION, float(intrinsic[1, 1]) * RESOLUTION)
    retained = hits & _admitted_pixels(depth, missed, focal, depth_edge_rtol, stats=stats)[ys, xs]
    kept = owners[retained].astype(np.int64, copy=False)
    if not kept.size:
        raise RuntimeError("observation admission removed every first-hit observation")
    keys = np.unique((kept[:, 0] * RESOLUTION + kept[:, 1]) * RESOLUTION + kept[:, 2])
    return torch.from_numpy(np.stack((keys // (RESOLUTION * RESOLUTION),
                                      keys // RESOLUTION % RESOLUTION,
                                      keys % RESOLUTION), axis=1).astype(np.int32))


def owner_rows(coords: torch.Tensor, owners: torch.Tensor, ordered: np.ndarray, order: np.ndarray) -> torch.Tensor:
    values = owners.cpu().numpy().astype(np.int64, copy=False)
    keys = (values[:, 0] * RESOLUTION + values[:, 1]) * RESOLUTION + values[:, 2]
    positions = np.searchsorted(ordered, keys)
    if positions.size and (np.any(positions >= ordered.size) or np.any(ordered[positions] != keys)):
        raise RuntimeError("visible owner is absent from the surface support")
    result = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
    if positions.size:
        result[torch.as_tensor(order[positions], device=coords.device)] = True
    return result


def boundary_distance_to_unobserved(surface_counts: torch.Tensor, observed_counts: torch.Tensor) -> torch.Tensor:
    """Exact sparse six-neighbour distance used by the observation message."""
    active = (surface_counts > 0).reshape(LATENT_RESOLUTION, LATENT_RESOLUTION, LATENT_RESOLUTION)
    observed = (observed_counts > 0).reshape_as(active)
    distance = torch.full(active.shape, -1, dtype=torch.int16, device=active.device)
    frontier = active & ~observed
    distance[frontier] = 0
    step = 0
    while bool(frontier.any()):
        expanded = torch.zeros_like(frontier)
        expanded[1:] |= frontier[:-1]; expanded[:-1] |= frontier[1:]
        expanded[:, 1:] |= frontier[:, :-1]; expanded[:, :-1] |= frontier[:, 1:]
        expanded[:, :, 1:] |= frontier[:, :, :-1]; expanded[:, :, :-1] |= frontier[:, :, 1:]
        frontier = expanded & active & (distance < 0)
        if not bool(frontier.any()):
            break
        step += 1
        distance[frontier] = step
    return distance.reshape(-1)


def encode_partial(encoder: Any, coords: torch.Tensor, visible: torch.Tensor, rgb: torch.Tensor, mra: torch.Tensor, shape_coords: torch.Tensor, normalization: dict[str, Any], sparse: Any, device: str) -> Any:
    """Encode visible PBR rows into the frozen 34-channel observation message."""
    selected = coords[visible]
    features = torch.cat((rgb[visible], mra[visible]), dim=1).clamp(0, 1).float() * 2.0 - 1.0
    encoded = encoder(sparse.SparseTensor(features, torch.cat((torch.zeros_like(selected[:, :1]), selected), dim=1).int()), sample_posterior=False)
    raw = encoded.feats.float()
    if raw.ndim != 2 or raw.shape[1] != 32:
        raise RuntimeError("material encoder must emit 32 channels")
    blocks = coords // (RESOLUTION // LATENT_RESOLUTION)
    keys = (blocks[:, 0] * LATENT_RESOLUTION + blocks[:, 1]) * LATENT_RESOLUTION + blocks[:, 2]
    all_counts = torch.bincount(keys, minlength=LATENT_RESOLUTION ** 3)
    observed_counts = torch.bincount(keys[visible], minlength=LATENT_RESOLUTION ** 3)
    xyz = encoded.coords[:, 1:].long()
    latent_keys = (xyz[:, 0] * LATENT_RESOLUTION + xyz[:, 1]) * LATENT_RESOLUTION + xyz[:, 2]
    # The bounded grid already has exact occupancy counts. Enumerate occupied
    # keys instead of sorting millions of three-column surface coordinates.
    # Linear-key order is the same lexicographic order as unique(..., dim=0).
    expected_observed = torch.nonzero(observed_counts, as_tuple=False).flatten()
    if (bool(((xyz < 0) | (xyz >= LATENT_RESOLUTION)).any())
            or not torch.equal(torch.sort(latent_keys).values, expected_observed)):
        raise RuntimeError("encoded observation support differs from floor(visible_surface/16)")
    coverage = (observed_counts[latent_keys].float() / all_counts[latent_keys].float()).unsqueeze(1)
    expected_keys = torch.nonzero(all_counts, as_tuple=False).flatten()
    shape_xyz = shape_coords.long()
    shape_keys = (shape_xyz[:, 0] * LATENT_RESOLUTION + shape_xyz[:, 1]) * LATENT_RESOLUTION + shape_xyz[:, 2]
    sorted_shape = torch.sort(shape_keys).values
    if expected_keys.shape[0] != shape_keys.shape[0] or not torch.equal(expected_keys, sorted_shape):
        raise RuntimeError("pooled material support does not exactly match the shape latent support")
    positions = torch.searchsorted(sorted_shape, latent_keys)
    if bool((positions >= sorted_shape.shape[0]).any()) or not torch.equal(sorted_shape[positions], latent_keys):
        raise RuntimeError("encoded observation support is not a subset of full material support")
    boundary = boundary_distance_to_unobserved(all_counts, observed_counts)[latent_keys].to(dtype=coverage.dtype).unsqueeze(1)
    if bool((boundary == 0).any()) or bool((boundary < -1).any()):
        raise RuntimeError("invalid observed material boundary distance")
    mean = torch.as_tensor(normalization["mean"], dtype=raw.dtype, device=device).reshape(1, 32)
    std = torch.as_tensor(normalization["std"], dtype=raw.dtype, device=device).reshape(1, 32)
    if mean.shape[1] != 32 or std.shape[1] != 32 or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("partial observation normalization must provide 32 finite positive standard deviations")
    return sparse.SparseTensor(torch.cat(((raw - mean) / std, coverage, boundary), dim=1), encoded.coords.int())

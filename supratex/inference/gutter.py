"""Opt-in, chart-isolated one-texel RGB padding for an exported UV atlas."""
from __future__ import annotations

import numpy as np
from PIL import Image


def pad_rgb(image: np.ndarray, chart: np.ndarray) -> tuple[np.ndarray, int]:
    """Copy covered donor RGB into a one-texel ring; reject mixed-chart donors.

    ``chart`` is -1 outside rasterized UV triangles. Cardinal donors take
    precedence over diagonals. Every donor comes from the original covered
    region, so padding never grows recursively or changes covered texels.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise ValueError("gutter requires an RGB/RGBA uint8 atlas")
    if chart.shape != image.shape[:2] or not np.issubdtype(chart.dtype, np.integer):
        raise ValueError("UV chart labels must be an integer array matching the atlas")
    height, width = chart.shape
    donor = np.full((height, width), -1, dtype=np.int32)
    selected_chart = np.full_like(donor, -1)
    conflict = np.zeros((height, width), dtype=bool)
    for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        sy, sx = slice(max(0, -dy), height - max(0, dy)), slice(max(0, -dx), width - max(0, dx))
        ty, tx = slice(max(0, dy), height - max(0, -dy)), slice(max(0, dx), width - max(0, -dx))
        source_chart = chart[sy, sx]
        target_chart = selected_chart[ty, tx]
        eligible = (chart[ty, tx] < 0) & (source_chart >= 0)
        conflict[ty, tx] |= eligible & (target_chart >= 0) & (target_chart != source_chart)
        choose = eligible & (target_chart < 0)
        rows, cols = np.nonzero(choose)
        donor[ty, tx][rows, cols] = (rows + sy.start) * width + cols + sx.start
        target_chart[choose] = source_chart[choose]
    write = (donor >= 0) & ~conflict
    after = image.copy()
    after[write, :3] = image.reshape(-1, image.shape[2])[donor[write], :3]
    return after, int(np.any(after[..., :3] != image[..., :3], axis=-1).sum())


def _rasterize_charts(uv: np.ndarray, faces: np.ndarray, size: tuple[int, int], device: str) -> np.ndarray:
    """Rasterize final GLB UVs in the same image-row convention as the baker."""
    import torch
    import nvdiffrast.torch as dr
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from trimesh.graph import face_adjacency

    # UV seams have separate vertex indices in the exporter's unwrapped mesh.
    # Sharing an entire edge joins faces into a chart; point contacts do not.
    adjacent = face_adjacency(faces=faces)
    graph = coo_matrix((np.ones(len(adjacent), dtype=np.uint8), (adjacent[:, 0], adjacent[:, 1])),
                       shape=(len(faces), len(faces)))
    _, face_chart = connected_components(graph, directed=False)
    with torch.inference_mode():
        coords = torch.as_tensor(uv.copy(), dtype=torch.float32, device=device)
        coords[:, 1] = 1 - coords[:, 1]  # Undo the baker's final GLB V flip.
        clip = torch.cat((coords * 2 - 1, torch.zeros_like(coords[:, :1]), torch.ones_like(coords[:, :1])), dim=1)[None]
        triangles = torch.as_tensor(faces, dtype=torch.int32, device=device).contiguous()
        labels = torch.as_tensor(face_chart, dtype=torch.int32, device=device)
        chart = torch.full(size, -1, dtype=torch.int32, device=device)
        context = dr.RasterizeCudaContext(device=device)
        for start in range(0, len(faces), 100000):
            rast, _ = dr.rasterize(context, clip, triangles[start:start + 100000], resolution=size)
            primitive = rast[0, ..., 3].long() - 1
            covered = primitive >= 0
            chart[covered] = labels[primitive[covered] + start]
        return chart.cpu().numpy()


def apply_gutter(mesh, *, device: str = "cuda") -> dict[str, int]:
    """Pad only the exported base-color RGB; preserve alpha and other content."""
    material = mesh.visual.material
    image = material.baseColorTexture
    if image is None or image.mode not in ("RGB", "RGBA"):
        raise ValueError("gutter requires an exported RGB/RGBA base-color texture")
    uv, faces = np.asarray(mesh.visual.uv), np.asarray(mesh.faces)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all() or np.any((uv < 0) | (uv > 1)):
        raise ValueError("gutter requires finite UV coordinates in [0, 1]")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("gutter requires indexed triangular faces")
    if faces.min() < 0 or faces.max() >= len(uv):
        raise ValueError("gutter faces reference invalid UV vertices")
    before = np.asarray(image)
    chart = _rasterize_charts(uv, faces, before.shape[:2], device)
    if not np.any(chart >= 0):
        raise ValueError("gutter UV rasterization produced no covered texels")
    after, changed = pad_rgb(before, chart)
    material.baseColorTexture = Image.fromarray(after)
    return {"covered_texels": int((chart >= 0).sum()), "changed_texels": changed, "radius": 1}

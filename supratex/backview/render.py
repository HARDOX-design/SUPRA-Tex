"""CUDA implementation of the v4 occupancy search and normal rendering."""

from __future__ import annotations

import math
import time

import numpy as np

from .camera import (BORDER_PX, COARSE_RES, FINE_RES, NEAR, FAR, PITCH_RANGE_DEG,
                     RESOLUTION, SCALE_RANGE, YAW_RANGE_DEG, normal_sign_check,
                     occupancy_score)


def _eyes(distance, params, target, device):
    import torch
    params = torch.as_tensor(params, dtype=torch.float32, device=device)
    yaw, pitch = math.pi + torch.deg2rad(params[:, 0]), torch.deg2rad(params[:, 1])
    offsets = (distance * params[:, 2, None]) * torch.stack(
        [torch.sin(yaw) * torch.cos(pitch), torch.sin(pitch), torch.cos(yaw) * torch.cos(pitch)], dim=-1)
    return offsets + torch.as_tensor(target, dtype=torch.float32, device=device)


def _look_at(eyes, target):
    import torch
    import utils3d
    target = torch.as_tensor(target, dtype=eyes.dtype, device=eyes.device).expand_as(eyes)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=eyes.dtype, device=eyes.device).expand_as(eyes)
    return utils3d.torch.extrinsics_look_at(eyes, target, up)


def _border(masks):
    return (masks[..., :BORDER_PX, :].flatten(-2).any(-1)
            | masks[..., -BORDER_PX:, :].flatten(-2).any(-1)
            | masks[..., :, :BORDER_PX].flatten(-2).any(-1)
            | masks[..., :, -BORDER_PX:].flatten(-2).any(-1))


class SilhouetteRasterizer:
    """Binary silhouettes using precisely the local MeshRenderer projection."""

    def __init__(self, vertices, faces, intrinsic):
        import torch
        import nvdiffrast.torch as dr
        from supratex.renderers.mesh_renderer import intrinsics_to_projection
        self.dr, self.faces = dr, faces
        self.context = dr.RasterizeCudaContext(device=vertices.device)
        self.vertices = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1).contiguous()
        self.projection = intrinsics_to_projection(intrinsic, NEAR, FAR)

    def masks(self, extrinsics, resolution):
        import torch
        masks = []
        # One camera per raster call bounds peak triangle-buffer memory.
        for extrinsic in extrinsics:
            full = self.projection @ extrinsic
            clip = (self.vertices @ full.T).unsqueeze(0).contiguous()
            raster, _ = self.dr.rasterize(self.context, clip, self.faces, (resolution, resolution))
            masks.append(raster[..., 3] > 0)
        return torch.cat(masks, dim=0)


class BackSearch:
    """Deterministic 256-pixel grid followed by 512-pixel pattern search."""

    def __init__(self, raster, camera, front_extrinsic, device):
        self.raster, self.camera = raster, camera
        self.front_extrinsic, self.device = front_extrinsic, device
        self.front_cache = {}

    def front(self, resolution):
        if resolution not in self.front_cache:
            mask = self.raster.masks(self.front_extrinsic[None], resolution)[0]
            self.front_cache[resolution] = (float(mask.float().mean()), bool(_border(mask)))
        return self.front_cache[resolution]

    def score(self, params, resolution):
        eyes = _eyes(self.camera["distance"], params, self.camera["back_look_at"], self.device)
        masks = self.raster.masks(_look_at(eyes, self.camera["back_look_at"]), resolution)
        occupancy = masks.flatten(1).float().mean(1).cpu().numpy()
        border = _border(masks).cpu().numpy()
        front_occupancy, front_border = self.front(resolution)
        params = np.asarray(params, dtype=np.float32)
        scores = occupancy_score(front_occupancy, occupancy, params[:, 0], params[:, 1], params[:, 2],
                                 border, front_border)
        return scores, occupancy

    def run(self):
        start = time.monotonic()
        yaws = np.arange(-YAW_RANGE_DEG, YAW_RANGE_DEG + 1e-6, 5.0)
        pitches = np.arange(-PITCH_RANGE_DEG, PITCH_RANGE_DEG + 1e-6, 5.0)
        scales = np.round(np.arange(SCALE_RANGE[0], SCALE_RANGE[1] + 1e-6, 0.05), 4)
        grid = np.array([(yaw, pitch, scale) for yaw in yaws for pitch in pitches for scale in scales],
                        dtype=np.float32)
        scores, occupancy = self.score(grid, COARSE_RES)
        # Stable ordering makes exact ties reproducible across repeated runs.
        order = np.argsort(-scores, kind="stable")
        top5 = [{"dyaw_deg": float(grid[index, 0]), "pitch_deg": float(grid[index, 1]),
                 "scale": float(grid[index, 2]), "back_occupancy": float(occupancy[index]),
                 "score": float(scores[index])} for index in order[:5]]
        best = grid[order[0]].copy()
        best_score = float(self.score(best[None], FINE_RES)[0][0])
        steps = np.array([2.5, 2.5, 0.025], dtype=np.float32)
        lower = np.array([-YAW_RANGE_DEG, -PITCH_RANGE_DEG, SCALE_RANGE[0]], dtype=np.float32)
        upper = np.array([YAW_RANGE_DEG, PITCH_RANGE_DEG, SCALE_RANGE[1]], dtype=np.float32)
        offsets = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                            if (a, b, c) != (0, 0, 0)], dtype=np.float32)
        evaluations = 0
        for _ in range(4):
            for _ in range(8):
                candidates = np.clip(best[None] + offsets * steps[None], lower, upper)
                scores, _ = self.score(candidates, FINE_RES)
                evaluations += len(candidates)
                winner = int(np.argmax(scores))
                if float(scores[winner]) <= best_score + 1e-6:
                    break
                best, best_score = candidates[winner].copy(), float(scores[winner])
            steps *= 0.5
        return {"dyaw_deg": float(best[0]), "pitch_deg": float(best[1]), "scale": float(best[2]),
                "fine_score": best_score,
                "coarse_grid": {"dyaw_deg": [-YAW_RANGE_DEG, YAW_RANGE_DEG, 5.0],
                                "pitch_deg": [-PITCH_RANGE_DEG, PITCH_RANGE_DEG, 5.0],
                                "scale": [SCALE_RANGE[0], SCALE_RANGE[1], 0.05],
                                "count": len(grid), "resolution": COARSE_RES},
                "coarse_top5": top5,
                "refine": {"resolution": FINE_RES, "evaluations": evaluations,
                           "final_step": (steps * 2).astype(float).tolist()},
                "seconds": round(time.monotonic() - start, 3)}


def _normal(renderer, mesh, extrinsic, intrinsic):
    import torch
    result = renderer.render(mesh, extrinsic, intrinsic, return_types=["mask", "normal"])
    cv = result["normal"] * 2.0 - 1.0
    normals = torch.nn.functional.normalize(torch.stack([-cv[0], cv[1], cv[2]], dim=0), dim=0)
    mask = result["mask"].reshape(RESOLUTION, RESOLUTION)
    rgb = (normals * 0.5 + 0.5).clamp(0, 1) * mask[None]
    rgb = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return rgb, normals.cpu().numpy(), mask.cpu().numpy()


def render_geometry(payload, camera, *, device="cuda", simplify=0):
    """Render a single mesh; image writes and provenance are owned by pack.py."""
    import torch
    import utils3d
    from supratex.renderers import MeshRenderer
    from supratex.representations import Mesh
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("back-view normal rendering requires CUDA and nvdiffrast")
    with torch.cuda.device(torch.device(device)), torch.inference_mode():
        vertices = payload["verts"].detach().to(device=device, dtype=torch.float32).contiguous()
        faces = payload["faces"].detach().to(device=device, dtype=torch.int32).contiguous()
        mesh = Mesh(vertices, faces)
        if simplify and len(faces) > simplify:
            mesh.simplify(target=simplify)
            vertices, faces = mesh.vertices.contiguous(), mesh.faces.int().contiguous()
        if camera["explicit"]:
            front = torch.as_tensor(camera["extrinsic"], device=device, dtype=torch.float32)
            intrinsic = torch.as_tensor(camera["intrinsic"], device=device, dtype=torch.float32)
        else:
            eye = torch.tensor(camera["front_eye"], device=device, dtype=torch.float32)
            front = _look_at(eye[None], camera["front_look_at"])[0]
            fov = torch.tensor(camera["fov"], device=device, dtype=torch.float32)
            intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        raster = SilhouetteRasterizer(vertices, faces, intrinsic)
        search = BackSearch(raster, camera, front, device).run()
        params = [[0.0, 0.0, 1.0], [search["dyaw_deg"], search["pitch_deg"], search["scale"]]]
        eyes = _eyes(camera["distance"], params, camera["back_look_at"], device)
        back_extrinsics = _look_at(eyes, camera["back_look_at"])
        masks = raster.masks(back_extrinsics, RESOLUTION).cpu().numpy()
        front_silhouette = raster.masks(front[None], RESOLUTION)[0].cpu().numpy()
        renderer = MeshRenderer({"resolution": RESOLUTION, "near": NEAR, "far": FAR,
                                 "ssaa": 1, "antialias": True}, device=device)
        back_rgb, back_normals, back_mask = _normal(renderer, mesh, back_extrinsics[1], intrinsic)
        front_rgb, front_normals, front_mask = _normal(renderer, mesh, front, intrinsic)
        return {"front_normal": front_rgb, "back_normal": back_rgb, "back_mask": back_mask > 0.5,
                "front_silhouette": front_silhouette, "exact_back_silhouette": masks[0],
                "back_silhouette": masks[1], "search": search, "faces": len(faces), "simplify": simplify,
                "front_extrinsic": front.cpu().numpy(), "back_extrinsic": back_extrinsics[1].cpu().numpy(),
                "back_eye": eyes[1].cpu().numpy(), "intrinsic": intrinsic.cpu().numpy(),
                "normal_sign_check": {"front": normal_sign_check(front_normals, front_mask),
                                      "back": normal_sign_check(back_normals, back_mask)}}

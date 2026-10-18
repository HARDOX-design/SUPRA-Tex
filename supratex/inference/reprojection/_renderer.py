"""Unlit full-camera rendering for measured atlas preservation gates."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping
DEFAULT_RESOLUTION = 1024

def _load_textured_mesh(glb: Path, rt: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any, Any, float, float]:
    (np, trimesh) = (rt['np'], rt['trimesh'])
    loaded = trimesh.load(glb, process=False)
    mesh = next(iter(loaded.geometry.values())) if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    visual = mesh.visual
    if vertices.size == 0 or faces.size == 0 or normals.shape != vertices.shape:
        raise ValueError('GLB mesh is empty or lacks usable vertex normals')
    if not hasattr(visual, 'uv') or visual.uv is None:
        raise ValueError('GLB mesh lacks UV coordinates')
    material = visual.material
    image = getattr(material, 'baseColorTexture', None)
    if image is None:
        raise ValueError('GLB mesh lacks a baseColorTexture')
    metallic_roughness_image = getattr(material, 'metallicRoughnessTexture', None)
    if metallic_roughness_image is None:
        raise ValueError('GLB mesh lacks metallicRoughnessTexture; refusing to label textured render as material-aware')
    texture = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    metallic_roughness = np.asarray(metallic_roughness_image.convert('RGB'), dtype=np.float32) / 255.0
    raw_factor = getattr(material, 'baseColorFactor', None)
    factor = np.asarray([1, 1, 1, 1] if raw_factor is None else raw_factor, dtype=np.float32)[:3]
    if factor.max(initial=1.0) > 1.0:
        factor /= 255.0
    metallic = getattr(material, 'metallicFactor', None)
    roughness = getattr(material, 'roughnessFactor', None)
    return (vertices, faces, normals, np.asarray(visual.uv, dtype=np.float32), texture.astype(np.float32), metallic_roughness.astype(np.float32), factor.astype(np.float32), float(1 if metallic is None else metallic), float(1 if roughness is None else roughness))

def _srgb_factor_torch(srgb: Any, factor: Any, torch: Any) -> Any:
    """Apply a glTF baseColorFactor in linear light and return sRGB samples."""
    linear = torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055).pow(2.4))
    linear = linear * torch.as_tensor(factor, dtype=srgb.dtype, device=srgb.device).view(1, 1, 1, 3)
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(0).pow(1.0 / 2.4) - 0.055).clamp(0.0, 1.0)

def _runtime_imports() -> dict[str, Any]:
    """Import CUDA dependencies only when rendering is actually requested."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import trimesh
    import nvdiffrast.torch as dr
    import utils3d
    from PIL import Image
    from supratex.renderers.mesh_renderer import intrinsics_to_projection
    from ._renderer import _load_textured_mesh
    return locals()

def _check_resolution(resolution: int) -> None:
    if resolution <= 0:
        raise ValueError('resolution must be positive')

def _camera_distance(camera: Mapping[str, Any]) -> float:
    try:
        distance = float(camera['distance']) * float(camera.get('mesh_scale', 1.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('camera requires canonical distance and optional mesh_scale') from error
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('camera distance * mesh_scale must be finite and positive')
    return distance

def _native_camera(camera: Mapping[str, Any], device: str, rt: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the baseline camera, accepting a full CV camera when supplied."""
    if 'extrinsic' in camera and ('projection_intrinsic' in camera or 'intrinsic' in camera):
        intrinsic_key = 'projection_intrinsic' if 'projection_intrinsic' in camera else 'intrinsic'
        extrinsic = rt['torch'].as_tensor(camera['extrinsic'], dtype=rt['torch'].float32, device=device)
        intrinsic = rt['torch'].as_tensor(camera[intrinsic_key], dtype=rt['torch'].float32, device=device)
        if extrinsic.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise ValueError('full camera requires 4x4 extrinsic and 3x3 intrinsic')
        return (extrinsic, intrinsic)
    distance = _camera_distance(camera)
    try:
        fov = float(camera['camera_angle_x'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('canonical camera requires camera_angle_x') from error
    if not 0 < fov < math.pi:
        raise ValueError('camera_angle_x must be in (0, pi)')
    (torch, utils3d) = (rt['torch'], rt['utils3d'])
    eye = torch.tensor((0.0, 0.0, distance), dtype=torch.float32, device=device)
    target = torch.zeros(3, dtype=torch.float32, device=device)
    up = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=device)
    extrinsic = utils3d.torch.extrinsics_look_at(eye, target, up)
    fov_t = torch.tensor(fov, dtype=torch.float32, device=device)
    return (extrinsic, utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t))

def _exported_glb_camera(camera: Mapping[str, Any], device: str, rt: Mapping[str, Any]):
    """Map exported vertices into the native camera, including off-axis poses."""
    (extrinsic, intrinsic) = _native_camera(camera, device, rt)
    sign = rt['torch'].diag(rt['torch'].tensor([-1.0, 1.0, -1.0, 1.0], device=device))
    return (extrinsic @ sign, intrinsic)

def _write_outputs(out: Path, albedo: Any, mra: Any, mask: Any, rt: Mapping[str, Any], save_attrs: bool) -> dict[str, str]:
    (np, Image) = (rt['np'], rt['Image'])
    out.parent.mkdir(parents=True, exist_ok=True)
    albedo_u8 = (np.clip(albedo, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    mask_u8 = (np.asarray(mask, dtype=np.float32) >= 0.5).astype(np.uint8) * 255
    Image.fromarray(albedo_u8, mode='RGB').save(out)
    mask_path = out.with_name(out.stem + '_mask.png')
    Image.fromarray(mask_u8, mode='L').save(mask_path)
    result = {'png': str(out), 'mask_png': str(mask_path)}
    if save_attrs:
        attrs_path = out.with_suffix('.npz')
        np.savez_compressed(attrs_path, albedo=np.asarray(albedo, dtype=np.float32), mra=np.asarray(mra, dtype=np.float32), mask=np.asarray(mask, dtype=np.uint8))
        result['attrs_npz'] = str(attrs_path)
    return result

def render_glb(glb: str | Path, camera: Mapping[str, Any], out: str | Path, resolution: int=DEFAULT_RESOLUTION, *, device: str='cuda', rt: Mapping[str, Any] | None=None, save_attrs: bool=True) -> dict[str, str]:
    """Render an exported GLB through the supplied native-frame camera.

    Sample base color and MRA directly with nvdiffrast, without lighting.
    """
    _check_resolution(resolution)
    runtime = _runtime_imports() if rt is None else dict(rt)
    distance = _camera_distance(camera)
    (extrinsic, intrinsic) = _exported_glb_camera(camera, device, runtime)
    values = runtime['_load_textured_mesh'](Path(glb), runtime)
    (vertices, faces, _normals, uv, texture, mr_texture, factor, metallic_factor, roughness_factor) = values
    (torch, dr) = (runtime['torch'], runtime['dr'])
    projection = runtime['intrinsics_to_projection'](intrinsic, max(0.2, distance - 1.0), distance + 1.0)
    vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device).contiguous()
    uv_t = torch.as_tensor(uv, dtype=torch.float32, device=device)
    uv_t = torch.stack((uv_t[:, 0], 1.0 - uv_t[:, 1]), dim=-1).contiguous()
    vertices_h = torch.cat((vertices_t, torch.ones_like(vertices_t[:, :1])), dim=-1)
    clip = vertices_h @ (projection @ extrinsic).T
    glctx = dr.RasterizeCudaContext(device=device)
    (raster, _) = dr.rasterize(glctx, clip[None].contiguous(), faces_t, (resolution, resolution))
    mask = raster[..., -1] > 0
    (uv_pixel, _) = dr.interpolate(uv_t[None], raster, faces_t)
    base = dr.texture(torch.as_tensor(texture, dtype=torch.float32, device=device)[None].contiguous(), uv_pixel, filter_mode='linear')
    base = _srgb_factor_torch(base, factor, torch)
    mra = dr.texture(torch.as_tensor(mr_texture, dtype=torch.float32, device=device)[None].contiguous(), uv_pixel, filter_mode='linear')
    mra = torch.stack((mra[..., 2] * metallic_factor, mra[..., 1] * roughness_factor, torch.ones_like(mra[..., 0])), dim=-1).clamp(0.0, 1.0)
    base = torch.where(mask[..., None], base, torch.zeros_like(base))
    mra = torch.where(mask[..., None], mra, torch.zeros_like(mra))
    return _write_outputs(Path(out), base[0].detach().cpu().numpy(), mra[0].detach().cpu().numpy(), mask[0].detach().cpu().numpy(), runtime, save_attrs)

def _as_rgb(value: Any, name: str):
    import numpy as np
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3 or (not np.isfinite(array).all()) or (array.min() < 0) or (array.max() > 1):
        raise ValueError(f'{name} must be finite HxWx3 in [0,1]')
    return array

def _masked_metrics(prediction: Any, target: Any, mask: Any) -> dict[str, float | int | None]:
    import numpy as np
    (pred, truth, valid) = (_as_rgb(prediction, 'prediction'), _as_rgb(target, 'target'), np.asarray(mask, dtype=bool))
    if pred.shape != truth.shape or valid.shape != pred.shape[:2]:
        raise ValueError('prediction, target, and mask shapes must agree')
    count = int(valid.sum())
    if count == 0:
        return {'valid_pixels': 0, 'masked_mae': None, 'psnr_db': None, 'gradient_vector_mae': None}
    delta = pred[valid] - truth[valid]
    mse = float(np.mean(delta * delta))
    psnr = 100.0 if mse <= 1e-12 else min(100.0, -10.0 * math.log10(mse))

    def sobel(image: Any):
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
        dx = padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:] - padded[:-2, :-2] - 2 * padded[1:-1, :-2] - padded[2:, :-2]
        dy = padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:] - padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        return np.stack((dx, dy), axis=-1)
    grad_delta = sobel(pred)[valid] - sobel(truth)[valid]
    return {'valid_pixels': count, 'masked_mae': float(np.abs(delta).mean()), 'psnr_db': float(psnr), 'gradient_vector_mae': float(np.abs(grad_delta).mean())}

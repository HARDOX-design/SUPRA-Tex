"""Invocation-local acceleration for native Pixal3D geometry conditioning.

The input images and conditioner weights must remain unchanged inside the
context. Only the repeatedly used sparse-structure conditioner caches image
features. Shape stages sample occupied queries directly and retain no maps.
No native class, module global, installed package, or random state is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from types import MethodType

import torch


@contextmanager
def _instance_method(instance, name, function):
    """Restore an instance's exact method override (including its absence)."""
    missing = object()
    previous = vars(instance).get(name, missing)
    setattr(instance, name, MethodType(function, instance))
    try:
        yield
    finally:
        if previous is missing:
            delattr(instance, name)
        else:
            setattr(instance, name, previous)


def _function_globals(method):
    return inspect.unwrap(getattr(method, "__func__", method)).__globals__


def _image_identity(image):
    # Keep the original image object(s) alive in the cache as well: identities
    # must not be recycled after a temporary list/tensor has gone away.
    if isinstance(image, torch.Tensor):
        try:
            version = image._version
        except RuntimeError:  # Tensors created under inference_mode lack this.
            version = None
        return (id(image), version, image.shape, image.dtype, image.device)
    return tuple(id(item) for item in image)


class NativeGeometryAcceleration:
    """Use with exactly one native geometry inference, including scale probes.

    ``cache_image_features`` reuses sparse-structure DINO feature maps across
    scale probes and final generation; these maps are released before shape
    conditioning. ``sparse_shape_sampling`` replaces dense shape projection
    with occupied-query sampling, preserving coordinate order and duplicates.
    The pipeline remains the owner of model placement and sampling RNG.
    """

    def __init__(self, pipe, *, cache_image_features=True,
                 sparse_shape_sampling=True, sparse_tensor_factory=None):
        self.pipe = pipe
        self.cache_image_features = cache_image_features
        self.sparse_shape_sampling = sparse_shape_sampling
        self.sparse_tensor_factory = sparse_tensor_factory
        self.stats = {"ss_feature_extractions": 0, "ss_feature_cache_hits": 0,
                      "shape_conditioning_calls": 0, "projection_coordinate_calls": 0,
                      "sampled_feature_queries": 0, "dense_feature_queries_avoided": 0}
        self._cache = None
        self._contexts = []
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("geometry acceleration context is already active")
        self._active = True
        try:
            if self.cache_image_features:
                model = self.pipe.image_cond_model_ss
                original = model.forward

                def forward(model, image, camera_angle_x=None, distance=None,
                            mesh_scale=None, transform_matrix=None):
                    return self._condition(model, original, image, camera_angle_x,
                                           distance, mesh_scale, transform_matrix,
                                           cache=True)

                self._install(model, "forward", forward)
            if self.cache_image_features or self.sparse_shape_sampling:
                original_shape = self.pipe.get_proj_cond_shape
                if self.sparse_shape_sampling and self.sparse_tensor_factory is None:
                    self.sparse_tensor_factory = _function_globals(original_shape)["SparseTensor"]

                def shape(pipe, image_cond_model, image, coords, *args, **kwargs):
                    # SS is finished; release its cached maps before shape VRAM
                    # is needed. Shape NAF maps never survive conditioning.
                    self._cache = None
                    if not self.sparse_shape_sampling:
                        return original_shape(image_cond_model, image, coords, *args, **kwargs)
                    return self._shape(image_cond_model, image, coords, *args, **kwargs)

                self._install(self.pipe, "get_proj_cond_shape", shape)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _install(self, instance, name, function):
        context = _instance_method(instance, name, function)
        context.__enter__()
        self._contexts.append(context)

    def __exit__(self, exc_type, exc_value, traceback):
        self._cache = None
        while self._contexts:
            self._contexts.pop().__exit__(exc_type, exc_value, traceback)
        self._active = False

    @torch.no_grad()
    def _condition(self, model, original, image, camera_angle_x, distance,
                   mesh_scale, transform_matrix=None, *, cache=False, coords=None):
        if getattr(model, "cls_only", False):
            raise ValueError("geometry acceleration requires projected conditioning")
        grid = model.proj_grid
        helpers = _function_globals(grid.forward)
        project = helpers["project_points_to_image_batch"]
        sample = helpers["sample_features"]
        image_points_norm = None
        captured = []

        def project_map(grid, features_map, camera_angle_x, distance, mesh_scale,
                        transform_matrix=None, BHWC=True):
            nonlocal image_points_norm
            if image_points_norm is None:
                batch = features_map.shape[0]
                # Keep all original low-dimensional projection arithmetic and
                # its matrix shapes. Selecting 3D points earlier can change
                # GEMM rounding; selecting 2D queries here avoids that change.
                points = grid.grid_points.expand(batch, -1, -1)
                points = points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
                assert transform_matrix is None, "transform_matrix is not None"
                transform = grid.front_view_transform_matrix.expand(batch, -1, -1).clone()
                transform[:, 1, 3] = -distance
                image_points, _, _ = project(points, transform, camera_angle_x, grid.image_resolution)
                image_points_norm = (image_points + 0.5) / grid.image_resolution * 2 - 1
                self.stats["projection_coordinate_calls"] += 1
                if coords is not None:
                    if batch != 1:
                        raise ValueError("native sparse shape conditioning requires one image")
                    resolution = grid.grid_resolution
                    indices = coords[:, 1:].long()
                    if bool((coords[:, 0] != 0).any() or (indices < 0).any()
                            or (indices >= resolution).any()):
                        raise ValueError("occupied coordinates are outside the native projection grid")
                    flat = (indices[:, 0] * resolution + indices[:, 1]) * resolution + indices[:, 2]
                    image_points_norm = image_points_norm[:, flat]
            if BHWC:
                features_map = features_map.permute(0, 3, 1, 2)
            queries = image_points_norm.shape[1]
            self.stats["sampled_feature_queries"] += queries * features_map.shape[0]
            if coords is not None:
                self.stats["dense_feature_queries_avoided"] += grid.grid_resolution ** 3 - queries
            return sample(features_map, image_points_norm).permute(0, 2, 1)

        identity = _image_identity(image) if cache else None
        cached = self._cache
        if cache and cached is not None and cached["model"] is model and cached["identity"] == identity:
            self.stats["ss_feature_cache_hits"] += 1
            projected = [project_map(grid, fmap, camera_angle_x, distance, mesh_scale,
                                     transform_matrix, BHWC=bhwc)
                         for fmap, bhwc in cached["maps"]]
            return cached["global"], torch.cat(projected, dim=-1) if len(projected) > 1 else projected[0]

        def capture(grid, features_map, camera_angle_x, distance, mesh_scale,
                    transform_matrix=None, BHWC=True):
            if cache:
                captured.append((features_map, BHWC))
            return project_map(grid, features_map, camera_angle_x, distance, mesh_scale,
                               transform_matrix, BHWC=BHWC)

        with _instance_method(grid, "forward", capture):
            result = original(image, camera_angle_x=camera_angle_x, distance=distance,
                              mesh_scale=mesh_scale, transform_matrix=transform_matrix)
        if cache:
            if not captured:
                raise RuntimeError("native conditioner did not expose projection feature maps")
            self.stats["ss_feature_extractions"] += 1
            self._cache = {"model": model, "identity": identity, "image": image,
                           "global": result[0], "maps": captured}
        return result

    @torch.no_grad()
    def _shape(self, model, image, coords, camera_angle_x=0.8575560450553894,
               distance=2.0, mesh_scale=1.0, grid_resolution_override=None):
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.dtype not in (torch.int32, torch.int64):
            raise ValueError("occupied coordinates must be an integer [N,4] tensor")
        original_resolution, original_grid = model.grid_resolution, model.proj_grid
        device, low_vram = self.pipe.device, self.pipe.low_vram
        try:
            if low_vram:
                model.to(device)
            if grid_resolution_override is not None and grid_resolution_override != original_resolution:
                model.grid_resolution = grid_resolution_override
                model.proj_grid = type(original_grid)(
                    grid_resolution=grid_resolution_override,
                    image_resolution=original_grid.image_resolution).to(device)
            z_global, z_proj = self._condition(
                model, model.forward, image, torch.tensor([camera_angle_x], device=device),
                torch.tensor([distance], device=device), torch.tensor([mesh_scale], device=device),
                coords=coords)
            features = z_proj[0]
            sparse = self.sparse_tensor_factory
            result = {"cond": {"global": z_global, "proj": sparse(feats=features, coords=coords)},
                      "neg_cond": {"global": torch.zeros_like(z_global),
                                   "proj": sparse(feats=torch.zeros_like(features), coords=coords)}}
            self.stats["shape_conditioning_calls"] += 1
            return result
        finally:
            model.grid_resolution, model.proj_grid = original_resolution, original_grid
            if low_vram:
                model.cpu()

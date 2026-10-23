"""Isolated adapter for native and paper Pixal3D geometry generators.

The prepared demo camera owns FOV and distance. This adapter calibrates scale
using the main coarse geometry model, then publishes the final mesh/camera in
the SUPRA-Tex canonical frame.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import sys


def calibrate_scale(probe, initial_scale: float = 1.0, *, target_padding: int = 3,
                    max_iterations: int = 2) -> tuple[float, dict]:
    """Keep the closest safe occupancy, with a bounded number of scale updates.

    ``probe(scale)`` returns single-sample [N,4] decoded grid indices. Invalid
    occupancy ends the search; inference/runtime errors propagate to the caller.
    """
    import torch

    # geometry.run validates these options before launching this runner.
    max_iterations = min(max_iterations, 2)
    lower, upper = max(1, target_padding - 1), target_padding + 1
    scale = best = float(initial_scale)
    best_padding, best_error = None, math.inf
    report = {"algorithm": "bounded_main_dense_padding", "initial_mesh_scale": scale,
              "target_padding": target_padding, "padding_tolerance": [lower, upper],
              "scale_bounds": [scale / 2, scale * 2], "max_iterations": max_iterations, "probes": []}
    reason = "max_iterations"
    for iteration in range(max_iterations + 1):
        coords = probe(scale)
        valid = (isinstance(coords, torch.Tensor) and coords.ndim == 2 and coords.shape[1] == 4
                 and coords.shape[0] > 0 and not coords.is_complex())
        if valid:
            valid = bool(torch.isfinite(coords).all() and (coords[:, 0] == 0).all()
                         and (coords[:, 1:] >= 0).all() and (coords[:, 1:] <= 63).all()
                         and (coords == coords.float().round()).all())
        state = {"mesh_scale": scale, "valid": valid}
        report["probes"].append(state)
        if not valid:
            reason = "invalid_occupancy"
            break
        xyz = coords[:, 1:].float()
        padding = min(float(xyz.min()), 63 - float(xyz.max()))
        offset = float((xyz - 31.5).abs().max())
        safe, error = padding >= lower, abs(padding - target_padding)
        state.update(padding=padding, max_offset=offset, occupied_voxels=len(coords), safe=safe)
        if safe and error < best_error:
            best, best_padding, best_error = scale, padding, error
        if lower <= padding <= upper:
            reason = "target_reached"
            break
        if iteration == max_iterations:
            break
        proposed = scale * (31.5 - target_padding) / offset
        if not math.isfinite(proposed) or not report["scale_bounds"][0] <= proposed <= report["scale_bounds"][1]:
            reason = "scale_out_of_bounds"
            break
        if math.isclose(proposed, scale, rel_tol=1e-6):
            reason = "scale_stalled"
            break
        scale = proposed
    report.update(selected_mesh_scale=best, selected_padding=best_padding,
                  converged=best_padding is not None and lower <= best_padding <= upper,
                  used_baseline=best == initial_scale, stop_reason=reason)
    return best, report


def traced_pipeline(base_class, camera: dict, receipt: dict, *, scale_search: bool = False):
    """Customize scale calibration while keeping the upstream generation path."""
    fov = float(camera["camera_angle_x"])
    if not math.isfinite(fov) or not 0 < fov < math.pi:
        raise ValueError("camera_angle_x must be finite and between zero and pi")
    distance = float(camera["distance"])
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("camera distance must be finite and positive")

    class GeometryPipeline(base_class):
        _optimizing_scale = False

        def estimate_fov(self, image_path):
            return fov

        def _optimize_mesh_scale(self, image_tensor, camera_angle_x_tensor, distance_tensor,
                                 initial_mesh_scale, dense_steps, dense_guidance_scale, dense_seed,
                                 dense_threshold, target_padding=3, max_iterations=2):
            import torch

            device = image_tensor.device
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            receipt["upstream_initial_generation_camera"] = {
                "camera_angle_x": float(camera_angle_x_tensor.item()),
                "distance": float(distance_tensor.item()), "mesh_scale": float(initial_mesh_scale)}
            # The upstream infer() re-derives distance using its 518px/padding
            # convention. Mutate its shared camera tensor before any probe so
            # final dense and both sparse stages retain the prepared demo camera.
            distance_tensor.fill_(distance)
            receipt["camera_distance_policy"] = "preserve_prepared_demo_camera"
            receipt["initial_generation_camera"] = {
                "camera_angle_x": float(camera_angle_x_tensor.item()),
                "distance": float(distance_tensor.item()), "mesh_scale": float(initial_mesh_scale)}

            def probe(scale):
                receipt["dense_probe_calls"] = receipt.get("dense_probe_calls", 0) + 1
                original_scheduler = self.dense_scheduler
                try:
                    # Probe state must not leak into another probe or final generation.
                    self.dense_scheduler = copy.deepcopy(original_scheduler)
                    with torch.random.fork_rng(devices=cuda_devices):
                        latents = self.infer_dense(
                            image_tensor, camera_angle_x_tensor, distance_tensor,
                            torch.tensor([scale], device=device, dtype=torch.float32),
                            dense_steps, dense_guidance_scale, dense_seed)
                        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                            return self.dense_vae.decode_mesh(latents, mc_threshold=dense_threshold, return_index=True)[0]
                finally:
                    self.dense_scheduler = original_scheduler

            if not scale_search:
                result, report = float(initial_mesh_scale), {"enabled": False, "probes": [], "stop_reason": "disabled"}
            else:
                self._optimizing_scale = True
                try:
                    result, report = calibrate_scale(probe, initial_mesh_scale, target_padding=target_padding,
                                                      max_iterations=max_iterations)
                finally:
                    self._optimizing_scale = False
            receipt["scale_search"] = report
            receipt["optimized_mesh_scale"] = float(result)
            print(f"[mesh_scale] selected={result:.6f}, stop={report['stop_reason']}, probes={len(report['probes'])}", flush=True)
            return result

        def infer_dense(self, image, camera_angle_x, distance, mesh_scale, *args, **kwargs):
            if not math.isclose(float(distance.item()), float(camera["distance"]), rel_tol=1e-5):
                raise RuntimeError("geometry generation did not retain the prepared demo camera distance")
            if not self._optimizing_scale:
                receipt["final_generation_camera"] = {
                    "camera_angle_x": float(camera_angle_x.item()),
                    "distance": float(distance.item()),
                    "mesh_scale": float(mesh_scale.item()),
                }
            return super().infer_dense(image, camera_angle_x, distance, mesh_scale, *args, **kwargs)

    return GeometryPipeline


def canonical_handoff(mesh, camera: dict, receipt: dict):
    """Undo upstream export rotation and scale while preserving projection."""
    import numpy as np
    import torch

    try:
        selected = receipt["final_generation_camera"]
        scale = float(selected["mesh_scale"])
        optimized = float(receipt["optimized_mesh_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("geometry source did not expose the two-stage scale/camera hooks") from error
    if not math.isfinite(scale) or scale <= 0 or not math.isclose(scale, optimized, rel_tol=1e-5):
        raise RuntimeError("final geometry camera does not match optimized mesh_scale")
    if not math.isfinite(selected["distance"]) or selected["distance"] <= 0:
        raise RuntimeError("final geometry distance must be finite and positive")
    if not math.isclose(selected["distance"], float(camera["distance"]), rel_tol=1e-5):
        raise RuntimeError("final geometry did not retain the prepared demo camera distance")
    if not math.isclose(selected["camera_angle_x"], float(camera["camera_angle_x"]), rel_tol=1e-5):
        raise RuntimeError("geometry generation did not retain the supplied MoGe-2 FOV")
    # Upstream exports R @ raw_xyz * (0.5 / scale), with R mapping
    # (x,y,z) -> (x,-z,y). Canonical points are raw_xyz * 0.5.
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    vertices = np.asarray(mesh.vertices, dtype=np.float32) @ rotation * scale
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("geometry generator returned invalid vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("geometry generator returned invalid faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("geometry generator returned out-of-range face indices")
    if float(np.abs(vertices).max()) > 0.5001:
        raise ValueError("converted geometry exceeds the canonical [-0.5, 0.5] volume")
    effective_distance = selected["distance"] * scale
    if effective_distance <= float(vertices[:, 2].max()):
        raise ValueError("optimized camera lies inside or behind the geometry")
    final_camera = dict(camera)
    final_camera.update(selected)
    final_camera["moge_initial_camera"] = {key: camera[key] for key in
                                           ("camera_angle_x", "distance", "mesh_scale") if key in camera}
    final_camera["geometry_scale_optimization"] = {
        "initial_mesh_scale": receipt["initial_mesh_scale"],
        "optimized_mesh_scale": scale,
        "effective_camera_distance": effective_distance,
        "objective": "dense_grid_boundary_padding",
    }
    geometry = {"verts": torch.from_numpy(vertices.copy()).float(),
                "faces": torch.from_numpy(faces.astype(np.int64, copy=True)).long()}
    return geometry, final_camera


def build_native_pipeline(model_root: Path):
    """Load only native geometry networks and their projection conditioners."""
    import torch
    local_naf = Path(torch.hub.get_dir()) / "valeoai_NAF_main"
    for artifact in (local_naf / "hubconf.py", local_naf.parent / "checkpoints/naf_release.pth"):
        if not artifact.is_file():
            raise FileNotFoundError(f"native geometry requires the local NAF cache: {artifact}")
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline, samplers
    from pixal3d.modules.projected_image_feature_extractor import DinoV3ProjFeatureExtractor
    from supratex.frontend.geometry_loading import load_native_geometry_model

    class GeometryOnlyPipeline(Pixal3DImageTo3DPipeline):
        model_names_to_load = ["sparse_structure_flow_model", "sparse_structure_decoder",
                               "shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder"]

    # Conditions already include foreground removal. Base loading avoids creating
    # another background-removal model or any texture networks/samplers.
    args = json.loads((model_root / "pipeline.json").read_text(encoding="utf-8"))["args"]
    networks = {name: load_native_geometry_model(model_root / checkpoint)
                for name, checkpoint in args["models"].items()
                if name in GeometryOnlyPipeline.model_names_to_load}
    missing = set(GeometryOnlyPipeline.model_names_to_load) - networks.keys()
    if missing:
        raise ValueError(f"Native geometry config is missing models: {sorted(missing)}")
    pipe = GeometryOnlyPipeline(networks)
    pipe._pretrained_args = args
    for name in ("sparse_structure", "shape_slat"):
        spec = args[f"{name}_sampler"]
        setattr(pipe, f"{name}_sampler", getattr(samplers, spec["name"])(**spec["args"]))
        setattr(pipe, f"{name}_sampler_params", spec["params"])
    pipe.shape_slat_normalization = args["shape_slat_normalization"]
    original_load = torch.hub.load

    def local_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "valeoai/NAF":
            kwargs.pop("trust_repo", None)
            kwargs["source"] = "local"
            repo_or_dir = str(local_naf)
        return original_load(repo_or_dir, model, *args, **kwargs)

    try:
        torch.hub.load = local_hub_load
        for name, size, grid, use_naf in (("ss", 512, 16, False),
                                         ("shape_512", 512, 32, True),
                                         ("shape_1024", 1024, 64, True)):
            conditioner = DinoV3ProjFeatureExtractor(
                model_name=str(model_root / "dino"), image_size=size, grid_resolution=grid,
                use_naf_upsample=use_naf, naf_target_size=512).eval()
            if use_naf:
                conditioner._load_naf()
            setattr(pipe, f"image_cond_model_{name}", conditioner)
    finally:
        torch.hub.load = original_load
    pipe._device = torch.device("cuda")
    pipe.low_vram = True
    return pipe


def infer_native(pipe, image, camera: dict, receipt: dict, *, seed: int,
                 initial_mesh_scale: float, scale_search: bool, target_padding: int, max_iterations: int):
    """Optionally calibrate native 64-cell occupancy, then run the Pixal3D shape cascade."""
    import torch

    device = torch.device(pipe.device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    sampler_ss = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": .7, "rescale_t": 5.}
    sampler_shape = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": .5, "rescale_t": 3.}
    condition = {key: float(camera[key]) for key in ("camera_angle_x", "distance")}
    receipt["initial_generation_camera"] = {**condition, "mesh_scale": initial_mesh_scale}
    receipt["camera_distance_policy"] = "preserve_prepared_demo_camera"

    def clear_cache():
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def probe(scale):
        original_sampler = pipe.sparse_structure_sampler
        receipt["dense_probe_calls"] += 1
        try:
            pipe.sparse_structure_sampler = copy.deepcopy(original_sampler)
            with torch.random.fork_rng(devices=cuda_devices):
                torch.default_generator.manual_seed(seed)
                if cuda_devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                cond = pipe.get_proj_cond_ss([image], **condition, mesh_scale=scale)
                coords = pipe.sample_sparse_structure(cond, resolution=64, sampler_params=sampler_ss)
                del cond
                clear_cache()
                return coords
        finally:
            pipe.sparse_structure_sampler = original_sampler

    if scale_search:
        selected, report = calibrate_scale(probe, initial_mesh_scale, target_padding=target_padding,
                                           max_iterations=max_iterations)
    else:
        # Default: keep Pixal3D's fixed mesh scale without dense probes.
        selected, report = initial_mesh_scale, {"enabled": False, "probes": [], "stop_reason": "disabled"}
    condition["mesh_scale"] = selected
    receipt.update(scale_search=report, optimized_mesh_scale=selected, final_generation_camera=dict(condition),
                   uses_dense_check_model=False, final_sparse_structure_resolution=32)
    print(f"[mesh_scale] selected={selected:.6f}, stop={report['stop_reason']}, probes={len(report['probes'])}", flush=True)
    torch.manual_seed(seed)
    cond = pipe.get_proj_cond_ss([image], **condition)
    coords = pipe.sample_sparse_structure(cond, resolution=32, sampler_params=sampler_ss)
    del cond
    clear_cache()
    cond = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512, [image], coords, **condition)
    slat = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_512"], coords, sampler_shape)
    del cond, coords
    clear_cache()
    decoder = pipe.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    try:
        hr_coords = decoder.upsample(slat, upsample_times=4)
    finally:
        decoder.cpu()
        decoder.low_vram = False
    # Match Pixal3D's 1536 cascade, including its quantization and token fallback.
    actual_hr_resolution = 1536
    max_num_tokens = 49152
    while True:
        grid_res = actual_hr_resolution // 16
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + .5) / 512 * (grid_res - 1)).round().int(),
        ], dim=1)
        coords = quant_coords.unique(dim=0)
        num_tokens = coords.shape[0]
        if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
            break
        actual_hr_resolution -= 128
    actual_grid_res = actual_hr_resolution // 16
    receipt.update(requested_resolution=1536, resolution=actual_hr_resolution,
                   max_num_tokens=max_num_tokens, num_tokens=int(num_tokens))
    print(f"[geometry] requested=1536, resolution={actual_hr_resolution}, tokens={num_tokens}, "
          f"max_num_tokens={max_num_tokens}", flush=True)
    del slat, hr_coords, quant_coords
    clear_cache()
    cond = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_1024, [image], coords,
                                   grid_resolution_override=actual_grid_res, **condition)
    slat = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_1024"], coords, sampler_shape)
    del cond, coords
    clear_cache()
    meshes, _ = pipe.decode_shape_slat(slat, actual_hr_resolution)
    if len(meshes) != 1 or not meshes[0].vertices.numel() or not meshes[0].faces.numel():
        raise RuntimeError("geometry decoder returned an empty mesh")
    meshes[0].fill_holes()
    mesh = meshes[0]
    vertices, faces = mesh.vertices.detach().float().cpu(), mesh.faces.detach().long().cpu()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not torch.isfinite(vertices).all()
            or float(vertices.abs().max()) > .5001 or faces.ndim != 2 or faces.shape[1] != 3
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("native geometry decoder returned invalid canonical mesh")
    if condition["distance"] * selected <= float(vertices[:, 2].max()):
        raise ValueError("optimized camera lies inside or behind the geometry")
    final_camera = {**camera, **condition,
                    "moge_initial_camera": {key: camera[key] for key in ("camera_angle_x", "distance", "mesh_scale") if key in camera},
                    "geometry_scale_optimization": {"initial_mesh_scale": initial_mesh_scale,
                                                    "optimized_mesh_scale": selected,
                                                    "effective_camera_distance": condition["distance"] * selected,
                                                    "objective": "dense_grid_boundary_padding"}}
    return {"verts": vertices, "faces": faces, "resolution": actual_hr_resolution}, final_camera


def run(package: Path, model_root: Path, input_dir: Path, output_dir: Path, *, seed: int = 42,
        initial_mesh_scale: float = 1.0, scale_search: bool = False, target_padding: int = 3,
        max_iterations: int = 2):
    import torch

    for name in ("geometry.pt", "camera.json", "geometry_receipt.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"geometry output already exists: {output_dir / name}")
    if not torch.cuda.is_available():
        raise RuntimeError("geometry generation requires CUDA")
    from supratex.inference.runtime import configure_inference_runtime
    runtime = configure_inference_runtime()
    camera = json.loads((input_dir / "camera.json").read_text(encoding="utf-8"))
    if any(key in camera for key in ("extrinsic", "intrinsic", "projection_intrinsic")):
        raise ValueError("geometry generation requires a canonical scalar camera")
    sys.path.insert(0, str(package.resolve()))
    native = (package / "pixal3d/pipelines/pixal3d_image_to_3d.py").is_file()
    kernel_config = None
    if native:
        from supratex.inference.kernel_config import configure_sparse_kernels
        kernel_config = configure_sparse_kernels()
    module = importlib.import_module("pixal3d.pipelines.pixal3d_image_to_3d" if native else "pixal3dpipeline2stage")
    if not Path(module.__file__).resolve().is_relative_to(package.resolve()):
        raise RuntimeError("geometry pipeline was imported outside the selected source package")
    receipt = {"schema": "supratex.geometry.v2", "generator": "Pixal3D paper two-stage",
               "runtime": runtime,
               "source_package": str(package.resolve()), "model_root": str(model_root.resolve()),
               "camera_estimator": camera.get("camera_estimator", "supplied camera"),
               "initial_mesh_scale": initial_mesh_scale, "scale_search": scale_search,
               "target_padding": target_padding,
               "max_iterations": max_iterations, "seed": seed, "dense_probe_calls": 0,
               "objective": "dense_grid_boundary_padding", "geometry_frame": "canonical [-0.5,0.5]",
               "scope": "paper geometry generation with prepared demo camera; optional bounded scale calibration"}
    if native:
        from PIL import Image
        from supratex.frontend.geometry_acceleration import NativeGeometryAcceleration
        receipt.update(generator="Pixal3D native geometry cascade",
                       sparse_kernel_config=kernel_config,
                       scope="native geometry generation with prepared demo camera; optional bounded scale calibration")
        pipeline = build_native_pipeline(model_root)
        with Image.open(input_dir / "condition_rgb.png") as image, torch.inference_mode(), NativeGeometryAcceleration(pipeline) as acceleration:
            geometry, final_camera = infer_native(pipeline, image.convert("RGB"), camera, receipt, seed=seed,
                                                  initial_mesh_scale=initial_mesh_scale, scale_search=scale_search,
                                                  target_padding=target_padding, max_iterations=max_iterations)
        receipt["conditioning_acceleration"] = dict(acceleration.stats)
    else:
        pipeline_class = traced_pipeline(module.Pixal3DPipeline2Stage, camera, receipt, scale_search=scale_search)
        pipeline = pipeline_class.from_pretrained(ckpt_dir=str(model_root), repo_id=None,
                                                  use_moge=False, use_dense_check=False)
        receipt["uses_dense_check_model"] = bool(pipeline.has_dense_check)
        with torch.inference_mode():
            mesh = pipeline.infer(image=str(input_dir / "condition_rgb.png"), dense_seed=seed,
                                  sparse_seed=seed, mesh_scale=initial_mesh_scale,
                                  optimize_mesh_scale=True, target_padding=target_padding,
                                  max_optim_iterations=max_iterations)
        geometry, final_camera = canonical_handoff(mesh, camera, receipt)
    receipt["vertices"] = int(geometry["verts"].shape[0])
    receipt["faces"] = int(geometry["faces"].shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(geometry, output_dir / "geometry.pt")
    (output_dir / "camera.json").write_text(json.dumps(final_camera, indent=2) + "\n", encoding="utf-8")
    # Publish the receipt last, after the complete final mesh/camera handoff.
    (output_dir / "geometry_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "model-root", "input-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-mesh-scale", type=float, default=1.0)
    parser.add_argument("--scale-search", action="store_true")
    parser.add_argument("--target-padding", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=2)
    args = parser.parse_args()
    run(args.package, args.model_root, args.input_dir, args.output_dir, seed=args.seed,
        initial_mesh_scale=args.initial_mesh_scale, scale_search=args.scale_search,
        target_padding=args.target_padding, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()

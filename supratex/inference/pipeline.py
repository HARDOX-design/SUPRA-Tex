"""Observation-guided PBR completion and GLB export.

The public boundary begins with a mesh/camera/RGBA observation plus albedo and
MRA maps.  Geometry prediction and 2D map prediction intentionally remain
outside this inference-only package.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from . import native_observation
from .checkpoint import load_completion_state_dict, validate_completion_checkpoint

RESOLUTION = native_observation.RESOLUTION
LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
# Released completion-model ABI. It is checked once, when the config is loaded.
MODEL_ABI = {"in_channels": 64, "out_channels": 32, "cond_channels": 0, "model_channels": 1536,
             "num_heads": 12, "num_blocks": 60, "base_block_policy": "alternating_u_x",
             "v11_init_policy": "odd_copy_even_identity", "direct_modulation_broadcast": True,
             "rope_freq": [1, 10000], "dtype": "float32"}


def _file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"missing {label}: {value}")
    return value


def _normalization(value: Any, label: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} normalization must be an object")
    result: dict[str, list[float]] = {}
    for key in ("mean", "std"):
        raw = value.get(key)
        if not isinstance(raw, list) or len(raw) != 32:
            raise ValueError(f"{label} normalization requires 32-value {key}")
        result[key] = [float(item) for item in raw]
    if not all(math.isfinite(item) for item in result["mean"] + result["std"]) or any(item <= 0 for item in result["std"]):
        raise ValueError(f"{label} normalization standard deviations must be positive")
    return result


def load_contract(model_config: str | Path, checkpoint: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load the public model ABI without inspecting or hashing weight bytes."""
    config_path, checkpoint_path = _file(model_config, "model config"), _file(checkpoint, "checkpoint")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dataset = config["dataset"]["args"]
        model = config["models"]["denoiser"]
        args = model["args"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("model config does not satisfy the Supra-Tex inference ABI") from error
    if config.get("format") == "supratex_completion" and config.get("schema_version") != 1:
        raise ValueError("unsupported Supra-Tex completion config schema_version")
    if model.get("name") != "InterleavedFlowTransformer":
        raise ValueError("model config must select InterleavedFlowTransformer")
    if int(dataset.get("resolution", -1)) != RESOLUTION or dataset.get("attrs") != ["base_color", "metallic", "roughness", "alpha"]:
        raise ValueError("model config must describe 1024 base-color/metallic/roughness/alpha inputs")
    mismatched = {key: args.get(key) for key, expected in MODEL_ABI.items() if args.get(key) != expected}
    if mismatched:
        raise ValueError(f"model config does not match the released completion ABI: {mismatched}")
    if dataset.get("bridge_z_obs_args", {}).get("partial_support_mode") != "partial_messages":
        raise ValueError("model config requires partial observation messages")
    _normalization(dataset.get("shape_slat_normalization"), "shape")
    material = _normalization(dataset.get("pbr_slat_normalization"), "material")
    for key in ("pbr_mean", "pbr_std"):
        expected = material["mean" if key.endswith("mean") else "std"]
        if [float(value) for value in args.get(key, [])] != expected:
            raise ValueError(f"model {key} must exactly match material normalization")
    return config, dataset, checkpoint_path


def _stem(path: str | Path, label: str) -> Path:
    stem = Path(path)
    _file(Path(f"{stem}.json"), f"{label} config")
    _file(Path(f"{stem}.safetensors"), f"{label} weights")
    return stem


def load_completion_contract(model_config: str | Path, checkpoint: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, list[float]]]:
    """Validate the model ABI and observation statistics without loading weights."""
    config, dataset, checkpoint_path = load_contract(model_config, checkpoint)
    validate_completion_checkpoint(checkpoint_path)
    path = Path(dataset["bridge_z_obs_args"]["partial_z_obs_normalization_path"])
    if not path.is_absolute():
        path = Path(model_config).expanduser().absolute().parent / path
    partial_stats = _normalization(json.loads(_file(path, "partial normalization").read_text(encoding="utf-8")), "partial observation")
    model_args = config["models"]["denoiser"]["args"]
    for key in ("partial_mean", "partial_std"):
        expected = partial_stats["mean" if key.endswith("mean") else "std"]
        if [float(value) for value in model_args.get(key, [])] != expected:
            raise ValueError(f"model {key} must exactly match partial-observation normalization")
    return config, dataset, checkpoint_path, partial_stats


def _completion_vae_root(trellis_root: str | Path) -> Path:
    root = Path(trellis_root).expanduser()
    return root / "ckpts" if (root / "ckpts").is_dir() else root


def completion_decoder_stems(config: Mapping[str, Any], dataset: Mapping[str, Any],
                             model_config: str | Path, trellis_root: str | Path) -> dict[str, Path]:
    """Resolve release VAE dependencies or preserve legacy config-relative paths."""
    base = Path(model_config).expanduser().absolute().parent
    standalone = config.get("format") == "supratex_completion"
    if standalone:
        base = _completion_vae_root(trellis_root)
    stems = {}
    for key in ("pretrained_shape_slat_dec", "pretrained_pbr_slat_dec"):
        value = dataset.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"model config requires {key}")
        path = Path(value).expanduser()
        if standalone and (path.name != value or value in (".", "..") or "\\" in value):
            raise ValueError(f"standalone completion {key} must be a bare sparse VAE stem")
        stems[key] = path if path.is_absolute() else base / path
    return stems


def validate_completion_artifacts(model_config: str | Path, checkpoint: str | Path,
                                  trellis_root: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, list[float]]]:
    """Check completion metadata, safe-file headers, and all VAE files before GPU work."""
    contract = load_completion_contract(model_config, checkpoint)
    config, dataset, _, _ = contract
    ckpts = _completion_vae_root(trellis_root)
    stems = [ckpts / "shape_enc_next_dc_f16c32_fp16", ckpts / "tex_enc_next_dc_f16c32_fp16"]
    stems.extend(completion_decoder_stems(config, dataset, model_config, trellis_root).values())
    for stem in stems:
        _stem(stem, "completion artifact")
    return contract


def _load_vae(models: Any, stem: Path) -> Any:
    """Build a sparse VAE on the meta device so weights are not randomly initialized first.

    Checkpoint tensors are cast to the constructor's dtypes, exactly as the
    copy-based loader does. Partial checkpoints or constructor-made tensors that
    a meta shell cannot materialize fall back to models.from_pretrained.
    """
    import torch
    from safetensors.torch import load_file

    config = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = getattr(models, config["name"])(**config["args"])
    expected = model.state_dict()
    state = load_file(f"{stem}.safetensors", device="cpu")
    if state.keys() != expected.keys() or any(state[name].shape != target.shape for name, target in expected.items()):
        return models.from_pretrained(str(stem))
    model.load_state_dict({name: state[name].to(dtype=target.dtype) for name, target in expected.items()},
                          strict=True, assign=True)
    if any(isinstance(value, torch.Tensor) and value.is_meta
           for module in model.modules() for value in (*module._parameters.values(), *module._buffers.values(), *vars(module).values())):
        return models.from_pretrained(str(stem))
    return model


def _load_case(input_dir: str | Path, torch: Any, image: Any) -> dict[str, Any]:
    root = Path(input_dir)
    required = {name: _file(root / name, name) for name in ("geometry.pt", "camera.json", "condition_rgba.png", "albedo.png", "mra.png")}
    geometry = torch.load(required["geometry.pt"], map_location="cpu", weights_only=True)
    try:
        vertices, faces = geometry["verts"].float().contiguous(), geometry["faces"].long().contiguous()
    except (KeyError, AttributeError) as error:
        raise ValueError("geometry.pt must contain tensor keys verts and faces") from error
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("geometry.pt must contain verts[N,3] and faces[M,3]")
    rgba = image.open(required["condition_rgba.png"]).convert("RGBA")
    albedo, mra = image.open(required["albedo.png"]).convert("RGB"), image.open(required["mra.png"]).convert("RGB")
    if rgba.size[0] != rgba.size[1]:
        raise ValueError("condition_rgba.png must be square")
    if albedo.size != (RESOLUTION, RESOLUTION) or mra.size != (RESOLUTION, RESOLUTION):
        raise ValueError("albedo.png and mra.png must be 1024 by 1024 RGB maps")
    try:
        camera = json.loads(required["camera.json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("camera.json must be valid JSON") from error
    if "camera_angle_x" not in camera or "distance" not in camera:
        raise ValueError("camera.json must provide canonical camera_angle_x and distance")
    return {"vertices": vertices, "faces": faces, "rgba": rgba, "albedo": albedo, "mra": mra, "camera": camera}


def _runtime() -> dict[str, Any]:
    from .runtime import configure_inference_runtime
    configure_inference_runtime()
    from .kernel_config import configure_sparse_kernels
    kernel_config = configure_sparse_kernels()
    if kernel_config["enabled"]:
        print(f"Sparse kernels: {kernel_config['preset']}", flush=True)
    import numpy as np
    import o_voxel
    import torch
    import torch.nn.functional as F
    import utils3d
    from PIL import Image, ImageFilter
    from supratex import models
    from supratex.modules import sparse
    from supratex.representations import MeshWithVoxel
    from .mesh_export import to_glb
    return locals()


def _camera(spec: Mapping[str, Any], device: str, torch: Any, utils3d: Any) -> tuple[Any, Any]:
    if "extrinsic" in spec and ("projection_intrinsic" in spec or "intrinsic" in spec):
        intrinsic = spec.get("projection_intrinsic", spec.get("intrinsic"))
        external, projection = torch.as_tensor(spec["extrinsic"], dtype=torch.float32, device=device), torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
        if external.shape != (4, 4) or projection.shape != (3, 3): raise ValueError("camera matrices must be 4x4 and 3x3")
        return external, projection
    if "distance" not in spec or "camera_angle_x" not in spec: raise ValueError("camera must supply distance and camera_angle_x")
    eye = torch.tensor([0., 0., float(spec["distance"]) * float(spec.get("mesh_scale", 1.0))], device=device)
    return utils3d.torch.extrinsics_look_at(eye, torch.zeros(3, device=device), torch.tensor([0., 1., 0.], device=device)), utils3d.torch.intrinsics_from_fov_xy(torch.tensor(float(spec["camera_angle_x"]), device=device), torch.tensor(float(spec["camera_angle_x"]), device=device))


def _shape(vertices: Any, faces: Any, encoder: Any, runtime: Mapping[str, Any], device: str) -> tuple[Any, Any]:
    torch, o_voxel, sparse = runtime["torch"], runtime["o_voxel"], runtime["sparse"]
    indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(vertices=vertices.float().cpu().contiguous(), faces=faces.long().cpu().contiguous(), grid_size=RESOLUTION, aabb=[[-.5, -.5, -.5], [.5, .5, .5]], face_weight=1., boundary_weight=.2, regularization_weight=1e-2, timing=False)
    offsets = (dual_vertices.float() * RESOLUTION - indices.float()).clamp(0, 1)
    coords = torch.cat((torch.zeros_like(indices[:, :1]), indices), dim=1).int().to(device)
    vertices_sparse = sparse.SparseTensor(offsets.to(device), coords)
    with torch.no_grad(): latent = encoder(vertices_sparse, vertices_sparse.replace(intersected.bool().to(device)))
    return latent, sparse.SparseTensor(torch.zeros(coords.shape[0], 6, device=device), coords)


def _partial(surface: Any, shape: Any, case: Mapping[str, Any], external: Any, projection: Any, encoder: Any, normalization: Mapping[str, Any], runtime: Mapping[str, Any], device: str) -> Any:
    torch, F, np, image, image_filter, sparse = runtime["torch"], runtime["F"], runtime["np"], runtime["Image"], runtime["ImageFilter"], runtime["sparse"]
    coords = surface.coords[:, 1:].long(); points = (coords.float() + .5) / RESOLUTION - .5
    camera = (torch.cat((points, torch.ones_like(points[:, :1])), dim=1) @ external.T)[:, :3]
    uvw = camera @ projection.T; uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-8)
    grid = torch.stack((uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1), dim=1).view(1, 1, -1, 2)
    def sampled(source: Any) -> Any:
        data = np.asarray(source.convert("RGB").resize((RESOLUTION, RESOLUTION), image.Resampling.LANCZOS), dtype=np.float32).copy() / 255.
        return F.grid_sample(torch.from_numpy(data).permute(2, 0, 1).to(device)[None], grid, mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T.clamp(0, 1)
    alpha = np.asarray(case["rgba"].getchannel("A"), dtype=np.uint8)
    mask = image.fromarray(np.where(alpha >= 254, 255, 0).astype(np.uint8)).resize((RESOLUTION, RESOLUTION), image.Resampling.NEAREST).filter(image_filter.MinFilter(5))
    trusted = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy() == 255)
    bits, ordered, order = native_observation.build_occupancy_index(coords)
    owners = native_observation.pixel_first_occupied(dict(case["camera"]), bits, trusted)
    visible = native_observation.owner_rows(coords, owners, ordered, order)
    if not bool(visible.any()): raise RuntimeError("the trusted alpha gate leaves no observed material surface")
    with torch.no_grad():
        return native_observation.encode_partial(encoder, coords, visible, sampled(case["albedo"]), sampled(case["mra"]), shape.coords[:, 1:], dict(normalization), sparse, device)


def _export(output: Path, representation: Any, texture_size: int, decimation_target: int, texture_format: str, runtime: Mapping[str, Any], *, gutter: bool = False, low_vram: bool = False) -> None:
    if output.suffix.lower() != ".glb": raise ValueError("output must have a .glb suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.publish.lock")
    try:
        handle = lock.open("x")
    except FileExistsError as error:
        raise FileExistsError(f"another process is publishing {output}") from error
    handle.close()
    if output.exists():
        lock.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}.glb")
    try:
        scene = runtime["to_glb"](vertices=representation.vertices, faces=representation.faces, attr_volume=representation.attrs, coords=representation.coords, attr_layout=LAYOUT, grid_size=RESOLUTION, aabb=[[-.5, -.5, -.5], [.5, .5, .5]], decimation_target=decimation_target, texture_size=texture_size, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True, **({"low_vram": True} if low_vram else {}))
        if gutter:
            from .gutter import apply_gutter
            print(f"Gutter postprocess: {apply_gutter(scene)}", flush=True)
        scene.apply_transform(runtime["np"].asarray([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=runtime["np"].float64))
        scene.export(str(temporary), extension_webp=texture_format == "webp")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def _stage_sparse(value: Any, device: str, *, keep_topology: bool = False) -> Any:
    """Transfer sparse data without retaining backend caches on its old device.

    SparseTensor.to()/cpu() share spatial/backend caches. Encoder subdivision
    maps are semantic: the decoders use them to reproduce the input surface,
    overriding predicted subdivisions. Preserve those maps when requested;
    convolution and attention caches are safe to recompute on the new device.
    """
    def transfer(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: transfer(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(transfer(child) for child in item)
        if hasattr(item, "detach") and hasattr(item, "to"):
            return item.detach().to(device)
        return item
    staged = type(value)(value.feats.detach().to(device), value.coords.detach().to(device),
                         shape=value._shape, scale=value._scale)
    if keep_topology:
        staged._spatial_cache = {
            scale: {name: transfer(item) for name, item in cache.items()
                    if name.startswith(("channel2spatial_", "upsample_")) or name == "shape"}
            for scale, cache in value._spatial_cache.items()
        }
    return staged


@contextmanager
def _completion_stage(model: Any, torch: Any, device: str, *, layer_offload: bool = False):
    """Own one model's residency, including cleanup after failed inference."""
    previous = getattr(model, "low_vram", False)
    try:
        if layer_offload:
            model.low_vram = True
        else:
            model.to(device)
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        model.to("cpu")
        if layer_offload:
            model.low_vram = previous
        torch.cuda.empty_cache()


def _run_completion_low_vram(*, case: Mapping[str, Any], back_case: Mapping[str, Any] | None,
                             ckpts: Path, config: Mapping[str, Any], dataset: Mapping[str, Any],
                             checkpoint_path: Path, partial_stats: Mapping[str, Any], resolved: Any,
                             runtime: Mapping[str, Any], device: str, seed: int, output: str | Path,
                             texture_size: int, decimation_target: int, texture_format: str, gutter: bool) -> dict[str, Any]:
    """Serialize completion stages and keep only their CPU handoffs alive."""
    torch, models = runtime["torch"], runtime["models"]
    print("Completion low VRAM: sequential models, CPU handoffs, decoder layer offload", flush=True)
    shape_encoder = _load_vae(models, _stem(ckpts / "shape_enc_next_dc_f16c32_fp16", "shape encoder"))
    with _completion_stage(shape_encoder, torch, device):
        shape, surface = _shape(case["vertices"], case["faces"], shape_encoder, runtime, device)
        shape_cpu = _stage_sparse(shape, "cpu", keep_topology=True)
        shape = _stage_sparse(shape, device)
    del shape_encoder
    texture_encoder = _load_vae(models, _stem(ckpts / "tex_enc_next_dc_f16c32_fp16", "material encoder"))
    dual_stats = {}
    with _completion_stage(texture_encoder, torch, device):
        if back_case is None:
            external, projection = _camera(case["camera"], device, torch, runtime["utils3d"])
            observation = _partial(surface, shape, case, external, projection, texture_encoder, partial_stats, runtime, device)
            del external, projection
        else:
            from .dual_view import encode_two_views
            observation, dual_stats = encode_two_views(surface, shape, case, back_case, texture_encoder,
                                                       partial_stats, runtime, device)
    del texture_encoder, surface
    torch.cuda.empty_cache()

    from supratex.pipelines.observation_guided_completion import build_completion_model, sample_material_latents
    state = load_completion_state_dict(checkpoint_path, torch_module=torch)
    model = build_completion_model(config, state, assign_weights=True)
    del state
    with _completion_stage(model, torch, device):
        shape_condition = shape.replace(feats=(shape.feats - torch.as_tensor(dataset["shape_slat_normalization"]["mean"], device=device)[None]) / torch.as_tensor(dataset["shape_slat_normalization"]["std"], device=device)[None])
        noise = shape.replace(feats=torch.randn((shape.coords.shape[0], 32), generator=torch.Generator(device=device).manual_seed(int(seed)), device=device))
        print("Observation cache: disabled (low VRAM)", flush=True)
        sampled = sample_material_latents(model, noise, concat_cond=shape_condition, partial_z_obs=observation, steps=25, rescale_t=3., sigma_min=1e-5, use_observation_cache=False).samples
        if not torch.equal(sampled.coords, noise.coords) or sampled.feats.shape != noise.feats.shape:
            raise RuntimeError("material sampler changed sparse support or feature shape")
        generated = shape.replace(feats=sampled.feats.float() * torch.as_tensor(dataset["pbr_slat_normalization"]["std"], device=device)[None] + torch.as_tensor(dataset["pbr_slat_normalization"]["mean"], device=device)[None])
        generated_cpu = _stage_sparse(generated, "cpu")
    del model, shape, shape_condition, noise, sampled, generated, observation
    torch.cuda.empty_cache()

    # Reuse the encoder's exact support in both decoders. Do not carry GPU
    # convolution caches or the unused decoded mesh into material decoding.
    shape_decoder = _load_vae(models, _stem(resolved(dataset["pretrained_shape_slat_dec"]), "shape decoder"))
    shape_decoder.set_resolution(RESOLUTION)
    with _completion_stage(shape_decoder, torch, device, layer_offload=True):
        shape = _stage_sparse(shape_cpu, device, keep_topology=True)
        decoded_shape, guide = shape_decoder(shape, return_subs=True, return_mesh=False)
        guide_cpu = [_stage_sparse(value, "cpu") for value in guide]
    del shape_decoder, shape, decoded_shape, guide
    generated_cpu._spatial_cache = shape_cpu._spatial_cache
    del shape_cpu
    torch.cuda.empty_cache()

    material_decoder = _load_vae(models, _stem(resolved(dataset["pretrained_pbr_slat_dec"]), "material decoder"))
    with _completion_stage(material_decoder, torch, device, layer_offload=True):
        generated = _stage_sparse(generated_cpu, device, keep_topology=True)
        guide = [_stage_sparse(value, device) for value in guide_cpu]
        decoded = material_decoder(generated, guide_subs=guide) * .5 + .5
        if decoded.feats.shape[1] != 6: raise RuntimeError("material decoder must emit six PBR channels")
        decoded_cpu = _stage_sparse(decoded, "cpu")
    del material_decoder, generated, generated_cpu, guide, guide_cpu, decoded
    torch.cuda.empty_cache()
    representation = runtime["MeshWithVoxel"](case["vertices"], case["faces"], [-.5, -.5, -.5], 1. / RESOLUTION, decoded_cpu.coords[:, 1:].int(), decoded_cpu.feats.float().clamp(0, 1), torch.Size([1, 6, RESOLUTION, RESOLUTION, RESOLUTION]), layout=LAYOUT)
    del decoded_cpu
    _export(Path(output), representation, texture_size, decimation_target, texture_format, runtime, gutter=gutter, low_vram=True)
    return dual_stats


def _run_completion(*, input_dir: str | Path, output: str | Path, model_config: str | Path, checkpoint: str | Path, trellis_root: str | Path, seed: int = 42, device: str = "cuda", texture_size: int = 4096, decimation_target: int = 500000, texture_format: str = "png", gutter: bool = False, back_case: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Complete PBR texture from public input maps and write one new GLB."""
    if device != "cuda": raise ValueError("public material completion currently requires device='cuda'")
    if texture_size <= 0 or decimation_target <= 0 or texture_format not in {"png", "webp"}: raise ValueError("invalid GLB export settings")
    config, dataset, checkpoint_path, partial_stats = validate_completion_artifacts(model_config, checkpoint, trellis_root)
    decoder_stems = completion_decoder_stems(config, dataset, model_config, trellis_root)
    decoder_paths = {dataset[key]: path for key, path in decoder_stems.items()}
    def resolved(value: str) -> Path:
        return decoder_paths[value]
    runtime = _runtime(); torch, models = runtime["torch"], runtime["models"]
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    case = _load_case(input_dir, torch, runtime["Image"]); ckpts = _completion_vae_root(trellis_root)
    if os.environ.get("SUPRATEX_LOW_VRAM", "0") == "1":
        return _run_completion_low_vram(case=case, back_case=back_case, ckpts=ckpts, config=config,
                                        dataset=dataset, checkpoint_path=checkpoint_path, partial_stats=partial_stats,
                                        resolved=resolved, runtime=runtime, device=device, seed=seed, output=output,
                                        texture_size=texture_size, decimation_target=decimation_target,
                                        texture_format=texture_format, gutter=gutter)
    shape_encoder = _load_vae(models, _stem(ckpts / "shape_enc_next_dc_f16c32_fp16", "shape encoder")).to(device).eval()
    texture_encoder = _load_vae(models, _stem(ckpts / "tex_enc_next_dc_f16c32_fp16", "material encoder")).to(device).eval()
    external, projection = _camera(case["camera"], device, torch, runtime["utils3d"])
    shape, surface = _shape(case["vertices"], case["faces"], shape_encoder, runtime, device)
    shape_encoder.to("cpu"); torch.cuda.empty_cache()
    dual_stats = {}
    if back_case is None:
        observation = _partial(surface, shape, case, external, projection, texture_encoder, partial_stats, runtime, device)
    else:
        from .dual_view import encode_two_views
        observation, dual_stats = encode_two_views(surface, shape, case, back_case, texture_encoder,
                                                   partial_stats, runtime, device)
    texture_encoder.to("cpu"); torch.cuda.empty_cache()
    from supratex.pipelines.observation_guided_completion import build_completion_model, sample_material_latents
    state = load_completion_state_dict(checkpoint_path, torch_module=torch)
    model = build_completion_model(config, state, assign_weights=True).to(device).eval()
    del state
    shape_condition = shape.replace(feats=(shape.feats - torch.as_tensor(dataset["shape_slat_normalization"]["mean"], device=device)[None]) / torch.as_tensor(dataset["shape_slat_normalization"]["std"], device=device)[None])
    noise = shape.replace(feats=torch.randn((shape.coords.shape[0], 32), generator=torch.Generator(device=device).manual_seed(int(seed)), device=device))
    # Cache fixed observation work within this sample only. Retain ample room
    # for attention activations; low-memory devices keep the uncached path.
    cache_bytes = 2 * (len(model.blocks) + len(model.refiners)) * observation.feats.shape[0] * model.model_channels * 2
    free_bytes, _ = torch.cuda.mem_get_info(noise.feats.device)
    use_cache = free_bytes >= cache_bytes * 1.25 + 8 * 1024 ** 3
    print(f"Observation cache: {'enabled' if use_cache else 'disabled (memory headroom)'}", flush=True)
    sampled = sample_material_latents(model, noise, concat_cond=shape_condition, partial_z_obs=observation, steps=25, rescale_t=3., sigma_min=1e-5, use_observation_cache=use_cache).samples
    if not torch.equal(sampled.coords, noise.coords) or sampled.feats.shape != noise.feats.shape:
        raise RuntimeError("material sampler changed sparse support or feature shape")
    generated = shape.replace(feats=sampled.feats.float() * torch.as_tensor(dataset["pbr_slat_normalization"]["std"], device=device)[None] + torch.as_tensor(dataset["pbr_slat_normalization"]["mean"], device=device)[None])
    model.to("cpu"); torch.cuda.empty_cache()
    shape_stem, material_stem = _stem(resolved(dataset["pretrained_shape_slat_dec"]), "shape decoder"), _stem(resolved(dataset["pretrained_pbr_slat_dec"]), "material decoder")
    shape_decoder, material_decoder = _load_vae(models, shape_stem).to(device).eval(), _load_vae(models, material_stem).to(device).eval()
    shape_decoder.set_resolution(RESOLUTION)
    with torch.no_grad(): _, guide = shape_decoder(shape, return_subs=True); decoded = material_decoder(generated, guide_subs=guide) * .5 + .5
    if decoded.feats.shape[1] != 6: raise RuntimeError("material decoder must emit six PBR channels")
    representation = runtime["MeshWithVoxel"](case["vertices"].to(device), case["faces"].to(device), [-.5, -.5, -.5], 1. / RESOLUTION, decoded.coords[:, 1:].int(), decoded.feats.float().clamp(0, 1), torch.Size([1, 6, RESOLUTION, RESOLUTION, RESOLUTION]), layout=LAYOUT)
    _export(Path(output), representation, texture_size, decimation_target, texture_format, runtime, gutter=gutter)
    return dual_stats


def run_completion(*, input_dir: str | Path, output: str | Path, model_config: str | Path,
                   checkpoint: str | Path, trellis_root: str | Path, seed: int = 42,
                   device: str = "cuda", texture_size: int = 4096,
                   decimation_target: int = 500000, texture_format: str = "png", gutter: bool = False,
                   back_view_dir: str | Path | None = None, work_dir: str | Path | None = None) -> None:
    """Complete one view by default, or validate and publish a dual-view result."""
    arguments = dict(input_dir=input_dir, output=output, model_config=model_config,
                     checkpoint=checkpoint, trellis_root=trellis_root, seed=seed, device=device,
                     texture_size=texture_size, decimation_target=decimation_target,
                     texture_format=texture_format, gutter=gutter)
    if back_view_dir is None:
        if work_dir is not None:
            raise ValueError("completion --work-dir is reserved for dual-view inference")
        _run_completion(**arguments)
        return
    from .dual_view import CAMERA_PROTOCOL, PROTOCOL, digest, load_back_case, write_json
    import shutil

    output = Path(output).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve() if work_dir else output.with_name(output.stem + "_dual_view")
    if output.suffix.lower() != ".glb":
        raise ValueError("output must have a .glb suffix")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if work.exists():
        raise FileExistsError(f"dual-view work directory must be new: {work}")
    from supratex.pipeline_config import DUAL_VIEW_EXPORT
    export = {"texture_size": texture_size, "texture_format": texture_format, "decimation_target": decimation_target}
    if gutter or export != DUAL_VIEW_EXPORT:
        raise ValueError("dual-view reprojection requires a 4096 PNG atlas, 500000-face target, and no --gutter")
    front_root, back_root = Path(input_dir).expanduser().resolve(), Path(back_view_dir).expanduser().resolve()
    back = load_back_case(front_root, back_root)
    work.mkdir(parents=True)
    receipt = {"protocol": PROTOCOL, "camera_protocol": CAMERA_PROTOCOL, "view": "dual",
               "status": "running", "stage": "texture", "front_input": str(front_root),
               "back_input": str(back_root), "output": str(output),
               "checkpoint": str(Path(checkpoint).expanduser().resolve()),
               "model_config": str(Path(model_config).expanduser().resolve()),
               "seed": seed, "sampler": "Euler", "steps": 25,
               "geometry_sha256": digest(front_root / "geometry.pt"),
               "cameras_sha256": digest(back_root / "cameras.json"),
               "front_material_receipt_sha256": digest(front_root / "material_receipt.json"),
               "back_material_receipt_sha256": digest(back_root / "material_receipt.json")}
    write_json(work / "receipt.json", receipt)
    try:
        native = work / "preds/mesh.glb"
        arguments.update(input_dir=front_root, output=native, back_case=back)
        stats = _run_completion(**arguments)
        # Reprojection runs in separate CUDA workers. Release the completion
        # allocator's inactive blocks before those processes create contexts.
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        write_json(work / "receipt_texture.json", {**receipt, "status": "success",
                   "dual_stats": stats, "sampling_calls": 1, "mesh": str(native), "mesh_sha256": digest(native)})
        receipt.update(stage="reprojection")
        write_json(work / "receipt.json", receipt)
        from .reprojection import run_dual_view, validate_dual_view
        final = run_dual_view(native, front_root, back_root, back["cameras"], work)
        validation = validate_dual_view(work, raise_on_invalid=True)
        # Model and reprojection work can be long-running. Revalidate the
        # original inputs and generation evidence before publishing the mesh.
        load_back_case(front_root, back_root)
        bindings = {"geometry_sha256": front_root / "geometry.pt",
                    "cameras_sha256": back_root / "cameras.json",
                    "front_material_receipt_sha256": front_root / "material_receipt.json",
                    "back_material_receipt_sha256": back_root / "material_receipt.json"}
        for key, path in bindings.items():
            if receipt[key] != digest(path):
                raise RuntimeError(f"input changed during dual-view completion: {key}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.dual-view.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as target, final.open("rb") as source:
                shutil.copyfileobj(source, target)
            # Atomic publication that cannot replace another writer's output.
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        receipt.update(status="valid", stage="complete", mesh=str(final), mesh_sha256=digest(final),
                       output_sha256=digest(output), dual_stats=stats,
                       postprocess_validation=validation, stitched_19_phase_checked=False)
        write_json(work / "receipt.json", receipt)
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(work / "receipt.json", receipt)
        raise

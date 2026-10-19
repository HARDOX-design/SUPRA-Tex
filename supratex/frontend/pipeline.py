"""Runnable Image-to-Material Translation handoff builder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .camera import estimate
from .firered import run as run_firered
from .preprocess import RMBG2, prepare_image
from .geometry import run as run_geometry
from .qwen_image import run as run_qwen, validate as validate_qwen


def predict_material_maps(rgba: Path, rgb: Path, output: Path, *,
                          material_backend: str, material_base: str | Path,
                          material_model: str | Path | None = None,
                          qwen_runtime: Path | None = None, qwen_deps: Path | None = None,
                          firered_package: Path | None = None, firered_manifest: Path | None = None,
                          material_weights: Path | None = None, material_spec: Path | None = None,
                          material_checkpoint_manifest: Path | None = None,
                          revision: str | None = None, cache_dir: Path | None = None,
                          local_files_only: bool = False, device: str = "cuda", key: str = "input",
                          receipt_path: Path | None = None,
                          cuda_visible_devices: str | None = None) -> tuple[Path, Path]:
    """Predict either view with the same material backend and artifact selection."""
    if receipt_path is not None and firered_manifest is not None:
        raise ValueError("material provenance does not support a custom FireRed manifest; "
                         "omit it to bind the current condition and use matching prompts for both views")
    if receipt_path is not None:
        from supratex.weights import resolve_model_dir
        material_base = resolve_model_dir(material_base, cache_dir=cache_dir, local_files_only=local_files_only)
        if material_model is not None:
            patterns = (["adapter_model.safetensors", "adapter_config.json",
                         "checkpoint-0100000.pt", "checkpoint-0100000.complete.json"]
                        if material_backend == "qwen-image" else
                        ["full_transformer.safetensors", "joint_contract.json", "checkpoint_complete.json"])
            material_model = resolve_model_dir(material_model, revision=revision, cache_dir=cache_dir,
                                               local_files_only=local_files_only, allow_patterns=patterns)
        from supratex.backview.material_identity import artifact_identity
        before_artifacts = artifact_identity({"base": material_base, "model": material_model,
                                              "weights": material_weights, "spec": material_spec,
                                              "checkpoint_manifest": material_checkpoint_manifest})
    if material_backend == "qwen-image":
        albedo, mra = run_qwen(qwen_runtime, material_base, material_model, rgba, output,
                              deps=qwen_deps, cache_dir=cache_dir, revision=revision,
                              local_files_only=local_files_only, device=device,
                              cuda_visible_devices=cuda_visible_devices)
    elif material_backend == "firered":
        albedo, mra = run_firered(firered_package, material_base, rgb, output,
                                firered_manifest, key=key, weights=material_weights,
                                spec=material_spec, checkpoint_manifest=material_checkpoint_manifest,
                                material_model=material_model, revision=revision,
                                cache_dir=cache_dir, local_files_only=local_files_only)
    else:
        raise ValueError("material backend must be qwen-image or firered")
    if receipt_path is not None:
        def identity(value):
            if value is None:
                return None
            path = Path(value).expanduser()
            return str(path.resolve()) if path.exists() else str(value)
        signature = {"backend": material_backend, "base": identity(material_base),
                     "model": identity(material_model), "revision": revision,
                     "runtime": identity(qwen_runtime if material_backend == "qwen-image" else firered_package),
                     "deps": identity(qwen_deps) if material_backend == "qwen-image" else None,
                     "weights": identity(material_weights), "spec": identity(material_spec),
                     "checkpoint_manifest": identity(material_checkpoint_manifest), "seed": 42}
        from supratex.backview.material_identity import artifact_identity
        signature["artifacts"] = artifact_identity(signature)
        if signature["artifacts"] != before_artifacts:
            raise RuntimeError("material artifacts changed during prediction; discard this view and use a new output")
        # Image digests bind the receipt to the exact maps consumed downstream.
        paths = {"condition_rgba.png": rgba, "condition_rgb.png": rgb,
                 "albedo.png": albedo, "mra.png": mra}
        receipt = {"status": "complete", "backend": material_backend, "model_signature": signature,
                   "files": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for name, path in paths.items()}}
        receipt_path = Path(receipt_path)
        if receipt_path.exists():
            raise FileExistsError(receipt_path)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return albedo, mra


def _split_frontend_gpus(material_backend: str, device: str) -> tuple[str, str]:
    """Pick the first two visible GPUs for geometry and Qwen material."""
    import os
    import torch

    if material_backend != "qwen-image" or device != "cuda" or torch.cuda.device_count() < 2:
        raise ValueError("--parallel-frontend requires the qwen-image backend, device 'cuda', "
                         "geometry generation, and at least two visible GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    names = [name.strip() for name in visible.split(",")] if visible else [str(index) for index in range(torch.cuda.device_count())]
    return names[0], names[1]


def run(input_image: Path, output: Path, *, moge_model: str | Path,
        firered_package: Path | None = None, firered_base: str | Path | None = None,
        material_backend: str = "qwen-image", material_base: str | Path | None = None,
        qwen_runtime: Path | None = None, qwen_deps: Path | None = None,
        firered_manifest: Path | None = None, material_weights: Path | None = None, material_spec: Path | None = None,
        material_checkpoint_manifest: Path | None = None, rmbg_model: str | Path | None = None, use_input_alpha: bool = False,
        device: str = "cuda", moge_module: str = "moge.model.v2", resolution_level: int = 9,
        key: str = "input", material_model: str | Path | None = None, revision: str | None = None,
        cache_dir: Path | None = None, local_files_only: bool = False,
        geometry_package: Path | None = None, geometry_model: str | Path | None = None,
        geometry_revision: str | None = None, geometry_initial_mesh_scale: float = 1.0,
        geometry_scale_search: bool = False, geometry_target_padding: int = 3, geometry_max_iterations: int = 2,
        geometry_seed: int = 42, record_material_provenance: bool = False,
        attention: str = "auto", low_vram: bool | None = None,
        parallel_frontend: bool = False) -> dict[str, str]:
    """Build the condition, canonical camera, and material-map handoff in one directory."""
    if record_material_provenance and firered_manifest is not None:
        raise ValueError("material provenance does not support a custom FireRed manifest; "
                         "omit it to bind the current condition and use matching prompts for both views")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("frontend output must be a fresh empty directory")
    if (geometry_package is None) != (geometry_model is None):
        raise ValueError("--geometry-package and --geometry-model must be supplied together")
    if material_backend not in {"qwen-image", "firered"}:
        raise ValueError("--material-backend must be qwen-image or firered")
    if parallel_frontend and geometry_package is None:
        raise ValueError("--parallel-frontend requires geometry generation")
    gpus = _split_frontend_gpus(material_backend, device) if parallel_frontend else None
    from supratex.inference.runtime import configure_inference_runtime
    configure_inference_runtime(attention=attention, low_vram=low_vram, device=device,
                                material_backend=material_backend)
    if material_base is not None and firered_base is not None:
        raise ValueError("--material-base and --firered-base cannot be combined")
    from supratex.weights import resolve_model_dir
    if material_model is not None and any(value is not None for value in
                                          (material_weights, material_spec, material_checkpoint_manifest)):
        raise ValueError("--material-model cannot be combined with explicit material artifact paths")
    if material_backend == "qwen-image":
        if any(value is not None for value in (firered_base, firered_manifest, material_weights,
                                              material_spec, material_checkpoint_manifest)):
            raise ValueError("FireRed base, manifest and explicit artifacts require --material-backend firered")
        from supratex.pipeline_config import runtime_root
        root = Path(__file__).resolve().parents[2]
        checkpoints = root / "ckpts"
        qwen_runtime, material_base, checkpoint, qwen_deps = validate_qwen(
            qwen_runtime or runtime_root(root) / "qwen", material_base or checkpoints / "qwen-base",
            material_model or checkpoints / "qwen-material", deps=qwen_deps,
            cache_dir=cache_dir, revision=revision, local_files_only=local_files_only)
        material_model = checkpoint.parent
    else:
        if qwen_runtime is not None or qwen_deps is not None:
            raise ValueError("--qwen-runtime and --qwen-deps require --material-backend qwen-image")
        material_base = material_base or firered_base
        if material_base is None:
            raise ValueError("FireRed requires --material-base or --firered-base")
        firered_package = firered_package or Path(__file__).resolve().parent / "material_model"
    if not Path(moge_model).expanduser().is_file():
        moge_model = resolve_model_dir(moge_model, cache_dir=cache_dir, local_files_only=local_files_only,
                                       allow_patterns=["model.pt"])
    else:
        moge_model = Path(moge_model).expanduser()
    if rmbg_model is not None:
        rmbg_model = resolve_model_dir(rmbg_model, cache_dir=cache_dir, local_files_only=local_files_only)
    output.mkdir(parents=True, exist_ok=True)
    estimator = RMBG2(rmbg_model, device=device) if rmbg_model else None
    try:
        rgba, rgb, foreground_source = prepare_image(input_image, output_dir=output, use_input_alpha=use_input_alpha,
                                                      background_model=estimator)
    finally:
        if estimator is not None:
            estimator.close()
    camera = estimate(rgb, moge_model, moge_module, resolution_level, device)
    camera["foreground_source"] = foreground_source
    camera_path = output / "camera.json"
    camera_path.write_text(json.dumps(camera, indent=2) + "\n", encoding="utf-8")
    # Material maps depend only on the foreground condition, not on geometry. With
    # --parallel-frontend both subprocesses run concurrently, one GPU each.
    geometry_result, geometry_job = {}, None
    if geometry_package is not None:
        def geometry_call() -> dict:
            return run_geometry(geometry_package, geometry_model, output, output / "geometry",
                                seed=geometry_seed, initial_mesh_scale=geometry_initial_mesh_scale,
                                scale_search=geometry_scale_search, target_padding=geometry_target_padding,
                                max_iterations=geometry_max_iterations, cache_dir=cache_dir,
                                revision=geometry_revision, local_files_only=local_files_only,
                                cuda_visible_devices=gpus[0] if gpus else None)
        if gpus:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=1)
            geometry_job = executor.submit(geometry_call)
            executor.shutdown(wait=False)
            print(f"Frontend: geometry on GPU {gpus[0]}, material on GPU {gpus[1]} in parallel", flush=True)
        else:
            geometry_result = geometry_call()
    material_dir = output / "material"
    try:
        albedo, mra = predict_material_maps(
            rgba, rgb, material_dir, material_backend=material_backend, material_base=material_base,
            material_model=material_model, qwen_runtime=qwen_runtime, qwen_deps=qwen_deps,
            firered_package=firered_package, firered_manifest=firered_manifest,
            material_weights=material_weights, material_spec=material_spec,
            material_checkpoint_manifest=material_checkpoint_manifest, revision=revision,
            cache_dir=cache_dir, local_files_only=local_files_only, device=device, key=key,
            receipt_path=output / "material_receipt.json" if record_material_provenance else None,
            cuda_visible_devices=gpus[1] if gpus else None)
    finally:
        if geometry_job is not None:
            # Always wait, so a material failure never leaves geometry running.
            geometry_result = geometry_job.result()
    if geometry_package is not None:
        geometry_path = output / "geometry.pt"
        geometry_path.write_bytes(Path(geometry_result["geometry"]).read_bytes())
        camera_path.write_bytes(Path(geometry_result["camera"]).read_bytes())
        if record_material_provenance:
            (output / "geometry_receipt.json").write_bytes(Path(geometry_result["receipt"]).read_bytes())
    # The downstream texture stage consumes a single canonical work directory.
    final_albedo, final_mra = output / "albedo.png", output / "mra.png"
    final_albedo.write_bytes(albedo.read_bytes())
    final_mra.write_bytes(mra.read_bytes())
    if material_backend == "firered":
        (output / "firered_albedo.png").write_bytes(albedo.read_bytes())
        (output / "firered_mra.png").write_bytes(mra.read_bytes())
    result = {"condition_rgba": str(rgba), "condition_rgb": str(rgb), "camera": str(camera_path),
              "albedo": str(final_albedo), "mra": str(final_mra), "foreground_source": foreground_source,
              "material_backend": material_backend}
    if geometry_result:
        result.update(geometry=str(geometry_path), geometry_receipt=geometry_result["receipt"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Supra-Tex Image-to-Material Translation frontend.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moge-model", required=True, help="local MoGe checkpoint/directory or Hugging Face namespace/repo")
    parser.add_argument("--moge-module", default="moge.model.v2")
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--rmbg-model", help="local RMBG-2 directory or Hugging Face namespace/repo")
    parser.add_argument("--use-input-alpha", action="store_true")
    parser.add_argument("--firered-package", type=Path, default=Path(__file__).resolve().parent / "material_model")
    parser.add_argument("--material-backend", choices=("qwen-image", "firered"), default="qwen-image")
    parser.add_argument("--material-base", help="selected backend base directory or Hugging Face namespace/repo")
    parser.add_argument("--qwen-runtime", type=Path, help="native Qwen source snapshot (default: bundled runtime)")
    parser.add_argument("--qwen-deps", type=Path, help="optional Qwen dependency overlay; otherwise use the active Python environment")
    parser.add_argument("--firered-base", help="FireRed-only alias for --material-base")
    parser.add_argument("--material-model", help="local material-model directory or Hugging Face namespace/repo")
    parser.add_argument("--revision", help="material-model Hugging Face revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--geometry-package", type=Path, help="source checkout for the original two-stage geometry generator")
    parser.add_argument("--geometry-model", help="local geometry model directory or Hugging Face namespace/repo")
    parser.add_argument("--geometry-revision", help="geometry model Hugging Face revision")
    parser.add_argument("--geometry-initial-mesh-scale", type=float, default=1.0)
    parser.add_argument("--geometry-scale-search", action="store_true", help="enable bounded mesh-scale calibration (default: Pixal3D fixed scale)")
    parser.add_argument("--parallel-frontend", action="store_true",
                        help="run geometry and Qwen material concurrently on the first two visible GPUs")
    parser.add_argument("--geometry-target-padding", type=int, default=3)
    parser.add_argument("--geometry-max-iterations", type=int, default=2)
    parser.add_argument("--firered-manifest", type=Path)
    parser.add_argument("--material-weights", type=Path)
    parser.add_argument("--material-spec", type=Path)
    parser.add_argument("--material-checkpoint-manifest", type=Path)
    parser.add_argument("--key", default="input")
    parser.add_argument("--record-material-provenance", action="store_true", help="write material bindings for optional dual-view inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "flash_attn_3", "flash_attn", "sdpa", "xformers"), default="auto")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None,
                        help="CPU offload for Qwen/3D; auto enabled on GPUs with up to 24 GiB")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, moge_model=args.moge_model, firered_package=args.firered_package,
                         material_backend=args.material_backend, material_base=args.material_base,
                         qwen_runtime=args.qwen_runtime, qwen_deps=args.qwen_deps,
                         firered_base=args.firered_base, firered_manifest=args.firered_manifest,
                         material_weights=args.material_weights, material_spec=args.material_spec,
                         material_checkpoint_manifest=args.material_checkpoint_manifest,
                         material_model=args.material_model, revision=args.revision, cache_dir=args.cache_dir,
                         local_files_only=args.local_files_only,
                         geometry_package=args.geometry_package, geometry_model=args.geometry_model,
                         geometry_revision=args.geometry_revision,
                         geometry_initial_mesh_scale=args.geometry_initial_mesh_scale,
                         geometry_scale_search=args.geometry_scale_search,
                         parallel_frontend=args.parallel_frontend,
                         geometry_target_padding=args.geometry_target_padding,
                         geometry_max_iterations=args.geometry_max_iterations,
                         rmbg_model=args.rmbg_model, use_input_alpha=args.use_input_alpha, device=args.device,
                         moge_module=args.moge_module, resolution_level=args.resolution_level, key=args.key,
                         record_material_provenance=args.record_material_provenance,
                         attention=args.attention, low_vram=args.low_vram), indent=2))


if __name__ == "__main__":
    main()

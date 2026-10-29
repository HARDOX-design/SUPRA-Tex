#!/usr/bin/env python3
"""Generate a textured PBR mesh from one RGB/RGBA image."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


def _pipeline_overrides(args: argparse.Namespace) -> dict:
    """Validate explicit model selections before resolving a release."""
    overrides = {}
    for component, attribute in (("completion", "pretrained"), ("material", "material_model"),
                                 ("qwen-base", "material_base"), ("geometry", "geometry_model"),
                                 ("vae", "trellis_root"), ("moge2", "moge_model"), ("rmbg2", "rmbg_model")):
        value = getattr(args, attribute)
        if value is not None:
            if component not in ("completion", "material") and not Path(value).expanduser().exists():
                raise ValueError(f"--pipeline requires a local --{attribute.replace('_', '-')} override")
            overrides[component] = value
    if args.revision is not None:
        if args.pretrained is None:
            raise ValueError("--revision requires --pretrained with --pipeline; use --pipeline-revision for the release")
        overrides["completion_revision"] = args.revision
    if args.material_revision is not None:
        if args.material_model is None:
            raise ValueError("--material-revision requires --material-model with --pipeline")
        overrides["material_revision"] = args.material_revision
    return overrides


def _prepare_pipeline(args: argparse.Namespace, overrides: dict, descriptor: dict, *, has_alpha: bool) -> dict:
    """Prepare the release already selected during argument validation."""
    from scripts.prepare_models import prepare_pipeline
    from scripts.run_example import read_config

    source = descriptor["repo_id"] if descriptor["kind"] == "hub" else args.pipeline
    revision = descriptor.get("revision") if descriptor["kind"] == "hub" else args.pipeline_revision
    prepared = prepare_pipeline(source, output_dir=args.models_dir, revision=revision,
                                cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                                with_rmbg=not has_alpha,
                                overrides=overrides)
    return read_config(prepared)


def _validate_inference_options(args: argparse.Namespace) -> None:
    """Check explicit values first, then call again after release defaults."""
    from scripts.prepare_models import _pipeline

    if args.back_view_dir is not None and not args.dual_view:
        raise ValueError("--back-view-dir requires --dual-view")
    _pipeline.validate_export_options(args)
    if not math.isfinite(args.geometry_initial_mesh_scale) or args.geometry_initial_mesh_scale <= 0:
        raise ValueError("geometry-initial-mesh-scale must be finite and positive")
    if not 1 <= args.geometry_target_padding <= 30 or args.geometry_max_iterations < 1:
        raise ValueError("geometry-target-padding must be in [1, 30]; geometry-max-iterations must be positive")
    if not 0 <= args.resolution_level <= 9:
        raise ValueError("resolution-level must be in [0, 9]")


def _validate_input_output(args: argparse.Namespace, *, require_rmbg: bool = True) -> tuple[Path, Path, Path, bool]:
    """Check local inputs and destinations without creating or downloading files."""
    image, output = args.input.expanduser().resolve(), args.output.expanduser().absolute()
    if output.suffix.lower() != ".glb":
        raise ValueError("--output must have a .glb suffix")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output = output.resolve()
    work = (args.work_dir or output.parent / f"{output.stem}_work").expanduser().absolute()
    if work.is_symlink() and not work.exists():
        raise FileExistsError(f"--work-dir must be a fresh empty directory: {work}")
    work = work.resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise FileExistsError(f"--work-dir must be a fresh empty directory: {work}")
    for destination in (output, work):
        for parent in destination.parents:
            if parent.exists():
                if not parent.is_dir():
                    raise NotADirectoryError(f"destination parent must be a directory: {parent}")
                break
    if not image.is_file():
        raise FileNotFoundError(f"input image is missing: {image}")
    from PIL import Image
    with Image.open(image) as source:
        low, high = source.getchannel("A").getextrema() if source.mode == "RGBA" else (255, 255)
        has_alpha = low < 255 and high > 0
        if args.use_input_alpha and not has_alpha:
            raise ValueError("--use-input-alpha requires a non-opaque RGBA input with foreground pixels")
        if require_rmbg and not has_alpha and not args.rmbg_model:
            raise ValueError("RGB/opaque input requires --rmbg-model")
        source.load()
    if args.dual_view and args.back_view_dir is None:
        import shutil
        if shutil.which("codex") is None:
            raise RuntimeError("automatic back generation requires Codex with gpt-6-astra and built-in image_gen")
    return image, output, work, has_alpha


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pipeline", help="SUPRA-Tex release directory, pipeline.json, or HF repo ID; prepares and loads all stages")
    parser.add_argument("--pipeline-revision", help="HF revision of the complete SUPRA-Tex release")
    parser.add_argument("--models-dir", type=Path, help="prepared model directory for --pipeline (default: ckpts/ready)")
    parser.add_argument("--input", type=Path,
                        help="source RGB/RGBA image (default: assets/spiny_whelk_shell.png, or "
                             "assets/cinema_signage.png with --dual-view)")
    parser.add_argument("--output", required=True, type=Path, help="new destination .glb file")
    parser.add_argument("--work-dir", type=Path, help="fresh intermediate directory (default: <output stem>_work beside the GLB)")
    parser.add_argument("--moge-model", help="local MoGe-2 checkpoint/directory or HF repo ID")
    parser.add_argument("--rmbg-model", help="local RMBG-2 directory or HF repo ID; required for RGB input")
    parser.add_argument("--use-input-alpha", action="store_true", help="require and reuse foreground RGBA alpha")
    parser.add_argument("--no-filter", "--no-image-depth", dest="no_filter", action="store_true",
                        help="disable the default single-view image/geometry depth pre-filter; dual-view does not use it")
    parser.add_argument("--resolution-level", type=int, default=9, help="MoGe-2 resolution level")
    parser.add_argument("--geometry-package", type=Path, help="native or paper geometry source checkout; see docs/geometry.md")
    parser.add_argument("--geometry-model", help="local geometry weights directory or HF repo ID")
    parser.add_argument("--geometry-revision", help="geometry weights HF revision")
    parser.add_argument("--geometry-initial-mesh-scale", type=float, default=1.0)
    parser.add_argument("--geometry-scale-search", action="store_true",
                        help="enable bounded mesh-scale calibration (default: Pixal3D fixed scale)")
    parser.add_argument("--parallel-frontend", action="store_true",
                        help="run geometry and Qwen material concurrently on the first two visible GPUs")
    parser.add_argument("--geometry-target-padding", type=int, default=3, help="scale-search target padding")
    parser.add_argument("--geometry-max-iterations", type=int, default=2, help="scale-search update limit")
    parser.add_argument("--firered-package", type=Path, default=Path(__file__).resolve().parent / "supratex/frontend/material_model")
    parser.add_argument("--material-backend", choices=("qwen-image", "firered"), default="qwen-image")
    parser.add_argument("--material-base", help="selected backend base model: local directory or HF repo ID")
    parser.add_argument("--firered-base", help="legacy FireRed base option; requires --material-backend firered")
    parser.add_argument("--qwen-runtime", type=Path, help="native Qwen source snapshot (default: bundled runtime)")
    parser.add_argument("--qwen-deps", type=Path, help="optional Qwen dependency overlay; otherwise use the active Python environment")
    parser.add_argument("--material-model", help="local backend checkpoint bundle or HF repo ID")
    parser.add_argument("--material-revision", help="material checkpoint HF revision")
    parser.add_argument("--material-weights", type=Path)
    parser.add_argument("--material-spec", type=Path)
    parser.add_argument("--material-checkpoint-manifest", type=Path)
    parser.add_argument("--pretrained", help="local Supra-Tex completion bundle or HF repo ID")
    parser.add_argument("--revision", help="completion bundle HF revision")
    parser.add_argument("--model-config", type=Path, help="completion ABI JSON; overrides bundle config")
    parser.add_argument("--checkpoint", type=Path, help="completion weights; overrides bundle checkpoint")
    parser.add_argument("--trellis-root", type=Path, help="TRELLIS.2 VAE directory (all four components for HF releases)")
    parser.add_argument("--cache-dir", type=Path, help="shared HF download cache for all adapters")
    parser.add_argument("--local-files-only", action="store_true", help="resolve models locally without Hub downloads")
    parser.add_argument("--attention", choices=("auto", "flash_attn_3", "flash_attn", "sdpa", "xformers"),
                        help="inference attention backend; auto selects by GPU and installed extensions")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None,
                        help="CPU offload for Qwen and 3D stages; auto enabled on GPUs with up to 24 GiB")
    parser.add_argument("--seed", type=int, help="geometry and completion seed (material runner uses its fixed seed 42)")
    parser.add_argument("--texture-size", type=int)
    parser.add_argument("--decimation-target", type=int)
    parser.add_argument("--texture-format", choices=("png", "webp"))
    parser.add_argument("--gutter", action="store_true", help="enable chart-isolated one-texel base-color gutter padding (default: disabled); "
                             "with --refine, the observation-guided gutter is used")
    parser.add_argument("--refine", action="store_true",
                        help="refine the visible texture after export: write the observed reference-view albedo into the atlas "
                             "and repair its UV gutters from the observation (sharper fine text and detail; needs the 4096 PNG "
                             "atlas; adds about 2 minutes on an RTX 3090)")
    parser.add_argument("--remove-floaters", action=argparse.BooleanOptionalAction, default=True,
                        help="before completion, drop disconnected geometry pieces that project mostly outside the "
                             "image foreground (default: on); extra geometry attached to the object is kept")
    parser.add_argument("--dual-view", action="store_true", help="generate a back view (needs the external Codex CLI) and texture with both views; includes --refine for both views")
    parser.add_argument("--back-view-dir", type=Path, help="reuse a prepared, geometry-bound back package instead of generating a new back")
    args = parser.parse_args(argv)
    from scripts.run_example import MODEL_KEYS, _pipeline_config, default_input, inference_defaults
    if args.input is None:
        args.input = default_input(args.dual_view)
    if not args.pipeline and (args.pipeline_revision is not None or args.models_dir is not None):
        parser.error("--pipeline-revision and --models-dir require --pipeline")
    material_paths = (args.material_weights, args.material_spec, args.material_checkpoint_manifest)
    if args.pipeline and args.material_backend != "qwen-image":
        parser.error("--pipeline requires --material-backend qwen-image")
    if args.material_backend == "qwen-image" and (args.firered_base or any(material_paths)):
        parser.error("FireRed options require --material-backend firered")
    try:
        # Reject explicit argument/input errors before fetching any metadata.
        _validate_inference_options(args)
        if args.pipeline:
            overrides = _pipeline_overrides(args)
            _, _, _, has_alpha = _validate_input_output(args, require_rmbg=False)
            from scripts.prepare_models import _pipeline
            manifest, descriptor = _pipeline.resolve_pipeline(
                args.pipeline, revision=args.pipeline_revision, cache_dir=args.cache_dir,
                local_files_only=args.local_files_only)
            defaults = manifest["defaults"]
        else:
            defaults = inference_defaults()
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        # Release defaults can introduce a dual-view conflict; check before downloads.
        _validate_inference_options(args)
        if args.pipeline:
            prepared = _prepare_pipeline(args, overrides, descriptor, has_alpha=has_alpha)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.pipeline:
        explicit_pretrained = args.pretrained is not None
        direct_overrides = {"geometry_package", "qwen_runtime", "qwen_deps", "model_config", "checkpoint"}
        for key in MODEL_KEYS:
            if prepared.get(key) is not None and (key not in direct_overrides or getattr(args, key) is None):
                if explicit_pretrained and key in ("model_config", "checkpoint"):
                    continue
                action = parser._option_string_actions["--" + key.replace("_", "-")]
                value = prepared[key]
                setattr(args, key, action.type(value) if action.type else value)
        args.local_files_only = True
        args._pipeline_config = prepared
    if not all((args.moge_model, args.geometry_package, args.geometry_model)):
        parser.error("provide --pipeline or all of --moge-model, --geometry-package, and --geometry-model")
    if args.material_backend == "qwen-image":
        args.qwen_runtime = args.qwen_runtime or _pipeline_config().runtime_root(Path(__file__).resolve().parent) / "qwen"
        args.material_base = args.material_base or str(Path(__file__).resolve().parent / "ckpts/qwen-base")
        args.material_model = args.material_model or str(Path(__file__).resolve().parent / "ckpts/qwen-material")
    else:
        if args.qwen_runtime is not None or args.qwen_deps is not None:
            parser.error("--qwen-runtime and --qwen-deps require --material-backend qwen-image")
        if args.material_base and args.firered_base:
            parser.error("provide only one of --material-base and --firered-base")
        if not (args.material_base or args.firered_base):
            parser.error("FireRed requires --material-base or --firered-base")
        if args.material_model and any(material_paths):
            parser.error("--material-model cannot be combined with explicit material artifact paths")
        if not args.material_model and not all(material_paths):
            parser.error("provide --material-model or all three --material-weights, --material-spec, --material-checkpoint-manifest")
    if not args.pretrained and not all((args.model_config, args.checkpoint, args.trellis_root)):
        parser.error("provide --pretrained or all of --model-config, --checkpoint, and --trellis-root")
    return args


def _completion_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve local/Hub artifacts and check missing completion files before GPU work."""
    from supratex.weights import COMPLETION_ALLOW_PATTERNS, resolve_completion_artifacts, resolve_model_dir
    root = None
    if args.pretrained:
        root = resolve_model_dir(args.pretrained, revision=args.revision, cache_dir=args.cache_dir,
                                 local_files_only=args.local_files_only,
                                 allow_patterns=COMPLETION_ALLOW_PATTERNS)
    return resolve_completion_artifacts(root, model_config=args.model_config,
                                        checkpoint=args.checkpoint, trellis_root=args.trellis_root)


def run(args: argparse.Namespace) -> Path:
    """Run the existing image frontend, then complete and export its generated mesh."""
    if getattr(args, "_pipeline_config", None) is not None:
        from scripts.run_example import runtime_environment
        environment = runtime_environment(args._pipeline_config, args.attention)
        for key in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND"):
            if key not in environment:
                os.environ.pop(key, None)
        os.environ.update(environment)
        for source in reversed(environment["PYTHONPATH"].split(os.pathsep)):
            if source not in sys.path:
                sys.path.insert(0, source)
    image, output, work, _ = _validate_input_output(args)
    from supratex.inference.runtime import configure_inference_runtime
    runtime = configure_inference_runtime(attention=getattr(args, "attention", "auto"),
                                          low_vram=getattr(args, "low_vram", None),
                                          material_backend=args.material_backend)
    # The adapters validate source compatibility and full checkpoint contents.
    from supratex.frontend.geometry import validate_source
    validate_source(args.geometry_package.expanduser())
    if args.material_backend == "qwen-image":
        from supratex.frontend.qwen_image import validate
        validate(args.qwen_runtime, args.material_base, args.material_model, deps=args.qwen_deps,
                 revision=args.material_revision, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    elif not (args.firered_package.expanduser() / "infer.py").is_file():
        raise FileNotFoundError(f"material entrypoint is missing: {args.firered_package / 'infer.py'}")
    material_paths = {}
    for name in ("material_weights", "material_spec", "material_checkpoint_manifest"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing {name}: {path}")
        material_paths[name] = path
    config, checkpoint, encoders = _completion_paths(args)
    from supratex.frontend.pipeline import run as run_frontend
    from supratex.inference import run_completion
    work.mkdir(parents=True, exist_ok=True)
    handoff = work / "input"
    filter_requested = not getattr(args, "no_filter", False)
    filter_enabled = filter_requested and not args.dual_view
    filter_disabled_reason = ("user_disabled" if not filter_requested else
                              "dual_view_unsupported" if args.dual_view else None)
    receipt = {"status": "running", "stage": "frontend", "input": str(image), "output": str(output),
               "work_dir": str(work), "seed": args.seed, "gutter": args.gutter,
               "material_backend": args.material_backend, "material_seed": 42,
               "runtime": runtime,
               "model_config": str(config), "checkpoint": str(checkpoint), "dual_view": args.dual_view,
               "refine": args.refine, "remove_floaters": args.remove_floaters,
               "image_depth_filter": {"requested": filter_requested, "enabled": filter_enabled,
                                      "disabled_reason": filter_disabled_reason}}
    def record() -> None:
        temporary = work / ".run.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(work / "run.json")
    record()
    try:
        print(f"[1/2] Foreground, MoGe-2, geometry, material maps -> {handoff}", flush=True)
        result = run_frontend(image, handoff, moge_model=args.moge_model, rmbg_model=args.rmbg_model,
                              use_input_alpha=args.use_input_alpha, resolution_level=args.resolution_level,
                              export_image_depth=filter_enabled,
                              geometry_package=args.geometry_package.expanduser().resolve(), geometry_model=args.geometry_model,
                              geometry_revision=args.geometry_revision, geometry_seed=args.seed,
                              geometry_initial_mesh_scale=args.geometry_initial_mesh_scale,
                              geometry_scale_search=args.geometry_scale_search,
                              parallel_frontend=args.parallel_frontend,
                              geometry_target_padding=args.geometry_target_padding,
                              geometry_max_iterations=args.geometry_max_iterations,
                              firered_package=args.firered_package.expanduser().resolve(), firered_base=args.firered_base,
                              material_backend=args.material_backend, material_base=args.material_base,
                              qwen_runtime=args.qwen_runtime if args.material_backend == "qwen-image" else None,
                              qwen_deps=args.qwen_deps if args.material_backend == "qwen-image" else None,
                              material_model=args.material_model, **material_paths,
                              revision=args.material_revision, cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                              **({"record_material_provenance": True} if args.dual_view else {}))
        optional = {"refine": True, "work_dir": work / "refine"} if args.refine else {}
        if args.dual_view:
            receipt.update(stage="back_view", frontend=result)
            record()
            if args.back_view_dir is None:
                from supratex.backview.workflow import build_back_view
                back_dir = build_back_view(handoff, work / "back_view", cache_dir=args.cache_dir,
                                           local_files_only=args.local_files_only)
            else:
                back_dir = args.back_view_dir.expanduser().resolve()
            optional = {"back_view_dir": back_dir, "work_dir": work / "dual_view"}
            receipt["back_view_dir"] = str(back_dir)
        receipt.update(stage="completion", frontend=result)
        record()
        print(f"[2/2] Surface observations, PBR completion, GLB export -> {output}", flush=True)
        try:
            run_completion(input_dir=handoff, output=output, model_config=config, checkpoint=checkpoint,
                           trellis_root=encoders, seed=args.seed, texture_size=args.texture_size,
                           decimation_target=args.decimation_target, texture_format=args.texture_format, gutter=args.gutter,
                           remove_floaters=args.remove_floaters, **optional)
        finally:
            # Copy the snapshot actually consumed by completion, including failed
            # runs, rather than the possibly older material-preparation review.
            completion_record = work / "dual_view" / "receipt.json"
            if args.dual_view and completion_record.is_file():
                completion = json.loads(completion_record.read_text(encoding="utf-8"))
                receipt["back_view_review"] = completion["back_view_review"]
                receipt["review_changed_since_preparation"] = completion["review_changed_since_preparation"]
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"completion did not emit a nonempty GLB: {output}")
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        record()
        raise
    receipt.update(status="complete", stage="complete")
    record()
    print(f"Textured mesh: {output}\nIntermediate files and run receipt: {work}", flush=True)
    return output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

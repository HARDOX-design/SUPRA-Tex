#!/usr/bin/env python3
"""Texture a supplied mesh with Supra-Tex PBR materials."""
from __future__ import annotations

import argparse
from pathlib import Path


def _validate_options(args: argparse.Namespace) -> None:
    from scripts.prepare_models import _pipeline

    if args.dual_view != (args.back_view_dir is not None):
        raise ValueError("--dual-view and --back-view-dir must be provided together")
    if args.work_dir is not None and not (args.dual_view or args.refine):
        raise ValueError("--work-dir requires --dual-view or --refine")
    _pipeline.validate_export_options(args)


def _validate_pipeline_paths(args: argparse.Namespace) -> None:
    """Reject local input/output errors before fetching a manifest or weights."""
    from scripts.prepare_models import require_file

    args.input_dir = args.input_dir.expanduser().absolute()
    args.output = args.output.expanduser().absolute()
    if args.work_dir is not None:
        args.work_dir = args.work_dir.expanduser().absolute()
    for name in ("camera.json", "condition_rgba.png", "albedo.png", "mra.png"):
        require_file(args.input_dir / name)
    if not any((args.input_dir / name).is_file() for name in ("geometry.pt", "geometry.npz")):
        raise FileNotFoundError(f"missing geometry.pt or geometry.npz in {args.input_dir}")
    for name in ("model_config", "checkpoint"):
        if getattr(args, name) is not None:
            require_file(getattr(args, name).expanduser())
    for name in ("trellis_root", "back_view_dir"):
        if getattr(args, name) is not None and not getattr(args, name).expanduser().is_dir():
            raise NotADirectoryError(f"--{name.replace('_', '-')} must be an existing directory")
    output = args.output.expanduser().absolute()
    if output.suffix.lower() != ".glb":
        raise ValueError("--output must have a .glb suffix")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if args.work_dir is not None:
        work = args.work_dir.expanduser().absolute()
        if work.exists() or work.is_symlink():
            raise FileExistsError(f"--work-dir must be a new directory: {work}")
    for destination in (output, args.work_dir):
        if destination is not None:
            for parent in destination.expanduser().absolute().parents:
                if parent.exists():
                    if not parent.is_dir():
                        raise NotADirectoryError(f"destination parent must be a directory: {parent}")
                    break


def _prepare_pipeline(args: argparse.Namespace, manifest: dict, source: dict,
                      completion: dict | None) -> None:
    """Resolve only completion and VAE weights, keeping the Hub cache reusable."""
    from scripts.prepare_models import COMPLETION_PATTERNS, VAE_STEMS, completion_checkpoint, require_file

    def snapshot(descriptor: dict, patterns: list[str]) -> Path:
        if descriptor["kind"] == "local":
            return Path(descriptor["path"])
        from huggingface_hub import snapshot_download
        try:
            return Path(snapshot_download(
                repo_id=descriptor["repo_id"], revision=descriptor.get("revision"),
                allow_patterns=patterns,
                cache_dir=str(args.cache_dir.expanduser()) if args.cache_dir is not None else None,
                local_files_only=args.local_files_only))
        except Exception:
            detail = "the required weights must already be cached" if args.local_files_only else "check the revision, network and your access"
            raise RuntimeError(f"Cannot load weights from {descriptor['repo_id']}; {detail}") from None

    if completion is not None:
        # Preserve standalone/legacy bundle overrides accepted by --pretrained.
        root = snapshot(completion, COMPLETION_PATTERNS + ["configs/*.json", "weights/material_completion.pt",
                                                         "weights/sparse_encoders/*.json", "weights/scvae/*.json"])
        args.pretrained = str(root)
    elif args.model_config is None or args.checkpoint is None:
        subdir = manifest["models"]["completion"]
        root = snapshot(source, [f"{subdir}/{pattern}" for pattern in COMPLETION_PATTERNS]) / subdir
        args.pretrained = str(root)
    else:
        root = None
    if root is not None:
        standalone = (root / "config.json").is_file()
        if args.model_config is None:
            require_file(root / ("config.json" if standalone else "configs/inference_model.json"))
            if standalone:
                require_file(root / "partial_normalization.json")
        if args.checkpoint is None:
            if standalone:
                completion_checkpoint(root)
            else:
                require_file(root / "weights/material_completion.pt")
    if args.trellis_root is None:
        entry = manifest["external_models"]["vae"]
        args.trellis_root = snapshot({"kind": "hub", **entry}, entry["allow_patterns"])
    vae = args.trellis_root.expanduser()
    vae = vae / "ckpts" if (vae / "ckpts").is_dir() else vae
    for stem in VAE_STEMS:
        for suffix in (".json", ".safetensors"):
            require_file(vae / (stem + suffix))
    args.local_files_only = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete a PBR GLB from a mesh, camera, RGBA image, and material maps.",
                                     allow_abbrev=False)
    parser.add_argument("--pipeline", help="SUPRA-Tex release directory, pipeline.json, or HF repo ID; loads completion and VAE weights")
    parser.add_argument("--pipeline-revision", help="HF revision of the complete SUPRA-Tex release")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "assets/examples/pitcher_plant",
                        help="directory containing geometry.pt or geometry.npz, camera.json, condition_rgba.png, "
                             "albedo.png, and mra.png (default: bundled assets/examples/pitcher_plant)")
    parser.add_argument("--output", required=True, type=Path, help="new destination GLB; an existing file is never replaced")
    parser.add_argument("--pretrained", help="local model bundle directory or Hugging Face namespace/repo")
    parser.add_argument("--revision", help="Hugging Face model revision")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    parser.add_argument("--local-files-only", action="store_true", help="use cached model files without network access")
    parser.add_argument("--attention", choices=("auto", "flash_attn_3", "flash_attn", "sdpa", "xformers"))
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None,
                        help="CPU offload; auto enabled on GPUs with up to 24 GiB")
    parser.add_argument("--model-config", type=Path, help="public Supra-Tex model ABI JSON; overrides bundle config")
    parser.add_argument("--checkpoint", type=Path, help="public material-completion checkpoint; overrides bundle weights")
    parser.add_argument("--trellis-root", type=Path, help="TRELLIS.2 VAE directory (all four components for HF releases)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--texture-size", type=int)
    parser.add_argument("--decimation-target", type=int)
    parser.add_argument("--texture-format", choices=("png", "webp"))
    parser.add_argument("--gutter", action="store_true", help="enable chart-isolated one-texel base-color gutter padding (default: disabled); "
                             "with --refine, the observation-guided gutter is used")
    parser.add_argument("--refine", action="store_true",
                        help="refine the visible texture after export: write the observed reference-view albedo into the atlas "
                             "and repair its UV gutters from the observation (sharper fine text and detail; needs the 4096 PNG "
                             "atlas; adds about 2 minutes on an RTX 3090)")
    parser.add_argument("--remove-floaters", action="store_true",
                        help="before completion, drop disconnected geometry pieces that project mostly outside the "
                             "image foreground; extra geometry attached to the object is kept")
    parser.add_argument("--dual-view", action="store_true", help="add front/back observation fusion from a prepared back view; includes --refine for both views")
    parser.add_argument("--back-view-dir", type=Path, help="prepared v4 back-view package with generated RGBA, material maps and receipts")
    parser.add_argument("--work-dir", type=Path, help="new directory for dual-view intermediate meshes and receipts")
    args = parser.parse_args(argv)
    if not args.pipeline and args.pipeline_revision is not None:
        parser.error("--pipeline-revision requires --pipeline")
    if args.pipeline and args.revision is not None and args.pretrained is None:
        parser.error("--revision requires --pretrained with --pipeline; use --pipeline-revision for the release")
    try:
        _validate_options(args)
        if args.pipeline:
            from scripts.prepare_models import _pipeline
            completion = _pipeline.model_source(args.pretrained, args.revision) if args.pretrained is not None else None
            _validate_pipeline_paths(args)
            manifest, source = _pipeline.resolve_pipeline(
                args.pipeline, revision=args.pipeline_revision, cache_dir=args.cache_dir,
                local_files_only=args.local_files_only)
            defaults = manifest["defaults"]
        else:
            from scripts.run_example import inference_defaults
            defaults = inference_defaults()
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        _validate_options(args)
        if args.pipeline:
            _prepare_pipeline(args, manifest, source, completion)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    if not args.pretrained and not all((args.model_config, args.checkpoint, args.trellis_root)):
        parser.error("provide --pipeline, --pretrained, or all of --model-config, --checkpoint, and --trellis-root")
    return args


def main() -> None:
    args = parse_args()
    from supratex.inference.runtime import configure_inference_runtime
    configure_inference_runtime(attention=args.attention, low_vram=args.low_vram)
    if args.pretrained:
        from supratex.weights import COMPLETION_ALLOW_PATTERNS, resolve_completion_artifacts, resolve_model_dir
        root = resolve_model_dir(args.pretrained, revision=args.revision, cache_dir=args.cache_dir,
                                 local_files_only=args.local_files_only,
                                 allow_patterns=COMPLETION_ALLOW_PATTERNS)
        args.model_config, args.checkpoint, args.trellis_root = resolve_completion_artifacts(
            root, model_config=args.model_config, checkpoint=args.checkpoint,
            trellis_root=args.trellis_root)
    from supratex.inference import run_completion
    optional = ({"back_view_dir": args.back_view_dir, "work_dir": args.work_dir} if args.dual_view else
                {"refine": True, "work_dir": args.work_dir} if args.refine else {})
    run_completion(input_dir=args.input_dir, output=args.output, model_config=args.model_config, checkpoint=args.checkpoint, trellis_root=args.trellis_root, seed=args.seed, texture_size=args.texture_size, decimation_target=args.decimation_target, texture_format=args.texture_format, gutter=args.gutter, remove_floaters=args.remove_floaters, **optional)


if __name__ == "__main__":
    main()

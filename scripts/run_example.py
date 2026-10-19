#!/usr/bin/env python3
"""Run the bundled RGBA example with prepared models and an isolated source path."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("moge_model", "rmbg_model", "geometry_package", "geometry_model", "qwen_runtime",
              "qwen_deps", "material_base", "material_model", "pretrained", "model_config", "checkpoint", "trellis_root")
PATH_KEYS = (*MODEL_KEYS, "torch_home")


def _pipeline_config():
    """Load the lightweight shared helpers without importing GPU model modules."""
    spec = importlib.util.spec_from_file_location("supratex_pipeline_config", ROOT / "supratex/pipeline_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inference_defaults(config: dict | None = None) -> dict:
    """Read defaults from the pipeline without importing GPU model modules."""
    module = _pipeline_config()
    defaults = module.default_pipeline()["defaults"]
    configured = (config or {}).get("defaults", {})
    if not isinstance(configured, dict):
        raise ValueError("models.json defaults must contain an object")
    defaults.update(configured)
    return module.validate_defaults(defaults)


def read_config(path: Path) -> dict:
    """Resolve paths relative to models.json, preserving metadata for preflight."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("models.json must contain an object")
    for key in PATH_KEYS:
        value = config.get(key)
        if value is None and key in ("rmbg_model", "qwen_deps", "model_config", "checkpoint"):
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"models.json requires a nonempty {key} path")
        resolved = Path(value).expanduser()
        # Normalize without following links: cache-backed files link to
        # opaque blobs, and weight validation needs the logical file name.
        config[key] = os.path.normpath(resolved if resolved.is_absolute() else path.parent / resolved)
    return config


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "ckpts/ready/models.json")
    parser.add_argument("--input", type=Path, help="default: assets/spiny_whelk_shell.png, or "
                        "assets/cinema_signage.png with --dual-view")
    parser.add_argument("--output", type=Path, required=True, help="destination .glb")
    parser.add_argument("--use-input-alpha", action="store_true", help="explicitly require the input foreground alpha")
    parser.add_argument("--attention", choices=("auto", "flash_attn_3", "flash_attn", "sdpa", "xformers"),
                        help="override the prepared pipeline's attention default")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--print-command", action="store_true", help="print without creating files or launching inference")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    protected = {"--" + key.replace("_", "-") for key in MODEL_KEYS}
    protected.update(("--material-backend", "--local-files-only", "--cache-dir", "--revision", "--material-revision", "--geometry-revision",
                      "--pipeline", "--pipeline-revision", "--models-dir"))
    if any(value.split("=", 1)[0] in protected for value in extra):
        parser.error("model paths and offline settings come from --config; additional arguments cannot override them")
    return args, extra


def has_foreground_alpha(path: Path) -> bool:
    from PIL import Image
    with Image.open(path) as image:
        # The inference foreground contract accepts only an actual RGBA
        # input; converting LA or palette transparency here would hide a
        # mode that the inference process itself rejects.
        if image.mode != "RGBA":
            return False
        low, high = image.getchannel("A").getextrema()
        return low < 255 and high > 0


def default_input(dual_view: bool) -> Path:
    """Bundled example: the whelk shell, or the cinema signage for dual-view."""
    return ROOT / "assets" / ("cinema_signage.png" if dual_view else "spiny_whelk_shell.png")


def build_command(args: argparse.Namespace, extra: list[str], config: dict) -> tuple[list[str], dict[str, str]]:
    defaults = inference_defaults(config)
    attention = args.attention or defaults["attention"]
    image = (args.input or default_input("--dual-view" in extra)).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"input image is missing: {image}")
    use_alpha = args.use_input_alpha or has_foreground_alpha(image)
    if not use_alpha and not config.get("rmbg_model"):
        raise ValueError("RGB or opaque input requires RMBG: prepare with --with-rmbg or --rmbg, then use that models.json")
    command = [sys.executable, str(ROOT / "infer.py"), "--input", str(image), "--output",
               str(args.output.expanduser().resolve()), "--material-backend", "qwen-image", "--local-files-only",
               "--attention", attention]
    for key in ("seed", "texture_size", "decimation_target", "texture_format"):
        command.extend(("--" + key.replace("_", "-"), str(defaults[key])))
    if use_alpha:
        command.append("--use-input-alpha")
    if args.low_vram is not None:
        command.append("--low-vram" if args.low_vram else "--no-low-vram")
    for key in MODEL_KEYS:
        if config.get(key):
            command.extend(("--" + key.replace("_", "-"), config[key]))
    command.extend(extra)
    return command, runtime_environment(config, attention)


def runtime_environment(config: dict, attention: str) -> dict[str, str]:
    """Use prepared caches and bundled sources consistently in both entrypoints."""
    runtime = _pipeline_config().runtime_root(ROOT)
    environment = os.environ.copy()
    environment.update(TORCH_HOME=config["torch_home"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", PYTHONNOUSERSITE="1",
                       PYTHONPATH=os.pathsep.join(str(path) for path in
                                                (ROOT, runtime / "moge", runtime / "utils3d")))
    if attention == "auto":
        environment.pop("ATTN_BACKEND", None)
        environment.pop("SPARSE_ATTN_BACKEND", None)
    else:
        environment.update(ATTN_BACKEND=attention, SPARSE_ATTN_BACKEND=attention)
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def main(argv: list[str] | None = None) -> int:
    args, extra = parse_args(argv)
    try:
        command, environment = build_command(args, extra, read_config(args.config))
        if args.print_command:
            print(shlex.join(command))
            return 0
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

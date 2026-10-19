"""Qwen Image 2.1 material adapter inference through its native CLI."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from supratex.weights import resolve_model_dir


def validate_adapter(checkpoint: Path) -> dict:
    """Validate a native adapter config and tensor headers without loading tensors."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    config = json.loads(checkpoint.with_name("adapter_config.json").read_text())
    if (not isinstance(config, dict) or config.get("format") != "supratex_qwen_material_adapter"
            or type(config.get("schema_version")) is not int or config["schema_version"] != 1
            or config.get("adapter_type") != "qwen_image_21_pbr_endpoint"
            or config.get("base_model_name_or_path") != "Qwen/Qwen-Image-2.1"
            or config.get("weight_file") != checkpoint.name
            or type(config.get("source_step")) is not int or config["source_step"] != 100000):
        raise ValueError("unsupported Qwen material adapter config")
    inference = config.get("inference_config")
    booleans = {"lora_output_projection", "omit_all_valid_mask", "bf16_autocast", "deterministic_algorithms"}
    required = booleans | {"resolution", "noise_seed", "lora_rank", "lora_alpha", "attention_backend"}
    if not isinstance(inference, dict) or set(inference) != required:
        raise ValueError("Qwen adapter requires complete inference config")
    if (type(inference["resolution"]) is not int or inference["resolution"] != 1024
            or type(inference["noise_seed"]) is not int or inference["noise_seed"] != 42
            or inference["attention_backend"] != "torch"
            or type(inference["lora_rank"]) is not int or inference["lora_rank"] <= 0
            or type(inference["lora_alpha"]) not in {int, float}
            or not math.isfinite(inference["lora_alpha"]) or inference["lora_alpha"] <= 0
            or any(type(inference[key]) is not bool for key in booleans)):
        raise ValueError("invalid Qwen adapter inference config")
    tensors = config.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Qwen adapter requires a nonempty tensor manifest")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if (metadata.get("format") != "pt" or metadata.get("adapter_format") != config["format"]
                or metadata.get("schema_version") != str(config["schema_version"])
                or set(reader.keys()) != set(tensors)):
            raise ValueError("Qwen adapter tensor keys or metadata mismatch")
        for name, specification in tensors.items():
            header = reader.get_slice(name)
            if (not isinstance(specification, dict)
                    or not isinstance(specification.get("shape"), list)
                    or any(type(size) is not int or size < 0 for size in specification["shape"])
                    or specification["shape"] != header.get_shape()
                    or specification.get("dtype") not in {"F64", "F32", "F16", "BF16"}
                    or specification["dtype"] != header.get_dtype()):
                raise ValueError(f"Qwen adapter tensor shape or dtype mismatch: {name}")
    return config


def validate(runtime: Path, base_model: str | Path, model_source: str | Path, *,
             deps: Path | None = None, cache_dir: Path | None = None,
             revision: str | None = None, local_files_only: bool = False
             ) -> tuple[Path, Path, Path, Path | None]:
    """Check local artifacts without importing the runtime or loading tensors."""
    runtime = Path(runtime).expanduser().resolve()
    for name in ("infer.py", "inference_utils.py", "inference_offload.py"):
        path = runtime / "examples/qwen_image_21/pbr" / name
        if not path.is_file():
            raise FileNotFoundError(f"Qwen native runtime is incomplete: {path}")
    if not (runtime / "diffsynth").is_dir():
        raise FileNotFoundError(runtime / "diffsynth")
    if deps is not None:
        deps = Path(deps).expanduser().resolve()
        if not deps.is_dir():
            raise NotADirectoryError(deps)
    root = resolve_model_dir(model_source, revision=revision, cache_dir=cache_dir,
                             local_files_only=local_files_only,
                             allow_patterns=["adapter_model.safetensors", "adapter_config.json",
                                             "checkpoint-0100000.pt", "checkpoint-0100000.complete.json"])
    checkpoint = root / "adapter_model.safetensors"
    if checkpoint.exists() or (root / "adapter_config.json").exists():
        validate_adapter(checkpoint)
    else:
        checkpoint = root / "checkpoint-0100000.pt"
        marker = json.loads(checkpoint.with_suffix(".complete.json").read_text())
        if (marker.get("status") != "complete" or marker.get("step") != 100000
                or marker.get("bytes") != checkpoint.stat().st_size or checkpoint.stat().st_size == 0):
            raise ValueError("Qwen original checkpoint requires a matching complete record at step 100000")
    base = resolve_model_dir(base_model, cache_dir=cache_dir, local_files_only=local_files_only)
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (base / "processor" / name).is_file():
            raise FileNotFoundError(base / "processor" / name)
    for component in ("vae", "text_encoder", "transformer"):
        directory = base / component
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(directory / "config.json")
        weights = list(directory.glob("*.safetensors"))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise ValueError(f"Qwen base component has missing or empty weights: {directory}")
        for index in directory.glob("*.safetensors.index.json"):
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
            if not shards or any(not (directory / name).is_file() for name in shards):
                raise ValueError(f"Qwen base component has missing indexed shards: {index}")
    return runtime, base, checkpoint, deps


def run(runtime: Path, base_model: str | Path, model_source: str | Path,
        condition: Path, output: Path, *, deps: Path | None = None,
        cache_dir: Path | None = None, revision: str | None = None,
        local_files_only: bool = False, device: str = "cuda:0",
        cuda_visible_devices: str | None = None) -> tuple[Path, Path]:
    """Preserve the frontend canvas and translate native MR into completion MRA."""
    from PIL import Image, ImageChops

    # The native CLI calls torch.cuda.set_device(), which requires an index.
    device = "cuda:0" if device == "cuda" else device
    output = Path(output).resolve()
    targets = [output / name for name in ("native", "input_rgba.png", "inference.log",
                                          "albedo.png", "mra.png", "qwen_receipt.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Qwen material outputs already exist; select a fresh output directory")
    runtime, base, checkpoint, deps = validate(
        runtime, base_model, model_source, deps=deps, cache_dir=cache_dir,
        revision=revision, local_files_only=local_files_only)
    condition = Path(condition).resolve()
    with Image.open(condition) as image:
        if image.mode != "RGBA" or image.width != image.height:
            raise ValueError("Qwen condition must be the frontend's square straight RGBA canvas")
        source_size = image.size
        square = image.copy() if image.size == (1024, 1024) else image.resize((1024, 1024), Image.Resampling.LANCZOS)
    output.mkdir(parents=True, exist_ok=True)
    input_path, native = output / "input_rgba.png", output / "native"
    square.save(input_path)
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(str(path) for path in (runtime, deps) if path is not None),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    command = [sys.executable, str(runtime / "examples/qwen_image_21/pbr/infer.py"),
               "--checkpoint", str(checkpoint), "--model-root", str(base),
               "--input", str(input_path), "--output-dir", str(native), "--device", device]
    with (output / "inference.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=runtime)
    report = json.loads((native / "REPORT.json").read_text())
    config = report.get("config", {})
    if (report.get("status") != "complete" or report.get("checkpoint_step") != 100000
            or config.get("resolution") != 1024 or config.get("noise_seed") != 42
            or config.get("attention_backend") != "torch"):
        raise ValueError("Qwen native report does not match the original 100k inference contract")
    coverage = square.getchannel("A")
    maps = []
    for name in ("albedo", "mra"):
        with Image.open(native / f"{name}.png") as image:
            if image.mode != "RGBA" or image.size != (1024, 1024):
                raise ValueError(f"Qwen native {name} must be RGBA 1024x1024")
            if ImageChops.difference(image.getchannel("A"), coverage).getbbox() is not None:
                raise ValueError(f"Qwen native {name} coverage differs from its input")
            if name == "mra":
                if image.getchannel("B").getextrema() != (255, 255):
                    raise ValueError("Qwen native MR.B must be the constant 255 placeholder")
                maps.append(Image.merge("RGB", (image.getchannel("R"), image.getchannel("G"), coverage)))
            else:
                maps.append(image.convert("RGB"))
    albedo, mra = output / "albedo.png", output / "mra.png"
    for image, path in zip(maps, (albedo, mra)):
        image.save(path)
    receipt = {
        "backend": "qwen-image", "checkpoint": str(checkpoint), "checkpoint_step": 100000,
        "base_model": str(base), "runtime": str(runtime), "condition": str(condition),
        "source_size": list(source_size), "canvas_size": [1024, 1024],
        "preprocessing": "square canvas resized once with LANCZOS; straight RGBA; no crop or compositing",
        "native_report": str(native / "REPORT.json"),
        "albedo": "sRGB; no alpha premultiplication",
        "mra": "R=linear metallic, G=linear roughness, B=input coverage; no predicted material transparency",
        "noise_seed": 42, "attention_backend": "torch", "status": "complete",
    }
    (output / "qwen_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return albedo, mra

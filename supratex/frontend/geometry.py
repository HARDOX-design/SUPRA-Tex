"""Run native or paper Pixal3D geometry with bounded mesh-scale calibration."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from supratex.weights import resolve_model_dir


def validate_source(package: Path) -> None:
    """Check the source checkout before loading any image-stage models."""
    if (package / "pixal3d/pipelines/pixal3d_image_to_3d.py").is_file():
        return
    for name in ("pixal3dpipeline.py", "pixal3dpipeline2stage.py"):
        if not (package / name).is_file():
            raise FileNotFoundError(f"Pixal3D two-stage source is missing: {package / name}")


NATIVE_GEOMETRY_MODELS = ("sparse_structure_flow_model", "sparse_structure_decoder",
                          "shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder")


def validate_model(package: Path, model_root: Path) -> str:
    """Require complete local artifacts; native model stems must stay in the root."""
    native = (package / "pixal3d/pipelines/pixal3d_image_to_3d.py").is_file()
    if native:
        config = model_root / "pipeline.json"
        if not config.is_file():
            raise FileNotFoundError(f"geometry model artifact is missing: {config}")
        models = json.loads(config.read_text())["args"]["models"]
        required = []
        for name in NATIVE_GEOMETRY_MODELS:
            stem = (model_root / models[name]).resolve()
            if not stem.is_relative_to(model_root.resolve()):
                raise ValueError(f"geometry model must be inside model root: {name}")
            required.extend(Path(str(stem) + suffix) for suffix in (".json", ".safetensors"))
        dino = model_root / "dino"
        required.append(dino / "config.json")
        index = dino / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("DINO checkpoint index requires a nonempty weight_map")
            for shard in weight_map.values():
                if (not isinstance(shard, str) or not shard or "\\" in shard or ":" in shard
                        or Path(shard).is_absolute() or ".." in Path(shard).parts):
                    raise ValueError("DINO weight shard must be inside its model directory")
                # HF snapshot files point to shared cache blobs outside the
                # snapshot. Validate the indexed name, then follow its link
                # only when checking that the required file exists.
                required.append(dino / shard)
        else:
            required.append(dino / "model.safetensors")
    else:
        required = []
        for stage in ("dense", "sparse512", "sparse1024"):
            for component in ("dit", "vae", "conditioner", "scheduler"):
                required.append(model_root / stage / component / "config.json")
                if component in ("dit", "vae"):
                    required.append(model_root / stage / component / "model.safetensors")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"geometry model artifact is missing: {path}")
    return "native" if native else "paper"


def _validate_inputs(package: Path, input_dir: Path) -> None:
    validate_source(package)
    for name in ("condition_rgb.png", "condition_rgba.png", "camera.json"):
        if not (input_dir / name).is_file():
            raise FileNotFoundError(f"geometry input is missing: {input_dir / name}")
    from PIL import Image
    with Image.open(input_dir / "condition_rgb.png") as rgb, Image.open(input_dir / "condition_rgba.png") as rgba:
        if rgb.mode != "RGB" or rgba.mode != "RGBA" or rgb.size != rgba.size or rgb.width != rgb.height:
            raise ValueError("geometry conditions must be matching square RGB and RGBA images")
        if rgba.getchannel("A").getbbox() is None:
            raise ValueError("geometry condition alpha must contain foreground")
        rgb.load()
        rgba.load()
    try:
        camera = json.loads((input_dir / "camera.json").read_text(encoding="utf-8"))
        if not isinstance(camera, dict):
            raise ValueError("camera must be an object")
        if any(key in camera for key in ("extrinsic", "intrinsic", "projection_intrinsic")):
            raise ValueError("geometry requires the canonical camera, without explicit matrices")
        distance, fov = float(camera["distance"]), float(camera["camera_angle_x"])
        if not math.isfinite(distance) or distance <= 0 or not math.isfinite(fov) or not 0 < fov < math.pi:
            raise ValueError("invalid distance or field of view")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("camera.json must provide finite positive distance and camera_angle_x in (0, pi), without explicit matrices") from error


def run(package: Path, model_source: str | Path, input_dir: Path, output_dir: Path, *, seed: int = 42,
        initial_mesh_scale: float = 1.0, scale_search: bool = False, target_padding: int = 3, max_iterations: int = 2,
        cuda_visible_devices: str | None = None,
        cache_dir: Path | None = None, revision: str | None = None,
        local_files_only: bool = False) -> dict[str, str]:
    """Generate a final mesh using the adapter's bounded scale calibration.

    The external implementation runs in a separate interpreter so its package
    imports and GPU resources do not share the material frontend's process.
    """
    package, input_dir, output_dir = (Path(path).expanduser().resolve() for path in (package, input_dir, output_dir))
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("geometry output must be a fresh empty directory")
    _validate_inputs(package, input_dir)
    if not math.isfinite(initial_mesh_scale) or initial_mesh_scale <= 0:
        raise ValueError("initial_mesh_scale must be finite and positive")
    if not 1 <= target_padding <= 30:
        raise ValueError("target_padding must be between 1 and 30 for the 64-cell grid")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    model_root = resolve_model_dir(model_source, revision=revision, cache_dir=cache_dir,
                                   local_files_only=local_files_only)
    # The upstream local loader permits missing weight files. Require them here
    # so an incomplete checkpoint can never silently run random initial weights.
    validate_model(package, model_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().with_name("geometry_runner.py")
    command = [sys.executable, str(runner), "--package", str(package), "--model-root", str(model_root),
               "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--seed", str(seed),
               "--initial-mesh-scale", str(initial_mesh_scale), "--target-padding", str(target_padding),
               "--max-iterations", str(max_iterations), *(["--scale-search"] if scale_search else [])]
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(package), str(source_root))) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if local_files_only:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    with (output_dir / "geometry.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, check=True, cwd=package, env=environment, stdout=log, stderr=subprocess.STDOUT)
    results = {"geometry": output_dir / "geometry.pt", "camera": output_dir / "camera.json",
               "receipt": output_dir / "geometry_receipt.json"}
    for path in results.values():
        if not path.is_file():
            raise FileNotFoundError(f"geometry runner did not emit expected output: {path}")
    for name in ("condition_rgb.png", "condition_rgba.png"):
        shutil.copyfile(input_dir / name, output_dir / name)
    return {key: str(path) for key, path in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final mesh with native or paper Pixal3D geometry and bounded scale calibration.")
    parser.add_argument("--package", type=Path, required=True, help="Native source checkout (pixal3d/pipelines/) or paper source (pixal3dpipeline2stage.py)")
    parser.add_argument("--model", required=True, help="Local geometry-model directory or Hugging Face namespace/repo")
    parser.add_argument("--input-dir", type=Path, required=True, help="Condition RGB/RGBA and MoGe camera directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh empty geometry output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-mesh-scale", type=float, default=1.0)
    parser.add_argument("--scale-search", action="store_true", help="enable bounded mesh-scale calibration (default: Pixal3D fixed scale)")
    parser.add_argument("--target-padding", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.package, args.model, args.input_dir, args.output_dir, seed=args.seed,
                         initial_mesh_scale=args.initial_mesh_scale, scale_search=args.scale_search, target_padding=args.target_padding,
                         max_iterations=args.max_iterations, cache_dir=args.cache_dir, revision=args.revision,
                         local_files_only=args.local_files_only), indent=2))


if __name__ == "__main__":
    main()

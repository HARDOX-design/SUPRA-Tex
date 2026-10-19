"""MoGe-2 camera adapter for the canonical Supra-Tex camera JSON contract."""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def camera_from_intrinsics(intrinsics: np.ndarray, width: int, height: int, *, resolution_level: int) -> dict:
    """Translate normalized MoGe intrinsics to the texture-stage camera fields."""
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("MoGe intrinsics must be 3x3")
    fx = float(intrinsics[0, 0]) * width
    if not math.isfinite(fx) or fx <= 0.0:
        raise ValueError("MoGe returned an invalid focal length")
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))
    focal_pixels = 16.0 / math.tan(camera_angle_x / 2.0) * 512.0 / 32.0
    distance = focal_pixels * -0.5 / -256.0
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("MoGe-derived camera distance is invalid")
    return {"camera_angle_x": float(camera_angle_x), "distance": float(distance), "mesh_scale": 1.0,
            "image_width": int(width), "image_height": int(height), "camera_estimator": "MoGe-2",
            "moge_resolution_level": int(resolution_level)}


def estimate(image: Path, model_path: Path, module_name: str = "moge.model.v2", resolution_level: int = 9,
             device: str = "cuda") -> dict:
    if not 0 <= resolution_level <= 9:
        raise ValueError("resolution_level must be in [0, 9]")
    module = importlib.import_module(module_name)
    model_cls = getattr(module, "MoGeModel", None)
    if model_cls is None:
        raise RuntimeError(f"{module_name} does not expose MoGeModel")
    import torch
    checkpoint = model_path / "model.pt" if model_path.is_dir() else model_path
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = model_cls.from_pretrained(str(checkpoint)).to(device).eval()
    tensor = result = None
    try:
        rgb = Image.open(image).convert("RGB")
        width, height = rgb.size
        tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
        with torch.inference_mode():
            result = model.infer(tensor, resolution_level=resolution_level, force_projection=True, apply_mask=True)
        return camera_from_intrinsics(result["intrinsics"].squeeze().float().cpu().numpy(), width, height,
                                      resolution_level=resolution_level)
    finally:
        model.cpu()
        del tensor, result, model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate canonical Supra-Tex camera parameters with MoGe-2.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--module", default="moge.model.v2")
    parser.add_argument("--resolution-level", type=int, default=9,
                        help="MoGe-2 detail level from 0 (fastest) to 9 (highest).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(estimate(args.image, args.model, args.module, args.resolution_level, args.device), indent=2) + "\n")


if __name__ == "__main__":
    main()

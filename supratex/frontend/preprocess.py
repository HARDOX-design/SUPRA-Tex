"""Foreground preparation for the Supra-Tex Image-to-Material Translation stage."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class AlphaEstimator(Protocol):
    def __call__(self, image: Image.Image) -> Image.Image: ...


class RMBG2:
    """Local RMBG-2 foreground adapter using its documented Transformers form."""

    def __init__(self, model: str | Path, *, device: str = "cuda", alpha_threshold: float | None = None):
        self.model_path = Path(model)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"RMBG-2 model directory does not exist: {self.model_path}")
        if alpha_threshold is not None and not 0.0 <= alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in [0, 1]")
        self.device, self.alpha_threshold, self._model = device, alpha_threshold, None

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoModelForImageSegmentation
                import torch
            except ImportError as error:  # pragma: no cover - optional runtime
                raise RuntimeError("RMBG-2 requires torch and transformers") from error
            self._torch = torch
            self._model = AutoModelForImageSegmentation.from_pretrained(
                str(self.model_path), trust_remote_code=True, local_files_only=True
            ).to(self.device).eval()
        return self._model

    def close(self):
        if self._model is not None:
            self._model.cpu()
            self._model = None
            if str(self.device).startswith("cuda"):
                self._torch.cuda.empty_cache()

    def __call__(self, image: Image.Image) -> Image.Image:
        model, torch = self._load(), self._torch
        rgb = image.convert("RGB")
        resized = rgb.resize((1024, 1024), Image.Resampling.LANCZOS)
        value = np.asarray(resized, dtype=np.float32) / 255.0
        value = (value - np.asarray((0.485, 0.456, 0.406), np.float32)) / np.asarray((0.229, 0.224, 0.225), np.float32)
        tensor = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = model(tensor)
        logits = prediction[-1] if isinstance(prediction, (tuple, list)) else prediction.logits
        alpha = logits[0].squeeze().sigmoid().float().cpu().numpy()
        mask = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        mask = mask.resize(rgb.size, Image.Resampling.LANCZOS)
        if self.alpha_threshold is not None:
            cutoff = round(self.alpha_threshold * 255)
            mask = mask.point(lambda value: 255 if value >= cutoff else 0)
        rgb.putalpha(mask)
        return rgb


def _crop_square(image: Image.Image) -> Image.Image:
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if not foreground.size:
        raise ValueError("foreground alpha has no pixels above the 0.8 threshold")
    x0, y0 = foreground[:, 1].min(), foreground[:, 0].min()
    x1, y1 = foreground[:, 1].max(), foreground[:, 0].max()
    side = int(max(x1 - x0, y1 - y0) * 1.1)
    if side <= 0:
        raise ValueError("foreground crop has zero extent")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # Same float box as Pixal3D's preprocess_image; PIL rounds it identically.
    return image.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))


def prepare_image(source: str | Path, *, output_dir: str | Path, use_input_alpha: bool = False,
                  already_cropped: bool = False, background_model: AlphaEstimator | None = None) -> tuple[Path, Path, str]:
    """Write aligned condition RGBA/RGB frames and return their paths and provenance."""
    image = Image.open(source)
    rgba_present = image.mode == "RGBA" and np.any(np.asarray(image)[..., 3] != 255)
    if use_input_alpha and not rgba_present:
        raise ValueError("--use-input-alpha requires a non-opaque RGBA input")
    if not already_cropped:
        scale = min(1.0, 1024.0 / max(image.size))
        if scale < 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    if rgba_present:
        rgba, provenance = image.convert("RGBA"), "input_alpha"
    else:
        if background_model is None:
            raise ValueError("no foreground alpha supplied; pass an explicit RMBG-2 model")
        rgba, provenance = background_model(image.convert("RGB")).convert("RGBA"), "rmbg2"
    if not already_cropped:
        rgba = _crop_square(rgba)
    data = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = Image.fromarray((np.clip(data[..., :3] * data[..., 3:4], 0, 1) * 255).astype(np.uint8))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgba_path, rgb_path = out / "condition_rgba.png", out / "condition_rgb.png"
    rgba.save(rgba_path)
    rgb.save(rgb_path)
    return rgba_path, rgb_path, provenance + ("_precropped" if already_cropped else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an Image-to-Material Translation condition image.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-input-alpha", action="store_true")
    parser.add_argument("--already-cropped", action="store_true")
    parser.add_argument("--rmbg-model", type=Path, help="Local RMBG-2 model directory for opaque images.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha-threshold", type=float)
    args = parser.parse_args()
    estimator = RMBG2(args.rmbg_model, device=args.device, alpha_threshold=args.alpha_threshold) if args.rmbg_model else None
    rgba, rgb, provenance = prepare_image(args.input, output_dir=args.output_dir, use_input_alpha=args.use_input_alpha,
                                          already_cropped=args.already_cropped, background_model=estimator)
    print(f"condition_rgba={rgba}\ncondition_rgb={rgb}\nforeground_source={provenance}")


if __name__ == "__main__":
    main()

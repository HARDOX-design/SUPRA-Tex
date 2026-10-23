"""Prepare generated back-view conditions and predict matching material maps."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image


def composite_black(rgba: Image.Image) -> Image.Image:
    """Use float32 alpha multiplication and truncation, without rounding."""
    values = np.asarray(rgba, dtype=np.float32) / 255.
    return Image.fromarray((np.clip(values[..., :3] * values[..., 3:4], 0, 1) * 255).astype(np.uint8))


def prepare_condition(source: Path, output_dir: Path, *, case_id: str = "input") -> dict[str, Any]:
    """Keep the original generated RGBA and record any trusted-alpha fallback."""
    from scipy import ndimage
    from supratex.inference.dual_view import digest, write_json

    with Image.open(source) as image:
        if image.mode != "RGBA" or image.size != (1024, 1024):
            raise ValueError("back condition must be RGBA 1024x1024")
        rgba = image.copy()
    alpha = np.asarray(rgba)[..., 3]
    foreground, trusted = int((alpha > 8).sum()), int((alpha >= 254).sum())
    reason = None
    if foreground < 64:
        reason = f"alpha foreground contains only {foreground} pixels"
    elif trusted < .5 * foreground:
        reason = f"trusted alpha covers only {trusted}/{foreground} foreground pixels"
    if reason is not None:
        rgb = np.asarray(composite_black(rgba))
        dark = rgb.max(axis=2) <= 12
        seeds = np.zeros(dark.shape, dtype=bool)
        for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            seeds[y, x] = dark[y, x]
        background = ndimage.binary_propagation(seeds, mask=dark)
        new_alpha = np.where(background, 0, 255).astype(np.uint8)
        rgba = Image.fromarray(np.dstack((rgb, new_alpha)))
        foreground = trusted = int((new_alpha > 8).sum())
    if foreground < 64:
        raise ValueError("back foreground has fewer than 64 pixels after preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgba_path, rgb_path = output_dir / "condition_rgba.png", output_dir / "condition_rgb.png"
    if rgba_path.exists() or rgb_path.exists():
        raise FileExistsError("back condition outputs already exist")
    rgba.save(rgba_path)
    composite_black(rgba).save(rgb_path)
    receipt = {"status": "complete", "case_id": case_id,
               "alpha_source": "corner_flood_fill_on_back_condition" if reason else "back_rgba",
               "alpha_fallback_reason": reason, "alpha_pixels": foreground,
               "alpha_trusted_pixels": trusted, "condition_size": [1024, 1024],
               "source_sha256": digest(source), "condition_rgba_sha256": digest(rgba_path),
               "condition_rgb_sha256": digest(rgb_path)}
    write_json(output_dir / "prepare.json", receipt)
    return receipt


def prepare_back_material(front_input_dir: Path, pack_dir: Path, run_root: Path, output_dir: Path,
                          *, case_id: str = "input", device: str = "cuda",
                          cache_dir: Path | None = None, local_files_only: bool = False) -> Path:
    """Consume an audited Astra result using the front material model selection."""
    from .pack import validate_pack
    from .harness import validate_result
    from supratex.frontend.pipeline import predict_material_maps
    from supratex.inference.dual_view import digest, validate_material_receipt, write_json, view_camera

    front_input_dir, pack_dir, run_root, output_dir = (Path(path).expanduser().resolve()
                                                     for path in (front_input_dir, pack_dir, run_root, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"back material output must be new: {output_dir}")
    cameras = validate_pack(pack_dir, input_dir=front_input_dir)
    front_receipt = validate_material_receipt(front_input_dir)
    report = validate_result(run_root, case_id=case_id)
    if report.get("valid") is not True:
        raise ValueError(f"generated back did not pass integrity and inference eligibility checks: {report}")
    for name in ("cameras.json", "back_mask.png", "back_normal.png", "front_rgb.png"):
        if digest(run_root / "inputs" / case_id / name) != digest(pack_dir / name):
            raise ValueError(f"harness input snapshot differs from this normal pack: {name}")
    generated = Path(report["back_rgba"])
    with Image.open(generated) as image, Image.open(pack_dir / "back_mask.png") as mask:
        if image.mode != "RGBA" or image.size != (1024, 1024):
            raise ValueError("generated back must be RGBA 1024x1024")
        if not np.array_equal(np.asarray(image)[..., 3] >= 128, np.asarray(mask.convert("L")) >= 128):
            raise ValueError("generated alpha differs from the v4 rendered mask")
    shutil.copytree(pack_dir, output_dir)
    shutil.copyfile(generated, output_dir / "back_rgba.png")
    record = {"status": "running", "stage": "prepare", "front_input": str(front_input_dir),
              "generation_run": str(run_root), "case_id": case_id,
              "back_view_review": report["back_view_review"]}
    write_json(output_dir / "back_material_run.json", record)
    try:
        prepare_condition(output_dir / "back_rgba.png", output_dir, case_id=case_id)
        generation = {"status": "valid", "worker_model": report.get("worker_model"),
                      "worker_reasoning_effort": report.get("worker_reasoning_effort"),
                      "generation_tool": report.get("generation_tool"),
                      "back_rgba_sha256": digest(output_dir / "back_rgba.png"),
                      "cameras_sha256": digest(output_dir / "cameras.json"),
                      "run_root": str(run_root), "case_id": case_id, "validation": report,
                      "back_view_review": report["back_view_review"]}
        write_json(output_dir / "generation_receipt.json", generation)
        camera = view_camera(cameras, "back")
        camera = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in camera.items()}
        write_json(output_dir / "camera.json", camera)
        signature = front_receipt["model_signature"]
        is_qwen = signature["backend"] == "qwen-image"
        path = lambda value: Path(value) if value is not None else None
        record.update(stage="material")
        write_json(output_dir / "back_material_run.json", record)
        albedo, mra = predict_material_maps(
            output_dir / "condition_rgba.png", output_dir / "condition_rgb.png", output_dir / "material",
            material_backend=signature["backend"], material_base=signature["base"],
            material_model=signature.get("model"), revision=signature.get("revision"),
            qwen_runtime=path(signature.get("runtime")) if is_qwen else None,
            qwen_deps=path(signature.get("deps")) if is_qwen else None,
            firered_package=path(signature.get("runtime")) if not is_qwen else None,
            material_weights=path(signature.get("weights")), material_spec=path(signature.get("spec")),
            material_checkpoint_manifest=path(signature.get("checkpoint_manifest")),
            cache_dir=cache_dir, local_files_only=local_files_only, device=device, key=case_id,
            receipt_path=output_dir / "material_receipt.json")
        shutil.copyfile(albedo, output_dir / "albedo.png")
        shutil.copyfile(mra, output_dir / "mra.png")
        back_receipt = validate_material_receipt(output_dir)
        if signature != back_receipt["model_signature"]:
            raise ValueError("back material configuration differs from the front")
        record.update(status="complete", stage="complete")
        write_json(output_dir / "back_material_run.json", record)
        return output_dir
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output_dir / "back_material_run.json", record)
        raise


def build_back_view(front_input_dir: Path, work_dir: Path, *, device: str = "cuda",
                    cache_dir: Path | None = None, local_files_only: bool = False) -> Path:
    """Render, generate, audit and translate one back view for end-to-end inference."""
    from .pack import render_pack
    from .harness import prepare_run, run_generation

    work_dir = Path(work_dir)
    if work_dir.exists():
        raise FileExistsError(f"back-view work directory must be new: {work_dir}")
    work_dir.mkdir(parents=True)
    pack = render_pack(front_input_dir, work_dir / "pack", device=device)
    prepare_run(pack, work_dir / "harness", case_id="input")
    run_generation(work_dir / "harness", workers=1, execute=True)
    return prepare_back_material(front_input_dir, pack, work_dir / "harness", work_dir / "input",
                                 device=device, cache_dir=cache_dir, local_files_only=local_files_only)

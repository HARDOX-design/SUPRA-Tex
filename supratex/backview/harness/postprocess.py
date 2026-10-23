#!/usr/bin/env python3
"""Deterministic, deliberately limited postprocessing for generated back views.

No ground-truth back RGB is read. A target mask constrains the final canvas; it
does not establish that generated internal geometry is pixel-aligned.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


RESAMPLE = Image.Resampling
SCHEMA_VERSION = 2
REVIEWED_FILES = ("back_rgba.png", "back_rgb.png", "overlay.png", "alignment.json")
_LOADED_POLICY_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_sha256() -> str:
    """Identify the loaded policy; a source edit requires a fresh interpreter."""
    if sha256(Path(__file__)) != _LOADED_POLICY_SHA256:
        raise ValueError("Postprocessing policy changed after import; restart before processing or reviewing")
    return _LOADED_POLICY_SHA256


def mask_image(path: Path) -> Image.Image:
    """Use grayscale values, or meaningful alpha for an RGBA mask export."""
    with Image.open(path) as opened:
        if "A" in opened.getbands():
            alpha = opened.getchannel("A")
            if alpha.getextrema()[0] < 255:
                return alpha.copy()
        return opened.convert("L")


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, border_value=0)


def mask_metrics(candidate: np.ndarray, target: np.ndarray) -> dict:
    intersection = int(np.count_nonzero(candidate & target))
    union = int(np.count_nonzero(candidate | target))
    target_area = int(np.count_nonzero(target))
    candidate_area = int(np.count_nonzero(candidate))
    cb, tb = boundary(candidate), boundary(target)
    result = {
        "iou": intersection / union if union else None,
        "target_coverage": intersection / target_area if target_area else None,
        "candidate_inside_target": intersection / candidate_area if candidate_area else None,
        "candidate_foreground_pixels": candidate_area,
        "target_foreground_pixels": target_area,
        "symmetric_boundary_mean_px": None,
        "symmetric_boundary_p95_px": None,
        "symmetric_boundary_max_px": None,
    }
    if np.any(cb) and np.any(tb):
        distances = np.concatenate((distance_transform_edt(~tb)[cb], distance_transform_edt(~cb)[tb]))
        result.update(
            symmetric_boundary_mean_px=float(distances.mean()),
            symmetric_boundary_p95_px=float(np.percentile(distances, 95)),
            symmetric_boundary_max_px=float(distances.max()),
        )
    return result


def alpha_evidence(image: Image.Image) -> dict:
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return {"reliable": False, "reason": "raw_image_has_no_alpha_channel"}
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    transparent = float(np.mean(alpha < 32))
    opaque = float(np.mean(alpha > 223))
    reliable = transparent >= 0.005 and opaque >= 0.005
    return {
        "reliable": reliable,
        "reason": "nontrivial_native_alpha" if reliable else "alpha_is_nearly_constant_or_has_no_opaque_object",
        "transparent_fraction_lt32": transparent,
        "opaque_fraction_gt223": opaque,
        "threshold": 128,
        "caveat": "Alpha measures the supplied silhouette only; it cannot verify internal surface correspondence.",
    }


def affine(image: Image.Image, scale: float, dx: float, dy: float, resample=RESAMPLE.BICUBIC):
    width, height = image.size
    cx, cy = width / 2.0, height / 2.0
    inverse = (1 / scale, 0, cx - (cx + dx) / scale, 0, 1 / scale, cy - (cy + dy) / scale)
    fill = (255, 255, 255, 0) if image.mode == "RGBA" else 0
    return image.transform(image.size, Image.Transform.AFFINE, inverse, resample=resample, fillcolor=fill)


def bounded_registration(source_alpha: Image.Image, target_alpha: Image.Image) -> dict:
    """Estimate global framing from real alpha, then refine without deforming parts."""
    width, height = target_alpha.size
    factor = min(1.0, 160.0 / max(width, height))
    small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    candidate = source_alpha.resize(small_size, RESAMPLE.NEAREST)
    target = np.asarray(target_alpha.resize(small_size, RESAMPLE.NEAREST)) >= 128
    sw, sh = small_size

    def score(scale, dx_fraction, dy_fraction):
        test = np.asarray(affine(candidate, scale, dx_fraction * sw, dy_fraction * sh, RESAMPLE.NEAREST)) >= 128
        union = np.count_nonzero(test | target)
        return float(np.count_nonzero(test & target) / union) if union else 0.0

    baseline = score(1, 0, 0)
    best = (baseline, 1.0, 0.0, 0.0)
    for scale in (0.98, 0.99, 1.0, 1.01, 1.02):
        for dx in (-0.02, -0.01, 0.0, 0.01, 0.02):
            for dy in (-0.02, -0.01, 0.0, 0.01, 0.02):
                current = score(scale, dx, dy)
                if current > best[0] + 1e-9:
                    best = (current, scale, dx, dy)
    # A generator may preserve the silhouette but enlarge the entire object.
    # Seed a *uniform* scale/translation using all alpha components. No partwise
    # warping, rotation, RGB segmentation, or forced target-alpha scoring.
    native_source = np.asarray(source_alpha) >= 128
    native_target_seed = np.asarray(target_alpha) >= 128
    sy, sx = np.where(native_source)
    ty, tx = np.where(native_target_seed)
    framing_seeds = []
    if sx.size and tx.size:
        source_center = ((sx.min() + sx.max()) / 2, (sy.min() + sy.max()) / 2)
        target_center = ((tx.min() + tx.max()) / 2, (ty.min() + ty.max()) / 2)
        ratios = ((tx.max() - tx.min() + 1) / (sx.max() - sx.min() + 1),
                  (ty.max() - ty.min() + 1) / (sy.max() - sy.min() + 1))
        for initial_scale in (min(ratios), max(ratios), float(np.sqrt(native_target_seed.sum() / native_source.sum()))):
            dx = (target_center[0] - width / 2 - initial_scale * (source_center[0] - width / 2)) / width
            dy = (target_center[1] - height / 2 - initial_scale * (source_center[1] - height / 2)) / height
            framing_seeds.append({'scale': float(initial_scale), 'dx_fraction': float(dx), 'dy_fraction': float(dy)})
            for delta in (-0.01, 0.0, 0.01):
                scale = initial_scale + delta
                if not (0.65 <= scale <= 1.35 and abs(dx) <= 0.15 and abs(dy) <= 0.15):
                    continue
                current = score(scale, dx, dy)
                if current > best[0] + 1e-9:
                    best = (current, scale, dx, dy)
    _, center_scale, center_dx, center_dy = best
    for scale in (center_scale - 0.005, center_scale, center_scale + 0.005):
        for dx in (center_dx - 0.005, center_dx, center_dx + 0.005):
            for dy in (center_dy - 0.005, center_dy, center_dy + 0.005):
                if not (0.65 <= scale <= 1.35 and abs(dx) <= 0.15 and abs(dy) <= 0.15):
                    continue
                current = score(scale, dx, dy)
                if current > best[0] + 1e-9:
                    best = (current, scale, dx, dy)
    improved = best[0] - baseline >= 0.002
    if not improved:
        best = (baseline, 1.0, 0.0, 0.0)
    # Native-resolution coordinate descent resolves offsets below the thumbnail
    # pixel size; the final step is 0.25 px / 0.0005 uniform scale.
    native_target = np.asarray(target_alpha) >= 128

    def native_score(scale, dx, dy):
        native = np.asarray(affine(source_alpha, scale, dx, dy, RESAMPLE.BICUBIC)) >= 128
        union = np.count_nonzero(native | native_target)
        return float(np.count_nonzero(native & native_target) / union) if union else 0.0

    native_baseline = native_score(1, 0, 0)
    parameters = [best[1], best[2] * width, best[3] * height]
    native_best = native_score(*parameters)
    if native_best < native_baseline:
        parameters, native_best = [1.0, 0.0, 0.0], native_baseline
    for scale_step, pixel_step in ((0.002, max(1.0, max(width, height) * 0.0025)), (0.001, 0.5), (0.0005, 0.25)):
        for axis in (1, 2, 0):
            step = scale_step if axis == 0 else pixel_step
            center = parameters.copy()
            for offset in (-2, -1, 1, 2):
                proposal = center.copy()
                proposal[axis] += offset * step
                if not (0.65 <= proposal[0] <= 1.35 and abs(proposal[1]) <= width * 0.15 and abs(proposal[2]) <= height * 0.15):
                    continue
                value = native_score(*proposal)
                if value > native_best + 1e-9:
                    parameters, native_best = proposal, value
    improved = native_best - native_baseline >= 0.002
    if not improved:
        parameters, native_best = [1.0, 0.0, 0.0], native_baseline
    return {
        "method": "bounded_uniform_scale_translation_from_raw_alpha",
        "search_canvas": list(small_size),
        "scale": parameters[0],
        "dx_px": parameters[1],
        "dy_px": parameters[2],
        "search_iou_before": baseline,
        "search_iou_after": best[0],
        "applied": improved,
        "native_refinement": {"translation_final_step_px": 0.25, "scale_final_step": 0.0005, "iou_before": native_baseline, "iou_after": native_best},
        "framing_seeds": framing_seeds,
        "large_framing_correction": bool(abs(parameters[0] - 1) > 0.02 or abs(parameters[1]) > width * 0.02 or abs(parameters[2]) > height * 0.02),
        "limits": {"scale_min": 0.65, "scale_max": 1.35, "translation_fraction_max": 0.15, "rotation": False, "nonrigid_warp": False},
    }


def texture_diagnostics(image: Image.Image, target: np.ndarray) -> dict:
    # Excluding the silhouette prevents a sharp mask edge inflating this score.
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    interior = binary_erosion(target, iterations=3, border_value=0)
    lap = -4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
    gy, gx = np.gradient(gray)
    return {
        "interior_pixel_count": int(interior.sum()),
        "laplacian_variance": float(np.var(lap[interior])) if interior.any() else None,
        "mean_gradient_magnitude": float(np.mean(np.hypot(gx, gy)[interior])) if interior.any() else None,
        "interpretation": "Image statistics only: not proof of text legibility, physically correct texture, or geometric correspondence.",
    }


def luminance_diagnostics(front_path: Path, generated: Image.Image, target: np.ndarray) -> dict:
    def summarize(rgb, foreground):
        pixels = np.asarray(rgb.convert("RGB"), dtype=np.float32)
        luma = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        values = luma[foreground]
        if not values.size:
            return None
        return {
            "foreground_pixels": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "std": float(values.std()),
        }

    with Image.open(front_path) as opened:
        opened.load()
        evidence = alpha_evidence(opened)
        front_mask = np.asarray(opened.convert("RGBA").getchannel("A")) >= 128 if evidence["reliable"] else None
        front = summarize(opened, front_mask) if front_mask is not None else None
    return {
        "definition": "0.2126 R + 0.7152 G + 0.0722 B on sRGB code values, range 0..255; not physical linear-light luminance",
        "front": front,
        "front_mask_source": "native_alpha_threshold_128" if evidence["reliable"] else "unavailable_no_reliable_native_alpha",
        "front_mask_evidence": evidence,
        "back": summarize(generated, target),
        "back_mask_source": "target_mask_threshold_128",
        "automatic_lighting_match_verdict": None,
        "brightness_or_histogram_adjustment_applied": False,
        "interpretation": "Diagnostics only. Visible materials and their area fractions differ between front and back, so equal means/histograms do not establish matching illumination. Visual review must check lighting_and_brightness.",
    }


def process(task: dict, run_root: Path) -> dict:
    case_id = str(task["case_id"])
    output = resolve_path(task.get("output_dir", f"results/{case_id}"), run_root)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw.png"
    validation = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "raw_present": raw_path.is_file(),
        "status": "missing_output",
        "interior_pixel_alignment_proven": False,
        "text_legibility_proven": False,
        "back_ground_truth_rgb_used": False,
    }
    if not raw_path.is_file():
        validation["message"] = "Generation has not produced results/<case_id>/raw.png. No substitute image was used."
        write_json(output / "validation.json", validation)
        return validation
    try:
        policy = policy_sha256()
        normal_path = resolve_path(task["back_normal"], run_root)
        front_path = resolve_path(task["front_rgb"], run_root)
        mask_path = resolve_path(task["back_mask"], run_root)
        target_alpha = mask_image(mask_path)
        target = np.asarray(target_alpha) >= 128
        if not target.any():
            raise ValueError("Target foreground mask is empty")
        # Bind the audit to the exact bytes processed, even if a worker selects
        # a newer raw.png while this job is running. Reports reject stale hashes.
        raw_bytes = raw_path.read_bytes()
        raw_digest = hashlib.sha256(raw_bytes).hexdigest()
        with Image.open(BytesIO(raw_bytes)) as opened:
            opened.load()
            original_size = opened.size
            evidence = alpha_evidence(opened)
            raw = opened.convert("RGBA")
        standardized = raw.resize(target_alpha.size, RESAMPLE.LANCZOS) if raw.size != target_alpha.size else raw.copy()
        raw_metrics = mask_metrics(np.asarray(standardized.getchannel("A")) >= 128, target) if evidence["reliable"] else None
        transform = {
            "method": "none_no_reliable_raw_alpha", "scale": 1.0,
            "dx_px": 0.0, "dy_px": 0.0, "applied": False,
        }
        registered = standardized
        preclip_metrics = raw_metrics
        if evidence["reliable"]:
            transform = bounded_registration(standardized.getchannel("A"), target_alpha)
            if transform["applied"]:
                proposed = affine(standardized, transform["scale"], transform["dx_px"], transform["dy_px"])
                proposed_metrics = mask_metrics(np.asarray(proposed.getchannel("A")) >= 128, target)
                # A better silhouette must not reduce supported target texture.
                coverage_decreased = proposed_metrics["target_coverage"] < raw_metrics["target_coverage"]
                if proposed_metrics["iou"] > raw_metrics["iou"] and not coverage_decreased:
                    registered, preclip_metrics = proposed, proposed_metrics
                else:
                    transform["rejected_proposal"] = {k: transform[k] for k in ("scale", "dx_px", "dy_px")}
                    transform["rejected_proposal_metrics"] = proposed_metrics
                    transform.update(scale=1.0, dx_px=0.0, dy_px=0.0, applied=False,
                                     rejection_reason="target_coverage_decreased" if coverage_decreased
                                     else "no_native_resolution_iou_improvement")

        # Preserve straight RGB wherever the generator supplied any alpha.
        # Baking source alpha into RGB here would apply coverage a second time
        # when the target alpha is composited, whitening antialiased edges.
        # Fully transparent source pixels have no supported color: fill those
        # with white before enforcing target alpha, without texture inpainting.
        final_pixels = np.asarray(registered).copy()
        final_pixels[final_pixels[:, :, 3] == 0, :3] = 255
        final = Image.fromarray(final_pixels)
        final.putalpha(target_alpha)
        final.save(output / "back_rgba.png")
        white = Image.new("RGBA", final.size, (255, 255, 255, 255))
        white.alpha_composite(final)
        final_rgb = white.convert("RGB")
        final_rgb.save(output / "back_rgb.png")
        with Image.open(normal_path) as opened:
            normal = opened.convert("RGB").resize(target_alpha.size, RESAMPLE.NEAREST)
        display_raw = Image.new("RGBA", registered.size, (255, 255, 255, 255))
        display_raw.alpha_composite(registered)
        overlay = np.asarray(Image.blend(normal, display_raw.convert("RGB"), 0.5)).copy()
        overlay[boundary(target)] = [0, 230, 230]
        if evidence["reliable"]:
            overlay[boundary(np.asarray(registered.getchannel("A")) >= 128)] = [255, 45, 180]
        Image.fromarray(overlay).save(output / "overlay.png")

        alignment = {
            "schema_version": SCHEMA_VERSION,
            "postprocessing_policy_sha256": policy,
            "case_id": case_id,
            "raw_sha256": raw_digest,
            "front_rgb_sha256": sha256(front_path),
            "back_normal_sha256": sha256(normal_path),
            "back_mask_sha256": sha256(mask_path),
            "output_sha256": {name: sha256(output / name) for name in ("back_rgba.png", "back_rgb.png", "overlay.png")},
            "normalization": {
                "raw_size": list(original_size), "target_size": list(target_alpha.size),
                "resampling": "Lanczos" if original_size != target_alpha.size else "none",
                "resize_scale_x": target_alpha.width / original_size[0],
                "resize_scale_y": target_alpha.height / original_size[1],
                "aspect_ratio_changed": abs(original_size[0] / original_size[1] - target_alpha.width / target_alpha.height) > 1e-6,
            },
            "raw_alpha_evidence": evidence,
            "raw_standardized_silhouette": raw_metrics,
            "registration": transform,
            "preclip_silhouette": preclip_metrics,
            "unsupported_target_pixels": int(np.count_nonzero(target & (np.asarray(registered.getchannel("A")) < 128))) if evidence["reliable"] else None,
            "unsupported_target_handling": "zero_alpha_rgb_filled_white_no_texture_inpainting",
            "final_alpha_exactly_equals_target_mask": bool(np.array_equal(np.asarray(final.getchannel("A")), np.asarray(target_alpha))),
            "final_mask_is_enforced_not_alignment_evidence": True,
            "interior_pixel_alignment_proven": False,
            "geometry_audit": "silhouette_only" if evidence["reliable"] else "insufficient_no_reliable_generated_alpha",
            "texture_diagnostics": texture_diagnostics(final_rgb, target),
            "luminance_diagnostics": luminance_diagnostics(front_path, final_rgb, target),
            "limitations": [
                "Final alpha equality is a deterministic target-mask constraint, not evidence of generated geometry correctness.",
                "Raw/preclip silhouette metrics are unavailable when the generated image has no trustworthy alpha.",
                "Neither a silhouette fit nor a normal-overlay establishes pixel-wise internal surface correspondence.",
                "Text spelling, pattern identity, material consistency and plausible unseen details require visual review.",
                "No ground-truth back texture was loaded or used.",
                "Front/back foreground luminance is diagnostic only; differing visible materials prevent an automatic lighting-equivalence verdict.",
            ],
        }
        write_json(output / "alignment.json", alignment)
        if policy_sha256() != policy:
            raise ValueError("Postprocessing policy changed during processing")
        validation.update(
            status="needs_visual_review", message="Postprocessing complete; internal geometric alignment and text legibility remain unproven.",
            dimensions_match_target=True,
            final_alpha_exactly_equals_target_mask=alignment["final_alpha_exactly_equals_target_mask"],
            raw_alpha_reliable=evidence["reliable"],
            geometry_audit=alignment["geometry_audit"],
            raw_sha256=alignment["raw_sha256"],
            visual_review_present=(output / "visual_review.json").is_file(),
            provenance_present=(output / "provenance.json").is_file(),
            postprocessing_policy_sha256=policy,
            processed_file_sha256={name: sha256(output / name) for name in REVIEWED_FILES},
        )
    except Exception as exc:
        validation.update(status="error", message=f"{type(exc).__name__}: {exc}")
    write_json(output / "validation.json", validation)
    return validation


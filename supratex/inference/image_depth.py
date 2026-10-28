"""Conservative image/geometry depth consistency for source-pixel admission.

An auxiliary depth map supplies evidence about image-to-surface correspondence;
it is not ground truth. Invalid or degenerate fits leave the existing admission
rules unchanged and report why the additional gate was disabled.
"""
from __future__ import annotations

import math

import numpy as np

IMAGE_DEPTH_GAP = 0.02
IMAGE_DEPTH_HIGHPASS = 41
IMAGE_DEPTH_BOUNDARY_RADIUS = 16
MIN_FIT_PIXELS = 1000


def image_depth_disagreement(depth: np.ndarray, image_depth: np.ndarray,
                             gap: float = IMAGE_DEPTH_GAP,
                             highpass: int = IMAGE_DEPTH_HIGHPASS,
                             boundary_radius: int = IMAGE_DEPTH_BOUNDARY_RADIUS,
                             stats: dict | None = None) -> np.ndarray:
    """Return suspicious source pixels, using camera-z depths on the same grid.

    ``gap`` is an absolute canonical length (unit object cube). ``highpass``
    and ``boundary_radius`` are pixels of the observation grid, normally 1024.
    The latter is a square/Chebyshev radius. Pixels without two finite positive
    depths are never rejected. A source pixel is rejected only when its local
    residual exceeds ``gap`` near a depth edge in the fitted image-depth map.
    ``stats`` receives JSON-safe fit diagnostics, including disabled reasons.
    """
    from scipy.ndimage import maximum_filter, median_filter, minimum_filter

    depth = np.asarray(depth)
    image_depth = np.asarray(image_depth)
    if depth.ndim != 2 or image_depth.shape != depth.shape:
        raise ValueError("geometry and image depths must be matching 2D maps")
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("image-depth gap must be finite and positive")
    if not isinstance(highpass, int) or highpass < 1 or highpass % 2 != 1:
        raise ValueError("image-depth highpass must be a positive odd size")
    if not isinstance(boundary_radius, int) or boundary_radius < 0:
        raise ValueError("image-depth boundary radius must be nonnegative")

    valid_image = np.isfinite(image_depth) & (image_depth > 0)
    both = np.isfinite(depth) & (depth > 0) & valid_image
    reject = np.zeros(depth.shape, dtype=bool)
    report = {"status": "disabled", "reason": "insufficient_valid_depth",
              "valid_pixels": int(both.sum()), "gap": float(gap),
              "highpass": highpass, "boundary_radius": boundary_radius,
              "rejected_pixels": 0}

    def disabled(reason: str) -> np.ndarray:
        report["reason"] = reason
        if stats is not None:
            stats.update(report)
        return reject

    if report["valid_pixels"] < MIN_FIT_PIXELS:
        return disabled("insufficient_valid_depth")
    x, y = image_depth[both].astype(np.float64), depth[both].astype(np.float64)
    keep = np.ones(x.size, dtype=bool)
    for iteration in range(3):
        count = int(keep.sum())
        if count < MIN_FIT_PIXELS:
            return disabled("insufficient_fit_inliers")
        xx, yy = x[keep], y[keep]
        # Center the fit to avoid an ill-conditioned [large-depth, 1] matrix.
        # A single depth layer cannot identify both scale and offset.
        xmean, ymean = float(xx.mean()), float(yy.mean())
        span = float(np.ptp(xx))
        if span <= 1e-6 * max(float(np.max(np.abs(xx))), 1.0):
            return disabled("degenerate_image_depth_span")
        dx, dy = xx - xmean, yy - ymean
        variance = float(np.mean(dx * dx))
        if variance <= np.finfo(np.float64).eps * max(xmean * xmean, 1.0):
            return disabled("degenerate_image_depth_variance")
        a = float(np.mean(dx * dy) / variance)
        b = ymean - a * xmean
        if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
            return disabled("invalid_affine_fit")
        residual = np.abs(y - (a * x + b))
        if not np.isfinite(residual).all():
            return disabled("nonfinite_affine_residual")
        report.update(fit_iterations=iteration + 1, fit_pixels=count,
                      scale=a, offset=b,
                      median_absolute_residual=float(np.median(residual)))
        if iteration < 2:
            # Include percentile ties. Strict '<' can discard every pixel of
            # an exactly aligned plane, or all of one of two depth layers.
            threshold = float(np.percentile(residual, 80))
            tolerance = 64 * np.finfo(np.float64).eps * max(float(np.max(np.abs(y))), 1.0)
            keep = residual <= threshold + tolerance

    fitted = np.full(depth.shape, np.nan, dtype=np.float32)
    fitted[valid_image] = (a * image_depth[valid_image].astype(np.float64) + b).astype(np.float32)
    if not np.isfinite(fitted[both]).all() or not (fitted[both] > 0).all():
        return disabled("invalid_fitted_depth")
    full = np.zeros(depth.shape, dtype=np.float32)
    full[both] = depth[both] - fitted[both]
    smooth = median_filter(full, size=highpass)
    maximum = maximum_filter(np.where(valid_image, fitted, -np.inf), size=3,
                             mode="constant", cval=-np.inf)
    minimum = minimum_filter(np.where(valid_image, fitted, np.inf), size=3,
                             mode="constant", cval=np.inf)
    with np.errstate(invalid="ignore"):
        boundary = valid_image & np.isfinite(maximum) & np.isfinite(minimum) & ((maximum - minimum) > gap)
    near = maximum_filter(boundary, size=2 * boundary_radius + 1, mode="constant", cval=0)
    reject[both] = (np.abs((full - smooth)[both]) > gap) & near[both]
    report.update(status="applied", reason=None, rejected_pixels=int(reject.sum()),
                  boundary_pixels=int(boundary.sum()))
    if stats is not None:
        stats.update(report)
    return reject

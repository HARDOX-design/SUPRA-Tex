"""Packed texture conversion and equivalent Telea gap filling for GLB export."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch


def quantize_texture(attrs: torch.Tensor) -> np.ndarray:
    """Quantize on the source device and transfer a contiguous uint8 atlas once.

    Match the exporter's float32 multiply, clamp, and truncation. Keeping this
    operation out of NumPy also avoids four strided float32 host transfers.
    The input is left intact for callers that retain the baked attributes.
    """
    return attrs.detach().float().mul(255).clamp_(0, 255).to(torch.uint8).contiguous().cpu().numpy()


def fill_texture_maps(packed: np.ndarray, mask: np.ndarray, layout: dict):
    """Preserve Telea radii while sharing the scalar maps' mask traversal.

    OpenCV treats the three color channels independently. Metallic, roughness,
    and alpha use the same radius, so they can share one three-channel call.
    That call runs alongside base color, which retains its separate radius.
    """
    missing = np.ascontiguousarray(~mask, dtype=np.uint8)
    color = np.ascontiguousarray(packed[..., layout["base_color"]])
    material = np.concatenate([packed[..., layout[key]] for key in
                               ("metallic", "roughness", "alpha")], axis=-1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        color_future = pool.submit(cv2.inpaint, color, missing, 3, cv2.INPAINT_TELEA)
        material = cv2.inpaint(material, missing, 1, cv2.INPAINT_TELEA)
        color = color_future.result()
    return color, material[..., 0:1], material[..., 1:2], material[..., 2:3]


def prepare_texture_maps(attrs: torch.Tensor, mask: torch.Tensor, layout: dict):
    return fill_texture_maps(quantize_texture(attrs), mask.cpu().numpy(), layout)

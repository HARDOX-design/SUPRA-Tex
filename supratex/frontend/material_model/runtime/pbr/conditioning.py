# Adapted for SUPRA-Tex distribution (2026): packaged local inference runtime.
# Upstream FireRed portions remain under Apache-2.0; see THIRD_PARTY_NOTICES.md.
# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""Image preprocessing and prompt conditioning for material inference."""
from __future__ import annotations

import math
from typing import List, Tuple

from PIL import Image
import torch
from torchvision import transforms


DEFAULT_SYSTEM_PROMPT = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

DEFAULT_SYSTEM_PROMPT_T2I = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"

ASPECT_RATIO_512 = {
    '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
    '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
    '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
    '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
    '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}

CONDITION_IMAGE_SIZE = 384 * 384


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """根据图像序列长度计算 flow 的 shift 系数（线性插值）。"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def get_embedding(source_exist, instructions, source_images, processor, text_encoder):
    def _extract_valid_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        valid_hidden_states = []
        for i in range(batch_size):
            valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            valid_hidden_states.append(hidden_states[i][valid_indices])
        return valid_hidden_states

    with torch.no_grad():
        conversations = []
        if not source_exist:
            source_images = [[] for _ in range(len(instructions))]
        for instruction, source_images_sample in zip(instructions, source_images):
            system_prompt = DEFAULT_SYSTEM_PROMPT_T2I if not source_exist else DEFAULT_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": []},
            ]
            for condition_image in source_images_sample:
                messages[1]["content"].append({"type": "image", "image": condition_image})
            messages[1]["content"].append({"type": "text", "text": instruction})
            conversations.append(messages)

        batch_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(text_encoder.device)

        encoder_hidden_states = text_encoder(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            pixel_values=batch_inputs["pixel_values"] if source_exist else None,
            image_grid_thw=batch_inputs["image_grid_thw"] if source_exist else None,
            output_hidden_states=True,
        )

        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = _extract_valid_hidden_states(hidden_states, batch_inputs["attention_mask"])
        start_idx = 64 if source_exist else 34
        split_hidden_states = [e[start_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

    return prompt_embeds, encoder_attention_mask

def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512) -> Tuple[List[float], float]:
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda r: abs(float(r) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def calculate_dimensions(target_area: float, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)

def resize_source_images_for_condition(
    source_images_list: List[List[Image.Image]],
    ref_height: int,
    ref_width: int,
    image_sample_size: int,
    condition_image_size: int = CONDITION_IMAGE_SIZE,
) -> List[List[Image.Image]]:
    """
    对送入 text_encoder 的 source 图像做 Resize + CenterCrop，再缩放到 condition_image_size 对应尺寸。
    参考尺寸为 (ref_height, ref_width)。source_images_list: 每样本多张 source 图的 list of list of PIL。
    返回相同结构的 list of list of PIL Image。
    """
    if not source_images_list:
        return source_images_list
    return [
        apply_condition_transform_to_images(sample, ref_height, ref_width, image_sample_size, condition_image_size)
        for sample in source_images_list
    ]

def apply_condition_transform_to_images(
    images: List[Image.Image],
    ref_height: int,
    ref_width: int,
    image_sample_size: int,
    condition_image_size: int = CONDITION_IMAGE_SIZE,
) -> List[Image.Image]:
    """
    对多张图像应用统一的 Resize+CenterCrop+最终缩放。参考尺寸 (ref_height, ref_width)。
    返回与 images 等长的 list of PIL Image。
    """
    if not images:
        return []
    h, w = float(ref_height), float(ref_width)
    aspect_ratio_sample_size = {
        k: [x / 512 * image_sample_size for x in ASPECT_RATIO_512[k]]
        for k in ASPECT_RATIO_512
    }
    closest_size, _ = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
    closest_size = [int(x / 16) * 16 for x in closest_size]
    closest_size = list(map(int, closest_size))
    if closest_size[0] / h > closest_size[1] / w:
        resize_size = (closest_size[0], int(w * closest_size[0] / h))
    else:
        resize_size = (int(h * closest_size[1] / w), closest_size[1])

    source_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(closest_size),
    ])

    result = []
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        cropped = source_transform(im)
        width_c, height_c = cropped.size
        aspect_ratio = width_c / height_c
        new_width, new_height = calculate_dimensions(condition_image_size, aspect_ratio)
        result.append(cropped.resize((new_width, new_height), Image.Resampling.LANCZOS))
    return result

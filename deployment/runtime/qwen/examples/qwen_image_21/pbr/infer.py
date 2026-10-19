#!/usr/bin/env python3
"""RGB/RGBA-only endpoint inference. No GT, data manifests, downloads or hashes."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from inference_utils import (
    DEFAULTS, PROMPT, build_model, checkpoint_component, decode_pair,
    endpoint_noise, explicit_load,
)
from inference_offload import (
    release_device_cache, sequential_cpu_offload, stage_on_device,
    text_encoder_offload_blocks,
)


def prepare_image(path, resolution):
    if resolution < 32 or resolution % 32:
        raise ValueError('resolution must be positive and divisible by 32')
    with Image.open(path) as source:
        had_alpha = 'A' in source.getbands() or 'transparency' in source.info
        original = source.convert('RGBA')
    width, height = original.size
    scale = min(1., resolution / max(width, height))
    content = (max(1, round(width * scale)), max(1, round(height * scale)))
    aligned = tuple(math.ceil(value / 32) * 32 for value in content)
    offset = tuple((aligned[i] - content[i]) // 2 + (resolution - aligned[i]) // 2 for i in range(2))
    resized = original.resize(content, Image.Resampling.LANCZOS) if original.size != content else original.copy()
    square = Image.new('RGBA', (resolution, resolution), (0, 0, 0, 0))
    square.paste(resized, offset)
    transform = dict(source_size=[width, height], content_size=list(content), padded_size=[resolution, resolution],
                     offset=list(offset), scale=scale, input_had_alpha=had_alpha,
                     resampling='LANCZOS, two-stage centered padding, no crop or alpha compositing',
                     coverage='input alpha retained' if had_alpha else 'all-one source coverage; no segmentation inferred')
    return original, square, transform


def load_checkpoint(path):
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        import math

        adapter = json.loads(path.with_name('adapter_config.json').read_text())
        if (not isinstance(adapter, dict) or adapter.get('format') != 'supratex_qwen_material_adapter'
                or type(adapter.get('schema_version')) is not int or adapter['schema_version'] != 1
                or adapter.get('adapter_type') != 'qwen_image_21_pbr_endpoint'
                or adapter.get('base_model_name_or_path') != 'Qwen/Qwen-Image-2.1'
                or adapter.get('weight_file') != path.name
                or type(adapter.get('source_step')) is not int or adapter['source_step'] != 100000):
            raise ValueError('unsupported Qwen material adapter config')
        inference = adapter.get('inference_config')
        booleans = {'lora_output_projection', 'omit_all_valid_mask', 'bf16_autocast', 'deterministic_algorithms'}
        required = booleans | {'resolution', 'noise_seed', 'lora_rank', 'lora_alpha', 'attention_backend'}
        if not isinstance(inference, dict) or set(inference) != required:
            raise ValueError('Qwen adapter requires complete inference config')
        if (type(inference['resolution']) is not int or inference['resolution'] != 1024
                or type(inference['noise_seed']) is not int or inference['noise_seed'] != 42
                or inference['attention_backend'] != 'torch'
                or type(inference['lora_rank']) is not int or inference['lora_rank'] <= 0
                or type(inference['lora_alpha']) not in {int, float}
                or not math.isfinite(inference['lora_alpha']) or inference['lora_alpha'] <= 0
                or any(type(inference[key]) is not bool for key in booleans)):
            raise ValueError('invalid Qwen adapter inference config')
        tensors = adapter.get('tensors')
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError('Qwen adapter requires a nonempty tensor manifest')
        with safe_open(str(path), framework='pt', device='cpu') as reader:
            metadata = reader.metadata() or {}
            if (metadata.get('format') != 'pt' or metadata.get('adapter_format') != adapter['format']
                    or metadata.get('schema_version') != str(adapter['schema_version'])
                    or set(reader.keys()) != set(tensors)):
                raise ValueError('Qwen adapter tensor keys or metadata mismatch')
            for name, specification in tensors.items():
                header = reader.get_slice(name)
                if (not isinstance(specification, dict)
                        or not isinstance(specification.get('shape'), list)
                        or any(type(size) is not int or size < 0 for size in specification['shape'])
                        or specification['shape'] != header.get_shape()
                        or specification.get('dtype') not in {'F64', 'F32', 'F16', 'BF16'}
                        or specification['dtype'] != header.get_dtype()):
                    raise ValueError(f'Qwen adapter tensor shape or dtype mismatch: {name}')
            weights = {name: reader.get_tensor(name) for name in reader.keys()}
        config = DEFAULTS | inference | {'model_root': adapter['base_model_name_or_path']}
        return config, weights, adapter['source_step']
    marker = json.loads(path.with_suffix('.complete.json').read_text())
    if marker.get('status') != 'complete' or marker.get('bytes') != path.stat().st_size:
        raise ValueError('checkpoint completion record is absent or mismatched')
    # SupraTex adaptation: public inference weights use restricted loading.
    # Legacy trainer checkpoints also contain NumPy uint32 RNG arrays; these
    # exact reconstruction types retain compatibility without arbitrary pickle.
    from numpy.core.multiarray import _reconstruct
    with torch.serialization.safe_globals([_reconstruct, (_reconstruct, 'numpy.core.multiarray._reconstruct'),
                                           (_reconstruct, 'numpy._core.multiarray._reconstruct'),
                                           np.ndarray, np.dtype, type(np.dtype('uint32'))]):
        state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'qwen_pbr_endpoint_v1' or marker.get('step') != state.get('step'):
        raise ValueError('unsupported checkpoint or inconsistent step')
    if not isinstance(state.get('config'), dict) or not isinstance(state.get('model'), dict):
        raise ValueError('checkpoint needs complete config and trainable state')
    config = DEFAULTS | state['config']
    for key in ('model_root', 'resolution', 'lora_rank', 'lora_alpha', 'noise_seed'):
        if key not in state['config']:
            raise ValueError(f'checkpoint config lacks {key}')
    return config, state['model'], state['step']


def strict_load_adapter(model, weights):
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if expected.keys() != weights.keys():
        raise ValueError('checkpoint adapter key set differs from constructed model')
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = weights[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f'checkpoint tensor mismatch/nonfinite: {name}')
            parameter.copy_(tensor)
    model.requires_grad_(False).eval()


def canonical_conditions(values, resolution):
    """Only retain consumed conditioning; material targets are never loaded."""
    values = dict(values)
    layout = values['edit_image_pad_mask'].bool()
    prompt = values.get('prompt_embeds')
    if prompt is None:
        text = values['prompt_text_embeds']
        if layout.ndim != 2 or layout.shape[0] != 1 or text.shape[1] != int((~layout[0]).sum()):
            raise ValueError('compact conditioning layout mismatch')
        prompt = text.new_zeros((1, layout.shape[1], text.shape[-1]))
        prompt[:, ~layout[0]] = text
    if prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[2] != 4096 or layout.shape != prompt.shape[:2]:
        raise ValueError('invalid prompt/layout shape')
    # Model replaces image-placeholder rows from RGB VAE. Canonical zeros allow
    # exact native-v2 comparison without changing consumed values.
    prompt = prompt.clone()
    prompt[:, layout[0]] = 0
    mask = values.get('prompt_embeds_mask')
    mask = torch.ones_like(layout) if mask is None else mask.bool()
    rgb = values['rgb']
    if rgb.shape != (1, 64, resolution // 16, resolution // 16) or mask.shape != layout.shape:
        raise ValueError('invalid RGB latent/mask shape')
    result = dict(rgb=rgb, prompt_embeds=prompt, prompt_embeds_mask=mask, edit_image_pad_mask=layout)
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite conditioning')
    return result


def load_conditions(path, resolution, device):
    values = {}
    keys = {'rgb', 'prompt_embeds', 'prompt_text_embeds', 'prompt_embeds_mask', 'edit_image_pad_mask'}
    with safe_open(str(path), framework='pt', device=str(device)) as reader:
        for key in keys & set(reader.keys()):
            values[key] = reader.get_tensor(key)
    return canonical_conditions(values, resolution)


def input_binding(path):
    path = Path(os.path.abspath(path)); stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def verify_cache_input(cache, image, resolution, model_references):
    metadata = json.loads(Path(cache).with_suffix('.json').read_text())
    binding = metadata.get('input', metadata.get('sources', {}).get('rgb'))
    if binding != input_binding(image) or metadata.get('resolution') != resolution:
        raise ValueError('cache conditioning is not bound to this exact input/stat/resolution')
    if metadata.get('model_root') not in model_references or metadata.get('prompt') != PROMPT:
        raise ValueError('cached model reference or prompt differs from checkpoint inference contract')
    return metadata['model_root']


def native_conditions(square, model_root, resolution, device, low_vram=False):
    from transformers import AutoProcessor
    from diffsynth.models.qwen_image_21_vae import QwenImage21VAE
    from diffsynth.models.qwen_image_21_text_encoder import QwenImage21TextEncoder
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, QwenImage21Unit_EditImageEmbedder, QwenImage21Unit_PromptEmbedder
    from diffsynth.utils.state_dict_converters.qwen_image_21_text_encoder import QwenImage21TextEncoderStateDictConverter
    processor = AutoProcessor.from_pretrained(str(model_root / 'processor'), local_files_only=True)
    processor.image_processor.size = {'shortest_edge': resolution ** 2, 'longest_edge': resolution ** 2}
    pipe = QwenImage21Pipeline(device=device, torch_dtype=torch.bfloat16)
    pipe.processor = processor
    loading_device = 'cpu' if low_vram else device
    vae = explicit_load(QwenImage21VAE, checkpoint_component(model_root, 'vae'), loading_device, torch.bfloat16)
    pipe.vae = vae
    with stage_on_device(vae, device, enabled=low_vram), torch.inference_mode():
        edited = QwenImage21Unit_EditImageEmbedder().process(pipe, edit_image=[square], height=resolution, width=resolution,
                                                         tiled=False, tile_size=256, tile_stride=192)
        if edited['edit_image'][0].size != square.size:
            raise ValueError('native edit path changed dimensions')
        rgb = edited['edit_latents'][0]
    encoder = explicit_load(QwenImage21TextEncoder, checkpoint_component(model_root, 'text_encoder'), loading_device,
                            torch.bfloat16, QwenImage21TextEncoderStateDictConverter)
    pipe.text_encoder = encoder
    blocks = text_encoder_offload_blocks(encoder) if low_vram else ()
    with sequential_cpu_offload(encoder, blocks, device, enabled=low_vram), torch.inference_mode():
        values = QwenImage21Unit_PromptEmbedder().process(pipe, prompt=PROMPT, edit_image=[square])
    conditions = canonical_conditions(dict(values, rgb=rgb), resolution)
    pipe.text_encoder = None
    del encoder, pipe, values, edited, blocks
    gc.collect(); torch.cuda.empty_cache()
    return conditions, vae


def predict(model, conditions, cfg):
    # Shape comes from observed RGB, never a GT or fabricated target tensor.
    shape = (conditions['rgb'].shape[0], 2, *conditions['rgb'].shape[1:])
    shape_carrier = conditions['rgb'].new_empty(shape)
    noise = endpoint_noise(shape_carrier, cfg['noise_seed'])
    mask = conditions['prompt_embeds_mask']
    if cfg['omit_all_valid_mask'] and bool(mask.all()):
        mask = None
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16_autocast']):
        velocity = model(latents=noise, timestep=torch.full((noise.shape[0],), 1000., device=noise.device),
                         prompt_embeds=conditions['prompt_embeds'], prompt_embeds_mask=mask,
                         edit_image_pad_mask=conditions['edit_image_pad_mask'], edit_latents=[conditions['rgb']],
                         use_gradient_checkpointing=False, use_flex_attention=False, kv_cache=None)
    prediction = noise.float() - velocity.float()
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('nonfinite material latent prediction')
    return prediction


def predict_and_decode(model, vae, conditions, cfg, low_vram=False):
    device = conditions['rgb'].device
    try:
        with sequential_cpu_offload(model, model.dit.transformer_blocks, device, enabled=low_vram):
            latent = predict(model, conditions, cfg)
    finally:
        if low_vram:
            # Qwen stores RoPE tables as an ordinary list, outside Module buffers.
            model.dit.pos_embed.freqs = [value.cpu() for value in model.dit.pos_embed.freqs]
            release_device_cache(device)
    with stage_on_device(vae, device, enabled=low_vram), torch.inference_mode():
        decoded = decode_pair(vae, latent)
    return latent, decoded


def output_maps(decoded, square, original, transform, restore_size):
    if decoded.shape[0:3] != (1, 2, 4) or not torch.isfinite(decoded).all():
        raise ValueError('invalid/nonfinite decoded material pair')
    arrays = np.clip(np.rint(decoded[0].detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255), 0, 255).astype(np.uint8)
    diagnostics = [Image.fromarray(array[:, :, 3], 'L') for array in arrays]
    maps = []
    for modality, array in enumerate(arrays):
        color = Image.fromarray(array[:, :, :3], 'RGB')
        coverage = square.getchannel('A')
        if restore_size:
            x, y = transform['offset']; w, h = transform['content_size']
            color = color.crop((x, y, x + w, y + h))
            if color.size != original.size:
                color = color.resize(original.size, Image.Resampling.LANCZOS)
            coverage = original.getchannel('A')  # exact original coverage, no second interpolation
        out = np.array(color.convert('RGBA'))
        out[:, :, 3] = np.array(coverage)
        if modality == 1:
            out[:, :, 2] = 255  # placeholder, not material opacity or AO prediction
        maps.append(Image.fromarray(out, 'RGBA'))
    return maps, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Native adapter_model.safetensors with sibling adapter_config.json, or legacy PT with completion marker')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model-root', type=Path, help='Explicit loading location only; checkpoint model reference retained')
    parser.add_argument('--conditions-cache', type=Path, help='Read RGB conditioning only; sibling JSON must bind input/model/resolution')
    parser.add_argument('--compare-cache', type=Path, help='Native path plus exact conditioning/latent/decoded comparison against cache')
    parser.add_argument('--comparison-atol', type=float, default=0.)
    parser.add_argument('--save-conditions', action='store_true')
    parser.add_argument('--restore-input-size', action='store_true')
    parser.add_argument('--save-predicted-coverage', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--low-vram', action='store_true', default=os.environ.get('SUPRATEX_LOW_VRAM') == '1',
                        help='CPU stage/block offload; preserves weights and sampling, uses more CPU RAM and transfers (also SUPRATEX_LOW_VRAM=1)')
    args = parser.parse_args()
    if args.conditions_cache and args.compare_cache:
        parser.error('--conditions-cache and --compare-cache are mutually exclusive')
    if args.comparison_atol < 0:
        parser.error('--comparison-atol must be nonnegative')
    cfg, weights, step = load_checkpoint(args.checkpoint)
    os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] = cfg['attention_backend']
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if cfg['deterministic_algorithms']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(cfg['deterministic_algorithms'])
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    reference_root = cfg['model_root']
    model_root = Path(os.path.abspath(args.model_root or reference_root))
    # An explicit load-location override is an operator-supplied alias, never an
    # implicit identity rewrite or a digest-based equivalence claim.
    model_references = {reference_root, str(model_root)}
    original, square, transform = prepare_image(args.input, cfg['resolution'])
    if args.output_dir.exists():
        raise FileExistsError('choose a fresh inference output directory')
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    if args.conditions_cache:
        verify_cache_input(args.conditions_cache, args.input, cfg['resolution'], model_references)
        conditions = load_conditions(args.conditions_cache, cfg['resolution'], device)
        from diffsynth.models.qwen_image_21_vae import QwenImage21VAE
        vae = explicit_load(QwenImage21VAE, checkpoint_component(model_root, 'vae'),
                            'cpu' if args.low_vram else device, torch.bfloat16)
    else:
        conditions, vae = native_conditions(square, model_root, cfg['resolution'], device, low_vram=args.low_vram)
    load_cfg = cfg | {'model_root': str(model_root)}
    model = build_model(load_cfg, 'cpu' if args.low_vram else device)
    strict_load_adapter(model, weights)
    del weights
    latent, decoded = predict_and_decode(model, vae, conditions, cfg, low_vram=args.low_vram)
    comparison = None
    if args.compare_cache:
        verify_cache_input(args.compare_cache, args.input, cfg['resolution'], model_references)
        cached = load_conditions(args.compare_cache, cfg['resolution'], device)
        differences = {key: float((conditions[key].float() - cached[key].float()).abs().max()) for key in conditions}
        other_latent, other_decoded = predict_and_decode(model, vae, cached, cfg, low_vram=args.low_vram)
        differences.update(latent=float((latent - other_latent).abs().max()), decoded=float((decoded - other_decoded).abs().max()))
        comparison = dict(max_abs=differences, atol=args.comparison_atol, passed=all(v <= args.comparison_atol for v in differences.values()))
        if not comparison['passed']:
            # Discriminate native/cache disagreement from repeated-kernel noise
            # while holding conditions, checkpoint and process constant.
            repeat_latent, repeat_decoded = predict_and_decode(model, vae, cached, cfg, low_vram=args.low_vram)
            comparison['cached_repeat_control_max_abs'] = dict(
                latent=float((other_latent-repeat_latent).abs().max()),
                decoded=float((other_decoded-repeat_decoded).abs().max()))
        (args.output_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        if not comparison['passed']:
            raise ValueError('native/cache comparison exceeded requested tolerance')
    maps, diagnostics = output_maps(decoded, square, original, transform, args.restore_input_size)
    for label, image, diag in zip(('albedo', 'mra'), maps, diagnostics):
        image.save(args.output_dir / f'{label}.png')
        if args.save_predicted_coverage:
            diag.save(args.output_dir / f'{label}_predicted_coverage.png')
    if args.save_conditions:
        save_file({k: v.detach().cpu().contiguous() for k, v in conditions.items()}, str(args.output_dir / 'conditions.safetensors'))
        (args.output_dir / 'conditions.json').write_text(json.dumps(dict(input=input_binding(args.input), resolution=cfg['resolution'],
            model_root=reference_root, prompt=PROMPT, transform=transform), indent=2) + '\n')
    report = dict(status='complete', checkpoint=os.path.abspath(args.checkpoint), checkpoint_step=step,
                  input=input_binding(args.input), loading_model_root=str(model_root), checkpoint_model_reference=reference_root,
                  model_override_is_operator_reference_not_hash_verified=bool(args.model_root), config=cfg, transform=transform,
                  accepted_model_reference_aliases=sorted(model_references),
                  low_vram=args.low_vram,
                  memory_policy='cpu_stages_and_sequential_transformer_blocks' if args.low_vram else 'resident_models',
                  restore_input_size=args.restore_input_size, restoration='crop padding; LANCZOS color interpolation if size changed; exact original coverage' if args.restore_input_size else 'training square canvas',
                  output_semantics='Albedo sRGB; MRA R=linear Metallic G=linear Roughness B=1 placeholder; A=input coverage',
                  boundary='Opaque material estimator. Input alpha retained as coverage; no segmentation, material transparency or hidden-layer recovery inferred.',
                  path='cached_conditioning_only' if args.conditions_cache else 'native_rgb_vae_and_per_image_vl', comparison=comparison,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  elapsed_seconds=time.perf_counter()-started)
    (args.output_dir / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='complete', output=str(args.output_dir), checkpoint_step=step)))


if __name__ == '__main__':
    main()

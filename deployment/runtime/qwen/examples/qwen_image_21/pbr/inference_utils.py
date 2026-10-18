"""Model construction, local checkpoint loading, and decoding for PBR inference."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file


# Only settings consumed by inference; legacy checkpoint fields are not defaults.
DEFAULTS = dict(
    lora_rank=32, lora_alpha=32, noise_seed=42, resolution=512,
    omit_all_valid_mask=False, bf16_autocast=False,
    attention_backend='torch', deterministic_algorithms=False,
    lora_output_projection=False,
)


PROMPT = ("Generate aligned albedo and metallic/roughness material maps for the object in the input image. "
          "Pack MRA with red=metallic, green=roughness, and blue=constant one placeholder. "
          "No observed ambient occlusion is available.")


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required local file is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_component(root, component):
    """Resolve an explicit component and verify every indexed shard before GPU work."""
    directory = root / component
    config = read_json(directory / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"Expected an object config: {directory / 'config.json'}")
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"Ambiguous safetensors indexes under {directory}")
    if indexes:
        index = read_json(indexes[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map: {indexes[0]}")
        names = sorted(set(weight_map.values()))
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError(f"Indexed shard path leaves component directory: {indexes[0]}")
        files = [directory / name for name in names]
    else:
        pattern = "model*.safetensors" if component == "text_encoder" else "diffusion_pytorch_model*.safetensors"
        files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No local safetensors checkpoint found under {directory}")
    for filename in files:
        if not filename.is_file() or filename.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty checkpoint shard: {filename}")
    keys = set()
    for filename in files:
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
        duplicates = keys & shard_keys
        if duplicates:
            raise ValueError(f"Duplicate state keys between shards: {sorted(duplicates)[:5]}")
        keys.update(shard_keys)
    if indexes and keys != set(weight_map):
        raise ValueError(f"Index and shard state keys disagree under {directory}")
    return {"directory": directory, "config": config, "files": files, "tensor_count": len(keys)}


def explicit_load(model_class, component, device, dtype, converter=None):
    """Instantiate the named class and strictly assign the local tensor state."""
    from diffsynth.core.vram.initialization import skip_model_initialization

    constructor = inspect.signature(model_class.__init__).parameters
    config = {key: value for key, value in component["config"].items() if key in constructor and key != "self"}
    log(f"Loading {model_class.__name__} explicitly ({len(component['files'])} shards, device={device}, dtype={dtype})")
    started = time.perf_counter()
    # Set the default dtype to preserve generated float32 text-encoder rotary buffers;
    # checkpoint parameters are assigned below and then cast explicitly.
    with skip_model_initialization():
        model = model_class(**config)
    state = {}
    for filename in component["files"]:
        state.update(load_file(str(filename), device="cpu"))
    if converter is not None:
        state = converter(state)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device=device, dtype=dtype).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError(f"{model_class.__name__}: parameters remain on meta after strict checkpoint load")
    model.requires_grad_(False)
    log(f"Loaded {model_class.__name__} in {time.perf_counter() - started:.2f}s")
    return model


def endpoint_noise(target, seed):
    # Draw the fixed endpoint field on CPU before moving it to the model device.
    noise = torch.randn(target.shape, generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32)
    return noise.to(device=target.device, dtype=target.dtype)


def build_model(cfg, device):
    from peft import LoraConfig, inject_adapter_in_model
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21_pbr import QwenImage21PBRModel
    dit = explicit_load(QwenImage21DiT, checkpoint_component(Path(cfg['model_root']), 'transformer'), device, torch.bfloat16)
    if len(dit.transformer_blocks) != 32:
        raise ValueError('Expected complete pretrained 32-layer backbone')
    lora = LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], lora_dropout=0., bias='none',
                     target_modules=['attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0']
                     + (['proj_out'] if cfg['lora_output_projection'] else []))
    dit = inject_adapter_in_model(lora, dit)
    model = QwenImage21PBRModel(dit).to(device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def decode_pair(vae, latents, clamp_output=True):
    """Decode the albedo and MRA latent maps in their fixed modality order."""
    decoded = []
    for index in range(2):
        z = latents[:, index].to(dtype=torch.bfloat16)
        value = vae.decode(z, tiled=False, clamp_output=clamp_output)
        decoded.append((value.float() + 1.) / 2.)
    return torch.stack(decoded, dim=1)

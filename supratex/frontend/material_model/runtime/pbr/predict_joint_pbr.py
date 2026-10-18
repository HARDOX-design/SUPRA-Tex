#!/usr/bin/env python
# Adapted for SUPRA-Tex distribution (2026): packaged local inference runtime.
# Upstream FireRed portions remain under Apache-2.0; see LICENSE.
"""Joint albedo/MRA/normal inference for the factorized PBR checkpoint.

The stock QwenImageEditPlusPipeline rejects prompt batches larger than one, so
it cannot exercise modality-axis attention.  This runner keeps the accepted
pipeline preprocessing/scheduler/decoder semantics, but folds one physical
item into the fixed ``albedo, mra, normal`` batch and advances all three streams
together.  Joint checkpoints are trained only at sigma=1, therefore this
runner intentionally accepts one-step, CFG-free inference.  The checkpoint
contract selects the recorded 512/1024 resolution for v2/v3.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(REPO))
MODALITIES = ("albedo", "mra", "normal")
JOINT_ROUTES = ("aligned", "self_slot", "blockdiag")


def log(message: str) -> None:
    print(f"[predict_joint_pbr {time.strftime('%H:%M:%S')}] {message}", flush=True)


def output_path(out_dir: Path, item: dict, modality: str) -> Path:
    return out_dir / f"{item['key']}_{modality}.png"


def load_items(path: Path, limit: int | None) -> list[dict]:
    required = {
        "key",
        "dataset",
        "sha",
        "asset_id",
        "view",
        "cond",
        "albedo_gt",
        "mra_gt",
        "normal_gt",
        "normal_valid",
        "instr_albedo",
        "instr_mra",
        "instr_normal",
        "split",
    }
    items = []
    seen = set()
    with path.open() as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            missing = required - set(item)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            key = item["key"]
            if not isinstance(key, str) or not key or Path(key).name != key or key in {".", ".."} or "\\" in key:
                raise ValueError(f"{path}:{line_number}: key must be a single safe filename component")
            if item["key"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate key {item['key']}")
            if item["split"] != "eval":
                raise ValueError(f"{path}:{line_number}: expected split=eval")
            seen.add(item["key"])
            items.append(item)
    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError(f"no evaluation items in {path}")
    return items


def validate_checkpoint(checkpoint: Path, route: str) -> SimpleNamespace:
    manifest_path = checkpoint / "checkpoint_complete.json"
    contract_path = checkpoint / "joint_contract.json"
    run_contract_path = checkpoint.parent / "RUN_CONTRACT.json"
    adapter_path = checkpoint / "joint_adapter.safetensors"
    full_transformer_path = checkpoint / "diffusion_pytorch_model.safetensors"
    for path in (manifest_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    common_expected = {
        "sites": [29, 39, 49, 59],
        "modalities": list(MODALITIES),
        "rank": 256,
        "heads": 4,
        "training_route": "aligned",
    }
    schema = contract.get("schema")
    if schema == "pbr_joint_checkpoint_v1":
        expected = {
            **common_expected,
            "schema": schema,
            "trainable_tensor_count": 60,
            "trainable_parameter_count": 7_385_088,
        }
        input_size = 512
    elif schema in ("pbr_joint_checkpoint_v2", "pbr_joint_checkpoint_v3"):
        if not run_contract_path.is_file():
            raise FileNotFoundError(run_contract_path)
        run_contract = json.loads(run_contract_path.read_text())
        input_size = run_contract.get("resolution")
        if input_size not in (512, 1024):
            raise ValueError(
                f"unsupported joint-v2/v3 run resolution: {input_size!r}"
            )
        expected = {
            **common_expected,
            "schema": schema,
            "joint_tensor_count": 60,
            "joint_parameter_count": 7_385_088,
            "trainable_modules": ["modality_joint_blocks."],
        }
        backbone_low_lr = [
            f"transformer_blocks.{index}." for index in range(29, 50)
        ]
        if schema == "pbr_joint_checkpoint_v2":
            expected.update(
                low_lr_modules=backbone_low_lr,
                low_lr_ratio=0.0025,
            )
            if contract.get("fresh_joint_w_up_init_std") not in (0.0, 1e-4):
                expected["fresh_joint_w_up_init_std"] = "0.0 or 1e-4"
            residual_mode = "raw"
        else:
            low_lr_modules = contract.get("low_lr_modules")
            if low_lr_modules == []:
                expected.update(low_lr_modules=[], low_lr_ratio=None)
            elif low_lr_modules == backbone_low_lr:
                expected.update(
                    low_lr_modules=backbone_low_lr,
                    low_lr_ratio=0.0025,
                )
            else:
                expected["low_lr_modules"] = "[] or transformer blocks 29 through 49"
            residual_mode = contract.get("residual_mode")
            if residual_mode != "rms_scaled":
                expected["residual_mode"] = "rms_scaled"
            if contract.get("fresh_joint_w_up_init_std") not in (0.0, 3e-3):
                expected["fresh_joint_w_up_init_std"] = "0.0 or 0.003"
            if contract.get("source_joint_adapter_action") not in (
                "fresh", "loaded", "reset"
            ):
                expected["source_joint_adapter_action"] = "fresh, loaded, or reset"
    else:
        raise ValueError(f"unsupported joint checkpoint schema: {schema!r}")
    bad = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if bad:
        raise ValueError(f"joint checkpoint contract mismatch: {bad}")
    if manifest.get("joint_contract") != contract:
        raise ValueError("checkpoint manifest is not bound to joint_contract.json")
    if schema == "pbr_joint_checkpoint_v1":
        if (manifest.get("global_step"), manifest.get("world_size"), manifest.get("seed")) \
                != (2000, 40, 42):
            raise ValueError(f"joint checkpoint step/world/seed mismatch: {manifest}")
    elif (
        manifest.get("format_version") != 2
        or not isinstance(manifest.get("global_step"), int)
        or manifest["global_step"] <= 0
    ):
        raise ValueError(f"joint-v2/v3 checkpoint manifest mismatch: {manifest}")
    if route in JOINT_ROUTES and schema == "pbr_joint_checkpoint_v1":
        if not adapter_path.is_file():
            raise FileNotFoundError(adapter_path)
    if route in JOINT_ROUTES and schema in (
        "pbr_joint_checkpoint_v2", "pbr_joint_checkpoint_v3"
    ):
        if not full_transformer_path.is_file():
            raise FileNotFoundError(full_transformer_path)
    return SimpleNamespace(
        schema=schema,
        input_size=input_size,
        adapter_path=adapter_path if schema == "pbr_joint_checkpoint_v1" else None,
        full_transformer_path=(
            full_transformer_path
            if schema in ("pbr_joint_checkpoint_v2", "pbr_joint_checkpoint_v3")
            else None
        ),
        residual_mode="raw" if schema != "pbr_joint_checkpoint_v3" else residual_mode,
    )


def validate_payloads(items: list[dict]) -> None:
    fields = ("cond", "albedo_gt", "mra_gt", "normal_gt", "normal_valid")
    for item in items:
        missing = [item[field] for field in fields if not Path(item[field]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{item['key']}: missing payload(s): {missing[:3]}"
            )


def load_source_rgb(path: str, size: int):
    from PIL import Image

    image = Image.open(path)
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def _flash_attention_available(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(module, name, None))
        for name in ("flash_attn_func", "flash_attn_varlen_func")
    )


def select_attention_backends(torch, device) -> tuple[str, str]:
    """Resolve inference attention without depending on the parent CLI package.

    The image transformer supports FA3 and native PyTorch SDPA. FA2 can still
    accelerate the text encoder when the image transformer uses native SDPA.
    An explicit SDPA selection applies to both components.
    """
    selected = os.environ.get("ATTN_BACKEND", "auto")
    if selected == "sdpa":
        return "native", "sdpa"
    supported = ("auto", "flash_attn_3", "flash_attn", "xformers", "naive", "flash_attn_4")
    if selected not in supported:
        raise ValueError(f"unsupported ATTN_BACKEND for FireRed: {selected!r}")

    device = torch.device(device)
    capability = (
        torch.cuda.get_device_capability(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else (0, 0)
    )
    use_fa3 = (
        selected in ("auto", "flash_attn_3")
        and capability[0] == 9
        and _flash_attention_available("flash_attn_interface")
    )
    if selected == "flash_attn_3" and not use_fa3:
        raise RuntimeError(
            "ATTN_BACKEND=flash_attn_3 requires Hopper (SM90) and an importable "
            "flash_attn_interface extension; select sdpa for native attention"
        )
    use_fa2 = capability[0] in (8, 9) and _flash_attention_available("flash_attn")
    if selected == "flash_attn" and not use_fa2:
        raise RuntimeError(
            "ATTN_BACKEND=flash_attn requires an Ampere/Ada/Hopper GPU and an "
            "importable flash_attn extension; select sdpa for native attention"
        )
    return "fa3" if use_fa3 else "native", "flash_attention_2" if use_fa2 else "sdpa"


def load_pipeline(args, device):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    image_backend, text_backend = select_attention_backends(torch, device)
    from pbr.model_loading import (
        _load_base_transformer,
        _load_exact_full_transformer,
        _load_exact_joint_adapter,
    )
    from pbr.qwen_pbr_factorized import (
        PBR_ATTENTION_FA3,
        PBR_ATTENTION_NATIVE,
        PRODUCTION_JOINT_HEADS,
        PRODUCTION_JOINT_MODALITIES,
        PRODUCTION_JOINT_RANK,
        PRODUCTION_JOINT_SITES,
    )

    model_args = SimpleNamespace(pretrained_model_name_or_path=str(args.base_model))
    transformer = _load_base_transformer(model_args, torch.bfloat16)
    if args.route == "base" or args.joint_spec.schema == "pbr_joint_checkpoint_v1":
        _load_exact_full_transformer(
            transformer,
            args.transformer_path,
            allow_joint_sidecar=args.route == "base",
        )
    if args.route in JOINT_ROUTES:
        transformer.attach_joint_modules(
            sites=PRODUCTION_JOINT_SITES,
            modalities=PRODUCTION_JOINT_MODALITIES,
            rank=PRODUCTION_JOINT_RANK,
            heads=PRODUCTION_JOINT_HEADS,
            residual_mode=args.joint_spec.residual_mode,
        )
        if args.joint_spec.schema == "pbr_joint_checkpoint_v1":
            _load_exact_joint_adapter(transformer, args.joint_spec.adapter_path)
        else:
            _load_exact_full_transformer(
                transformer, args.joint_spec.full_transformer_path
            )
    transformer.set_pbr_attention_backend(
        PBR_ATTENTION_FA3 if image_backend == "fa3" else PBR_ATTENTION_NATIVE,
        device=device,
        dtype=torch.bfloat16,
    )
    transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    transformer.eval()

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(args.base_model),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config["shift_terminal"] = None
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    pipe.to(device)
    pipe.text_encoder.set_attn_implementation(text_backend)
    pipe.set_progress_bar_config(disable=True)
    log(f"attention image={image_backend} text={text_backend}")
    return pipe


def make_generators(torch, device, seed: int):
    # Keep per-modality noise independent, deterministic, and invariant to GPU
    # sharding.  All routes use these exact three streams.
    offsets = (104_729, 209_759, 314_159)
    return [
        torch.Generator(device=device).manual_seed(seed + offset)
        for offset in offsets
    ]


def encode_source(pipe, source, device, dtype, seed: int):
    import numpy as np
    import torch

    pixels = torch.from_numpy(
        np.asarray(source, dtype=np.float32).copy()
    ).permute(2, 0, 1).unsqueeze(0)
    pixels = (pixels / 127.5 - 1.0).to(device=device, dtype=dtype).unsqueeze(2)
    generator = torch.Generator(device=device).manual_seed(seed)
    source_latents = pipe._encode_vae_image(pixels, generator=generator)
    _, channels, _, height, width = source_latents.shape
    return pipe._pack_latents(source_latents, 1, channels, height, width)


def encode_prompts(pipe, item: dict, source, input_size: int, device, dtype):
    from pbr.conditioning import get_embedding
    from pbr.conditioning import resize_source_images_for_condition

    prompts = [item[f"instr_{modality}"] for modality in MODALITIES]
    source_lists = resize_source_images_for_condition(
        [[source] for _ in MODALITIES],
        ref_height=input_size,
        ref_width=input_size,
        image_sample_size=input_size,
    )
    embeds, mask = get_embedding(
        True,
        prompts,
        source_lists,
        pipe.processor,
        pipe.text_encoder,
    )
    return embeds.to(device=device, dtype=dtype), mask.to(device=device)


def generate_triplet(pipe, item: dict, args, device):
    import numpy as np
    import torch

    from pbr.conditioning import calculate_shift

    source = load_source_rgb(item["cond"], args.input_size)
    dtype = pipe.transformer.dtype
    prompt_embeds, prompt_mask = encode_prompts(
        pipe, item, source, args.input_size, device, dtype
    )
    source_latents = encode_source(pipe, source, device, dtype, args.seed)
    source_latents = source_latents.expand(len(MODALITIES), -1, -1)

    channels = pipe.transformer.config.in_channels // 4
    generators = make_generators(torch, device, args.seed)
    latents, _ = pipe.prepare_latents(
        images=None,
        batch_size=len(MODALITIES),
        num_channels_latents=channels,
        height=args.input_size,
        width=args.input_size,
        dtype=dtype,
        device=device,
        generator=generators,
        latents=None,
    )
    target_seq_len = latents.shape[1]
    grid = args.input_size // pipe.vae_scale_factor // 2
    img_shapes = [[(1, grid, grid), (1, grid, grid)] for _ in MODALITIES]

    sigmas = np.linspace(1.0, 1.0, 1)
    mu = calculate_shift(
        target_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    pipe.scheduler.set_begin_index(0)

    with torch.inference_mode():
        for timestep in pipe.scheduler.timesteps:
            hidden_states = torch.cat([latents, source_latents], dim=1)
            model_timestep = timestep.expand(len(MODALITIES)).to(dtype) / 1000
            transformer_kwargs = {
                "hidden_states": hidden_states,
                "timestep": model_timestep,
                "guidance": None,
                "encoder_hidden_states_mask": prompt_mask,
                "encoder_hidden_states": prompt_embeds,
                "img_shapes": img_shapes,
                "attention_kwargs": {},
                "return_dict": False,
            }
            if args.route in JOINT_ROUTES:
                transformer_kwargs.update(
                    pbr_joint_batch_size=1,
                    pbr_joint_target_seq_len=target_seq_len,
                    pbr_joint_route=args.route,
                )
            with pipe.transformer.cache_context("cond"):
                prediction = pipe.transformer(**transformer_kwargs)[0]
            prediction = prediction[:, :target_seq_len]
            latents = pipe.scheduler.step(
                prediction, timestep, latents, return_dict=False
            )[0]

        unpacked = pipe._unpack_latents(
            latents, args.input_size, args.input_size, pipe.vae_scale_factor
        ).to(pipe.vae.dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(unpacked.device, unpacked.dtype)
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked / std + mean
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0][:, :, 0]
        images = pipe.image_processor.postprocess(decoded, output_type="pil")
    if len(images) != len(MODALITIES):
        raise RuntimeError(f"decoder returned {len(images)} images, expected 3")
    return images


def save_triplet(out_dir: Path, item: dict, images) -> None:
    for modality, image in zip(MODALITIES, images):
        destination = output_path(out_dir, item, modality)
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}.png")
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--items-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--route", choices=("base",) + JOINT_ROUTES, required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")

    for path in (args.base_model / "model_index.json", args.transformer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.joint_spec = validate_checkpoint(args.joint_checkpoint, args.route)
    if (
        args.input_size != args.joint_spec.input_size
        or args.steps != 1
        or args.cfg != 1.0
    ):
        parser.error(
            f"{args.joint_spec.schema} evaluation is pinned to "
            f"--input-size {args.joint_spec.input_size} --steps 1 --cfg 1"
        )
    items = load_items(args.items_jsonl, args.limit)
    validate_payloads(items)
    shard_items = items[args.shard_index::args.num_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        item for item in shard_items
        if not all(output_path(args.out_dir, item, modality).stat().st_size > 0
                   if output_path(args.out_dir, item, modality).is_file() else False
                   for modality in MODALITIES)
    ]
    log(
        f"route={args.route} shard={args.shard_index}/{args.num_shards} "
        f"items={len(shard_items)} pending={len(pending)} seed={args.seed} "
        f"protocol={args.input_size}/s1/cfg1 schema={args.joint_spec.schema}"
    )
    if args.dry_run:
        for item in shard_items[:10]:
            log(f"DRY {item['key']} -> {args.out_dir}")
        log("dry run complete; no model or GPU was loaded")
        return
    if not pending:
        log("nothing to do")
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    log(f"loading model on {torch.cuda.get_device_name(device)}")
    pipe = load_pipeline(args, device)
    log("model ready")

    started = time.time()
    for index, item in enumerate(pending, 1):
        images = generate_triplet(pipe, item, args, device)
        save_triplet(args.out_dir, item, images)
        if index == 1 or index % 10 == 0 or index == len(pending):
            elapsed = time.time() - started
            rate = elapsed / index
            eta = rate * (len(pending) - index)
            log(
                f"{index}/{len(pending)} key={item['key']} "
                f"avg={rate:.2f}s/item eta={eta / 60:.1f}min"
            )

    manifest = {
        "schema": "joint_pbr_prediction_shard_v1",
        "route": args.route,
        "resolution": args.input_size,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "items_jsonl": str(args.items_jsonl.absolute()),
        "joint_checkpoint": str(args.joint_checkpoint.absolute()),
        "joint_checkpoint_schema": args.joint_spec.schema,
        "base_transformer": str(args.transformer_path.absolute()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_items": len(shard_items),
        "n_outputs": len(shard_items) * len(MODALITIES),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    manifest_path = args.out_dir / (
        f"prediction_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    )
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    log(f"DONE outputs={manifest['n_outputs']} manifest={manifest_path}")


if __name__ == "__main__":
    main()

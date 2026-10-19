"""Device-only offload for frozen, serial endpoint inference.

Weights are loaded on CPU before using these contexts. Only device placement
changes: parameter/buffer dtypes, forward calls, and activation tensors stay
unchanged. CPU RAM must hold the full model; transfers trade latency for VRAM.
"""
from contextlib import contextmanager
import gc

import torch


def release_device_cache(device):
    gc.collect()
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


@contextmanager
def stage_on_device(model, device, enabled=True):
    """Keep a model on the execution device for one encode/decode stage."""
    if not enabled:
        yield
        return
    try:
        model.to(device=device)
        yield
    finally:
        model.to(device='cpu')
        release_device_cache(device)


@contextmanager
def sequential_cpu_offload(model, blocks, device, enabled=True):
    """Run one selected subtree at a time on device; keep the rest resident.

    Selected modules must be disjoint and called through ``Module.__call__``.
    Their descendants (including LoRA adapters) move together, without dtype
    conversion. Parameters outside the selected subtrees stay on device until
    the stage ends. Forward hooks preserve module identity and state-dict keys.
    This deliberately supports frozen inference only, without prefetching or
    concurrent/reentrant calls.
    """
    if not enabled:
        yield
        return
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('sequential CPU offload requires a frozen inference model')
    blocks = tuple(blocks)
    selected = {id(block) for block in blocks}
    members = {id(module) for module in model.modules()}
    if len(selected) != len(blocks) or not selected <= members:
        raise ValueError('offload blocks must be distinct submodules of the model')
    if any(id(child) in selected for block in blocks for child in tuple(block.modules())[1:]):
        raise ValueError('offload blocks must not contain one another')

    def move_resident(module):
        if id(module) in selected:
            return
        # Module.to() on a parent would also move all selected blocks at once.
        module._apply(lambda tensor: tensor.to(device=device), recurse=False)
        for child in module.children():
            move_resident(child)

    def onload(module, _args):
        if torch.is_grad_enabled():
            raise RuntimeError('sequential CPU offload requires inference_mode or no_grad')
        module.to(device=device)

    def offload(module, _args, _output):
        module.to(device='cpu')

    handles = []
    try:
        model.to(device='cpu')
        for block in blocks:
            handles.append(block.register_forward_pre_hook(onload))
            handles.append(block.register_forward_hook(offload, always_call=True))
        move_resident(model)
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.to(device='cpu')
        release_device_cache(device)


def text_encoder_offload_blocks(encoder):
    """Qwen3-VL stage units; the unused vocabulary head stays on CPU."""
    backbone = encoder.model.model
    language = backbone.language_model
    return (
        backbone.visual, language.embed_tokens, *language.layers,
        language.norm, language.rotary_emb, encoder.model.lm_head,
    )

"""Strict native-checkpoint inference for static-O Interleaved flow transformer texture flow."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch

from ..models.interleaved_flow_transformer import InterleavedFlowTransformer
from .samplers.flow_euler import FlowEulerSampler


_NORMALIZATION_KEYS = ('partial_mean', 'partial_std', 'pbr_mean', 'pbr_std')


def build_completion_model(
    config: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor], *,
    assign_weights: bool = False,
) -> InterleavedFlowTransformer:
    """Strict-load a native static-O Interleaved flow transformer state.

    ``config`` must already have passed ``supratex.inference.pipeline.load_contract``,
    which checks the released model ABI and normalization statistics.
    """
    args = dict(config['models']['denoiser']['args'])
    # The inference loader already places checkpoint tensors on the target
    # device. A meta shell avoids allocating/initializing a second full model
    # and copying its weights through CPU memory before moving it back to CUDA.
    with torch.device('meta') if assign_weights else nullcontext():
        model = InterleavedFlowTransformer(**args)
    target = model.state_dict()
    missing = sorted(set(target) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(target))
    invalid = sorted(
        key for key in set(target) & set(state_dict)
        if not isinstance(state_dict[key], torch.Tensor)
        or state_dict[key].is_meta
        or state_dict[key].shape != target[key].shape
        or state_dict[key].dtype != target[key].dtype
    )
    if missing or unexpected or invalid:
        raise RuntimeError(
            'Invalid native Interleaved flow transformer checkpoint before load: '
            f'missing={missing}, unexpected={unexpected}, type_shape_dtype_mismatch={invalid}'
        )
    # Normalization buffers are stored in the checkpoint; they must equal the
    # config statistics that the rest of the pipeline uses.
    mismatched_normalization = [
        key for key in _NORMALIZATION_KEYS
        if not torch.equal(state_dict[key].detach().cpu(), torch.as_tensor(args[key], dtype=torch.float32))
    ]
    if mismatched_normalization:
        raise RuntimeError(
            'Invalid native Interleaved flow transformer checkpoint before load: '
            f'normalization_mismatch={mismatched_normalization}'
        )
    model.load_state_dict(state_dict, strict=True, assign=assign_weights)
    if assign_weights:
        from ..modules.sparse.attention.rope import SparseRotaryPositionEmbedder
        for module in model.modules():
            if isinstance(module, SparseRotaryPositionEmbedder):
                # RoPE frequencies are a plain attribute, absent from the
                # checkpoint and nn.Module.to(); materialize them explicitly.
                frequencies = torch.arange(module.freq_dim, dtype=torch.float32, device='cpu') / module.freq_dim
                module.freqs = module.rope_freq[0] / (module.rope_freq[1] ** frequencies)
    return model.eval()


def sample_material_latents(
    model: InterleavedFlowTransformer,
    noise: Any,
    *,
    concat_cond: Any,
    partial_z_obs: Any,
    partial_z_obs_keep_mask: Any = None,
    steps: int = 25,
    rescale_t: float = 3.0,
    sigma_min: float = 1e-5,
    use_observation_cache: bool = False,
    verbose: bool = False,
) -> Any:
    """Sample static-O Interleaved flow transformer with native Euler settings and an opt-in O K/V cache."""
    sampler = FlowEulerSampler(sigma_min=sigma_min)
    autocast = (torch.autocast('cuda', dtype=torch.bfloat16)
                if getattr(noise, 'device', torch.device('cpu')).type == 'cuda' else nullcontext())
    with torch.no_grad(), autocast:
        cache_context = (model.observation_cache(noise, concat_cond, partial_z_obs,
                                                 partial_z_obs_keep_mask)
                         if use_observation_cache else nullcontext(None))
        with cache_context as observation_cache:
            return sampler.sample(
                model, noise, cond=None, steps=steps, rescale_t=rescale_t,
                verbose=verbose, concat_cond=concat_cond, partial_z_obs=partial_z_obs,
                partial_z_obs_keep_mask=partial_z_obs_keep_mask,
                observation_cache=observation_cache,
            )

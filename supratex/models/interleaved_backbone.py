"""Packed joint O/X attention for the fixed-stride observation attention-ILV policy.

This is a distinct architecture with the same observation attention parameter layout and V11
loader interface but a different block mapping, not an equivalent optimization. Observation
state is static at step zero, then depends on current X after its first joint
refresh at a nonterminal even block, so cross-denoising O K/V caches are
deliberately unsupported.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..modules import sparse as sp
from ..modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from .observation_attention import _ObservationCache, _PSABlock
from .efficient_attention import EfficientObservationAttentionModel


class InterleavedBackbone(EfficientObservationAttentionModel):
    """observation attention Fast with fixed stride-two packed joint O/X refreshes.

    Even nonterminal blocks run one varlen attention over each sample's
    ``[O; X]`` sequence and split its output back into O and X rows.  Odd
    blocks retain the directed X query over that same sample-local K/V set,
    without updating O or its MLP.  This class never materializes or consumes
    an observation cache.
    """
    def __init__(self, direct_modulation_broadcast: bool = False,
                 safe_empty_cache: bool = False, terminal_o_kv_only: bool = True,
                 o_refresh_stride: int = 2, compact_observation_cache: bool = False,
                 hot_observation_cache: bool = False, use_observation_cache: bool = False,
                 v11_init_policy: str = 'odd_copy_even_identity',
                 **kwargs: Any):
        if v11_init_policy != 'odd_copy_even_identity':
            raise ValueError("interleaved backbone fixes v11_init_policy='odd_copy_even_identity'")
        if not terminal_o_kv_only:
            raise ValueError('interleaved backbone fixes terminal_o_kv_only=True')
        if o_refresh_stride != 2:
            raise ValueError('interleaved backbone fixes o_refresh_stride=2')
        for name, value in (
            ('compact_observation_cache', compact_observation_cache),
            ('hot_observation_cache', hot_observation_cache),
            ('use_observation_cache', use_observation_cache),
        ):
            if value:
                raise ValueError(f'interleaved backbone does not support {name}=True')
        # This is deliberately not module state: observation attention checkpoints retain their
        # state_dict ABI, while a load has one frozen, reviewable expansion rule.
        self.v11_init_policy = v11_init_policy
        super().__init__(
            direct_modulation_broadcast=direct_modulation_broadcast,
            safe_empty_cache=safe_empty_cache,
            terminal_o_kv_only=True,
            o_refresh_stride=2,
            **kwargs,
        )

    def initialize_weights(self) -> None:
        """Initialize active odd blocks and identity-inserted even blocks.

        The observation attention base initializes its former tail as identity blocks.  ILV
        instead places the active V11 copies at odd positions, so restore the
        ordinary residual-output initialization for odd tail blocks and zero
        exactly the two residual outputs of every even inserted block.
        """
        super().initialize_weights()
        output_std = math.sqrt(2.0 / (5.0 * self.model_channels))
        for index, block in enumerate(self.blocks):
            if index % 2:
                nn.init.normal_(block.self_attn.to_out.weight, std=output_std)
                nn.init.zeros_(block.self_attn.to_out.bias)
                nn.init.normal_(block.mlp.mlp[2].weight, std=output_std)
                nn.init.zeros_(block.mlp.mlp[2].bias)
            else:
                nn.init.zeros_(block.self_attn.to_out.weight)
                nn.init.zeros_(block.self_attn.to_out.bias)
                nn.init.zeros_(block.mlp.mlp[2].weight)
                nn.init.zeros_(block.mlp.mlp[2].bias)

    def migrate_v11_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Strictly expand V11's 30 blocks into ILV identity/copy pairs.

        Parent validation remains the sole source classifier: it rejects every
        unknown tensor, shape mismatch, and V11 block beyond the frozen source
        depth before it writes.  Once that validation succeeds, this method
        remaps directly from the untouched source dictionary, preventing an
        in-place 0 -> 1, 1 -> 3 overlap from corrupting a later source block.
        """
        if self.num_blocks != 60:
            raise RuntimeError('interleaved backbone V11 migration requires the formal 60-block target')
        inventory = super().migrate_v11_state_dict(state_dict)
        target = self.state_dict()
        remapped: Dict[str, torch.Tensor] = {}
        seeded: List[str] = []
        zeroed: List[str] = []

        for source_key in inventory['copied']:
            match = re.match(r'^blocks\.(\d+)\.(.*)$', source_key)
            if match is None:
                continue
            source_index, suffix = int(match.group(1)), match.group(2)
            if source_index >= 30:
                # The parent classifier has already made this impossible; keep
                # this local invariant explicit beside the ILV index arithmetic.
                raise RuntimeError(f'unclassified V11 tensor outside frozen 30-block source: {source_key}')
            suffix = suffix.replace('norm3.', 'norm2.')
            for destination_index in (2 * source_index, 2 * source_index + 1):
                destination = f'blocks.{destination_index}.{suffix}'
                if destination not in target:
                    raise RuntimeError(f'ILV destination is missing: {destination}')
                remapped[destination] = state_dict[source_key]

        for index in range(0, self.num_blocks, 2):
            for suffix in (
                'self_attn.to_out.weight', 'self_attn.to_out.bias',
                'mlp.mlp.2.weight', 'mlp.mlp.2.bias',
            ):
                destination = f'blocks.{index}.{suffix}'
                remapped[destination] = torch.zeros_like(target[destination])
                zeroed.append(destination)
            seeded.extend(
                key for key in target if key.startswith(f'blocks.{index}.') and key not in zeroed
            )

        self.load_state_dict(remapped, strict=False)
        loaded = set(remapped)
        loaded.update(key for key in inventory['copied'] if not key.startswith('blocks.'))
        return {
            'copied': inventory['copied'],
            'discarded': inventory['discarded'],
            'seeded': sorted(seeded),
            'zeroed': sorted(zeroed),
            'new': sorted(set(target) - loaded),
        }

    def load_v11_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        return self.migrate_v11_state_dict(state_dict)

    def _cache_signature(self, x: sp.SparseTensor, shape: sp.SparseTensor,
                         observed: sp.SparseTensor, keep: Optional[torch.Tensor]
                         ) -> Tuple[Any, ...]:
        """Tag the class should a caller inspect identity without creating a cache."""
        return super()._cache_signature(x, shape, observed, keep) + (('psailv_cache_disabled',),)

    def observation_cache(self, x: sp.SparseTensor, concat_cond: sp.SparseTensor,
                          partial_z_obs: sp.SparseTensor,
                          partial_z_obs_keep_mask: Optional[torch.Tensor] = None) -> _ObservationCache:
        raise RuntimeError('interleaved backbone disables observation_cache; packed joint attention requires uncached O K/V')

    @staticmethod
    def _joined_layout(o_layout: List[slice], x_layout: List[slice]) -> List[slice]:
        start = 0
        result = []
        for os, xs in zip(o_layout, x_layout):
            stop = start + (os.stop - os.start) + (xs.stop - xs.start)
            result.append(slice(start, stop))
            start = stop
        return result

    def _attention(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor,
                   o_layout: List[slice], x: torch.Tensor, x_tensor: sp.SparseTensor,
                   cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, *,
                   compute_o_attention: Optional[bool] = None, **_: Any
                   ) -> Tuple[Optional[torch.Tensor], torch.Tensor,
                              Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run the fixed ILV attention policy without a baseline O->O fallback."""
        if cached_kv is not None:
            raise RuntimeError('interleaved backbone disables observation_cache; cached K/V consumption is unsupported')
        if compute_o_attention is None:
            raise RuntimeError('interleaved backbone requires an explicit O-attention policy from its block scheduler')

        qx, kx, vx = self._project_qkv(block, x, x_tensor.coords, x_tensor.layout)
        if len(o):
            qo, ko, vo = self._project_qkv(block, o, o_coords, o_layout)
        else:
            qo = ko = vo = None

        joint_layout = self._joined_layout(o_layout, x_tensor.layout)
        joint_k = []
        joint_v = []
        joint_q = []
        for os, xs in zip(o_layout, x_tensor.layout):
            if len(o):
                joint_k.append(torch.cat((ko.feats[os], kx.feats[xs]), dim=0))
                joint_v.append(torch.cat((vo.feats[os], vx.feats[xs]), dim=0))
                if compute_o_attention:
                    joint_q.append(torch.cat((qo.feats[os], qx.feats[xs]), dim=0))
            else:
                joint_k.append(kx.feats[xs])
                joint_v.append(vx.feats[xs])
                if compute_o_attention:
                    joint_q.append(qx.feats[xs])
        kv = sp.VarLenTensor(torch.stack((torch.cat(joint_k, dim=0), torch.cat(joint_v, dim=0)), dim=1), joint_layout)

        if not compute_o_attention:
            x_out = sparse_scaled_dot_product_attention(qx, kv).feats
            return None, block.self_attn.to_out(x_out.reshape(len(x_out), self.model_channels)), None

        joint_q_tensor = sp.VarLenTensor(torch.cat(joint_q, dim=0), joint_layout)
        joint_out = sparse_scaled_dot_product_attention(joint_q_tensor, kv).feats
        o_parts, x_parts = [], []
        start = 0
        for os, xs in zip(o_layout, x_tensor.layout):
            o_count, x_count = os.stop - os.start, xs.stop - xs.start
            o_parts.append(joint_out[start:start + o_count])
            x_parts.append(joint_out[start + o_count:start + o_count + x_count])
            start += o_count + x_count
        o_out = torch.cat(o_parts, dim=0)
        x_out = torch.cat(x_parts, dim=0)
        return (block.self_attn.to_out(o_out.reshape(len(o_out), self.model_channels)),
                block.self_attn.to_out(x_out.reshape(len(x_out), self.model_channels)), None)

    def _block(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor,
               o_layout: List[slice], x: torch.Tensor, x_tensor: sp.SparseTensor,
               mod_x: torch.Tensor, mod_o: torch.Tensor, terminal: bool,
               cache: Optional[_ObservationCache] = None, index: int = 0
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache is not None:
            raise RuntimeError('interleaved backbone disables observation_cache; cache consumption is unsupported')
        return super()._block(block, o, o_coords, o_layout, x, x_tensor, mod_x, mod_o,
                              terminal, None, index)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Any = None,
                concat_cond: Optional[sp.SparseTensor] = None,
                partial_z_obs: Optional[sp.SparseTensor] = None,
                partial_z_obs_keep_mask: Optional[torch.Tensor] = None,
                observation_cache: Optional[_ObservationCache] = None, **kwargs: Any) -> sp.SparseTensor:
        if observation_cache is not None:
            raise RuntimeError('interleaved backbone disables observation_cache; pass observation_cache=None')
        for name in ('use_observation_cache', 'compact_observation_cache', 'hot_observation_cache'):
            if name in kwargs:
                value = kwargs.pop(name)
                if value is not False:
                    raise RuntimeError(f'interleaved backbone disables {name}; O depends on current X')
        if self.o_refresh_stride != 2 or self.terminal_o_kv_only is not True:
            raise RuntimeError('interleaved backbone requires its fixed stride2/terminal-KV-only policy')
        return super().forward(x, t, cond, concat_cond, partial_z_obs,
                               partial_z_obs_keep_mask, None, **kwargs)

"""Interleaved flow transformer: alternate full-X and U-only updates with static observations.

The observation attention-ILV weights remain a 60-block backbone. O is always read-only;
the production policy uses the original even blocks for U-only updates and
the original odd blocks for full-X updates. Explicitly configured,
identity-seeded post-block refiners remain available for engineering tests.
No O-update schedule or transition phase exists.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..modules import sparse as sp
from ..modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from ..modules.utils import convert_module_to, manual_cast
from .observation_attention import _PSABlock, _join_exact_rows, _layouts_from_batch, _live_zero
from .interleaved_backbone import InterleavedBackbone


@dataclass
class _PreparedObservation:
    hidden: torch.Tensor
    coords: torch.Tensor
    layout: List[slice]
    urows: torch.Tensor
    ucoords: torch.Tensor
    ubatch: torch.Tensor
    u_layout: List[slice]


class _StaticObservationCache:
    """Invocation-local, eval-only cache for static O preparation and layer K/V."""
    def __init__(self, model: 'InterleavedFlowTransformer', x: sp.SparseTensor, shape: sp.SparseTensor,
                 observed: sp.SparseTensor, keep: Optional[torch.Tensor]):
        if model.training or torch.is_grad_enabled():
            raise RuntimeError('observation_cache is inference-only; training cannot retain detached O K/V')
        tensors = {
            'x.coords': x.coords, 'concat_cond.coords': shape.coords, 'concat_cond.feats': shape.feats,
            'partial_z_obs.coords': observed.coords, 'partial_z_obs.feats': observed.feats,
        }
        if keep is not None:
            tensors['partial_z_obs_keep_mask'] = keep
        inference = next((name for name, tensor in tensors.items() if torch.is_inference(tensor)), None)
        if inference is not None:
            raise RuntimeError(
                f'observation_cache cannot safely validate inference-mode tensor {inference}; '
                'pass regular tensors under torch.no_grad()'
            )
        model_tensor = next(
            (name for name, tensor in [*model.named_parameters(), *model.named_buffers()]
             if torch.is_inference(tensor)),
            None,
        )
        if model_tensor is not None:
            raise RuntimeError(
                f'observation_cache cannot safely validate inference-mode model tensor {model_tensor}; '
                'construct the model outside torch.inference_mode()'
            )
        self.model = model
        self.signature = model._cache_signature(x, shape, observed, keep)
        self.kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.prepared: Optional[_PreparedObservation] = None
        self.zero_time: Optional[torch.Tensor] = None
        self.zero_time_signature: Optional[Tuple[Any, ...]] = None
        self.bytes = 0
        self.closed = False

    @property
    def preprocessing_bytes(self) -> int:
        """Resident preparation tensors, separate from the existing K/V byte count."""
        tensors = [] if self.prepared is None else [
            self.prepared.hidden, self.prepared.coords, self.prepared.urows,
            self.prepared.ucoords, self.prepared.ubatch,
        ]
        if self.zero_time is not None:
            tensors.append(self.zero_time)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def validate(self, x: sp.SparseTensor, shape: sp.SparseTensor, observed: sp.SparseTensor,
                 keep: Optional[torch.Tensor]) -> None:
        if self.closed or self.model.training or torch.is_grad_enabled():
            raise RuntimeError('observation_cache is closed or used outside eval/no_grad')
        if self.signature != self.model._cache_signature(x, shape, observed, keep):
            raise RuntimeError('observation_cache inputs, model weights, dtype, device, or policy changed; rebuild the context')

    def get(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.kv.get(key)

    def record(self, key: str, k: torch.Tensor, v: torch.Tensor) -> None:
        if key not in self.kv:
            self.kv[key] = (k.detach(), v.detach())
            self.bytes += (k.numel() * k.element_size()) + (v.numel() * v.element_size())

    def close(self) -> None:
        self.kv.clear()
        self.prepared = None
        self.zero_time = None
        self.zero_time_signature = None
        self.closed = True

    def __enter__(self) -> '_StaticObservationCache':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class InterleavedFlowTransformer(InterleavedBackbone):
    """Inherited backbone with permanently static O and optional U refiners.

    O receives no Q projection, attention output, or MLP execution. It still
    receives its layer-specific normalization/modulation and K/V projection so
    every X layer attends over that layer's ``[O; X]`` keys and values.
    """
    def __init__(self, refinement_indices: Sequence[int] = (),
                 base_block_policy: str = 'alternating_u_x',
                 allow_test_config: bool = False,
                 direct_modulation_broadcast: bool = True, **kwargs: Any):
        retired = {'o_update_scale', 'o_transition_steps'} & kwargs.keys()
        if retired:
            raise ValueError(f'Interleaved flow transformer O is always static; removed options: {sorted(retired)}')
        if base_block_policy not in {'alternating_u_x', 'all_x'}:
            raise ValueError("base_block_policy must be 'alternating_u_x' or 'all_x'")
        super().__init__(direct_modulation_broadcast=direct_modulation_broadcast, **kwargs)
        if self.num_blocks % 2:
            raise ValueError('Interleaved flow transformer requires an even number of base blocks')
        if self.num_blocks != 60 and not allow_test_config:
            raise ValueError('non-60-block Interleaved flow transformer test configurations require allow_test_config=True')
        self.base_block_policy = base_block_policy
        indices = self._validate_refinement_indices(refinement_indices)
        self.refinement_indices = tuple(indices)
        self.refiners = nn.ModuleDict({str(index): self._new_refiner(index) for index in indices})

    def _validate_refinement_indices(self, indices: Sequence[int]) -> Tuple[int, ...]:
        result = tuple(indices)
        if tuple(sorted(set(result))) != result or any(index < 0 or index >= self.num_blocks for index in result):
            raise ValueError('refinement_indices must be unique, sorted, valid zero-based base-block indices')
        return result

    def _new_refiner(self, index: int) -> _PSABlock:
        # The attention module does not retain a public rope-frequency field;
        # a structural clone is therefore the exact construction contract.
        block = copy.deepcopy(self.blocks[index])
        with torch.no_grad():
            nn.init.zeros_(block.self_attn.to_out.weight)
            nn.init.zeros_(block.self_attn.to_out.bias)
            nn.init.zeros_(block.mlp.mlp[2].weight)
            nn.init.zeros_(block.mlp.mlp[2].bias)
        return block

    def convert_to(self, dtype: torch.dtype) -> None:
        """Keep post-parent refiners on the same explicit inference dtype."""
        super().convert_to(dtype)
        if hasattr(self, 'refiners'):
            self.refiners.apply(lambda module: convert_module_to(module, dtype=dtype))

    def _cache_signature(self, x: sp.SparseTensor, shape: sp.SparseTensor, observed: sp.SparseTensor,
                         keep: Optional[torch.Tensor]) -> Tuple[Any, ...]:
        """Extend the parent identity signature with static-O cache policy."""
        return super()._cache_signature(x, shape, observed, keep) + (
            ('interleaved_flow_transformer_static_o_kv', self.base_block_policy, self.direct_modulation_broadcast, self.dtype,
             torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
             torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu')),
        )

    def observation_cache(self, x: sp.SparseTensor, concat_cond: sp.SparseTensor,
                          partial_z_obs: sp.SparseTensor,
                          partial_z_obs_keep_mask: Optional[torch.Tensor] = None) -> _StaticObservationCache:
        """Create an invocation-local cache for static O preparation and K/V."""
        return _StaticObservationCache(self, x, concat_cond, partial_z_obs, partial_z_obs_keep_mask)

    def _prepare_observation(self, x: sp.SparseTensor, shape: sp.SparseTensor,
                             observed: sp.SparseTensor, keep: Optional[torch.Tensor]) -> _PreparedObservation:
        """Prepare the time-independent O stream and the complementary U support."""
        if not torch.equal(shape.coords, x.coords):
            raise AssertionError('Interleaved flow transformer requires coordinate-aligned PBR32 x and shape concat_cond')
        batch_size = x.shape[0]
        keep = torch.ones(batch_size, device=x.device, dtype=torch.bool) if keep is None else keep.to(x.device, torch.bool)
        if keep.shape != (batch_size,):
            raise AssertionError(f'partial_z_obs_keep_mask must be [B={batch_size}]')
        active = keep[observed.coords[:, 0].long()]
        ocoords, ofeats = observed.coords[active], observed.feats[active]
        rows = _join_exact_rows(x.coords, ocoords)
        o_layout = _layouts_from_batch(ocoords[:, 0], batch_size)
        raw = ofeats[:, :32].float() * self.partial_std + self.partial_mean
        opbr = (raw - self.pbr_mean) / self.pbr_std
        oinput = torch.cat((opbr.to(ofeats.dtype), shape.feats[rows]), -1)
        # Preserve the parent addition/cast ordering under autocast.
        o = self.input_layer(sp.VarLenTensor(oinput, o_layout)).feats
        o = o + self.observation_metadata(ofeats[:, 32:34].to(oinput.dtype)) + self.observation_role.to(oinput.dtype)
        o = manual_cast(o, self.dtype)
        observed_rows = torch.zeros(len(x.feats), device=x.device, dtype=torch.bool)
        observed_rows[rows] = True
        urows = torch.nonzero(~observed_rows, as_tuple=False).flatten()
        ucoords = x.coords[urows]
        ubatch = ucoords[:, 0].long()
        u_layout = _layouts_from_batch(ubatch, batch_size)
        return _PreparedObservation(o, ocoords, o_layout, urows, ucoords, ubatch, u_layout)

    def migrate_v11_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        raise RuntimeError('Interleaved flow transformer requires a interleaved backbone parent state; V11 migration is not a valid parent path')

    def load_v11_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        raise RuntimeError('Interleaved flow transformer requires a interleaved backbone parent state; V11 migration is not a valid parent path')

    def load_interleaved_backbone_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Strictly seed Interleaved flow transformer from an exact interleaved backbone state before writing.

        Base keys must be complete and shape-exact.  Refiners inherit the
        matching base block, then their two residual outputs are zeroed, making
        each newly inserted operation exactly identity at migration time.
        """
        target = self.state_dict()
        base = {key for key in target if not key.startswith('refiners.')}
        source_keys = set(state_dict)
        missing = sorted(base - source_keys)
        unexpected = sorted(source_keys - base)
        non_tensors = sorted(key for key in base & source_keys if not isinstance(state_dict[key], torch.Tensor))
        mismatched = sorted(key for key in base & source_keys if isinstance(state_dict[key], torch.Tensor)
                            and (target[key].shape != state_dict[key].shape or target[key].dtype != state_dict[key].dtype))
        if missing or unexpected or non_tensors or mismatched:
            raise RuntimeError('invalid interleaved backbone parent state before write: '
                               f'missing={missing}, unexpected={unexpected}, non_tensors={non_tensors}, '
                               f'shape_or_dtype_mismatch={mismatched}')
        updates = {key: state_dict[key] for key in base}
        seeded: List[str] = []
        zeroed: List[str] = []
        for index, refiner in self.refiners.items():
            for suffix, value in refiner.state_dict().items():
                destination = f'refiners.{index}.{suffix}'
                source = f'blocks.{index}.{suffix}'
                if source not in state_dict or target[destination].shape != state_dict[source].shape:
                    raise RuntimeError(f'invalid refiner seed source before write: {source}')
                updates[destination] = state_dict[source]
                seeded.append(destination)
            for suffix in ('self_attn.to_out.weight', 'self_attn.to_out.bias',
                           'mlp.mlp.2.weight', 'mlp.mlp.2.bias'):
                destination = f'refiners.{index}.{suffix}'
                updates[destination] = torch.zeros_like(target[destination])
                zeroed.append(destination)
        self.load_state_dict(updates, strict=True)
        return {'inherited': sorted(base), 'new': sorted(set(target) - base),
                'seeded': sorted(seeded), 'zeroed': sorted(zeroed)}

    def _mod(self, value: torch.Tensor, layout: List[slice], batch: torch.Tensor) -> torch.Tensor:
        return self._direct_modulation(value, layout) if self.direct_modulation_broadcast else value[batch]

    @staticmethod
    def _obatch(o: torch.Tensor, layout: List[slice]) -> torch.Tensor:
        if not len(o):
            return torch.empty(0, device=o.device, dtype=torch.long)
        return torch.cat([torch.full((item.stop - item.start,), index, device=o.device, dtype=torch.long)
                          for index, item in enumerate(layout)])

    def _qkv(self, block: _PSABlock, hidden: torch.Tensor, coords: torch.Tensor,
             layout: List[slice], which: Iterable[int]) -> Dict[int, sp.VarLenTensor]:
        """Project exactly requested Q/K/V slices; static O never materializes Q."""
        heads, dim = self.num_heads, self.model_channels // self.num_heads
        selected = tuple(which)
        if selected == (0, 1, 2):
            q, k, v = self._project_qkv(block, hidden, coords, layout)
            return {0: q, 1: k, 2: v}
        weight = block.self_attn.to_qkv.weight.reshape(3, self.model_channels, self.model_channels)
        bias = block.self_attn.to_qkv.bias.reshape(3, self.model_channels)
        result: Dict[int, sp.VarLenTensor] = {}
        if selected == (1, 2):
            # One sliced K/V projection is the actual static-O saving; two
            # independent linears would preserve semantics but lose the budget.
            packed = F.linear(hidden, weight[1:].reshape(2 * self.model_channels, self.model_channels),
                              bias[1:].reshape(2 * self.model_channels)).reshape(-1, 2, heads, dim)
            token = sp.VarLenTensor(packed, layout)
            result[1], result[2] = token.unbind(1)
        for item in selected:
            if item in result:
                continue
            feats = F.linear(hidden, weight[item], bias[item]).reshape(-1, heads, dim)
            result[item] = sp.VarLenTensor(feats, layout)
        if 0 in result and block.self_attn.qk_rms_norm:
            result[0] = block.self_attn.q_rms_norm(result[0])
        if 1 in result and block.self_attn.qk_rms_norm:
            result[1] = block.self_attn.k_rms_norm(result[1])
        if len(coords) and (0 in result or 1 in result):
            q = result.get(0); k = result.get(1)
            if q is None:
                dummy = sp.VarLenTensor(torch.zeros_like(result[1].feats), layout)
                _, k = block.self_attn.rope(sp.SparseTensor(dummy.feats, coords), sp.SparseTensor(k.feats, coords))
                result[1] = sp.VarLenTensor(k.feats, layout)
            elif k is None:
                q, _ = block.self_attn.rope(sp.SparseTensor(q.feats, coords), sp.SparseTensor(torch.zeros_like(q.feats), coords))
                result[0] = sp.VarLenTensor(q.feats, layout)
            else:
                q, k = block.self_attn.rope(sp.SparseTensor(q.feats, coords), sp.SparseTensor(k.feats, coords))
                result[0], result[1] = sp.VarLenTensor(q.feats, layout), sp.VarLenTensor(k.feats, layout)
        return result

    def _x_attention(self, block: _PSABlock, o: torch.Tensor, ocoords: torch.Tensor, o_layout: List[slice],
                     x: torch.Tensor, x_tensor: sp.SparseTensor,
                     cache: Optional[_StaticObservationCache] = None, cache_key: Optional[str] = None) -> torch.Tensor:
        xqkv = self._qkv(block, x, x_tensor.coords, x_tensor.layout, (0, 1, 2))
        cached_kv = None if cache is None or cache_key is None else cache.get(cache_key)
        if cached_kv is not None:
            oqkv = {1: sp.VarLenTensor(cached_kv[0], o_layout), 2: sp.VarLenTensor(cached_kv[1], o_layout)}
        else:
            oqkv = self._qkv(block, o, ocoords, o_layout, (1, 2)) if len(o) else {}
            if cache is not None and cache_key is not None and len(o):
                cache.record(cache_key, oqkv[1].feats, oqkv[2].feats)
        keys: List[torch.Tensor] = []; values: List[torch.Tensor] = []; layout: List[slice] = []; start = 0
        for os, xs in zip(o_layout, x_tensor.layout):
            key = torch.cat(([oqkv[1].feats[os], xqkv[1].feats[xs]]) if len(o) else [xqkv[1].feats[xs]], 0)
            value = torch.cat(([oqkv[2].feats[os], xqkv[2].feats[xs]]) if len(o) else [xqkv[2].feats[xs]], 0)
            keys.append(key); values.append(value); layout.append(slice(start, start + len(key))); start += len(key)
        kv = sp.VarLenTensor(torch.stack((torch.cat(keys), torch.cat(values)), dim=1), layout)
        xout = sparse_scaled_dot_product_attention(xqkv[0], kv).feats
        return block.self_attn.to_out(xout.reshape(len(x_tensor.feats), self.model_channels))

    def _base_block(self, block: _PSABlock, o: torch.Tensor, ocoords: torch.Tensor, o_layout: List[slice],
                    h: torch.Tensor, x: sp.SparseTensor, tx: torch.Tensor, to: torch.Tensor,
                    cache: Optional[_StaticObservationCache] = None, cache_key: Optional[str] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        sx, scx, gx, smx, scm, gmx = block._modulation(tx)
        xo = block.norm1(h) * (1 + self._mod(scx, x.layout, x.batch_boardcast_map)) + self._mod(sx, x.layout, x.batch_boardcast_map)
        cached_kv = None if cache is None or cache_key is None else cache.get(cache_key)
        if cached_kv is None:
            so, sco, _go, _smo, _scmo, _gmo = block._modulation(to)
            obatch = self._obatch(o, o_layout) if not self.direct_modulation_broadcast else torch.empty(0, device=o.device, dtype=torch.long)
            oo = block.norm1(o) * (1 + self._mod(sco, o_layout, obatch)) + self._mod(so, o_layout, obatch) if len(o) else o
        else:
            oo = o
        ax = self._x_attention(block, oo, ocoords, o_layout, xo, x, cache, cache_key)
        h = h + ax * self._mod(gx, x.layout, x.batch_boardcast_map)
        h = h + block.mlp.mlp(x.replace(block.norm2(h) * (1 + self._mod(scm, x.layout, x.batch_boardcast_map)) + self._mod(smx, x.layout, x.batch_boardcast_map))).feats * self._mod(gmx, x.layout, x.batch_boardcast_map)
        return o, h

    def _refine(self, block: _PSABlock, o: torch.Tensor, ocoords: torch.Tensor, o_layout: List[slice], h: torch.Tensor,
                x: sp.SparseTensor, tx: torch.Tensor, to: torch.Tensor, urows: torch.Tensor, u_layout: List[slice],
                cache: Optional[_StaticObservationCache] = None, cache_key: Optional[str] = None) -> torch.Tensor:
        if not len(urows):
            return h + _live_zero(block.parameters(), h)
        sx, scx, gx, smx, scm, gmx = block._modulation(tx)
        xo = block.norm1(h) * (1 + self._mod(scx, x.layout, x.batch_boardcast_map)) + self._mod(sx, x.layout, x.batch_boardcast_map)
        cached_kv = None if cache is None or cache_key is None else cache.get(cache_key)
        if cached_kv is None:
            so, sco, _go, _smo, _scmo, _gmo = block._modulation(to)
            obatch = self._obatch(o, o_layout) if not self.direct_modulation_broadcast else torch.empty(0, device=o.device, dtype=torch.long)
            oo = block.norm1(o) * (1 + self._mod(sco, o_layout, obatch)) + self._mod(so, o_layout, obatch) if len(o) else o
        else:
            oo = o
        u = h[urows]
        if cache is not None and cache.prepared is not None:
            ucoords, ubatch = cache.prepared.ucoords, cache.prepared.ubatch
        else:
            ucoords, ubatch = x.coords[urows], x.coords[urows, 0].long()
        uo = xo[urows]
        # U is Q-only; K/V deliberately include O plus every current X row.
        uq = self._qkv(block, uo, ucoords, u_layout, (0,))[0]
        xkv = self._qkv(block, xo, x.coords, x.layout, (1, 2))
        cached_kv = None if cache is None or cache_key is None else cache.get(cache_key)
        if cached_kv is not None:
            okv = {1: sp.VarLenTensor(cached_kv[0], o_layout), 2: sp.VarLenTensor(cached_kv[1], o_layout)}
        else:
            okv = self._qkv(block, oo, ocoords, o_layout, (1, 2)) if len(o) else {}
            if cache is not None and cache_key is not None and len(o):
                cache.record(cache_key, okv[1].feats, okv[2].feats)
        parts: List[torch.Tensor] = []; values: List[torch.Tensor] = []; layout: List[slice] = []; start = 0
        for os, xs in zip(o_layout, x.layout):
            key = torch.cat(([okv[1].feats[os], xkv[1].feats[xs]]) if len(o) else [xkv[1].feats[xs]], 0)
            value = torch.cat(([okv[2].feats[os], xkv[2].feats[xs]]) if len(o) else [xkv[2].feats[xs]], 0)
            parts.append(key); values.append(value); count = key.shape[0]; layout.append(slice(start, start + count)); start += count
        kv = sp.VarLenTensor(torch.stack((torch.cat(parts), torch.cat(values)), 1), layout)
        u = u + block.self_attn.to_out(sparse_scaled_dot_product_attention(uq, kv).feats.reshape(len(u), self.model_channels)) * self._mod(gx, u_layout, ubatch)
        u = u + block.mlp.mlp(sp.SparseTensor(block.norm2(u) * (1 + self._mod(scm, u_layout, ubatch)) + self._mod(smx, u_layout, ubatch), ucoords, shape=x.shape)).feats * self._mod(gmx, u_layout, ubatch)
        h = h.clone(); h[urows] = u
        return h

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Any = None, concat_cond: Optional[sp.SparseTensor] = None,
                partial_z_obs: Optional[sp.SparseTensor] = None, partial_z_obs_keep_mask: Optional[torch.Tensor] = None,
                observation_cache: Optional[_StaticObservationCache] = None, **kwargs: Any) -> sp.SparseTensor:
        retired = {'o_update_scale', 'o_transition_steps'} & kwargs.keys()
        if retired:
            raise ValueError(f'Interleaved flow transformer O is always static; removed options: {sorted(retired)}')
        if observation_cache is not None:
            if not isinstance(observation_cache, _StaticObservationCache) or observation_cache.model is not self:
                raise RuntimeError('observation_cache belongs to a different Interleaved flow transformer model')
            observation_cache.validate(x, concat_cond, partial_z_obs, partial_z_obs_keep_mask)
        if kwargs:
            disabled = next((name for name in ('use_observation_cache', 'compact_observation_cache', 'hot_observation_cache') if name in kwargs and kwargs[name] is not False), None)
            if disabled is not None: raise RuntimeError(f'Interleaved flow transformer disables {disabled}')
        if concat_cond is None or x.feats.shape[1] != 32 or concat_cond.feats.shape != x.feats.shape:
            raise AssertionError('Interleaved flow transformer requires coordinate-aligned PBR32 x and shape concat_cond')
        batch_size = x.shape[0]
        if t.shape != (batch_size,): raise AssertionError(f't must be [B={batch_size}]')
        if partial_z_obs is None or partial_z_obs.feats.ndim != 2 or partial_z_obs.feats.shape[1] != 34:
            raise AssertionError('Interleaved flow transformer requires 34-channel partial_z_obs (possibly empty)')
        prepared = None if observation_cache is None else observation_cache.prepared
        if prepared is None:
            prepared = self._prepare_observation(x, concat_cond, partial_z_obs, partial_z_obs_keep_mask)
            if observation_cache is not None:
                observation_cache.prepared = prepared
        o, ocoords, o_layout = prepared.hidden, prepared.coords, prepared.layout
        urows, u_layout = prepared.urows, prepared.u_layout
        h = self.input_layer(x.replace(torch.cat((x.feats, concat_cond.feats), -1))).feats
        h = manual_cast(h, self.dtype)
        tx = manual_cast(self.adaLN_modulation(self.t_embedder(t)), h.dtype)
        zero_time_signature = (t.dtype, t.device, h.dtype)
        if observation_cache is not None and observation_cache.zero_time_signature == zero_time_signature:
            to = observation_cache.zero_time
        else:
            to = manual_cast(self.adaLN_modulation(self.t_embedder(torch.zeros_like(t))), h.dtype)
            if observation_cache is not None:
                observation_cache.zero_time = to
                observation_cache.zero_time_signature = zero_time_signature
        def run_group(observation: torch.Tensor, generated: torch.Tensor, start: int, stop: int) -> Tuple[torch.Tensor, torch.Tensor]:
            for index in range(start, stop):
                if self.base_block_policy == 'alternating_u_x' and not index % 2:
                    generated = self._refine(self.blocks[index], observation, ocoords, o_layout,
                                             generated, x, tx, to, urows, u_layout,
                                             observation_cache, f'blocks.{index}')
                else:
                    observation, generated = self._base_block(self.blocks[index], observation, ocoords, o_layout,
                                                              generated, x, tx, to,
                                                              observation_cache, f'blocks.{index}')
                key = str(index)
                if key in self.refiners:
                    generated = self._refine(self.refiners[key], observation, ocoords, o_layout,
                                             generated, x, tx, to, urows, u_layout,
                                             observation_cache, f'refiners.{key}')
            return observation, generated
        for start in range(0, len(self.blocks), self.checkpoint_group_size):
            stop = min(start + self.checkpoint_group_size, len(self.blocks))
            if self.use_checkpoint and self.training:
                o, h = torch.utils.checkpoint.checkpoint(lambda a, b, begin=start, end=stop: run_group(a, b, begin, end), o, h, use_reentrant=False)
            else: o, h = run_group(o, h, start, stop)
        if not len(o): h = h + _live_zero(self.observation_metadata.parameters(), h) + self.observation_role.sum().to(h.dtype) * 0
        return self._finalize(x, h)

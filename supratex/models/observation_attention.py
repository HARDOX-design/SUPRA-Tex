"""observation attention: directed pure-self-attention texture flow denoiser.

The observation stream is deliberately a separate variable-length stream.  It
never receives values from generated-shell tokens; generated tokens attend to
their sample's observations and generated tokens through a second varlen call.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
import re

from ..modules import sparse as sp
from ..modules.sparse.attention import SparseMultiHeadAttention
from ..modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from ..modules.sparse.transformer.blocks import SparseFeedForwardNet
from ..modules.norm import LayerNorm32
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from .sparse_structure_flow import TimestepEmbedder


def _live_zero(parameters: Iterable[nn.Parameter], reference: torch.Tensor) -> torch.Tensor:
    """Keep every observation-only parameter present in an empty-DDP graph."""
    total = reference.new_zeros(())
    for parameter in parameters:
        total = total + parameter.sum().to(dtype=reference.dtype) * 0
    return total


def _layouts_from_batch(batch: torch.Tensor, batch_size: int) -> List[slice]:
    counts = torch.bincount(batch.long(), minlength=batch_size).tolist()
    start = 0
    layouts = []
    for count in counts:
        layouts.append(slice(start, start + count))
        start += count
    return layouts


def _join_exact_rows(full_coords: torch.Tensor, observed_coords: torch.Tensor) -> torch.Tensor:
    if observed_coords.numel() == 0:
        return torch.empty(0, device=full_coords.device, dtype=torch.long)
    if full_coords.ndim != 2 or observed_coords.ndim != 2 or full_coords.shape[1] != observed_coords.shape[1]:
        raise AssertionError('partial_z_obs coordinate rank mismatch')
    joined = torch.cat((full_coords, observed_coords), dim=0)
    _, inverse = torch.unique(joined, dim=0, return_inverse=True)
    full_ids, obs_ids = inverse[:len(full_coords)], inverse[len(full_coords):]
    if torch.unique(full_ids).numel() != len(full_ids):
        raise AssertionError('x contains duplicate coordinates')
    if torch.unique(obs_ids).numel() != len(obs_ids):
        raise AssertionError('partial_z_obs contains duplicate coordinates')
    order, rows = torch.sort(full_ids)
    position = torch.searchsorted(order, obs_ids)
    valid = position < len(order)
    safe = position.clamp(max=len(order) - 1)
    valid = valid & (order[safe] == obs_ids)
    if not bool(valid.all()):
        raise AssertionError('partial_z_obs coordinates must be an exact subset of current x coordinates')
    return rows[safe]


def _tensor_identity(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Mutation-sensitive identity, deliberately without inspecting payload values."""
    return (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype), str(tensor.device), tensor._version)


def _sparse_identity(tensor: sp.SparseTensor, include_features: bool) -> Tuple[Any, ...]:
    """Sparse support/layout identity; values are included only when cache-dependent."""
    return (
        _tensor_identity(tensor.coords),
        _tensor_identity(tensor.feats) if include_features else None,
        tuple(tensor.shape),
        tuple((item.start, item.stop, item.step) for item in tensor.layout),
    )


class _ObservationCache:
    """Invocation-local, eval-only O K/V cache.  It never survives ``with`` exit."""
    def __init__(self, model: 'ObservationAttentionModel', x: sp.SparseTensor, shape: sp.SparseTensor,
                 observed: sp.SparseTensor, keep: Optional[torch.Tensor]):
        if model.training or torch.is_grad_enabled():
            raise RuntimeError('observation_cache is inference-only; training cannot retain detached O K/V')
        self.model = model
        self.signature = model._cache_signature(x, shape, observed, keep)
        self.kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * model.num_blocks
        self.bytes = 0
        self.closed = False

    def validate(self, x: sp.SparseTensor, shape: sp.SparseTensor, observed: sp.SparseTensor,
                 keep: Optional[torch.Tensor]) -> None:
        if self.closed or self.model.training or torch.is_grad_enabled():
            raise RuntimeError('observation_cache is closed or used outside eval/no_grad')
        if self.signature != self.model._cache_signature(x, shape, observed, keep):
            raise RuntimeError('observation_cache inputs, model weights, dtype, or device changed; rebuild the context')

    def record(self, index: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.kv[index] is None:
            self.kv[index] = (k.detach(), v.detach())
            self.bytes += (k.numel() * k.element_size()) + (v.numel() * v.element_size())

    def close(self) -> None:
        self.kv.clear(); self.closed = True

    def __enter__(self) -> '_ObservationCache':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _PSABlock(nn.Module):
    """A V11-compatible self-attention/MLP block with no cross-attention slot."""
    def __init__(self, channels: int, heads: int, mlp_ratio: float, rope_freq: Tuple[float, float], qk_rms_norm: bool):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        # V11's norm3 maps here.  Its cross-attention norm2 is intentionally absent.
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(channels, heads, type='self', attn_mode='full', use_rope=True,
                                                   rope_freq=rope_freq, qk_rms_norm=qk_rms_norm)
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _modulation(self, mod: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)


class ObservationAttentionModel(nn.Module):
    """60-block directed observation/self-attention flow model.

    ``partial_*`` and ``pbr_*`` are explicit 32-vector normalization statistics;
    an observation's first 32 values are converted through raw latent units only
    for its shared input embedding.
    """
    def __init__(self, resolution: int = 1024, in_channels: int = 64, model_channels: int = 1536,
                 cond_channels: int = 0, out_channels: int = 32, num_blocks: int = 60,
                 num_heads: int = 12, mlp_ratio: float = 5.3334, rope_freq: Tuple[float, float] = (1., 10000.),
                 dtype: str = 'float32', use_checkpoint: bool = True, share_mod: bool = True,
                 qk_rms_norm: bool = True, partial_mean: Optional[Sequence[float]] = None,
                 partial_std: Optional[Sequence[float]] = None, pbr_mean: Optional[Sequence[float]] = None,
                 pbr_std: Optional[Sequence[float]] = None, checkpoint_group_size: int = 1,
                 _test_tiny: bool = False, **_: Any):
        super().__init__()
        if not _test_tiny and (num_blocks, model_channels, num_heads) != (60, 1536, 12):
            raise ValueError('observation attention is frozen at 60 blocks, width 1536, and 12 heads')
        if (in_channels != 64 or out_channels != 32 or not share_mod) and not _test_tiny:
            raise ValueError('observation attention requires shared [PBR32|shape32] input and PBR32 output')
        if checkpoint_group_size < 1:
            raise ValueError('checkpoint_group_size must be positive')
        stats = (partial_mean, partial_std, pbr_mean, pbr_std)
        if any(value is None for value in stats) or any(len(value) != 32 for value in stats):
            raise ValueError('partial_mean, partial_std, pbr_mean, and pbr_std must each be explicit 32-vectors')
        self.resolution, self.in_channels, self.model_channels = resolution, in_channels, model_channels
        self.cond_channels, self.out_channels, self.num_blocks, self.num_heads = cond_channels, out_channels, num_blocks, num_heads
        self.mlp_ratio, self.use_checkpoint, self.share_mod, self.qk_rms_norm = mlp_ratio, use_checkpoint, share_mod, qk_rms_norm
        self.checkpoint_group_size = int(checkpoint_group_size)
        self.dtype = str_to_dtype(dtype)
        self.register_buffer('partial_mean', torch.tensor(partial_mean, dtype=torch.float32), persistent=True)
        self.register_buffer('partial_std', torch.tensor(partial_std, dtype=torch.float32), persistent=True)
        self.register_buffer('pbr_mean', torch.tensor(pbr_mean, dtype=torch.float32), persistent=True)
        self.register_buffer('pbr_std', torch.tensor(pbr_std, dtype=torch.float32), persistent=True)
        self.t_embedder = TimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.observation_metadata = nn.Linear(2, model_channels, bias=True)
        self.observation_role = nn.Parameter(torch.zeros(model_channels))
        self.blocks = nn.ModuleList([_PSABlock(model_channels, num_heads, mlp_ratio, rope_freq, qk_rms_norm) for _ in range(num_blocks)])
        self.out_layer = sp.SparseLinear(model_channels, out_channels)
        self.initialize_weights()
        self.convert_to(self.dtype)

    def convert_to(self, dtype: torch.dtype) -> None:
        """Honor an explicit inference dtype; formal V13 training keeps FP32 parameters under AMP."""
        self.dtype = dtype
        self.blocks.apply(lambda module: convert_module_to(module, dtype=dtype))

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=math.sqrt(2.0 / (5.0 * self.model_channels)))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.input_layer.weight, std=1.0 / math.sqrt(self.in_channels))
        nn.init.zeros_(self.input_layer.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=.02)
        nn.init.zeros_(self.adaLN_modulation[-1].weight); nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.observation_metadata.weight); nn.init.zeros_(self.observation_metadata.bias); nn.init.zeros_(self.observation_role)
        nn.init.zeros_(self.out_layer.weight); nn.init.zeros_(self.out_layer.bias)
        for block in self.blocks[30:]:
            nn.init.zeros_(block.self_attn.to_out.weight); nn.init.zeros_(block.self_attn.to_out.bias)
            nn.init.zeros_(block.mlp.mlp[2].weight); nn.init.zeros_(block.mlp.mlp[2].bias)

    def _cache_signature(self, x: sp.SparseTensor, shape: sp.SparseTensor, observed: sp.SparseTensor,
                         keep: Optional[torch.Tensor]) -> Tuple[Any, ...]:
        """Only tensor identity/version metadata is used; no payload digest is computed."""
        weights = tuple((name, parameter.data_ptr(), parameter._version, str(parameter.dtype), str(parameter.device))
                        for name, parameter in self.named_parameters())
        return (_sparse_identity(x, include_features=False), _sparse_identity(shape, include_features=True),
                _sparse_identity(observed, include_features=True),
                None if keep is None else _tensor_identity(keep),
                _tensor_identity(self.partial_mean), _tensor_identity(self.partial_std),
                _tensor_identity(self.pbr_mean), _tensor_identity(self.pbr_std), weights)

    def observation_cache(self, x: sp.SparseTensor, concat_cond: sp.SparseTensor, partial_z_obs: sp.SparseTensor,
                          partial_z_obs_keep_mask: Optional[torch.Tensor] = None) -> _ObservationCache:
        """Create an explicit, invocation-local eval/no-grad O K/V cache context."""
        return _ObservationCache(self, x, concat_cond, partial_z_obs, partial_z_obs_keep_mask)

    def _finalize(self, x: sp.SparseTensor, hidden: torch.Tensor) -> sp.SparseTensor:
        """Exact V11 final cast -> parameter-free LayerNorm -> copied output layer."""
        h = x.replace(manual_cast(hidden, x.dtype))
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h)

    def _attention(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor, o_layout: List[slice],
                   x: torch.Tensor, x_tensor: sp.SparseTensor,
                   cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Exactly two varlen attention calls: O->O and X->[O;X]."""
        def qkv(hidden: torch.Tensor, coords: torch.Tensor, layout: List[slice]):
            packed = block.self_attn.to_qkv(hidden).reshape(-1, 3, self.num_heads, self.model_channels // self.num_heads)
            token = sp.VarLenTensor(packed, layout)
            q, k, v = token.unbind(1)
            if block.self_attn.qk_rms_norm:
                q, k = block.self_attn.q_rms_norm(q), block.self_attn.k_rms_norm(k)
            if len(coords):
                q_sparse, k_sparse = sp.SparseTensor(q.feats, coords), sp.SparseTensor(k.feats, coords)
                q_sparse, k_sparse = block.self_attn.rope(q_sparse, k_sparse)
                q, k = sp.VarLenTensor(q_sparse.feats, layout), sp.VarLenTensor(k_sparse.feats, layout)
            return q, k, v
        qx, kx, vx = qkv(x, x_tensor.coords, x_tensor.layout)
        if cached_kv is not None:
            ko, vo = cached_kv
            o_out = o
            kv_feats, kv_layout = [], []
            start = 0
            for os, xs in zip(o_layout, x_tensor.layout):
                joined = torch.cat((ko[os], kx.feats[xs]), dim=0)
                joined_v = torch.cat((vo[os], vx.feats[xs]), dim=0)
                kv_feats.append(torch.stack((joined, joined_v), dim=1)); kv_layout.append(slice(start, start + len(joined))); start += len(joined)
            kv = sp.VarLenTensor(torch.cat(kv_feats, dim=0), kv_layout)
            built_kv = None
        elif len(o):
            qo, ko, vo = qkv(o, o_coords, o_layout)
            o_out = sparse_scaled_dot_product_attention(qo, ko, vo).feats
            kv_feats, kv_layout = [], []
            start = 0
            for os, xs in zip(o_layout, x_tensor.layout):
                joined = torch.cat((ko.feats[os], kx.feats[xs]), dim=0)
                joined_v = torch.cat((vo.feats[os], vx.feats[xs]), dim=0)
                kv_feats.append(torch.stack((joined, joined_v), dim=1)); kv_layout.append(slice(start, start + len(joined))); start += len(joined)
            kv = sp.VarLenTensor(torch.cat(kv_feats, dim=0), kv_layout)
            built_kv = (ko.feats, vo.feats)
        else:
            o_out = o
            kv = sp.VarLenTensor(torch.stack((kx.feats, vx.feats), dim=1), x_tensor.layout)
            built_kv = (o, o)
        x_out = sparse_scaled_dot_product_attention(qx, kv).feats
        return (block.self_attn.to_out(o_out.reshape(len(o_out), self.model_channels)),
                block.self_attn.to_out(x_out.reshape(len(x_out), self.model_channels)), built_kv)

    def _block(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor, o_layout: List[slice],
               x: torch.Tensor, x_tensor: sp.SparseTensor, mod_x: torch.Tensor, mod_o: torch.Tensor, terminal: bool,
               cache: Optional[_ObservationCache] = None, index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        sx, scx, gx, smx, scm, gmx = block._modulation(mod_x)
        so, sco, go, smo, scmo, gmo = block._modulation(mod_o)
        xo = block.norm1(x) * (1 + scx[x_tensor.batch_boardcast_map]) + sx[x_tensor.batch_boardcast_map]
        oo = block.norm1(o) * (1 + sco[torch.cat([torch.full((s.stop-s.start,), i, device=o.device, dtype=torch.long) for i,s in enumerate(o_layout)])] if len(o) else 1) + (so[torch.cat([torch.full((s.stop-s.start,), i, device=o.device, dtype=torch.long) for i,s in enumerate(o_layout)])] if len(o) else 0)
        cached_kv = None if cache is None else cache.kv[index]
        ao, ax, built_kv = self._attention(block, oo, o_coords, o_layout, xo, x_tensor, cached_kv)
        if cache is not None and built_kv is not None:
            cache.record(index, *built_kv)
        x = x + ax * gx[x_tensor.batch_boardcast_map]
        if not terminal and cached_kv is None:
            obatch = torch.cat([torch.full((s.stop-s.start,), i, device=o.device, dtype=torch.long) for i,s in enumerate(o_layout)]) if len(o) else torch.empty(0, device=o.device, dtype=torch.long)
            o = o + ao * go[obatch]
            o_mlp = sp.VarLenTensor(
                block.norm2(o) * (1 + scmo[obatch]) + smo[obatch], o_layout,
            )
            o = o + block.mlp(o_mlp).feats * gmo[obatch]
        x = x + block.mlp.mlp(x_tensor.replace(block.norm2(x) * (1 + scm[x_tensor.batch_boardcast_map]) + smx[x_tensor.batch_boardcast_map])).feats * gmx[x_tensor.batch_boardcast_map]
        return o, x

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Any = None, concat_cond: Optional[sp.SparseTensor] = None,
                partial_z_obs: Optional[sp.SparseTensor] = None, partial_z_obs_keep_mask: Optional[torch.Tensor] = None,
                observation_cache: Optional[_ObservationCache] = None, **_: Any) -> sp.SparseTensor:
        if concat_cond is None:
            raise AssertionError('observation attention requires concat_cond=aligned shape sparse tensor')
        if x.feats.shape[1] != 32 or concat_cond.feats.shape != x.feats.shape or not torch.equal(concat_cond.coords, x.coords):
            raise AssertionError('x and concat_cond must be coordinate-aligned PBR32 and shape32 tensors')
        batch_size = x.shape[0]
        if t.shape != (batch_size,): raise AssertionError(f't must be [B={batch_size}]')
        if partial_z_obs is None: raise AssertionError('observation attention requires partial_z_obs (possibly empty)')
        if partial_z_obs.feats.ndim != 2 or partial_z_obs.feats.shape[1] != 34: raise AssertionError('partial_z_obs must have 34 channels')
        keep = torch.ones(batch_size, device=x.device, dtype=torch.bool) if partial_z_obs_keep_mask is None else partial_z_obs_keep_mask.to(x.device, torch.bool)
        if keep.shape != (batch_size,): raise AssertionError(f'partial_z_obs_keep_mask must be [B={batch_size}]')
        if observation_cache is not None:
            if observation_cache.model is not self:
                raise RuntimeError('observation_cache belongs to a different observation attention model')
            observation_cache.validate(x, concat_cond, partial_z_obs, partial_z_obs_keep_mask)
        active = keep[partial_z_obs.coords[:, 0].long()]
        ocoords, ofeats = partial_z_obs.coords[active], partial_z_obs.feats[active]
        rows = _join_exact_rows(x.coords, ocoords)
        o_layout = _layouts_from_batch(ocoords[:, 0], batch_size)
        raw = ofeats[:, :32].float() * self.partial_std + self.partial_mean
        opbr = (raw - self.pbr_mean) / self.pbr_std
        oinput = torch.cat((opbr.to(ofeats.dtype), concat_cond.feats[rows]), dim=-1)
        o = self.input_layer(sp.VarLenTensor(oinput, o_layout)).feats + self.observation_metadata(ofeats[:, 32:34].to(oinput.dtype)) + self.observation_role.to(oinput.dtype)
        h = self.input_layer(x.replace(torch.cat((x.feats, concat_cond.feats), dim=-1))).feats
        o, h = manual_cast(o, self.dtype), manual_cast(h, self.dtype)
        # Existing flow trainer supplies the baseline-scaled timestep (t*1000).
        tx = manual_cast(self.adaLN_modulation(self.t_embedder(t)), h.dtype)
        to = manual_cast(self.adaLN_modulation(self.t_embedder(torch.zeros_like(t))), h.dtype)
        def run_group(observation: torch.Tensor, generated: torch.Tensor, start: int, stop: int) -> Tuple[torch.Tensor, torch.Tensor]:
            for index in range(start, stop):
                observation, generated = self._block(
                    self.blocks[index], observation, ocoords, o_layout, generated, x, tx, to,
                    index == len(self.blocks) - 1, observation_cache, index,
                )
            return observation, generated

        for start in range(0, len(self.blocks), self.checkpoint_group_size):
            stop = min(start + self.checkpoint_group_size, len(self.blocks))
            if self.use_checkpoint and self.training and observation_cache is None:
                # The closure captures static sparse layouts/coordinates.  Tensor-only
                # grouped segments reduce retained O/X inter-block activations.
                o, h = torch.utils.checkpoint.checkpoint(
                    lambda observation, generated, begin=start, end=stop: run_group(observation, generated, begin, end),
                    o, h, use_reentrant=False,
                )
            else:
                o, h = run_group(o, h, start, stop)
        if not len(o): h = h + _live_zero(self.observation_metadata.parameters(), h) + self.observation_role.sum().to(h.dtype) * 0
        return self._finalize(x, h)

    def migrate_v11_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Strictly classify every V11 tensor and load the frozen first-30 mapping."""
        target = self.state_dict(); copied: List[str] = []; discarded: List[str] = []; updates: Dict[str, torch.Tensor] = {}
        for source_key, value in state_dict.items():
            block_match = re.match(r'^blocks\.(\d+)\.(.*)$', source_key)
            if block_match is not None and int(block_match.group(1)) >= 30:
                raise RuntimeError(f'unclassified V11 tensor outside frozen 30-block source: {source_key}')
            discard = (
                source_key == 'partial_zobs_injector.weight'
                or re.match(r'^blocks\.([0-9]|[12][0-9])\.norm2\.(weight|bias)$', source_key) is not None
                or re.match(r'^blocks\.([0-9]|[12][0-9])\.cross_attn\.(to_q|to_kv|to_out)\.(weight|bias)$', source_key) is not None
                or re.match(r'^blocks\.([0-9]|[12][0-9])\.cross_attn\.[qk]_rms_norm\.gamma$', source_key) is not None
            )
            if discard:
                discarded.append(source_key); continue
            dest = source_key.replace('.norm3.', '.norm2.') if block_match is not None else source_key
            if dest not in target: raise RuntimeError(f'unclassified V11 tensor: {source_key}')
            if target[dest].shape != value.shape: raise RuntimeError(f'shape mismatch {source_key}: {tuple(value.shape)} != {tuple(target[dest].shape)}')
            updates[dest] = value; copied.append(source_key)
        required = [
            key for key in target
            if not (
                key.startswith(('observation_metadata.', 'observation_role'))
                or key.startswith(('partial_', 'pbr_'))
                or (key.startswith('blocks.') and int(key.split('.')[1]) >= 30)
            )
        ]
        missing = [key for key in required if key not in updates]
        if missing: raise RuntimeError(f'missing required V11 tensors: {missing}')
        self.load_state_dict(updates, strict=False)
        return {'copied': sorted(copied), 'discarded': sorted(discarded), 'new': sorted(set(target) - set(updates))}

    def load_v11_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        return self.migrate_v11_state_dict(state_dict)

"""Optional O-stream refresh policy for the frozen observation attention architecture.

The default policy deliberately delegates to :class:`ObservationAttentionModel` so it
remains bit-for-bit on the established path.  The opt-in policy retains an
observation K/V projection at every layer because every X layer attends to its
own [O; X] keys and values, while avoiding O updates on selected layers.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ..modules import sparse as sp
from ..modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from .observation_attention import (
    ObservationAttentionModel,
    _ObservationCache,
    _PSABlock,
)


class ObservationStreamModel(ObservationAttentionModel):
    """observation attention with opt-in sparse O-stream updates.

    ``o_refresh_stride`` refreshes O at blocks ``0, stride, 2 * stride, ...``
    before the terminal block.  All blocks still project the current O state
    to their own K/V for X's unchanged joint ``[O; X]`` self-attention.  When
    terminal relief is disabled, the inherited terminal O attention is retained
    so the stride experiment changes no other policy.
    """
    def __init__(self, terminal_o_kv_only: bool = False, o_refresh_stride: int = 1,
                 **kwargs: Any):
        if o_refresh_stride < 1:
            raise ValueError('o_refresh_stride must be positive')
        super().__init__(**kwargs)
        # Plain policy attributes intentionally add no checkpoint state.
        self.terminal_o_kv_only = terminal_o_kv_only
        self.o_refresh_stride = o_refresh_stride

    def _cache_signature(self, x: sp.SparseTensor, shape: sp.SparseTensor, observed: sp.SparseTensor,
                         keep: Optional[torch.Tensor]) -> Tuple[Any, ...]:
        return super()._cache_signature(x, shape, observed, keep) + (
            ('o_refresh_policy', self.terminal_o_kv_only, self.o_refresh_stride),
        )

    def _project_qkv(self, block: _PSABlock, hidden: torch.Tensor, coords: torch.Tensor,
                     layout: List[slice]) -> Tuple[sp.VarLenTensor, sp.VarLenTensor, sp.VarLenTensor]:
        """Retain the parent's full QKV projection for numerical/checkpoint parity."""
        packed = block.self_attn.to_qkv(hidden).reshape(
            -1, 3, self.num_heads, self.model_channels // self.num_heads,
        )
        token = sp.VarLenTensor(packed, layout)
        q, k, v = token.unbind(1)
        if block.self_attn.qk_rms_norm:
            q, k = block.self_attn.q_rms_norm(q), block.self_attn.k_rms_norm(k)
        if len(coords):
            q_sparse, k_sparse = sp.SparseTensor(q.feats, coords), sp.SparseTensor(k.feats, coords)
            q_sparse, k_sparse = block.self_attn.rope(q_sparse, k_sparse)
            q, k = sp.VarLenTensor(q_sparse.feats, layout), sp.VarLenTensor(k_sparse.feats, layout)
        return q, k, v

    def _attention(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor, o_layout: List[slice],
                   x: torch.Tensor, x_tensor: sp.SparseTensor,
                   cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, *,
                   compute_o_attention: Optional[bool] = None
                   ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Materialize O->O only for a refresh or inherited terminal behavior.

        We deliberately still use ``to_qkv`` for O on K/V-only layers.  Slicing
        its weight would be a new numerical path and is not necessary to retain
        distinct per-layer O K/V for the unchanged X attention.
        """
        # Preserve the exact inherited helper for the parent _block call path.
        # Besides keeping default parity literal, this accepts its positional
        # ``cached_kv`` call convention.
        if compute_o_attention is None:
            return super()._attention(block, o, o_coords, o_layout, x, x_tensor, cached_kv)
        qx, kx, vx = self._project_qkv(block, x, x_tensor.coords, x_tensor.layout)
        if cached_kv is not None:
            ko, vo = cached_kv
            built_kv = None
        elif len(o):
            qo, ko, vo = self._project_qkv(block, o, o_coords, o_layout)
            built_kv = (ko.feats, vo.feats)
        else:
            qo = None
            ko = vo = o
            built_kv = (o, o)

        if not len(o) and cached_kv is None:
            kv = sp.VarLenTensor(torch.stack((kx.feats, vx.feats), dim=1), x_tensor.layout)
        else:
            kv_feats, kv_layout = [], []
            start = 0
            for os, xs in zip(o_layout, x_tensor.layout):
                joined_k = torch.cat((ko[os], kx.feats[xs]), dim=0) if cached_kv is not None else torch.cat((ko.feats[os], kx.feats[xs]), dim=0)
                joined_v = torch.cat((vo[os], vx.feats[xs]), dim=0) if cached_kv is not None else torch.cat((vo.feats[os], vx.feats[xs]), dim=0)
                kv_feats.append(torch.stack((joined_k, joined_v), dim=1))
                kv_layout.append(slice(start, start + len(joined_k)))
                start += len(joined_k)
            kv = sp.VarLenTensor(torch.cat(kv_feats, dim=0), kv_layout)
        x_out = sparse_scaled_dot_product_attention(qx, kv).feats
        o_out = None
        if compute_o_attention:
            # A cache is eval/no-grad and therefore never permits an O update.
            assert cached_kv is None and qo is not None
            o_out = block.self_attn.to_out(
                sparse_scaled_dot_product_attention(qo, ko, vo).feats.reshape(len(o), self.model_channels),
            )
        return o_out, block.self_attn.to_out(x_out.reshape(len(x_out), self.model_channels)), built_kv

    @staticmethod
    def _obatch(o: torch.Tensor, o_layout: List[slice]) -> torch.Tensor:
        if not len(o):
            return torch.empty(0, device=o.device, dtype=torch.long)
        return torch.cat([
            torch.full((item.stop - item.start,), index, device=o.device, dtype=torch.long)
            for index, item in enumerate(o_layout)
        ])

    def _block(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor, o_layout: List[slice],
               x: torch.Tensor, x_tensor: sp.SparseTensor, mod_x: torch.Tensor, mod_o: torch.Tensor, terminal: bool,
               cache: Optional[_ObservationCache] = None, index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Default O semantics remain the literal parent block, including its
        # terminal dead O work.  The parent forward dynamically calls this.
        if self.o_refresh_stride == 1 and (not self.terminal_o_kv_only or not terminal):
            return super()._block(block, o, o_coords, o_layout, x, x_tensor, mod_x, mod_o, terminal, cache, index)
        sx, scx, gx, smx, scm, gmx = block._modulation(mod_x)
        so, sco, go, smo, scmo, gmo = block._modulation(mod_o)
        obatch = self._obatch(o, o_layout)
        xo = block.norm1(x) * (1 + scx[x_tensor.batch_boardcast_map]) + sx[x_tensor.batch_boardcast_map]
        oo = block.norm1(o) * (1 + sco[obatch]) + so[obatch] if len(o) else o
        cached_kv = None if cache is None else cache.kv[index]
        refresh_o = bool(len(o)) and not terminal and index % self.o_refresh_stride == 0 and cached_kv is None
        # With the terminal relief disabled, retain the inherited dead terminal
        # O attention so a stride-only experiment changes no other policy.
        compute_o_attention = refresh_o or (
            terminal and not self.terminal_o_kv_only and bool(len(o)) and cached_kv is None
        )
        ao, ax, built_kv = self._attention(
            block, oo, o_coords, o_layout, xo, x_tensor,
            compute_o_attention=compute_o_attention, cached_kv=cached_kv,
        )
        if cache is not None and built_kv is not None:
            cache.record(index, *built_kv)
        x = x + ax * gx[x_tensor.batch_boardcast_map]
        if refresh_o:
            assert ao is not None
            o = o + ao * go[obatch]
            o_mlp = sp.VarLenTensor(block.norm2(o) * (1 + scmo[obatch]) + smo[obatch], o_layout)
            o = o + block.mlp(o_mlp).feats * gmo[obatch]
        x = x + block.mlp.mlp(
            x_tensor.replace(block.norm2(x) * (1 + scm[x_tensor.batch_boardcast_map]) + smx[x_tensor.batch_boardcast_map]),
        ).feats * gmx[x_tensor.batch_boardcast_map]
        return o, x

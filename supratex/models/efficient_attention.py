"""Opt-in modulation and empty-cache fixes for the frozen observation attention-O model.

Both policies are deliberately plain attributes rather than module state, so a
observation attention-O checkpoint loads strictly without an ABI migration.  Their defaults
keep the inherited execution path literal.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ..modules import sparse as sp
from .observation_attention import _ObservationCache, _PSABlock
from .observation_stream import ObservationStreamModel


class EfficientObservationAttentionModel(ObservationStreamModel):
    """observation attention-O with independently opt-in execution-only policies.

    ``direct_modulation_broadcast`` avoids the per-token modulation gather.
    At B=1 it retains the singleton modulation row for PyTorch broadcasting;
    at B>1 it expands each sample's own singleton row and concatenates those
    pieces, preserving ragged sample boundaries.  The changed BF16 reduction
    path is mathematically equivalent but is not a bit-exact gradient claim.

    ``safe_empty_cache`` repairs only the cached all-empty-O representation.
    It is default-off because the inherited literal policy is frozen; enable it
    for an eval cache that may be populated by an all-dropped observation call.
    """
    def __init__(self, direct_modulation_broadcast: bool = False,
                 safe_empty_cache: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.direct_modulation_broadcast = direct_modulation_broadcast
        self.safe_empty_cache = safe_empty_cache

    def _cache_signature(self, x: sp.SparseTensor, shape: sp.SparseTensor,
                         observed: sp.SparseTensor, keep: Optional[torch.Tensor]
                         ) -> Tuple[Any, ...]:
        return super()._cache_signature(x, shape, observed, keep) + (
            ('fast_policy', self.direct_modulation_broadcast, self.safe_empty_cache),
        )

    def _attention(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor,
                   o_layout: List[slice], x: torch.Tensor, x_tensor: sp.SparseTensor,
                   cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, **kwargs: Any):
        if self.safe_empty_cache and cached_kv is not None and cached_kv[0].ndim == 2:
            head_dim = self.model_channels // self.num_heads
            cached_kv = tuple(item.reshape(0, self.num_heads, head_dim) for item in cached_kv)
        return super()._attention(block, o, o_coords, o_layout, x, x_tensor, cached_kv, **kwargs)

    @staticmethod
    def _direct_modulation(value: torch.Tensor, layout: List[slice]) -> torch.Tensor:
        """Return token-aligned modulation without an indexed batch gather."""
        if len(layout) == 1:
            return value
        parts = [value[index:index + 1].expand(item.stop - item.start, -1)
                 for index, item in enumerate(layout)]
        return torch.cat(parts, dim=0)

    def _block(self, block: _PSABlock, o: torch.Tensor, o_coords: torch.Tensor,
               o_layout: List[slice], x: torch.Tensor, x_tensor: sp.SparseTensor,
               mod_x: torch.Tensor, mod_o: torch.Tensor, terminal: bool,
               cache: Optional[_ObservationCache] = None, index: int = 0
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.direct_modulation_broadcast:
            return super()._block(block, o, o_coords, o_layout, x, x_tensor, mod_x, mod_o,
                                  terminal, cache, index)

        sx, scx, gx, smx, scm, gmx = block._modulation(mod_x)
        so, sco, go, smo, scmo, gmo = block._modulation(mod_o)
        x_layout = x_tensor.layout
        xo = block.norm1(x) * (1 + self._direct_modulation(scx, x_layout)) + self._direct_modulation(sx, x_layout)
        oo = (block.norm1(o) * (1 + self._direct_modulation(sco, o_layout)) + self._direct_modulation(so, o_layout)
              if len(o) else o)
        cached_kv = None if cache is None else cache.kv[index]
        refresh_o = bool(len(o)) and not terminal and index % self.o_refresh_stride == 0 and cached_kv is None
        compute_o_attention = refresh_o or (
            terminal and not self.terminal_o_kv_only and bool(len(o)) and cached_kv is None
        )
        ao, ax, built_kv = self._attention(
            block, oo, o_coords, o_layout, xo, x_tensor,
            compute_o_attention=compute_o_attention, cached_kv=cached_kv,
        )
        if cache is not None and built_kv is not None:
            cache.record(index, *built_kv)
        x = x + ax * self._direct_modulation(gx, x_layout)
        if refresh_o:
            assert ao is not None
            o = o + ao * self._direct_modulation(go, o_layout)
            o_mlp = sp.VarLenTensor(
                block.norm2(o) * (1 + self._direct_modulation(scmo, o_layout)) + self._direct_modulation(smo, o_layout),
                o_layout,
            )
            o = o + block.mlp(o_mlp).feats * self._direct_modulation(gmo, o_layout)
        x = x + block.mlp.mlp(x_tensor.replace(
            block.norm2(x) * (1 + self._direct_modulation(scm, x_layout)) + self._direct_modulation(smx, x_layout),
        )).feats * self._direct_modulation(gmx, x_layout)
        return o, x

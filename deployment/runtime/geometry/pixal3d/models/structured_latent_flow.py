from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder


class _PartialZObsInnovationGate(nn.Module):
    """Residual, per-observed-token correction for the partial z_obs carrier."""

    def __init__(self, model_channels: int, message_channels: int, dim: int, time_dim: int, num_blocks: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(num_blocks))
        self.embed_y = nn.Linear(message_channels, dim, bias=True)
        self.time_proj = nn.Linear(model_channels, time_dim, bias=True)
        self.embed_x = nn.Linear(model_channels + time_dim, dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim + 2 + time_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, 1, bias=True),
        )
        self.proj = nn.Linear(dim, model_channels, bias=True)


_LOCAL_ONE_HOP_OFFSETS = (
    (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    (0, 0, 1), (0, 1, 0), (1, 0, 0),
)
_LOCAL_TWO_HOP_OFFSETS = tuple(sorted(
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if abs(dx) + abs(dy) + abs(dz) <= 2
))
assert len(_LOCAL_TWO_HOP_OFFSETS) == 25


def _exact_coordinate_rows(
    full_coords: torch.Tensor,
    candidate_coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact full-support rows and a validity mask for candidate coords."""
    assert full_coords.ndim == candidate_coords.ndim == 2
    assert full_coords.shape[1] == candidate_coords.shape[1]
    assert full_coords.shape[0] > 0, 'local attention requires nonempty full sparse support'
    all_coords = torch.cat([full_coords, candidate_coords], dim=0)
    _, inverse = torch.unique(all_coords, dim=0, return_inverse=True)
    full_ids = inverse[:full_coords.shape[0]]
    candidate_ids = inverse[full_coords.shape[0]:]
    assert torch.unique(full_ids).numel() == full_ids.numel(), \
        'x contains duplicate coordinates'
    ordered_ids, ordered_rows = torch.sort(full_ids)
    positions = torch.searchsorted(ordered_ids, candidate_ids)
    in_range = positions < ordered_ids.numel()
    safe_positions = positions.clamp(max=ordered_ids.numel() - 1)
    valid = in_range & (ordered_ids[safe_positions] == candidate_ids)
    rows = torch.where(valid, ordered_rows[safe_positions], -torch.ones_like(safe_positions))
    return rows, valid


def _build_two_hop_observed_topology(
    full_coords: torch.Tensor,
    observed_rows: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Pack exact batch-local G(S) distance<=2 observed source/query pairs."""
    assert full_coords.ndim == 2 and full_coords.shape[1] == 4, \
        f'local attention requires [N,4] integer coords, got {tuple(full_coords.shape)}'
    assert not torch.is_floating_point(full_coords), 'local attention coordinates must be integer'
    assert observed_rows.ndim == 1 and observed_rows.dtype == torch.long
    # Validate full-coordinate uniqueness even for an all-absent batch.
    _exact_coordinate_rows(full_coords, full_coords.new_empty((0, full_coords.shape[1])))
    if observed_rows.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=full_coords.device)
        return {
            'query_rows': empty,
            'source_indices': empty,
            'offset_indices': empty,
            'segment_indices': empty,
            'neighbour_counts': empty,
        }
    assert observed_rows.min() >= 0 and observed_rows.max() < full_coords.shape[0], \
        'observed row lies outside full sparse support'
    assert torch.unique(observed_rows).numel() == observed_rows.numel(), \
        'observed rows must be unique'

    device = full_coords.device
    one_hop = torch.tensor(
        _LOCAL_ONE_HOP_OFFSETS, dtype=full_coords.dtype, device=device,
    )
    source_coords = full_coords[observed_rows]
    candidate_hop1 = source_coords[:, None, :].expand(-1, len(_LOCAL_ONE_HOP_OFFSETS), -1).clone()
    candidate_hop1[:, :, 1:] += one_hop[None]
    hop1_rows, hop1_valid = _exact_coordinate_rows(
        full_coords, candidate_hop1.reshape(-1, full_coords.shape[1]),
    )
    source_ids_hop1 = torch.arange(
        observed_rows.numel(), device=device, dtype=torch.long,
    )[:, None].expand(-1, len(_LOCAL_ONE_HOP_OFFSETS)).reshape(-1)
    hop1_rows = hop1_rows[hop1_valid]
    source_ids_hop1 = source_ids_hop1[hop1_valid]

    if hop1_rows.numel() != 0:
        candidate_hop2 = full_coords[hop1_rows][:, None, :].expand(
            -1, len(_LOCAL_ONE_HOP_OFFSETS), -1,
        ).clone()
        candidate_hop2[:, :, 1:] += one_hop[None]
        hop2_rows, hop2_valid = _exact_coordinate_rows(
            full_coords, candidate_hop2.reshape(-1, full_coords.shape[1]),
        )
        source_ids_hop2 = source_ids_hop1[:, None].expand(
            -1, len(_LOCAL_ONE_HOP_OFFSETS),
        ).reshape(-1)
        hop2_rows = hop2_rows[hop2_valid]
        source_ids_hop2 = source_ids_hop2[hop2_valid]
    else:
        hop2_rows = torch.empty(0, dtype=torch.long, device=device)
        source_ids_hop2 = torch.empty(0, dtype=torch.long, device=device)

    source_ids = torch.cat([
        torch.arange(observed_rows.numel(), device=device, dtype=torch.long),
        source_ids_hop1,
        source_ids_hop2,
    ])
    query_rows = torch.cat([observed_rows, hop1_rows, hop2_rows])
    # Multiple graph paths to the same query/source pair are one neighbourhood entry.
    unique_pairs = torch.unique(torch.stack([query_rows, source_ids], dim=1), dim=0)
    query_rows = unique_pairs[:, 0]
    source_ids = unique_pairs[:, 1]
    query_coords = full_coords[query_rows]
    selected_source_coords = source_coords[source_ids]
    assert torch.equal(query_coords[:, 0], selected_source_coords[:, 0]), \
        'local attention constructed a cross-sample edge'
    relative = query_coords[:, 1:] - selected_source_coords[:, 1:]
    offset_table = torch.tensor(
        _LOCAL_TWO_HOP_OFFSETS, dtype=full_coords.dtype, device=device,
    )
    offset_matches = (relative[:, None, :] == offset_table[None, :, :]).all(dim=-1)
    assert torch.equal(offset_matches.sum(dim=1), torch.ones_like(offset_matches[:, 0], dtype=torch.long)), \
        'local attention produced an invalid two-hop offset'
    offset_indices = offset_matches.to(dtype=torch.long).argmax(dim=1)

    # Canonical per-query real-KV order is lexicographic relative offset, independent
    # of either full-support or observed-message row ordering.
    order = torch.argsort(query_rows * len(_LOCAL_TWO_HOP_OFFSETS) + offset_indices)
    query_rows = query_rows[order]
    source_ids = source_ids[order]
    offset_indices = offset_indices[order]
    active_query_rows, segment_indices, counts = torch.unique_consecutive(
        query_rows, return_inverse=True, return_counts=True,
    )
    assert counts.max() <= len(_LOCAL_TWO_HOP_OFFSETS), \
        'local observed neighbourhood exceeds frozen 25-source cap'
    return {
        'query_rows': active_query_rows,
        'source_indices': source_ids,
        'offset_indices': offset_indices,
        'segment_indices': segment_indices,
        'neighbour_counts': counts,
    }


class _PartialZObsLocalAttention(nn.Module):
    """Shared packed two-hop observed-innovation attention for V12-A."""

    def __init__(
        self,
        model_channels: int,
        message_channels: int,
        dim: int,
        time_dim: int,
        num_heads: int,
        num_active_blocks: int,
    ):
        super().__init__()
        assert dim == 128 and time_dim == 32 and num_heads == 8
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.alpha = nn.Parameter(torch.zeros(num_active_blocks))
        self.time_proj = nn.Linear(model_channels, time_dim, bias=True)
        self.embed_x = nn.Linear(model_channels + time_dim, dim, bias=True)
        self.embed_y = nn.Linear(message_channels, dim, bias=True)
        self.key_proj = nn.Linear(dim, dim, bias=True)
        self.value_proj = nn.Linear(dim, dim, bias=True)
        self.relative_offset = nn.Parameter(torch.empty(len(_LOCAL_TWO_HOP_OFFSETS), dim))
        self.null_key = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.out_proj = nn.Linear(dim, model_channels, bias=False)

    @staticmethod
    def _rms_norm(value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(
        self,
        hidden: torch.Tensor,
        full_batch: torch.Tensor,
        t_base: torch.Tensor,
        message: torch.Tensor,
        observed_rows: torch.Tensor,
        topology: Dict[str, torch.Tensor],
        block_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert 0 <= block_index < self.alpha.numel(), 'terminal block has no V12-A residual'
        query_rows = topology['query_rows']
        source_indices = topology['source_indices']
        offset_indices = topology['offset_indices']
        segment_indices = topology['segment_indices']
        counts = topology['neighbour_counts']
        assert full_batch.ndim == 1 and full_batch.shape[0] == hidden.shape[0]
        assert full_batch.numel() == 0 or (
            full_batch.min() >= 0 and full_batch.max() < t_base.shape[0]
        ), 'full sparse batch coordinate is outside timestep batch'
        assert message.shape == (observed_rows.numel(), 34)

        if query_rows.numel() == 0:
            live_zero = sum(parameter.sum() for parameter in self.parameters()) * 0
            empty_delta = hidden.new_empty((0, hidden.shape[1])) + live_zero
            zero = hidden.new_zeros(())
            return empty_delta, {
                'null_attention_mean': zero,
                'real_neighbour_entropy_mean': zero,
                'effective_real_neighbour_count_mean': zero,
                'active_queries': zero,
                'singleton_queries': zero,
                'packed_pairs': zero,
            }

        branch_dtype = self.embed_x.weight.dtype
        hidden_branch = hidden.to(dtype=branch_dtype)
        time = self.time_proj(t_base.to(dtype=branch_dtype))
        x_embed = self.embed_x(torch.cat([
            F.layer_norm(hidden_branch, hidden_branch.shape[-1:], eps=1e-6),
            time[full_batch],
        ], dim=-1))
        innovation = self.embed_y(message.to(dtype=branch_dtype)) - x_embed[observed_rows]
        query = self._rms_norm(
            x_embed[query_rows].reshape(-1, self.num_heads, self.head_dim)
        )
        real_key = self.key_proj(innovation)[source_indices]
        real_key = real_key + self.relative_offset[offset_indices]
        real_key = self._rms_norm(real_key.reshape(-1, self.num_heads, self.head_dim))
        real_value = self.value_proj(innovation)[source_indices].reshape(
            -1, self.num_heads, self.head_dim,
        )
        real_logits = (query[segment_indices] * real_key).sum(dim=-1) / np.sqrt(self.head_dim)
        null_key = self._rms_norm(self.null_key)
        null_logits = (query * null_key[None]).sum(dim=-1) / np.sqrt(self.head_dim)

        segment_heads = segment_indices[:, None].expand(-1, self.num_heads)
        maxima = null_logits.clone()
        maxima.scatter_reduce_(0, segment_heads, real_logits, reduce='amax', include_self=True)
        real_exp = torch.exp(real_logits - maxima[segment_indices])
        null_exp = torch.exp(null_logits - maxima)
        real_sum = torch.zeros_like(null_exp).scatter_add(0, segment_heads, real_exp)
        denominator = null_exp + real_sum
        real_weight = real_exp / denominator[segment_indices]
        null_weight = null_exp / denominator
        attended = hidden_branch.new_zeros(
            (query_rows.numel(), self.num_heads, self.head_dim),
        )
        attended.index_add_(
            0, segment_indices, real_weight[..., None] * real_value,
        )
        delta = self.out_proj(attended.reshape(query_rows.numel(), self.dim))
        delta = delta * self.alpha[block_index].to(dtype=delta.dtype)

        # Entropy is conditional on selecting real evidence.  Null-vs-real
        # acceptance is reported independently as null_attention_mean; mixing
        # that mass into this distribution would make singleton entropy nonzero.
        real_conditional_weight = real_exp / real_sum[segment_indices].clamp_min(
            torch.finfo(real_sum.dtype).tiny
        )
        entropy_terms = -real_conditional_weight * torch.log(
            real_conditional_weight.clamp_min(
                torch.finfo(real_conditional_weight.dtype).tiny
            )
        )
        real_entropy = torch.zeros_like(null_weight).scatter_add(
            0, segment_heads, entropy_terms,
        )
        return delta, {
            'null_attention_mean': null_weight.mean(),
            'real_neighbour_entropy_mean': real_entropy.mean(),
            'effective_real_neighbour_count_mean': real_entropy.exp().mean(),
            'active_queries': delta.new_tensor(float(query_rows.numel())),
            'singleton_queries': delta.new_tensor(float((counts == 1).sum())),
            'packed_pairs': delta.new_tensor(float(source_indices.numel())),
        }


class _V12LACrossRope(nn.Module):
    """Apply the same 3D rotary basis to independently packed Q and K rows."""

    def __init__(
        self,
        head_dim: int,
        rope_freq: Tuple[float, float],
        reserved_dims: int = 2,
    ):
        super().__init__()
        content_dim = head_dim - reserved_dims
        assert content_dim > 0 and content_dim % 6 == 0
        self.head_dim = head_dim
        self.content_dim = content_dim
        freq_dim = content_dim // 2 // 3
        frequencies = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
        frequencies = rope_freq[0] / (rope_freq[1] ** frequencies)
        self.register_buffer('frequencies', frequencies, persistent=False)

    def forward(self, value: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        assert value.ndim == 3 and value.shape[-1] == self.head_dim
        assert coords.shape == (value.shape[0], 3)
        if value.shape[0] == 0:
            return value
        phases = torch.outer(
            coords.reshape(-1).float(), self.frequencies,
        ).reshape(coords.shape[0], self.content_dim // 2)
        phases = torch.polar(torch.ones_like(phases), phases).unsqueeze(1)
        content = torch.view_as_complex(
            value[..., :self.content_dim].float().reshape(
                value.shape[0], value.shape[1], -1, 2,
            )
        )
        rotated = torch.view_as_real(content * phases).reshape(
            value.shape[0], value.shape[1], self.content_dim,
        ).to(dtype=value.dtype)
        return torch.cat([rotated, value[..., self.content_dim:]], dim=-1)


class _PartialZObsLearnedAttention(nn.Module):
    """Five-site V12LA observation attention with an exact carrier anchor."""

    def __init__(
        self,
        model_channels: int,
        num_heads: int,
        sites: Sequence[int],
        rope_freq: Tuple[float, float],
        null_prior: float = 0.5,
    ):
        super().__init__()
        self.sites = tuple(int(site) for site in sites)
        assert self.sites == (5, 10, 15, 20, 25), self.sites
        assert num_heads == 12
        assert model_channels % num_heads == 0
        assert (model_channels // num_heads - 2) > 0
        assert (model_channels // num_heads - 2) % 6 == 0
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.head_dim = model_channels // num_heads
        self.content_dim = self.head_dim - 2
        self.site_to_slot = {site: slot for slot, site in enumerate(self.sites)}
        self.null_prior = float(null_prior)
        assert 0.0 < self.null_prior < 1.0, \
            f'learned-attention null prior must lie in (0,1), got {self.null_prior}'
        self.null_logit_offset = float(
            np.log(self.null_prior) - np.log1p(-self.null_prior)
        )

        # K/V are source-only and shared across all five sites. Q and the
        # zero-start deviation/output matrix are site-specific.
        self.key_proj = nn.Linear(model_channels, model_channels, bias=False)
        self.value_proj = nn.Linear(model_channels, model_channels, bias=False)
        self.query_proj = nn.ModuleList([
            nn.Linear(model_channels, model_channels, bias=False)
            for _ in self.sites
        ])
        self.delta_proj = nn.ModuleList([
            nn.Linear(model_channels, model_channels, bias=False)
            for _ in self.sites
        ])
        self.null_key = nn.Parameter(torch.empty(num_heads, self.content_dim))
        self.rope = _V12LACrossRope(self.head_dim, rope_freq, reserved_dims=2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.key_proj.weight)
        # Attention mixes the already trained V11 carrier basis at warm start;
        # the value map may then adapt, but does not begin as a random rewrite.
        nn.init.eye_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.null_key)
        for query_proj in self.query_proj:
            nn.init.xavier_uniform_(query_proj.weight)
        for delta_proj in self.delta_proj:
            nn.init.zeros_(delta_proj.weight)

    @staticmethod
    def _rms_norm(value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return normalized.to(dtype=value.dtype)

    @staticmethod
    def _anchor(
        carrier: torch.Tensor,
        observed_rows: torch.Tensor,
        full_rows: int,
    ) -> torch.Tensor:
        """Evaluate fixed coordinate/null attention without an additive scatter."""
        assert carrier.ndim == 2
        assert observed_rows.ndim == 1 and observed_rows.dtype == torch.long
        assert carrier.shape[0] == observed_rows.numel()
        assert full_rows >= observed_rows.numel()
        if observed_rows.numel() != 0:
            assert observed_rows.min() >= 0 and observed_rows.max() < full_rows
            assert torch.unique(observed_rows).numel() == observed_rows.numel()
        # Index zero is the exact-zero null value. Every observed row is then
        # redirected to its own carrier row; unmatched rows keep selecting null.
        values = torch.cat([carrier.new_zeros((1, carrier.shape[1])), carrier], dim=0)
        anchor_index = torch.zeros(full_rows, dtype=torch.long, device=carrier.device)
        anchor_index[observed_rows] = torch.arange(
            1, observed_rows.numel() + 1, dtype=torch.long, device=carrier.device,
        )
        return values[anchor_index]

    def build_cache(
        self,
        full_coords: torch.Tensor,
        full_layout: List[slice],
        carrier: torch.Tensor,
        observed_rows: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build the exact anchor and the shared per-sample O+null K/V cache."""
        assert full_coords.ndim == 2 and full_coords.shape[1] == 4
        assert carrier.shape == (observed_rows.numel(), self.model_channels)
        assert len(full_layout) > 0
        anchor = self._anchor(carrier, observed_rows, full_coords.shape[0])
        observed_coords = full_coords[observed_rows]
        observed_batch = observed_coords[:, 0].long()
        assert observed_batch.numel() == 0 or (
            observed_batch.min() >= 0 and observed_batch.max() < len(full_layout)
        )

        branch_dtype = self.key_proj.weight.dtype
        carrier_branch = carrier.to(dtype=branch_dtype)
        carrier_norm = F.layer_norm(
            carrier_branch, (self.model_channels,), eps=1e-6,
        )
        real_key = self.key_proj(carrier_norm).reshape(
            -1, self.num_heads, self.head_dim,
        )
        real_key = self.rope(real_key, observed_coords[:, 1:])
        real_key = self._rms_norm(real_key[..., :self.content_dim])
        real_key = torch.cat([
            real_key,
            real_key.new_zeros((real_key.shape[0], self.num_heads, 2)),
        ], dim=-1)
        real_value = self.value_proj(carrier_branch).reshape(
            -1, self.num_heads, self.head_dim,
        )
        null_content = self._rms_norm(
            self.null_key.to(dtype=real_key.dtype)
        )

        packed_kv = []
        observed_counts = []
        for batch_index in range(len(full_layout)):
            source_indices = torch.nonzero(
                observed_batch == batch_index, as_tuple=False,
            ).flatten()
            observed_count = int(source_indices.numel())
            observed_counts.append(observed_count)
            # One null competes with O real keys.  log(O) cancels its cardinality
            # disadvantage; the constant logit offset sets the null mass only in
            # the equal-content-logit reference case, not after learned Q/K drift.
            null_reserved = real_key.new_zeros((self.num_heads, 2))
            null_reserved[:, 0] = np.sqrt(self.head_dim) * (
                np.log(max(observed_count, 1)) + self.null_logit_offset
            )
            null_key = torch.cat([null_content, null_reserved], dim=-1).unsqueeze(0)
            null_value = real_value.new_zeros((1, self.num_heads, self.head_dim))
            keys = torch.cat([real_key[source_indices], null_key], dim=0)
            values = torch.cat([real_value[source_indices], null_value], dim=0)
            packed_kv.append(torch.stack([keys, values], dim=1))

        kv = sp.VarLenTensor.from_tensor_list(packed_kv)
        return {
            'anchor': anchor,
            'kv': kv,
            'observed_rows': observed_rows,
            'observed_counts': tuple(observed_counts),
        }

    def forward(
        self,
        hidden: torch.Tensor,
        full_coords: torch.Tensor,
        full_layout: List[slice],
        cache: Dict[str, Any],
        block_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert block_index in self.site_to_slot, block_index
        assert hidden.shape == (full_coords.shape[0], self.model_channels)
        slot = self.site_to_slot[block_index]
        branch_dtype = self.query_proj[slot].weight.dtype
        hidden_branch = hidden.to(dtype=branch_dtype)
        query = self.query_proj[slot](F.layer_norm(
            hidden_branch, (self.model_channels,), eps=1e-6,
        )).reshape(-1, self.num_heads, self.head_dim)
        query = self.rope(query, full_coords[:, 1:])
        query_content = self._rms_norm(query[..., :self.content_dim])
        query_reserved = query.new_zeros((query.shape[0], self.num_heads, 2))
        query_reserved[..., 0] = 1
        query = torch.cat([query_content, query_reserved], dim=-1)
        query_varlen = sp.VarLenTensor(query, full_layout)
        attended = sp.sparse_scaled_dot_product_attention(
            query_varlen, cache['kv'],
        ).feats.reshape(-1, self.model_channels)

        anchor = cache['anchor'].to(dtype=attended.dtype)
        deviation_input = attended - anchor
        correction = self.delta_proj[slot](deviation_input)
        inject = anchor + correction
        eps = torch.finfo(inject.dtype).eps
        observed_rows = cache['observed_rows']
        observed_mask = torch.zeros(
            inject.shape[0], dtype=torch.bool, device=inject.device,
        )
        observed_mask[observed_rows] = True
        return inject, {
            'correction_to_anchor_norm_ratio': correction.norm() / anchor.norm().clamp_min(eps),
            'correction_to_hidden_norm_ratio': correction.norm() / hidden_branch.norm().clamp_min(eps),
            'observed_correction_l2_norm': correction[observed_mask].float().norm(),
            'unobserved_correction_l2_norm': correction[~observed_mask].float().norm(),
            'attention_output_l2_norm': attended.float().norm(),
            'anchor_l2_norm': anchor.float().norm(),
            'delta_weight_l2_norm': self.delta_proj[slot].weight.float().norm(),
        }

class SLatFlowModel(nn.Module):
    """
    Structured Latent Flow Model for 3D generation.
    
    Supports two conditioning modes:
    - "cross": Standard cross-attention with image features
    - "proj": View-aligned projection attention with camera-aware features
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_attn_mode: Literal["cross", "proj", "gated_proj"] = "cross",
        proj_in_channels: Optional[int] = None,
        vae_in_channels: Optional[int] = None,
        zobs_inject: bool = False,
        zobs_inject_in_channels: int = 33,
        zobs_inject_scale: float = 1.0,
        partial_zobs_inject: bool = False,
        partial_zobs_in_channels: int = 34,
        partial_zobs_inject_position: Literal['post_block', 'pre_mlp'] = 'post_block',
        legacy_zobs_zero_input: bool = False,
        partial_zobs_innovation_gate: bool = False,
        innovation_gate_dim: int = 128,
        innovation_gate_time_dim: int = 32,
        partial_zobs_innovation_gate_train_only: bool = False,
        partial_zobs_local_attention: bool = False,
        local_attention_dim: int = 128,
        local_attention_time_dim: int = 32,
        local_attention_heads: int = 8,
        partial_zobs_local_attention_train_only: bool = False,
        partial_zobs_learned_attention: bool = False,
        learned_attention_sites: Sequence[int] = (5, 10, 15, 20, 25),
        learned_attention_heads: int = 12,
        learned_attention_null_prior: float = 0.5,
        partial_zobs_learned_attention_train_only: bool = False,
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.image_attn_mode = image_attn_mode
        self.proj_in_channels = proj_in_channels
        self.vae_in_channels = vae_in_channels
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        # Trained-bridge: single shared per-block injector of [z_obs (x) vis (32) | vis (1)] -> model_channels.
        self.zobs_inject = zobs_inject
        self.zobs_inject_scale = float(zobs_inject_scale)
        self.zobs_injector = sp.SparseLinear(zobs_inject_in_channels, model_channels) if zobs_inject else None
        # v8-partial: the observation lives only on its observed support.  This adapter is
        # deliberately bias-free: absent rows never acquire a learned content/fill value.
        self.partial_zobs_inject = partial_zobs_inject
        self.partial_zobs_in_channels = partial_zobs_in_channels
        assert partial_zobs_inject_position in ('post_block', 'pre_mlp'), \
            f'partial_zobs_inject_position must be post_block or pre_mlp, got {partial_zobs_inject_position!r}'
        self.partial_zobs_inject_position = partial_zobs_inject_position
        if partial_zobs_inject_position == 'pre_mlp':
            assert not any((
                partial_zobs_innovation_gate,
                partial_zobs_local_attention,
                partial_zobs_learned_attention,
            )), 'partial_zobs_inject_position=pre_mlp is incompatible with post-block experimental branches'
        self.legacy_zobs_zero_input = legacy_zobs_zero_input
        self.partial_zobs_injector = (
            sp.SparseLinear(partial_zobs_in_channels, model_channels, bias=False)
            if partial_zobs_inject else None
        )
        self.partial_zobs_innovation_gate = partial_zobs_innovation_gate
        self.partial_zobs_innovation_gate_train_only = partial_zobs_innovation_gate_train_only
        if partial_zobs_innovation_gate_train_only:
            assert partial_zobs_innovation_gate, \
                'partial_zobs_innovation_gate_train_only requires partial_zobs_innovation_gate=true'
        if partial_zobs_innovation_gate:
            assert partial_zobs_inject, 'partial_zobs_innovation_gate requires partial_zobs_inject=true'
            assert partial_zobs_in_channels == 34, \
                f'partial_zobs_innovation_gate requires 34 input channels, got {partial_zobs_in_channels}'
            assert innovation_gate_dim == 128, f'innovation_gate_dim is fixed at 128, got {innovation_gate_dim}'
            assert innovation_gate_time_dim == 32, \
                f'innovation_gate_time_dim is fixed at 32, got {innovation_gate_time_dim}'
            assert model_channels == 1536, f'partial_zobs_innovation_gate requires 1536 model channels, got {model_channels}'
            assert num_blocks == 30, f'partial_zobs_innovation_gate requires 30 blocks, got {num_blocks}'
            self.innovation_gate = _PartialZObsInnovationGate(
                model_channels, partial_zobs_in_channels, innovation_gate_dim,
                innovation_gate_time_dim, num_blocks,
            )
        else:
            self.innovation_gate = None
        self.partial_zobs_local_attention = partial_zobs_local_attention
        self.partial_zobs_local_attention_train_only = partial_zobs_local_attention_train_only
        if partial_zobs_local_attention_train_only:
            assert partial_zobs_local_attention, \
                'partial_zobs_local_attention_train_only requires partial_zobs_local_attention=true'
        assert sum(bool(value) for value in (
            partial_zobs_innovation_gate,
            partial_zobs_local_attention,
            partial_zobs_learned_attention,
        )) <= 1, 'partial-zobs experimental attention/gate branches are mutually exclusive'
        if partial_zobs_local_attention:
            assert partial_zobs_inject, 'partial_zobs_local_attention requires partial_zobs_inject=true'
            assert partial_zobs_in_channels == 34, \
                f'partial_zobs_local_attention requires 34 input channels, got {partial_zobs_in_channels}'
            assert local_attention_dim == 128, \
                f'local_attention_dim is fixed at 128, got {local_attention_dim}'
            assert local_attention_time_dim == 32, \
                f'local_attention_time_dim is fixed at 32, got {local_attention_time_dim}'
            assert local_attention_heads == 8, \
                f'local_attention_heads is fixed at 8, got {local_attention_heads}'
            assert model_channels == 1536, \
                f'partial_zobs_local_attention requires 1536 model channels, got {model_channels}'
            assert num_blocks == 30, \
                f'partial_zobs_local_attention requires 30 blocks, got {num_blocks}'
            self.local_obs_attention = _PartialZObsLocalAttention(
                model_channels, partial_zobs_in_channels, local_attention_dim,
                local_attention_time_dim, local_attention_heads, num_blocks - 1,
            )
        else:
            self.local_obs_attention = None

        self.partial_zobs_learned_attention = partial_zobs_learned_attention
        self.partial_zobs_learned_attention_train_only = \
            partial_zobs_learned_attention_train_only
        if partial_zobs_learned_attention_train_only:
            assert partial_zobs_learned_attention, \
                'partial_zobs_learned_attention_train_only requires partial_zobs_learned_attention=true'
        if partial_zobs_learned_attention:
            assert partial_zobs_inject, \
                'partial_zobs_learned_attention requires partial_zobs_inject=true'
            assert partial_zobs_in_channels == 34, \
                f'partial_zobs_learned_attention requires 34 input channels, got {partial_zobs_in_channels}'
            assert model_channels == 1536, \
                f'partial_zobs_learned_attention requires 1536 model channels, got {model_channels}'
            assert num_blocks == 30, \
                f'partial_zobs_learned_attention requires 30 blocks, got {num_blocks}'
            assert learned_attention_heads == 12, \
                f'partial_zobs_learned_attention requires 12 heads, got {learned_attention_heads}'
            assert tuple(learned_attention_sites) == (5, 10, 15, 20, 25), \
                f'V12LA sites are fixed at [5,10,15,20,25], got {learned_attention_sites}'
            self.learned_attention = _PartialZObsLearnedAttention(
                model_channels,
                learned_attention_heads,
                learned_attention_sites,
                rope_freq,
                learned_attention_null_prior,
            )
        else:
            self.learned_attention = None

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                image_attn_mode=image_attn_mode,
                proj_in_channels=proj_in_channels,
                vae_in_channels=vae_in_channels,
            )
            for _ in range(num_blocks)
        ])
            
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)
        if self.partial_zobs_innovation_gate_train_only:
            self._freeze_for_partial_zobs_innovation_gate()
        if self.partial_zobs_local_attention_train_only:
            self._freeze_for_partial_zobs_local_attention()
        if self.partial_zobs_learned_attention_train_only:
            self._freeze_for_partial_zobs_learned_attention()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))
        if self.learned_attention is not None:
            self.learned_attention.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)
            
        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            
            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                # Handle cross, proj, and gated_proj modes
                if self.image_attn_mode in ("proj", "gated_proj"):
                    block.cross_attn.cross_attn_block.to_out.apply(_scaled_init)
                else:
                    block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)
            
            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)
            
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        # Trained-bridge: zero-init the z_obs injector LAST (after _basic_init), so step-0 injection == 0.
        if getattr(self, 'zobs_injector', None) is not None:
            nn.init.zeros_(self.zobs_injector.weight)
            nn.init.zeros_(self.zobs_injector.bias)
        if getattr(self, 'partial_zobs_injector', None) is not None:
            nn.init.zeros_(self.partial_zobs_injector.weight)
        if getattr(self, 'innovation_gate', None) is not None:
            nn.init.zeros_(self.innovation_gate.alpha)
        if getattr(self, 'local_obs_attention', None) is not None:
            # V12-A freezes vanilla Xavier initialization independently of the
            # scaled V11 backbone initialization mode. Only alpha starts at zero.
            for module in self.local_obs_attention.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(self.local_obs_attention.relative_offset)
            nn.init.xavier_uniform_(self.local_obs_attention.null_key)
            nn.init.zeros_(self.local_obs_attention.alpha)
        if getattr(self, 'learned_attention', None) is not None:
            self.learned_attention.reset_parameters()

    def _freeze_for_partial_zobs_innovation_gate(self) -> None:
        """Make the V12 gate the sole trainable branch before an optimizer is built."""
        assert self.innovation_gate is not None
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.innovation_gate.parameters():
            parameter.requires_grad_(True)
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        expected = [f'innovation_gate.{name}' for name in self.innovation_gate.state_dict()]
        assert trainable == expected, (trainable, expected)
        assert sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad) == 473663

    def _freeze_for_partial_zobs_local_attention(self) -> None:
        """Make the V12-A local attention the sole trainable warm-start branch."""
        assert self.local_obs_attention is not None
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.local_obs_attention.parameters():
            parameter.requires_grad_(True)
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        expected = [f'local_obs_attention.{name}' for name in self.local_obs_attention.state_dict()]
        assert trainable == expected, (trainable, expected)
        assert len(trainable) == 14
        assert sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad) == 487485

    def _freeze_for_partial_zobs_learned_attention(self) -> None:
        """Make the V12LA module the sole trainable warm-start branch."""
        assert self.learned_attention is not None
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.learned_attention.parameters():
            parameter.requires_grad_(True)
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        expected = [
            f'learned_attention.{name}'
            for name, _ in self.learned_attention.named_parameters()
        ]
        assert trainable == expected, (trainable, expected)
        assert len(trainable) == 13, trainable

    def _partial_zobs_message(
        self,
        x: sp.SparseTensor,
        partial_z_obs: sp.SparseTensor,
        partial_z_obs_keep_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project a variable-support observation and join its messages onto ``x`` rows.

        The return value stays variable-length: projected messages plus their exact full-row
        indices.  Neither observation content nor a zero-filled model-channel tensor is
        materialized on full support. ``torch.unique(..., dim=0)`` makes the coordinate join
        exact and rejects duplicate or non-subset partial coordinates instead of assuming a
        row order.
        """
        adapter = self.partial_zobs_injector
        assert adapter is not None
        assert partial_z_obs.feats.ndim == 2 and \
            partial_z_obs.feats.shape[-1] == self.partial_zobs_in_channels, \
            f'partial_z_obs must be [N,{self.partial_zobs_in_channels}], got {tuple(partial_z_obs.feats.shape)}'
        assert partial_z_obs.coords.ndim == 2 and partial_z_obs.coords.shape[1] == x.coords.shape[1], \
            f'partial_z_obs coordinate rank mismatch: {tuple(partial_z_obs.coords.shape)} vs {tuple(x.coords.shape)}'

        active = torch.ones(partial_z_obs.feats.shape[0], dtype=torch.bool, device=partial_z_obs.feats.device)
        if partial_z_obs_keep_mask is not None:
            assert partial_z_obs_keep_mask.ndim == 1 and partial_z_obs_keep_mask.shape[0] == x.shape[0], \
                f'partial_z_obs_keep_mask must be [B={x.shape[0]}], got {tuple(partial_z_obs_keep_mask.shape)}'
            keep = partial_z_obs_keep_mask.to(device=partial_z_obs.feats.device, dtype=torch.bool)
            active = keep[partial_z_obs.coords[:, 0].long()]

        active_coords = partial_z_obs.coords[active]
        active_feats = partial_z_obs.feats[active]
        # F.linear on a zero-row tensor is intentional: it keeps the adapter parameters in the
        # graph on all-drop batches, so DDP sees a zero (rather than None) gradient.
        active_messages = F.linear(active_feats, adapter.weight, None)

        full_rows = torch.empty(0, dtype=torch.long, device=x.feats.device)
        if active_coords.shape[0] != 0:
            all_coords = torch.cat([x.coords, active_coords], dim=0)
            _, inverse = torch.unique(all_coords, dim=0, return_inverse=True)
            x_ids = inverse[:x.coords.shape[0]]
            partial_ids = inverse[x.coords.shape[0]:]
            assert torch.unique(x_ids).numel() == x_ids.numel(), 'x contains duplicate coordinates'
            assert torch.unique(partial_ids).numel() == partial_ids.numel(), 'partial_z_obs contains duplicate coordinates'
            ordered_x_ids, ordered_x_rows = torch.sort(x_ids)
            positions = torch.searchsorted(ordered_x_ids, partial_ids)
            assert (positions < ordered_x_ids.numel()).all() and \
                torch.equal(ordered_x_ids[positions], partial_ids), \
                'partial_z_obs coordinates must be an exact subset of current x coordinates'
            full_rows = ordered_x_rows[positions]

        return active_messages, full_rows, active_feats, active_coords[:, 0].long()

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor], Dict[str, Union[torch.Tensor, sp.SparseTensor]], Tuple],
        concat_cond: Optional[sp.SparseTensor] = None,
        z_obs: Optional[sp.SparseTensor] = None,
        partial_z_obs: Optional[sp.SparseTensor] = None,
        partial_z_obs_keep_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> sp.SparseTensor:
        """
        Forward pass.

        Args:
            x: SparseTensor input
            t: Timestep tensor [B]
            cond: Conditioning tensor. For "cross" mode: list of tensors or tensor.
                  For "proj" mode: dict {'global': global_cond, 'proj': proj_cond}
                  or tuple of (global_cond, proj_cond)
            concat_cond: Optional concatenation condition
            z_obs: Optional 33ch [z_obs(32)|vis(1)] sparse tensor, same coords as x. Feeds ONLY
                   the zobs_injector (v6 patchcross path: concat_cond is the plain 32ch shape
                   latent and z_obs never enters the input_layer). When None, the legacy 97ch
                   concat slice below applies — shipped/v4 behavior is bit-identical.
            partial_z_obs: Optional variable-length v8 observation
                [normalized_z_obs(32)|coverage(1)|boundary_distance(1)].
                Its coordinates must be a subset of x; it is consumed only by the dedicated,
                bias-free partial adapter and is never converted to full-support content.

        Returns:
            SparseTensor output
        """
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)

        # Trained-bridge: compute the per-block injection vector ONCE (shared zero-init linear, added each block).
        # Sources, in priority order: separate z_obs kwarg (v6), else the trailing channels of a
        # 97-ch [x_t|shape|z_obs|vis] input (v4). z_obs*vis==0 at vis=0 in both.
        zobs_inject = None
        _injector = getattr(self, 'zobs_injector', None)
        if z_obs is not None:
            # Fail fast: without an injector the kwarg would be a silent no-op, and in the 64ch
            # patchcross architecture the injector is the ONLY z_obs consumption path.
            assert _injector is not None, \
                'z_obs kwarg passed but the denoiser was built without zobs_inject=true'
            # Full coords equality is asserted at data-load; a row-count check catches gross mismatch.
            assert z_obs.feats.shape[-1] == 33 and z_obs.feats.shape[0] == x.feats.shape[0], \
                f'z_obs kwarg mismatch: {tuple(z_obs.feats.shape)} vs x rows {x.feats.shape[0]}'
            _zo = z_obs.feats[:, :32]
            _vis = z_obs.feats[:, 32:33]
            zobs_inject = manual_cast(self.zobs_injector(z_obs.replace(torch.cat([_zo * _vis, _vis], dim=-1))), self.dtype)
            if not getattr(self, '_zobs_kwarg_logged', False):  # in-log proof hook (once per process)
                print('[zobs_injector] consuming separate z_obs kwarg (patchcross path)', flush=True)
                self._zobs_kwarg_logged = True
        elif _injector is not None and x.feats.shape[-1] >= 97:
            _zo = x.feats[:, 64:96]
            _vis = x.feats[:, 96:97]
            zobs_inject = manual_cast(self.zobs_injector(x.replace(torch.cat([_zo * _vis, _vis], dim=-1))), self.dtype)
        elif _injector is not None and self.legacy_zobs_zero_input:
            # 64ch v8 shell with a legacy injector: preserve the legacy all-zero z_obs mode,
            # including its learned per-block bias, without materialising observation content.
            # Evaluate one zero row and broadcast the result: this is exactly W*0+b, keeps a
            # zero (non-None) weight gradient under DDP, and avoids an N-row 33ch GEMM.
            zero_output = F.linear(
                x.feats.new_zeros((1, _injector.in_features)),
                _injector.weight,
                _injector.bias,
            )
            zobs_inject = manual_cast(
                x.replace(zero_output.expand(x.feats.shape[0], -1)), self.dtype
            )
        elif _injector is not None and not getattr(self, '_zobs_none_warned', False):
            # Injector present but NO z_obs source (64ch input, kwarg omitted): the trained
            # per-block bias is NOT applied — a third mode distinct from "z_obs dropped" (an
            # all-zero 33ch tensor still adds the bias each block). A nozobs arm on a patchcross
            # model must pass the zero tensor, never omit the kwarg.
            print('[zobs_injector] WARNING: injector present but no z_obs source (kwarg omitted, '
                  'input < 97ch) — per-block bias NOT applied; pass an all-zero 33ch z_obs for a '
                  'true nozobs arm', flush=True)
            self._zobs_none_warned = True

        partial_zobs_inject = None
        partial_zobs_rows = None
        partial_zobs_feats = None
        partial_zobs_batch = None
        if partial_z_obs is not None:
            assert self.partial_zobs_injector is not None, \
                'partial_z_obs passed but denoiser was built without partial_zobs_inject=true'
            partial_zobs_inject, partial_zobs_rows, partial_zobs_feats, partial_zobs_batch = self._partial_zobs_message(
                x, partial_z_obs, partial_z_obs_keep_mask
            )
            partial_zobs_inject = manual_cast(partial_zobs_inject, self.dtype)
        elif self.partial_zobs_injector is not None:
            raise AssertionError('partial_zobs_inject=true requires a separate partial_z_obs tensor')

        if zobs_inject is not None and self.zobs_inject_scale != 1.0:
            zobs_inject = zobs_inject.replace(zobs_inject.feats * self.zobs_inject_scale)
            if not getattr(self, '_zobs_inject_scale_logged', False):
                print(f'[zobs_injector] output scale={self.zobs_inject_scale:g}', flush=True)
                self._zobs_inject_scale_logged = True

        # Keep the pre-AdaLN embedding for the innovation gate; blocks retain their existing
        # modulation input exactly.
        t_base = self.t_embedder(t)
        t_emb = t_base
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)

        innovation_time = None
        innovation_y = None
        innovation_gate_y = None
        if self.innovation_gate is not None and partial_zobs_feats is not None:
            # Observation-only terms are shared by all blocks. Empty rows deliberately pass
            # through these linears too, so an all-drop batch keeps the full gate graph live.
            gate_dtype = self.innovation_gate.embed_y.weight.dtype
            innovation_gate_y = partial_zobs_feats.to(dtype=gate_dtype)
            innovation_y = self.innovation_gate.embed_y(innovation_gate_y)
            innovation_time = self.innovation_gate.time_proj(t_base)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
            
        # Handle different conditioning modes
        if self.image_attn_mode == 'proj':
            if isinstance(cond, dict):
                global_cond = cond['global']
                proj_cond = cond['proj']
            else:
                global_cond, proj_cond = cond
            if isinstance(global_cond, list):
                global_cond = sp.VarLenTensor.from_tensor_list(global_cond)
            global_cond = manual_cast(global_cond, self.dtype)
            proj_cond = manual_cast(proj_cond, self.dtype)
            cond = (global_cond, proj_cond)
        elif self.image_attn_mode == 'gated_proj':
            global_cond = cond['global']
            if isinstance(global_cond, list):
                global_cond = sp.VarLenTensor.from_tensor_list(global_cond)
            global_cond = manual_cast(global_cond, self.dtype)
            proj_semantic = manual_cast(cond['proj_semantic'], self.dtype)
            proj_color = manual_cast(cond['proj_color'], self.dtype)
            cond = {'global': global_cond, 'proj_semantic': proj_semantic, 'proj_color': proj_color}
        else:
            if isinstance(cond, list):
                cond = sp.VarLenTensor.from_tensor_list(cond)
            cond = manual_cast(cond, self.dtype)

        # Inference-only probe hook. When the attribute is absent (the default, and the
        # only state any training path produces) this is None and the loop below is
        # byte-identical to the unhooked path: no constructor arg, no config key, and no
        # entry in the inference ABI contract. A probe sets it to a length-num_blocks
        # sequence to rescale the shared partial message per block.
        _partial_scales = getattr(self, 'partial_zobs_layer_scales', None)
        # Optional evaluator-only callable: ``fn(gate_value, active_batch, block_index)``.
        # It is an instance attribute rather than a constructor/config argument, so no model
        # state or training ABI changes; callers can implement neutral, constant, or shuffled
        # gate-value probes deterministically.
        _gate_value_transform = getattr(self, 'partial_zobs_innovation_gate_value_transform', None)
        _innovation_ratio_stats = [] if self.innovation_gate is not None else None
        _innovation_gate_mean_stats = [] if self.innovation_gate is not None else None
        _innovation_observed_rows = 0 if partial_zobs_rows is None else int(partial_zobs_rows.numel())
        _local_attention = getattr(self, 'local_obs_attention', None)
        _local_topology = None
        _local_full_batch = None
        _local_ratio_base_stats = [] if _local_attention is not None else None
        _local_ratio_hidden_stats = [] if _local_attention is not None else None
        _local_null_stats = [] if _local_attention is not None else None
        _local_entropy_stats = [] if _local_attention is not None else None
        _local_effective_neighbour_stats = [] if _local_attention is not None else None
        _local_observed_residual_stats = [] if _local_attention is not None else None
        _local_unobserved_residual_stats = [] if _local_attention is not None else None
        _local_query_is_observed = None
        _local_neighbourhood_histogram = None
        if _local_attention is not None:
            assert partial_zobs_rows is not None and partial_zobs_feats is not None
            _local_topology = _build_two_hop_observed_topology(x.coords, partial_zobs_rows)
            _local_full_batch = x.coords[:, 0].long()
            _active_observed_mask = torch.zeros(
                x.coords.shape[0], dtype=torch.bool, device=x.coords.device,
            )
            _active_observed_mask[partial_zobs_rows] = True
            _local_query_is_observed = _active_observed_mask[
                _local_topology['query_rows']
            ]
            _local_neighbourhood_histogram = torch.bincount(
                _local_topology['neighbour_counts'], minlength=26,
            )[:26]
            _local_neighbourhood_histogram[0] = (
                x.coords.shape[0] - _local_topology['query_rows'].numel()
            )
            assert int(_local_neighbourhood_histogram.sum()) == x.coords.shape[0]
            assert int(_local_neighbourhood_histogram[1:].sum()) == \
                _local_topology['query_rows'].numel()
        _learned_attention = getattr(self, 'learned_attention', None)
        _learned_cache = None
        _learned_stats = [] if _learned_attention is not None else None
        if _learned_attention is not None:
            assert partial_zobs_inject is not None and partial_zobs_rows is not None
            assert _partial_scales is None, \
                'partial_zobs_layer_scales is not defined for the five-site V12LA operator'
            _learned_cache = _learned_attention.build_cache(
                x.coords,
                x.layout,
                partial_zobs_inject,
                partial_zobs_rows,
            )
        partial_zobs_inject_position = getattr(
            self, 'partial_zobs_inject_position', 'post_block',
        )
        for _block_index, block in enumerate(self.blocks):
            if partial_zobs_inject is not None and partial_zobs_inject_position == 'pre_mlp':
                _pre_mlp_scale = 1.0 if _partial_scales is None else float(
                    _partial_scales[_block_index]
                )
                h = block(
                    h,
                    t_emb,
                    cond,
                    partial_zobs_rows=partial_zobs_rows,
                    partial_zobs_inject=partial_zobs_inject,
                    partial_zobs_inject_scale=_pre_mlp_scale,
                )
            else:
                h = block(h, t_emb, cond)
            if zobs_inject is not None:
                h = h + zobs_inject
            _innovation_ratio = None
            _innovation_gate_mean = None
            _local_delta = None
            _local_stats = None
            if _local_attention is not None and _block_index < _local_attention.alpha.numel():
                assert _local_topology is not None and _local_full_batch is not None
                _local_delta, _local_stats = _local_attention(
                    h.feats,
                    _local_full_batch,
                    t_base,
                    partial_zobs_feats,
                    partial_zobs_rows,
                    _local_topology,
                    _block_index,
                )
                _local_delta = manual_cast(_local_delta, self.dtype)
                _local_ratio_base = _local_delta.norm() / partial_zobs_inject.norm().clamp_min(
                    torch.finfo(partial_zobs_inject.dtype).eps
                )
                _local_hidden = h.feats[_local_topology['query_rows']]
                _local_ratio_hidden = _local_delta.norm() / _local_hidden.norm().clamp_min(
                    torch.finfo(_local_hidden.dtype).eps
                )
                assert _local_query_is_observed is not None
                _local_observed_residual = _local_delta[
                    _local_query_is_observed
                ].float().norm()
                _local_unobserved_residual = _local_delta[
                    ~_local_query_is_observed
                ].float().norm()
            if _learned_attention is not None and _block_index in _learned_attention.site_to_slot:
                assert _learned_cache is not None
                _learned_inject, _site_stats = _learned_attention(
                    h.feats,
                    h.coords,
                    h.layout,
                    _learned_cache,
                    _block_index,
                )
                h = h.replace(h.feats + manual_cast(_learned_inject, self.dtype))
                _learned_stats.append({
                    name: value.detach() for name, value in _site_stats.items()
                })
            if partial_zobs_inject is not None and _learned_attention is None and \
                    partial_zobs_inject_position == 'post_block':
                layer_inject = partial_zobs_inject
                if self.innovation_gate is not None:
                    gate = self.innovation_gate
                    # Every operation intentionally accepts an empty leading dimension. This
                    # keeps all innovation parameters in the all-drop DDP graph.
                    assert innovation_gate_y is not None and innovation_y is not None and innovation_time is not None
                    t_obs = innovation_time[partial_zobs_batch]
                    h_obs = h.feats[partial_zobs_rows].to(dtype=gate.embed_y.weight.dtype)
                    x_embed = gate.embed_x(torch.cat([h_obs, t_obs], dim=-1))
                    innovation = innovation_y - x_embed
                    gate_input = torch.cat([
                        innovation,
                        innovation_gate_y[:, 32:34],
                        t_obs,
                    ], dim=-1)
                    gate_value = torch.sigmoid(gate.mlp(gate_input))
                    if _gate_value_transform is not None:
                        gate_value = _gate_value_transform(
                            gate_value, partial_zobs_batch, _block_index
                        )
                        assert gate_value.shape == (innovation.shape[0], 1), \
                            f'gate value transform must return [Nobs,1], got {tuple(gate_value.shape)}'
                    delta = gate.proj(innovation) * gate_value
                    delta = delta * gate.alpha[_block_index].to(dtype=delta.dtype)
                    if _block_index == len(self.blocks) - 1:
                        # Keep the frozen V11 base carrier at the terminal site,
                        # but never add a newly learned observed-only shortcut:
                        # no attention block remains to transport/smooth it into U.
                        delta = delta * 0
                    # Detached telemetry for the training log. This is before the legacy
                    # layer-scale hook and therefore measures the learned branch itself.
                    _innovation_ratio = delta.norm() / partial_zobs_inject.norm().clamp_min(
                        torch.finfo(partial_zobs_inject.dtype).eps
                    )
                    _innovation_gate_mean = (
                        gate_value.mean() if gate_value.numel() != 0 else delta.new_zeros(())
                    )
                    layer_inject = partial_zobs_inject + manual_cast(delta, self.dtype)
                if partial_zobs_rows.numel() != 0:
                    if _partial_scales is None:
                        h = h.replace(h.feats.index_add(
                            0, partial_zobs_rows, layer_inject
                        ))
                    else:
                        h = h.replace(h.feats.index_add(
                            0, partial_zobs_rows, layer_inject,
                            alpha=float(_partial_scales[_block_index]),
                        ))
                else:
                    # Preserve every empty carrier/gate GEMM in the graph on all-drop batches.
                    h = h.replace(h.feats + layer_inject.sum() * 0)
                if _local_delta is not None:
                    _local_scale = 1.0 if _partial_scales is None else float(
                        _partial_scales[_block_index]
                    )
                    if _local_topology['query_rows'].numel() != 0:
                        h = h.replace(h.feats.index_add(
                            0, _local_topology['query_rows'], _local_delta,
                            alpha=_local_scale,
                        ))
                    else:
                        # The module returns an exact-zero parameter edge for all 14
                        # trainable tensors, so DDP observes finite numerical-zero grads.
                        h = h.replace(h.feats + _local_delta.sum() * _local_scale)
            if self.innovation_gate is not None:
                if _innovation_ratio is None:
                    _innovation_ratio = h.feats.new_zeros(())
                    _innovation_gate_mean = h.feats.new_zeros(())
                _innovation_ratio_stats.append(_innovation_ratio.detach())
                _innovation_gate_mean_stats.append(_innovation_gate_mean.detach())
            if _local_attention is not None and _block_index < _local_attention.alpha.numel():
                assert _local_stats is not None
                _local_ratio_base_stats.append(_local_ratio_base.detach())
                _local_ratio_hidden_stats.append(_local_ratio_hidden.detach())
                _local_null_stats.append(_local_stats['null_attention_mean'].detach())
                _local_entropy_stats.append(_local_stats['real_neighbour_entropy_mean'].detach())
                _local_effective_neighbour_stats.append(
                    _local_stats['effective_real_neighbour_count_mean'].detach()
                )
                _local_observed_residual_stats.append(
                    _local_observed_residual.detach()
                )
                _local_unobserved_residual_stats.append(
                    _local_unobserved_residual.detach()
                )

        if self.innovation_gate is not None:
            # Instance-only telemetry: deliberately neither buffer nor state-dict entry.
            self.partial_zobs_innovation_gate_last_stats = {
                'delta_to_base_norm_ratio': torch.stack(_innovation_ratio_stats),
                'gate_mean': torch.stack(_innovation_gate_mean_stats),
                'observed_rows': _innovation_observed_rows,
            }
        if _local_attention is not None:
            # Block 29 is structurally absent: every telemetry vector has length 29.
            assert _local_query_is_observed is not None
            assert _local_neighbourhood_histogram is not None
            _full_support_rows = int(x.coords.shape[0])
            _active_queries = int(_local_topology['query_rows'].numel())
            _empty_queries = _full_support_rows - _active_queries
            _singleton_queries = int(
                (_local_topology['neighbour_counts'] == 1).sum()
            )
            self.partial_zobs_local_attention_last_stats = {
                'delta_to_base_norm_ratio': torch.stack(_local_ratio_base_stats),
                'delta_to_hidden_norm_ratio': torch.stack(_local_ratio_hidden_stats),
                'null_attention_mean': torch.stack(_local_null_stats),
                'real_neighbour_entropy_mean': torch.stack(_local_entropy_stats),
                'effective_real_neighbour_count_mean': torch.stack(
                    _local_effective_neighbour_stats
                ),
                'observed_residual_l2_norm': torch.stack(
                    _local_observed_residual_stats
                ),
                'unobserved_residual_l2_norm': torch.stack(
                    _local_unobserved_residual_stats
                ),
                'observed_rows': _innovation_observed_rows,
                'full_support_rows': _full_support_rows,
                'active_queries': _active_queries,
                'empty_queries': _empty_queries,
                'singleton_queries': _singleton_queries,
                'observed_active_queries': int(_local_query_is_observed.sum()),
                'unobserved_active_queries': int((~_local_query_is_observed).sum()),
                'active_query_fraction': _active_queries / max(_full_support_rows, 1),
                'empty_query_fraction': _empty_queries / max(_full_support_rows, 1),
                'singleton_query_fraction': _singleton_queries / max(_active_queries, 1),
                'neighbourhood_size_histogram': _local_neighbourhood_histogram.detach(),
                'packed_pairs': int(_local_topology['source_indices'].numel()),
            }
        if _learned_attention is not None:
            assert _learned_cache is not None
            assert len(_learned_stats) == len(_learned_attention.sites)
            self.partial_zobs_learned_attention_last_stats = {
                name: torch.stack([site[name] for site in _learned_stats])
                for name in _learned_stats[0]
            }
            self.partial_zobs_learned_attention_last_stats.update({
                'sites': _learned_attention.sites,
                'observed_rows': int(partial_zobs_rows.numel()),
                'full_support_rows': int(x.coords.shape[0]),
                'observed_counts': _learned_cache['observed_counts'],
            })

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


# Published Pixal3D geometry configs use this name. The former elastic mixin
# only controlled training memory; inference uses the same SLatFlowModel.
ElasticSLatFlowModel = SLatFlowModel

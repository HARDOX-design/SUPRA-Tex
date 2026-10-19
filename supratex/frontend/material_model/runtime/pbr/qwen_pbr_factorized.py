# Adapted for SUPRA-Tex distribution (2026): packaged local inference runtime.
# Upstream FireRed portions remain under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Factorized modality-axis attention for joint RGB-to-PBR prediction.

The stock Qwen image transformer continues to process each modality as an
independent folded-batch item.  Small registered residual modules then mix the
three modalities at equal latent-grid positions after selected complete Qwen
blocks.  The final up projection is zero initialized, so attaching the modules
is an exact no-op before training.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask,
)
from diffusers.utils import apply_lora_scale, deprecate


PRODUCTION_JOINT_MODALITIES = ("albedo", "mra", "normal")
PRODUCTION_JOINT_SITES = (29, 39, 49, 59)
PRODUCTION_JOINT_RANK = 256
PRODUCTION_JOINT_HEADS = 4

JOINT_RESIDUAL_RAW = "raw"
JOINT_RESIDUAL_RMS_SCALED = "rms_scaled"
JOINT_RESIDUAL_MODES = (JOINT_RESIDUAL_RAW, JOINT_RESIDUAL_RMS_SCALED)

PBR_ATTENTION_NATIVE = "native"
PBR_ATTENTION_FA3 = "fa3"
PBR_ATTENTION_BACKENDS = (PBR_ATTENTION_NATIVE, PBR_ATTENTION_FA3)

ROUTE_ALIGNED = 0
ROUTE_SELF_SLOT = 1
ROUTE_BLOCKDIAG = 2
_ROUTE_NAME_TO_CODE = {
    "aligned": ROUTE_ALIGNED,
    "self_slot": ROUTE_SELF_SLOT,
    "blockdiag": ROUTE_BLOCKDIAG,
}


@lru_cache(maxsize=1)
def _load_flash_attention_3_ops():
    try:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "FireRed image attention requires the t2h FlashAttention-3 "
            "flash_attn_interface extension"
        ) from error
    return flash_attn_func, flash_attn_varlen_func


@lru_cache(maxsize=16)
def validate_fa3_runtime(device: torch.device | str, dtype: torch.dtype) -> None:
    """Fail closed unless the selected worker can execute Hopper BF16/FP16 FA3."""

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FireRed FA3 image attention requires a CUDA device")
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            "FireRed FA3 image attention requires bfloat16 or float16, "
            f"got {dtype}"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability < (9, 0):
        raise RuntimeError(
            "FireRed FA3 image attention requires Hopper compute capability >= 9.0, "
            f"got {capability[0]}.{capability[1]}"
        )
    _load_flash_attention_3_ops()


def prepare_fa3_key_mask_plan(
    attention_mask: torch.Tensor,
    query_seq_len: int,
) -> dict[str, torch.Tensor | int]:
    """Pack arbitrary valid K/V positions once for all Qwen transformer blocks.

    Qwen joint attention lays tokens out as ``[text, image]``.  Because text is
    padded before the image suffix, its key mask is not necessarily a valid
    prefix.  Diffusers' generic FA3-varlen backend slices ``K/V[:valid_len]``
    and would therefore drop real image tokens.  This plan gathers the true
    boolean-mask positions while leaving every query in its original RoPE
    position.
    """

    if attention_mask.dtype != torch.bool or attention_mask.ndim != 2:
        raise ValueError(
            "FA3 Qwen attention_mask must be a 2D bool tensor, "
            f"got shape={tuple(attention_mask.shape)} dtype={attention_mask.dtype}"
        )
    batch_size, key_seq_len = attention_mask.shape
    query_seq_len = int(query_seq_len)
    if batch_size <= 0 or key_seq_len <= 0 or query_seq_len <= 0:
        raise ValueError("FA3 Qwen attention dimensions must be positive")

    key_lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    if not bool((key_lengths > 0).all().item()):
        raise ValueError("every FA3 Qwen sample must retain at least one key token")
    key_indices = attention_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    cu_seqlens_k = torch.zeros(
        batch_size + 1,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    cu_seqlens_k[1:] = key_lengths.cumsum(dim=0)
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * query_seq_len,
        query_seq_len,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    return {
        "key_indices": key_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": query_seq_len,
        "max_seqlen_k": int(key_lengths.max().item()),
    }


def flash_attention_3_with_key_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    mask_plan: dict[str, torch.Tensor | int] | None = None,
) -> torch.Tensor:
    """Run exact non-causal FA3 attention with Qwen's arbitrary key mask."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FA3 Qwen Q/K/V must have shape [B, S, H, D]")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError(
            "FA3 Qwen requires equal Q/K/V shapes, got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    if query.device != key.device or key.device != value.device:
        raise ValueError("FA3 Qwen Q/K/V must be on the same device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("FA3 Qwen Q/K/V must have the same dtype")
    validate_fa3_runtime(query.device, query.dtype)
    flash_attn_func, flash_attn_varlen_func = _load_flash_attention_3_ops()

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if attention_mask is None:
        return flash_attn_func(
            query,
            key,
            value,
            softmax_scale=None,
            causal=False,
        )

    if attention_mask.shape != query.shape[:2]:
        raise ValueError(
            "FA3 Qwen key mask must match [B, S], got "
            f"{tuple(attention_mask.shape)} for {tuple(query.shape[:2])}"
        )
    if attention_mask.device != query.device:
        raise ValueError("FA3 Qwen key mask must be on the Q/K/V device")
    if mask_plan is None:
        mask_plan = prepare_fa3_key_mask_plan(attention_mask, query.shape[1])

    batch_size, seq_len, heads, head_dim = query.shape
    key_indices = mask_plan["key_indices"]
    if not isinstance(key_indices, torch.Tensor) or key_indices.device != query.device:
        raise ValueError("invalid FA3 key_indices mask plan")
    flat_key = key.reshape(batch_size * seq_len, heads, head_dim)
    flat_value = value.reshape(batch_size * seq_len, heads, head_dim)
    packed_key = flat_key.index_select(0, key_indices)
    packed_value = flat_value.index_select(0, key_indices)
    packed_query = query.reshape(batch_size * seq_len, heads, head_dim)
    output = flash_attn_varlen_func(
        packed_query,
        packed_key,
        packed_value,
        mask_plan["cu_seqlens_q"],
        mask_plan["cu_seqlens_k"],
        int(mask_plan["max_seqlen_q"]),
        int(mask_plan["max_seqlen_k"]),
        softmax_scale=None,
        causal=False,
    )
    return output.reshape(batch_size, seq_len, heads, head_dim)


class QwenDoubleStreamAttnProcessorFA3:
    """Qwen double-stream processor using mask-correct Hopper FA3 kernels."""

    _attention_backend = "_flash_3"
    _parallel_config = None

    def __init__(self) -> None:
        _load_flash_attention_3_ops()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        fa3_key_indices: torch.Tensor | None = None,
        fa3_cu_seqlens_q: torch.Tensor | None = None,
        fa3_cu_seqlens_k: torch.Tensor | None = None,
        fa3_max_seqlen_q: int | None = None,
        fa3_max_seqlen_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenDoubleStreamAttnProcessorFA3 requires encoder_hidden_states"
            )

        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        mask_plan = None
        if attention_mask is not None:
            required_plan = (
                fa3_key_indices,
                fa3_cu_seqlens_q,
                fa3_cu_seqlens_k,
                fa3_max_seqlen_q,
                fa3_max_seqlen_k,
            )
            if any(item is None for item in required_plan):
                raise ValueError("FA3 Qwen attention mask plan is incomplete")
            mask_plan = {
                "key_indices": fa3_key_indices,
                "cu_seqlens_q": fa3_cu_seqlens_q,
                "cu_seqlens_k": fa3_cu_seqlens_k,
                "max_seqlen_q": fa3_max_seqlen_q,
                "max_seqlen_k": fa3_max_seqlen_k,
            }
        joint_hidden_states = flash_attention_3_with_key_mask(
            joint_query,
            joint_key,
            joint_value,
            attention_mask=attention_mask,
            mask_plan=mask_plan,
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output


def _normalize_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(item) for item in modalities)
    if result != PRODUCTION_JOINT_MODALITIES:
        raise ValueError(
            "the first factorized PBR model fixes modality order to "
            f"{PRODUCTION_JOINT_MODALITIES}, got {result}"
        )
    return result


def _normalize_sites(sites: Sequence[int], num_layers: int) -> tuple[int, ...]:
    result = tuple(int(site) for site in sites)
    if len(set(result)) != len(result):
        raise ValueError(f"joint sites must be unique, got {result}")
    if any(site < 0 or site >= num_layers for site in result):
        raise ValueError(f"joint sites must be within [0, {num_layers - 1}], got {result}")
    return result


def _normalize_residual_mode(residual_mode: str) -> str:
    result = str(residual_mode)
    if result not in JOINT_RESIDUAL_MODES:
        raise ValueError(
            f"joint residual mode must be one of {JOINT_RESIDUAL_MODES}, got {result!r}"
        )
    return result


class ModalityAxisJointAttention(nn.Module):
    """Mix equal-position modality tokens while keeping objects/tokens separate.

    Inputs use object-major folded-batch order: ``[B*M, L, D]`` where M is the
    fixed modality count.  Attention is over the donor-modality axis only.  The
    checkpointable forward accepts tensors exclusively.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = PRODUCTION_JOINT_RANK,
        heads: int = PRODUCTION_JOINT_HEADS,
        modalities: Sequence[str] = PRODUCTION_JOINT_MODALITIES,
        residual_mode: str = JOINT_RESIDUAL_RAW,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.heads = int(heads)
        self.modalities = _normalize_modalities(modalities)
        self.num_modalities = len(self.modalities)
        self.residual_mode = _normalize_residual_mode(residual_mode)

        if self.hidden_dim <= 0 or self.rank <= 0 or self.heads <= 0:
            raise ValueError("hidden_dim, rank, and heads must all be positive")
        if self.rank % self.heads != 0:
            raise ValueError(f"rank={self.rank} must be divisible by heads={self.heads}")
        self.head_dim = self.rank // self.heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.w_down = nn.Linear(self.hidden_dim, self.rank)
        self.modality_embedding = nn.Parameter(torch.empty(self.num_modalities, self.rank))
        self.to_q = nn.Linear(self.rank, self.rank)
        self.to_k = nn.Linear(self.rank, self.rank)
        self.to_v = nn.Linear(self.rank, self.rank)
        self.to_out = nn.Linear(self.rank, self.rank)
        self.w_up = nn.Linear(self.rank, self.hidden_dim)

        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_up.weight)
        nn.init.zeros_(self.w_up.bias)

    @staticmethod
    def _route_value(route_code: torch.Tensor) -> int:
        if not isinstance(route_code, torch.Tensor) or route_code.numel() != 1:
            raise ValueError("route_code must be a scalar tensor")
        route = int(route_code.detach().item())
        if route not in (ROUTE_ALIGNED, ROUTE_SELF_SLOT, ROUTE_BLOCKDIAG):
            raise ValueError(f"unknown joint route code: {route}")
        return route

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_code: torch.Tensor,
        modality_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a modality-mixed residual update for target-prefix tokens."""

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [B*M, L, D], got {tuple(hidden_states.shape)}"
            )
        folded_batch, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden_dim}")
        if folded_batch % self.num_modalities != 0:
            raise ValueError(
                f"folded batch {folded_batch} is not divisible by M={self.num_modalities}"
            )
        if seq_len <= 0:
            raise ValueError("the target token sequence must be non-empty")

        route = self._route_value(route_code)
        if route == ROUTE_BLOCKDIAG and self.training:
            raise ValueError("blockdiag is an evaluation-only intervention")

        if not isinstance(modality_attention_mask, torch.Tensor):
            raise ValueError("modality_attention_mask must be a tensor")
        if tuple(modality_attention_mask.shape) != (self.num_modalities, self.num_modalities):
            raise ValueError(
                "modality_attention_mask must have shape "
                f"[{self.num_modalities}, {self.num_modalities}], got "
                f"{tuple(modality_attention_mask.shape)}"
            )
        attention_mask = modality_attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not bool(attention_mask.any(dim=-1).all().item()):
            raise ValueError("every modality query must have at least one permitted donor")
        if route == ROUTE_BLOCKDIAG:
            expected = torch.eye(self.num_modalities, dtype=torch.bool, device=hidden_states.device)
            if not torch.equal(attention_mask, expected):
                raise ValueError("blockdiag route requires an identity modality attention mask")

        batch_size = folded_batch // self.num_modalities
        # Object-major fold: flat index = object_index * M + modality_index.
        x = hidden_states.reshape(batch_size, self.num_modalities, seq_len, hidden_dim)
        x = x.permute(0, 2, 1, 3)  # [B, L, M, D]
        z = self.w_down(self.norm(x))  # [B, L, M, R]

        modality_embedding = self.modality_embedding.to(dtype=z.dtype)
        query_input = z + modality_embedding.view(1, 1, self.num_modalities, self.rank)

        if route == ROUTE_SELF_SLOT:
            # Each query supplies its own content to all donor slots.  Donor
            # modality embeddings keep the slots distinct before K/V.
            donor_input = z.unsqueeze(3).expand(-1, -1, -1, self.num_modalities, -1)
        else:
            # Every query sees the real aligned modality tokens at this object
            # and latent-grid position.  Blockdiag changes only the edge mask.
            donor_input = z.unsqueeze(2).expand(-1, -1, self.num_modalities, -1, -1)
        donor_input = donor_input + modality_embedding.view(
            1, 1, 1, self.num_modalities, self.rank
        )

        query = self.to_q(query_input).reshape(
            batch_size, seq_len, self.num_modalities, self.heads, self.head_dim
        )
        query = query.permute(0, 1, 3, 2, 4)  # [B, L, H, Q, dh]

        key = self.to_k(donor_input).reshape(
            batch_size,
            seq_len,
            self.num_modalities,
            self.num_modalities,
            self.heads,
            self.head_dim,
        )
        value = self.to_v(donor_input).reshape_as(key)
        key = key.permute(0, 1, 4, 2, 3, 5)  # [B, L, H, Q, K, dh]
        value = value.permute(0, 1, 4, 2, 3, 5)

        logits = (query.unsqueeze(-2) * key).sum(dim=-1) * self.scale
        mask = attention_mask.view(1, 1, 1, self.num_modalities, self.num_modalities)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        context = (weights.unsqueeze(-1) * value).sum(dim=-2)
        context = context.permute(0, 1, 3, 2, 4).reshape(
            batch_size, seq_len, self.num_modalities, self.rank
        )
        delta = self.w_up(self.to_out(context))
        if self.residual_mode == JOINT_RESIDUAL_RMS_SCALED:
            # Qwen image residual states can reach 1e6--1e7 in BF16 even though
            # every following block consumes a LayerNorm-normalized view.  A
            # raw adapter update therefore disappears at the residual add.  In
            # the repaired parameterization, the adapter predicts a normalized
            # update and this detached RMS converts it back to the host unit.
            token_rms = torch.linalg.vector_norm(
                x, dim=-1, keepdim=True, dtype=torch.float32
            ) * (hidden_dim**-0.5)
            delta = delta * token_rms.detach().to(dtype=delta.dtype)
        output = x + delta
        return output.permute(0, 2, 1, 3).reshape(folded_batch, seq_len, hidden_dim)


class QwenImagePBRFactorizedTransformer2DModel(QwenImageTransformer2DModel):
    """Qwen image transformer with registered factorized PBR joint modules."""

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        joint_sites: Sequence[int] = (),
        joint_modalities: Sequence[str] = PRODUCTION_JOINT_MODALITIES,
        joint_rank: int = PRODUCTION_JOINT_RANK,
        joint_heads: int = PRODUCTION_JOINT_HEADS,
        joint_residual_mode: str = JOINT_RESIDUAL_RAW,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            zero_cond_t=zero_cond_t,
            use_additional_t_cond=use_additional_t_cond,
            use_layer3d_rope=use_layer3d_rope,
        )
        self._pbr_attention_backend = PBR_ATTENTION_NATIVE
        self.modality_joint_blocks = nn.ModuleDict()
        self.joint_modalities = _normalize_modalities(joint_modalities)
        self.joint_rank = int(joint_rank)
        self.joint_heads = int(joint_heads)
        self.joint_residual_mode = _normalize_residual_mode(joint_residual_mode)

        normalized_sites = _normalize_sites(joint_sites, len(self.transformer_blocks))
        if normalized_sites:
            self._attach_joint_modules(
                sites=normalized_sites,
                modalities=self.joint_modalities,
                rank=self.joint_rank,
                heads=self.joint_heads,
                residual_mode=self.joint_residual_mode,
                update_config=True,
            )

    @property
    def joint_sites(self) -> tuple[int, ...]:
        return tuple(int(site) for site in self.modality_joint_blocks.keys())

    def _attach_joint_modules(
        self,
        sites: Sequence[int],
        modalities: Sequence[str],
        rank: int,
        heads: int,
        residual_mode: str,
        *,
        update_config: bool,
    ) -> None:
        if len(self.modality_joint_blocks) != 0:
            raise RuntimeError(f"joint modules are already attached at sites {self.joint_sites}")
        normalized_sites = _normalize_sites(sites, len(self.transformer_blocks))
        if not normalized_sites:
            raise ValueError("attach_joint_modules requires at least one site")
        normalized_modalities = _normalize_modalities(modalities)
        rank = int(rank)
        heads = int(heads)
        residual_mode = _normalize_residual_mode(residual_mode)
        if rank <= 0 or heads <= 0 or rank % heads != 0:
            raise ValueError(f"joint rank={rank} must be positive and divisible by heads={heads}")

        self.modality_joint_blocks.update(
            {
                str(site): ModalityAxisJointAttention(
                    hidden_dim=self.inner_dim,
                    rank=rank,
                    heads=heads,
                    modalities=normalized_modalities,
                    residual_mode=residual_mode,
                )
                for site in normalized_sites
            }
        )
        self.joint_modalities = normalized_modalities
        self.joint_rank = rank
        self.joint_heads = heads
        self.joint_residual_mode = residual_mode
        if update_config:
            self.register_to_config(
                architecture_type="QwenImagePBRFactorized",
                joint_sites=list(normalized_sites),
                joint_modalities=list(normalized_modalities),
                joint_rank=rank,
                joint_heads=heads,
                joint_residual_mode=residual_mode,
                supported_joint_routes=["self_slot", "aligned"],
                blockdiag_eval_only=True,
            )

    def attach_joint_modules(
        self,
        sites: Sequence[int] = PRODUCTION_JOINT_SITES,
        modalities: Sequence[str] = PRODUCTION_JOINT_MODALITIES,
        rank: int = PRODUCTION_JOINT_RANK,
        heads: int = PRODUCTION_JOINT_HEADS,
        residual_mode: str = JOINT_RESIDUAL_RAW,
    ) -> None:
        """Attach initialized modules after an exact base-only load."""

        self._attach_joint_modules(
            sites=sites,
            modalities=modalities,
            rank=rank,
            heads=heads,
            residual_mode=residual_mode,
            update_config=True,
        )

    def set_pbr_attention_backend(
        self,
        backend: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Select the Qwen image-transformer attention kernel for this runtime."""

        backend = str(backend)
        if backend not in PBR_ATTENTION_BACKENDS:
            raise ValueError(
                f"pbr attention backend must be one of {PBR_ATTENTION_BACKENDS}, "
                f"got {backend!r}"
            )
        if backend == PBR_ATTENTION_FA3:
            if device is None or dtype is None:
                raise ValueError("FA3 backend selection requires runtime device and dtype")
            validate_fa3_runtime(device, dtype)
            processor_factory = QwenDoubleStreamAttnProcessorFA3
        else:
            processor_factory = QwenDoubleStreamAttnProcessor2_0

        for block in self.transformer_blocks:
            block.attn.set_processor(processor_factory())
        self._pbr_attention_backend = backend

    @staticmethod
    def _runtime_int(value: int | torch.Tensor | None, name: str) -> int:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar")
            value = int(value.detach().item())
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    @staticmethod
    def _runtime_route(value: str | int | torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if isinstance(value, str):
            if value not in _ROUTE_NAME_TO_CODE:
                raise ValueError(f"unknown pbr_joint_route: {value!r}")
            code = _ROUTE_NAME_TO_CODE[value]
        elif isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("pbr_joint_route tensor must be scalar")
            code = int(value.detach().item())
        elif isinstance(value, int):
            code = value
        else:
            raise ValueError("pbr_joint_route must be self_slot, aligned, or blockdiag")
        if code not in (ROUTE_ALIGNED, ROUTE_SELF_SLOT, ROUTE_BLOCKDIAG):
            raise ValueError(f"unknown pbr_joint_route code: {code}")
        return torch.tensor(code, dtype=torch.int64, device=device)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
        pbr_joint_batch_size: int | torch.Tensor | None = None,
        pbr_joint_target_seq_len: int | torch.Tensor | None = None,
        pbr_joint_route: str | int | torch.Tensor | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """Run stock Qwen blocks and insert target-only modality adapters."""

        has_joint_modules = len(self.modality_joint_blocks) > 0
        runtime_values = (pbr_joint_batch_size, pbr_joint_target_seq_len, pbr_joint_route)
        if has_joint_modules and any(value is None for value in runtime_values):
            raise ValueError(
                "joint models require pbr_joint_batch_size, pbr_joint_target_seq_len, "
                "and pbr_joint_route"
            )
        if not has_joint_modules and any(value is not None for value in runtime_values):
            raise ValueError("joint runtime arguments were provided before joint modules were attached")

        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `encoder_hidden_states_mask` instead. "
                "The mask-based approach is more flexible and supports variable-length sequences.",
                standard_warn=False,
            )

        if has_joint_modules:
            batch_size = self._runtime_int(pbr_joint_batch_size, "pbr_joint_batch_size")
            target_seq_len = self._runtime_int(
                pbr_joint_target_seq_len, "pbr_joint_target_seq_len"
            )
            if batch_size <= 0:
                raise ValueError("pbr_joint_batch_size must be positive")
            expected_folded_batch = batch_size * len(self.joint_modalities)
            if hidden_states.shape[0] != expected_folded_batch:
                raise ValueError(
                    "object-major folded batch mismatch: expected "
                    f"B*M={expected_folded_batch}, got {hidden_states.shape[0]}"
                )
            if target_seq_len <= 0 or target_seq_len > hidden_states.shape[1]:
                raise ValueError(
                    f"pbr_joint_target_seq_len must be in [1, {hidden_states.shape[1]}], "
                    f"got {target_seq_len}"
                )
            route_code = self._runtime_route(pbr_joint_route, hidden_states.device)
            if int(route_code.item()) == ROUTE_BLOCKDIAG:
                modality_attention_mask = torch.eye(
                    len(self.joint_modalities), dtype=torch.bool, device=hidden_states.device
                )
            else:
                modality_attention_mask = torch.ones(
                    (len(self.joint_modalities), len(self.joint_modalities)),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
        else:
            target_seq_len = 0
            route_code = None
            modality_attention_mask = None

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum(prod(shape) for shape in sample[1:]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )
        image_rotary_emb = self.pos_embed(
            img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device
        )

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            folded_batch, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones(
                (folded_batch, image_seq_len), dtype=torch.bool, device=hidden_states.device
            )
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask
            if self._pbr_attention_backend == PBR_ATTENTION_FA3:
                reserved = {
                    "fa3_key_indices",
                    "fa3_cu_seqlens_q",
                    "fa3_cu_seqlens_k",
                    "fa3_max_seqlen_q",
                    "fa3_max_seqlen_k",
                }
                conflict = reserved.intersection(block_attention_kwargs)
                if conflict:
                    raise ValueError(f"reserved FA3 attention kwargs supplied: {sorted(conflict)}")
                mask_plan = prepare_fa3_key_mask_plan(
                    joint_attention_mask,
                    query_seq_len=joint_attention_mask.shape[1],
                )
                block_attention_kwargs.update(
                    {
                        "fa3_key_indices": mask_plan["key_indices"],
                        "fa3_cu_seqlens_q": mask_plan["cu_seqlens_q"],
                        "fa3_cu_seqlens_k": mask_plan["cu_seqlens_k"],
                        "fa3_max_seqlen_q": mask_plan["max_seqlen_q"],
                        "fa3_max_seqlen_k": mask_plan["max_seqlen_k"],
                    }
                )

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )

            mixer_key = str(index_block)
            if mixer_key in self.modality_joint_blocks:
                mixer = self.modality_joint_blocks[mixer_key]
                target_hidden = hidden_states[:, :target_seq_len]
                target_hidden = mixer(
                    target_hidden,
                    route_code,
                    modality_attention_mask,
                )
                if target_seq_len == hidden_states.shape[1]:
                    hidden_states = target_hidden
                else:
                    # The adapter never reads or writes the common source suffix.
                    hidden_states = torch.cat(
                        [target_hidden, hidden_states[:, target_seq_len:]], dim=1
                    )

            if controlnet_block_samples is not None:
                interval_control = int(
                    np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
                )
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

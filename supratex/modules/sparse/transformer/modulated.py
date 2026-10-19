from typing import *
import torch
import torch.nn as nn
import torch.utils.checkpoint
from ..basic import VarLenTensor, SparseTensor
from ..attention import SparseMultiHeadAttention, SparseProjectAttention, SparseGatedProjectAttention
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
    Supports two image attention modes:
    - "cross": Standard cross-attention with image features
    - "proj": Projection-based attention with view-aligned features
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["cross", "proj", "gated_proj"] = "cross",
        proj_in_channels: Optional[int] = None,
        vae_in_channels: Optional[int] = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Build cross attention based on mode
        if image_attn_mode == "cross":
            self.cross_attn = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
        elif image_attn_mode == "proj":
            _proj_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseProjectAttention(cross_attn_block, channels, _proj_in)
        elif image_attn_mode == "gated_proj":
            _dino_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            _vae_in = vae_in_channels if vae_in_channels is not None else 16
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseGatedProjectAttention(cross_attn_block, channels, _dino_in, _vae_in)
        else:
            raise ValueError(f"Unknown image attention mode: {image_attn_mode}")
            
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: Union[torch.Tensor, VarLenTensor],
        partial_zobs_rows: Optional[torch.Tensor] = None,
        partial_zobs_inject: Optional[torch.Tensor] = None,
        partial_zobs_inject_scale: float = 1.0,
    ) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        # Inference-only causal probe hooks. Both attributes are absent by
        # default, leaving training and normal inference unchanged. A folded
        # zero-context constant preserves the learned residual while bypassing
        # pre-norm, Q/K/V, attention, and output projection during sampling.
        folded = getattr(self, "folded_zero_image_cross_attention", None)
        if folded is not None:
            assert folded.ndim == 1 and folded.shape[0] == x.feats.shape[1]
            x = x.replace(x.feats + folded.to(device=x.feats.device, dtype=x.feats.dtype))
        elif not getattr(self, "skip_image_cross_attention", False):
            h = x.replace(self.norm2(x.feats))
            h = self.cross_attn(h, context)
            x = x + h
        if partial_zobs_rows is not None or partial_zobs_inject is not None:
            assert partial_zobs_rows is not None and partial_zobs_inject is not None, \
                'partial z_obs rows and messages must be passed together'
            assert partial_zobs_rows.ndim == 1 and \
                partial_zobs_inject.shape == (partial_zobs_rows.numel(), x.feats.shape[1]), \
                'partial z_obs messages must be [Nobs, channels]'
            # This residual is deliberately before norm3/AdaLN/MLP. index_add also keeps the
            # zero-row projected-message graph live for all-drop checkpointed batches.
            x = x.replace(x.feats.index_add(
                0, partial_zobs_rows, partial_zobs_inject,
                alpha=partial_zobs_inject_scale,
            ))
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: Union[torch.Tensor, VarLenTensor],
        partial_zobs_rows: Optional[torch.Tensor] = None,
        partial_zobs_inject: Optional[torch.Tensor] = None,
        partial_zobs_inject_scale: float = 1.0,
    ) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                mod,
                context,
                partial_zobs_rows,
                partial_zobs_inject,
                partial_zobs_inject_scale,
                use_reentrant=False,
            )
        else:
            return self._forward(
                x,
                mod,
                context,
                partial_zobs_rows,
                partial_zobs_inject,
                partial_zobs_inject_scale,
            )

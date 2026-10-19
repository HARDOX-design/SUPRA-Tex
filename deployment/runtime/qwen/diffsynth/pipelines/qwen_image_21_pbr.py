"""Joint Albedo/MRA latent prediction using the Qwen-Image 2.1 backbone."""

import torch
from torch import nn

from .qwen_image_21 import patchify, unpatchify


class QwenImage21PBRModel(nn.Module):
    """Predict two separate material maps with a shared noisy target group.

    The modality axis is ordered ``(albedo, mra)``. MRA pixel semantics belong
    to the dataset contract; this module does not infer or repack AO/alpha.
    Conditions remain clean, and each map retains its own spatial RoPE block.
    """

    modalities = ("albedo", "mra")

    def __init__(self, dit):
        super().__init__()
        self.dit = dit
        channels = dit.img_in.in_features
        if channels != dit.out_channels:
            raise ValueError("PBR requires equal input and output latent channels")
        self.modality_embeddings = nn.Parameter(torch.zeros(2, channels))

    def forward(
        self,
        latents,
        timestep,
        prompt_embeds,
        prompt_embeds_mask,
        edit_image_pad_mask,
        edit_latents=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        use_flex_attention=False,
        kv_cache=None,
        gradient_checkpointing_interval=1,
    ):
        if latents.ndim != 5 or latents.shape[1] != len(self.modalities):
            raise ValueError("Expected latents shaped [batch, 2, channels, height, width]")
        batch_size, _, channels, height, width = latents.shape
        if channels != self.modality_embeddings.shape[1]:
            raise ValueError("Target latent channels do not match the DiT")
        if height % 2 or width % 2:
            raise ValueError("Target latent dimensions must be even (pixel dimensions divisible by 32)")
        if prompt_embeds.shape[0] != batch_size or edit_image_pad_mask.shape[0] != batch_size:
            raise ValueError("Condition and target batch sizes must match")
        if not torch.equal(edit_image_pad_mask, edit_image_pad_mask[:1].expand_as(edit_image_pad_mask)):
            raise ValueError("Batch elements must share the same image-token layout")
        edit_latents = [] if edit_latents is None else edit_latents
        for condition in edit_latents:
            if condition.ndim != 4 or condition.shape[:2] != (batch_size, channels):
                raise ValueError("Each condition must be shaped [batch, channels, height, width]")

        target_tokens = []
        for index in range(len(self.modalities)):
            tokens = patchify(latents[:, index])
            embedding = self.modality_embeddings[index].to(device=tokens.device, dtype=tokens.dtype)
            target_tokens.append(tokens + embedding)
        hidden_states = torch.cat([*[patchify(image) for image in edit_latents], *target_tokens], dim=1)
        target_seq_len = len(self.modalities) * height * width
        target_image_pad_mask = torch.ones(
            (batch_size, target_seq_len // 4), dtype=torch.bool, device=latents.device
        )
        image_pad_mask = torch.cat([edit_image_pad_mask.bool(), target_image_pad_mask], dim=1)
        img_shapes = [[
            *[(1, image.shape[2], image.shape[3]) for image in edit_latents],
            *[(1, height, width) for _ in self.modalities],
        ] for _ in range(batch_size)]
        model_output = self.dit(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            img_mask=image_pad_mask,
            kv_cache=kv_cache,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_flex_attention=use_flex_attention,
            num_target_images=len(self.modalities),
            gradient_checkpointing_interval=gradient_checkpointing_interval,
        )
        target_output = model_output[:, -target_seq_len:]
        return torch.stack([
            unpatchify(tokens, height, width)
            for tokens in target_output.split(height * width, dim=1)
        ], dim=1)

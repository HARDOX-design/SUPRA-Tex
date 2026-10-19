# SUPRA-Tex: 3D Generation with Super-Aligned Texture

The inference pipeline has three main stages:

1. **Image-to-Material Translation** estimates reference-view albedo, metallic
   and roughness from the reference image.
2. **Native Observation Encoding** performs filtered surface projection and
   sparse coordinate encoding with the pretrained material VAE. Only visible
   support is encoded. Each observation carries 32 latent channels, coverage
   and boundary distance (34 channels total).
3. **Observation-Guided Completion** samples complete material latents using
   the **Interleaved Flow Transformer**. Observation tokens remain fixed and
   supply keys/values; even blocks update unobserved cells and odd blocks
   update the full surface. The generation and observation streams share the
   same 3D coordinate frame. Block indices are zero-based; the released model
   has 60 blocks.

For image-to-3D generation, foreground extraction, MoGe-2 camera estimation,
and Pixal3D geometry generation prepare the mesh and reference view before
observation encoding. For an existing mesh, `texturing.py` starts from the
prepared geometry, camera, foreground image, and material maps.
After completion, the material VAE decodes the sampled latents and the exporter
bakes the material maps into a textured GLB. See [the frontend](frontend.md),
[geometry](geometry.md), and [texturing](texturing.md) guides for those inputs
and commands.

The implementation is organized as follows:

| Component | Implementation |
| --- | --- |
| Image-to-Material Translation | [frontend/pipeline.py](../supratex/frontend/pipeline.py) |
| Native observation encoding | [inference/native_observation.py](../supratex/inference/native_observation.py) |
| Interleaved flow transformer | [models/interleaved_flow_transformer.py](../supratex/models/interleaved_flow_transformer.py) |
| Observation attention and shared streams | [models/observation_attention.py](../supratex/models/observation_attention.py), [models/observation_stream.py](../supratex/models/observation_stream.py) |
| Interleaved backbone and efficient attention | [models/interleaved_backbone.py](../supratex/models/interleaved_backbone.py), [models/efficient_attention.py](../supratex/models/efficient_attention.py) |
| Observation-guided completion sampler | [pipelines/observation_guided_completion.py](../supratex/pipelines/observation_guided_completion.py) |
| Completion and GLB export | [inference/pipeline.py](../supratex/inference/pipeline.py), [inference/mesh_export.py](../supratex/inference/mesh_export.py) |

# Qwen runtime third-party notices

This directory contains the vendored Qwen material-model runtime, including
`diffsynth`. Its top-level [Apache License 2.0](LICENSE) is retained. Incorporated
third-party portions keep their own applicable licenses and copyright notices;
the top-level license does not convert them to Apache-2.0 or to the SUPRA-Tex
project's MIT license.

Paths in this notice are relative to `deployment/runtime/qwen/`. The repository
[third-party notices](../../../THIRD_PARTY_NOTICES.md) provide the wider source
map. The [source provenance record](../SOURCE_PROVENANCE.json) identifies this
as a vendored runtime without an identified upstream Git revision.
The distribution includes the inference entry point and its shared local
training, validation, and encoding helpers.

## Identified MIT subcomponents

| Distributed source | Upstream source | Full license text |
| --- | --- | --- |
| [`diffsynth/models/ace_step_residual_fsq.py`](diffsynth/models/ace_step_residual_fsq.py) | [`lucidrains/vector-quantize-pytorch/residual_fsq.py`](https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py), as credited in the file header; Copyright (c) 2020 Phil Wang. | [vector-quantize-pytorch MIT license](../../../THIRD_PARTY_NOTICES.md#vector-quantize-pytorch-mit). |
| The timestep embedding in [`diffsynth/models/step1x_connector.py`](diffsynth/models/step1x_connector.py) | [`openai/glide-text2im/glide_text2im/nn.py`](https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py), as credited in the implementation; Copyright (c) 2021 OpenAI. | [OpenAI GLIDE MIT license](../../../THIRD_PARTY_NOTICES.md#openai-glide-mit). |

The MIT texts above were obtained from the respective official upstream
[vector-quantize-pytorch license](https://raw.githubusercontent.com/lucidrains/vector-quantize-pytorch/master/LICENSE)
and [GLIDE license](https://raw.githubusercontent.com/openai/glide-text2im/main/LICENSE)
on 2026-10-14. These URLs do not establish an exact revision for this runtime
snapshot. Apache-2.0 source headers elsewhere in this directory remain in place
and refer to the bundled [Apache-2.0 text](LICENSE).

## Excluded YuE2 audio functionality

This source release excludes the YuE2 audio VAE formerly at
`diffsynth/models/yue2_vae.py`, its model-registry entry, and the dependent
music-generation pipeline formerly at `diffsynth/pipelines/yue2.py`. Their
source-permission records for the VAE's incorporated stable-audio-tools and
NVIDIA portions were incomplete. The excluded code is not distributed here,
and no replacement license or permission is inferred from the runtime's
top-level Apache-2.0 license.

Supra-Tex's Qwen-Image-2.1 material inference uses the separate
[image VAE](diffsynth/models/qwen_image_21_vae.py) and
[image pipeline](diffsynth/pipelines/qwen_image_21.py), which remain included
with their model-registry entries. Other registry entries are unchanged;
the retained YuE2 model/tokenizer helpers do not provide the removed audio
pipeline.

This notice records identified sources; it is not a completed audit of every
source reference in this vendored snapshot. Other per-file source and copyright
attributions remain applicable.

## Model weights and adapters

This is a source-code notice. It does not license Qwen base model weights,
trained adapters, or other model artifacts. Each separately distributed weight
artifact must carry its applicable license and model card; the source runtime's
Apache-2.0 license and the project's MIT license do not determine those terms.

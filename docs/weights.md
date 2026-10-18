# Model weights

The project model repository is [supra3d/supra-tex](https://huggingface.co/supra3d/supra-tex).
Its [completion/](https://huggingface.co/supra3d/supra-tex/tree/main/completion)
and [material/](https://huggingface.co/supra3d/supra-tex/tree/main/material)
subdirectories contain the two project models;
all other weights remain external. The model repository's root `pipeline.json` selects these
components, pinned external dependencies and inference defaults. The source
preparation entrypoint reads it to obtain the complete model set:

```bash
python scripts/prepare_models.py
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/spiny_whelk_shell.glb
```

See [source setup](quickstart.md) for upstream access requirements and local
model overrides. The standalone directory descriptions below apply to the
downloaded subdirectories.
Sizes below use decimal GB/MB.

With the default repository and output directory, preparation writes
`ckpts/ready/models.json`, downloads the project models under
`ckpts/ready/project/`, and places external models under
`ckpts/ready/external/`. It also assembles `ckpts/ready/geometry/` and the NAF
cache under `ckpts/ready/torch/`. The example runner reads these paths from
`models.json`. Add `--with-rmbg` during preparation for RGB or opaque inputs;
the bundled example already has foreground alpha.

The same preparation flow accepts a local model repository root with
`--repo /path/to/supra-tex`. Use `--pipeline /path/to/pipeline.json` to select an
explicit local manifest, and `--local-files-only` to require local or cached
models. For direct image-to-3D inference, `infer.py --pipeline` prepares the
selected pipeline before running:

```bash
python infer.py --pipeline supra3d/supra-tex \
  --input path/to/image.png --output outputs/object.glb
```

`--pipeline-revision` selects the Hub snapshot and `--models-dir` selects the
prepared model directory. Explicit inference arguments override manifest
defaults. Model architecture and normalization remain in the component configs.

For an existing mesh and prepared material observations, use the same manifest
with `texturing.py`:

```bash
python texturing.py --pipeline supra3d/supra-tex \
  --input-dir inputs/object --output outputs/object.glb
```

This entrypoint resolves only the completion model and TRELLIS VAEs, using
the manifest defaults with explicit CLI overrides. It accepts a local release
directory or manifest file as well. `--pipeline-revision`, `--cache-dir`, and
`--local-files-only` control Hub resolution; `--trellis-root` supplies existing
local VAEs. See [texturing](texturing.md) for component overrides.

| We publish | Payload | License |
| --- | --- | --- |
| SUPRA-Tex 3D Completion | Full 2.1B-parameter model; default `denoiser_step0030000.safetensors`, approximately 8.38 GB, one safetensors file | MIT |
| SUPRA-Tex Material Adapter | All 257 learned tensors; approximately 134.25 MB, one safetensors file | Qwen Research License |

The material adapter is **Built with Qwen**. Its
[research license](../THIRD_PARTY_NOTICES.md#qwen-research-license) applies to
the adapter. The source repository's MIT license does not replace those terms.

## Load the project models

The standalone completion model directory contains:

```text
completion/
  config.json
  partial_normalization.json
  denoiser_step0030000.safetensors      # default weights
  model.safetensors.index.json          # earlier release, kept as a fallback
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
```

`denoiser_step0030000.safetensors` is the default completion checkpoint: model
preparation downloads only it with `config.json` and `partial_normalization.json`,
and a completion directory that contains it uses it automatically. Both
checkpoints share the same architecture and normalization statistics. A directory
without it falls back to `model.safetensors` or the sharded index; select any
other weights explicitly with `--checkpoint`.

For the standalone `--pretrained` invocation without `--pipeline`, supply the
externally downloaded TRELLIS.2 VAEs through `--trellis-root`:

```bash
python texturing.py \
  --input-dir inputs/object \
  --output outputs/object.glb \
  --pretrained /path/to/completion \
  --trellis-root /path/to/TRELLIS.2-4B
```

The VAE root may be the upstream repository root with a `ckpts/` directory,
that `ckpts/` directory itself, or another folder containing all four required
pairs listed below. Without an explicit argument the standalone release
looks under `completion/vae/`. External weights are not downloaded
implicitly by the completion loader.

When using `--pretrained` with this combined repository, pass the downloaded
`completion/` directory, not the repository root. Keep the separate `--trellis-root`
argument. The loader also accepts `namespace/repo` or `hf://namespace/repo`
for standalone repositories whose model files are at the root; `--revision`,
`--cache-dir` and `--local-files-only` control that Hub loading path.
`--model-config`, `--checkpoint` and `--trellis-root` can also be supplied
explicitly, overriding bundle defaults.
A single `model.safetensors` is supported as well as the indexed shards.

The material directory contains:

```text
material/
  adapter_config.json
  adapter_model.safetensors
```

Use it in the frontend or complete `infer.py` command:

```bash
--material-backend qwen-image \
--material-base /path/to/Qwen-Image-2.1 \
--material-model /path/to/material
```

The adapter retains attention LoRA **and modality embeddings**, in the native
SUPRA-Tex/Qwen PBR format. The 134 MB export excludes optimizer and RNG state;
it preserves every learned tensor and does not quantize the model. Its size
is not the full Qwen model size or a GPU memory requirement. The native
runtime loads the base first, then applies the learned state. Generic PEFT
or Transformers `AutoModel` loading is not supported by this format.

The default runtime produces albedo and metallic/roughness with fixed 1024
resolution, noise seed 42, Torch attention, and single-step endpoint prediction.
Input alpha supplies coverage in the canonical MRA blue channel. See
[the frontend guide](frontend.md) for image preparation and runtime dependencies.

## Download external models from their publishers

| External dependency | Needed for | Source |
| --- | --- | --- |
| Qwen-Image-2.1 | Material prediction | [Qwen](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| Four TRELLIS.2 VAEs | 3D texture completion | [Microsoft TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) |
| Pixal3D geometry | RGB-to-mesh | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) |
| MoGe-2 | Camera estimation | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| DINOv3 and NAF | Geometry conditioning | [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [NAF](https://github.com/valeoai/NAF) |
| RMBG-2 | Foreground extraction when input alpha is unavailable | [BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) |

These weights are not included in the project model subdirectories. Retain their upstream
license and model-card files; gated models require access from their publishers.
Required file patterns and recorded revisions are listed in the source
repository's [`configs/pipeline.json`](../configs/pipeline.json), under `external_models`.

For Qwen, keep all files under `transformer/`, `text_encoder/`, `vae/`, and
`processor/`, including every weight file/shard, the component configs,
shard indexes, tokenizer and processor assets.

For TRELLIS.2, each stem requires both `.json` and `.safetensors`:

```text
shape_enc_next_dc_f16c32_fp16
tex_enc_next_dc_f16c32_fp16
shape_dec_next_dc_f16c32_fp16
tex_dec_next_dc_f16c32_fp16
```

Standalone completion reads its architecture and all necessary normalization
values from the project config/statistics. It does not need the entire
upstream geometry `pipeline.json`; RGB geometry generation separately needs
Pixal3D's own model layout, described in [Geometry](geometry.md).

## Legacy and optional formats

Existing PT bundles remain accepted: `configs/inference_model.json`,
`configs/partial_normalization.json`, `weights/material_completion.pt`,
`weights/sparse_encoders/` and `weights/scvae/`. In this legacy layout decoder
paths remain relative to the model config. Existing Qwen
`checkpoint-0100000.pt` plus its matching `.complete.json` also remain supported;
safetensors exports use their JSON adapter config instead of that PT marker.

The optional `firered` backend and its full Transformer/contract files remain
supported by the code, but FireRed weights are outside this project release.
The default selected release uses the Qwen material backend.

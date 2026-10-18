# Image-to-Material Translation frontend

Supra-Tex begins with foreground isolation, canonical-camera estimation, and
material prediction. The default `qwen-image` backend uses the Supra-Tex
Material Adapter based on Qwen-Image-2.1; `firered` selects the bundled factorized
modality-axis joint-attention model. Both produce the aligned albedo and MRA
maps used by surface-observation encoding.

Follow the fresh-checkout environment commands and CUDA extension instructions
in [Installation](installation.md). They install the pinned Python dependencies
and bundled MoGe-2/utils3d packages; the `frontend` extra alone is insufficient
for the native Qwen runtime.
The Qwen backend uses its native DiffSynth runtime and torch attention. The
FireRed backend selects FA3 for its material Transformer on compatible Hopper
setups and native Torch attention otherwise; its text encoder selects FA2 or
SDPA. The default Qwen backend supports `--low-vram` CPU offload. See
[GPU compatibility](gpu_compatibility.md) for device settings and validation
status, and [Weights](weights.md) for model artifacts.

After the [model preparation step](quickstart.md), run the frontend using the
prepared local paths. For an RGB input, prepare RMBG-2 as well:

```bash
python scripts/prepare_models.py --repo supra3d/supra-tex --with-rmbg
python -m supratex.frontend.pipeline --input input.png --output work \
  --rmbg-model ckpts/ready/external/rmbg2 \
  --moge-model ckpts/ready/external/moge2 \
  --material-backend qwen-image \
  --material-base ckpts/ready/external/qwen-base \
  --material-model ckpts/ready/project/material
```

These paths assume the default preparation directory and no local model
overrides. The standalone frontend does not read `models.json`; when using a
different preparation directory or local overrides, supply the corresponding
paths recorded there. Without explicit material paths, this CLI falls back to
`ckpts/qwen-base` and `ckpts/qwen-material` under the source root.
The bundled Qwen runtime is selected automatically and uses packages installed in the active
Python environment. `--qwen-runtime` can select another compatible local runtime;
`--qwen-deps` optionally adds an existing dependency overlay. Base and model sources
also accept Hugging Face `namespace/repo` IDs; `--cache-dir` and
`--local-files-only` apply to resolution, and `--revision` selects the material
checkpoint revision. Runtime and dependency paths must be local directories.
For the same options on `infer.py`, use `--material-revision` instead of
`--revision` (which selects the completion model there).
Use a fresh output directory. The handoff contains `condition_rgba.png`,
`condition_rgb.png`, `camera.json`, `albedo.png`, and `mra.png`.

For an existing non-opaque foreground RGBA, omit `--with-rmbg` during preparation
and replace `--rmbg-model` with `--use-input-alpha` in the frontend command.
Foreground preprocessing writes a straight-color
`condition_rgba.png` and a matching black-premultiplied `condition_rgb.png`
after reducing the longest input dimension to at most 1024 pixels and applying
a square foreground crop with 10% padding. Qwen receives the straight
RGBA frame, resized once to its 1024-square model canvas; FireRed receives
the RGB condition. Do not premultiply Qwen's RGB channels before attaching alpha.

Foreground extraction uses input alpha or RMBG-2. Depth-based rejection occurs
later during 3D observation construction: alpha erosion and first-hit depth
discontinuity filtering discard unreliable observation pixels. It does not
erode the saved foreground image using MoGe depth. See [Texturing](texturing.md).

Supply geometry separately in the canonical frame, or set
`TORCH_HOME="$PWD/ckpts/ready/torch"` and add
`--geometry-package deployment/runtime/geometry --geometry-model ckpts/ready/geometry`
to generate the final mesh with Pixal3D's fixed mesh scale (add
`--geometry-scale-search` for bounded scale calibration). This uses the NAF
cache created by model preparation. See the
[geometry frontend](geometry.md) for source requirements and the matching
mesh/camera handoff.

## Qwen-Image material backend

With the default preparation layout, the model and runtime paths are:

```text
ckpts/ready/
  external/qwen-base/                # complete Qwen-Image-2.1 base
  project/material/
    adapter_model.safetensors
    adapter_config.json
deployment/runtime/
  qwen/                             # bundled DiffSynth runtime
```

Keep the bundled runtime tree, including `diffsynth/` and
`examples/qwen_image_21/pbr/{infer,inference_utils,inference_offload}.py`.
The adapter requires its native JSON config and complete base model, including
processor, VAE, text encoder, and Transformer. The safe file includes learned
modality embeddings as well as LoRA and uses the Supra-Tex native loader.
Legacy `checkpoint-0100000.pt` plus its matching `.complete.json` also remain
supported. Neither adapter format replaces the external base model.

Inference fixes the native prompts, 1024 resolution, seed 42, sigma 1,
one endpoint-prediction step, and torch attention. The outputs are albedo and
metallic/roughness only; there is no normal or AO prediction. Native RGBA
outputs remain under `material/native/`. Canonical `albedo.png` and `mra.png`
are RGB: MRA red is metallic, green is roughness, and blue is the input
coverage. Native MR blue is a constant 255 placeholder, which the adapter
replaces with input coverage for completion. This coverage is not a prediction
of material transparency. Metallic and roughness use linear 0–255 encoding;
albedo is sRGB.

## FireRed material backend

Select FireRed explicitly with its own base and material bundle:

```bash
python -m supratex.frontend.pipeline --input input.png --output work \
  --rmbg-model /models/rmbg2 --moge-model /models/moge2 \
  --material-backend firered --material-base /models/firered-base \
  --material-model /models/firered-material
```

The legacy `--firered-base` option also supplies the FireRed base path.
For this backend only, replace `--material-model` with all three explicit
artifact arguments:

```bash
--material-weights full_transformer.safetensors \
--material-spec joint_contract.json \
--material-checkpoint-manifest checkpoint_complete.json
```

Individual foreground, camera, and FireRed stages remain available:

```bash
python -m supratex.frontend.preprocess --input input.png --rmbg-model /models/rmbg2 \
  --output-dir work --device cuda
python -m supratex.frontend.camera --image work/condition_rgb.png --model /models/moge2 \
  --output work/camera.json --device cuda
python -m supratex.frontend.firered --base-model /models/firered-base \
  --condition work/condition_rgb.png --weights full_transformer.safetensors \
  --spec joint_contract.json --checkpoint-manifest checkpoint_complete.json --output work/material
```

The FireRed adapter invokes the selected package's `infer.py`. The bundled
package accepts a JSONL record containing the condition image and instructions;
the adapter creates this record automatically. Additional fields retained for
checkpoint compatibility are not used as material targets during inference.
After decoding, the
adapter fixes the blue channel of `<key>_mra.png` to 255 (normalized opacity 1)
everywhere, preserving metallic in red, roughness in green, and the albedo pixels.
The checkpoint's native material-alpha prompt is retained; its predicted blue
channel is replaced before the maps reach the 3D material encoder. Foreground
coverage continues to come from the separate condition mask.
The adapter copies its `<key>_albedo.png` and `<key>_mra.png` outputs to the canonical
`albedo.png` and `mra.png` primary material handoff names. Compatibility
copies named `firered_albedo.png` and `firered_mra.png` are also written.

The bundled FireRed package is source code only. It retains the factorized
`ModalityAxisJointAttention` modules and the exact `modality_joint_blocks.*`
parameter names used by the checkpoint. The base model, factorized transformer
weights, joint contract, and completion manifest must be supplied explicitly;
no model weight is included in this repository. The frontend requires a
checkpoint that produces 1024-by-1024 RGB maps. A compatible external package
may be selected with `--firered-package` in the complete frontend, or `--package`
in the standalone FireRed command; its own entrypoint contract then applies.

## Camera handoff

The camera receipt has `camera_angle_x`, `distance`, and `mesh_scale`. These
are the canonical fields consumed by the surface-observation path: camera at
`(0, 0, distance * mesh_scale)`, looking at the origin with +Y up. MoGe-2
intrinsics are normalized and its horizontal focal length determines both FOV
and canonical distance. The optional `image_width`, `image_height`, and model
receipt fields are provenance only.

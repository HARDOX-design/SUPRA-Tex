# Two-stage geometry frontend

The optional geometry frontend supports the public native Pixal3D pipeline
and the historical `paper` implementation. By default both keep Pixal3D's fixed
mesh scale; `--geometry-scale-search` adds bounded mesh-scale calibration with
the main geometry model. Only the final mesh and its matching canonical camera
are passed to SUPRA-Tex. Intermediate meshes are not saved by
this adapter.

## Requirements

The native geometry source is bundled at `deployment/runtime/geometry`.
The [source setup entrypoints](quickstart.md) install its dependencies and
prepare matching weights, DINO and the local NAF cache. No separate Pixal3D
source checkout is needed for the default path. Geometry requires CUDA and
runs in a separate process using the same interpreter. An external compatible
checkout can still be selected explicitly for the historical paper backend.

### Native Pixal3D

The adapter automatically selects the native backend when the source checkout
contains `pixal3d/pipelines/pixal3d_image_to_3d.py`. Its model root must contain
`pipeline.json`, the five geometry model stems referenced by
`args.models`, and a local DINOv3 model under `dino/`. A typical local layout is:

```text
ckpts/ready/geometry/
  pipeline.json
  ckpts/
    ss_flow_img_dit_1_3B_64_bf16.{json,safetensors}
    ss_dec_conv3d_16l8_fp16.{json,safetensors}
    slat_flow_img2shape_dit_1_3B_512_bf16.{json,safetensors}
    slat_flow_img2shape_dit_1_3B_1024_bf16.{json,safetensors}
    shape_dec_next_dc_f16c32_fp16.{json,safetensors}
  dino/
    config.json
    model.safetensors
```

The five model keys are `sparse_structure_flow_model`,
`sparse_structure_decoder`, `shape_slat_flow_model_512`,
`shape_slat_flow_model_1024`, and `shape_slat_decoder`. Their JSON and weight
files must remain inside the model root. DINO may instead use
`model.safetensors.index.json` with all its referenced local shards. The
repository ignores `ckpts/`, so these local checkpoint copies stay outside Git.
Only geometry networks are loaded. Model preparation assembles this directory
from the Pixal3D and DINOv3 downloads; the unmodified upstream Pixal3D snapshot
alone is not the complete model root expected here.
NAF requires the Torch Hub source at `${TORCH_HOME}/hub/valeoai_NAF_main` and
weights at `${TORCH_HOME}/hub/checkpoints/naf_release.pth`.
For the default prepared layout, set `TORCH_HOME="$PWD/ckpts/ready/torch"` when
running the standalone frontend or geometry CLI from the source root.
`infer.py --pipeline` and `scripts/run_example.py` set it from the prepared
configuration automatically.

The five native networks are constructed on `meta` and receive validated local
checkpoint tensors directly. Loading preserves the original constructor's
per-layer dtypes, including BF16 flow blocks and FP32 decoder boundary layers;
missing keys and incompatible shapes fail explicitly. This avoids initializing
a full set of parameters that would immediately be overwritten.
`scripts/convert_geometry_weights.py` can write `*.runtime.safetensors` copies
already cast to these dtypes beside the originals; when present they are loaded
instead, with identical results and about half the bytes read.

### Paper Pixal3D

For the historical backend, supply the [Pixal3D paper source](https://github.com/TencentARC/Pixal3D/tree/paper)
containing `pixal3dpipeline.py`, `pixal3dpipeline2stage.py`, and its `pixal3d/`
package, without the native source entry above. The adapter disables the
source's MoGe-v1 loader and reuses the prepared MoGe-2 camera. This backend
requires its own compatible model layout:

```text
geometry-weights/
  dense/
    dit/{config.json,model.safetensors}
    vae/{config.json,model.safetensors}
    conditioner/config.json
    scheduler/config.json
  sparse512/
    dit/{config.json,model.safetensors}
    vae/{config.json,model.safetensors}
    conditioner/config.json
    scheduler/config.json
  sparse1024/
    dit/{config.json,model.safetensors}
    vae/{config.json,model.safetensors}
    conditioner/config.json
    scheduler/config.json
```

All required configs and weights are checked before geometry launches. Paper scale
probes always use the main dense model (`use_dense_check=False`); optional
`dense/scale_init` files are neither required nor loaded. DINO/NAF dependencies
and their pretrained caches follow the external source's loading policy; they
are not included in the geometry checkpoint directory by this adapter.

## Complete image frontend

After [preparing the models](quickstart.md) with `--with-rmbg` for RGB inputs,
add the two geometry arguments to the [material frontend](frontend.md):

```bash
TORCH_HOME="$PWD/ckpts/ready/torch" python -m supratex.frontend.pipeline \
  --input inputs/reference.png --output inputs/object \
  --rmbg-model ckpts/ready/external/rmbg2 \
  --moge-model ckpts/ready/external/moge2 \
  --geometry-package deployment/runtime/geometry \
  --geometry-model ckpts/ready/geometry \
  --material-base ckpts/ready/external/qwen-base \
  --material-model ckpts/ready/project/material
```

The output includes `geometry.pt`, the final `camera.json`, condition RGB/RGBA,
albedo, and MRA, ready for `texturing.py`. For a foreground RGBA input, use
`--use-input-alpha` in place of `--rmbg-model`.

Geometry follows Pixal3D by default: the mesh scale stays at
`--geometry-initial-mesh-scale 1.0` and no scale probes run.
`--geometry-scale-search` enables the bounded calibration described below, with
`--geometry-target-padding 3` and `--geometry-max-iterations 2` by default.
Each probe adds one 64-cell sparse-structure sampling (about 10 s on an
RTX 3090; up to three probes). The model
source accepts a local directory or a `namespace/repo` containing the complete
model-root layout above; `--geometry-revision` selects its Hub revision.
`--cache-dir` and `--local-files-only` also apply to
geometry checkpoint resolution. Offline HF/Transformers flags are forwarded
to the geometry process; external torch-hub dependencies need their caches
prepared separately.

When scale search is enabled, the native adapter reuses sparse-structure image
features across scale probes and final sparse-structure conditioning within the
same inference. Shape stages
sample DINO/NAF features only at occupied coordinates, preserving the original
projection arithmetic and coordinate order. The receipt records these operations
under `conditioning_acceleration`.

## Geometry only

If the condition images and MoGe camera are already prepared, generate the
final mesh without loading the material model:

```bash
TORCH_HOME="$PWD/ckpts/ready/torch" python -m supratex.frontend.geometry \
  --package deployment/runtime/geometry --model ckpts/ready/geometry \
  --input-dir inputs/prepared --output-dir outputs/geometry
```

Use a fresh output directory. It receives `geometry.pt`, `camera.json`,
`geometry_receipt.json`, `geometry.log`, and copies of the condition images.
The original inputs are preserved. The receipt records the selected scale,
camera actually used for final generation, check-model usage, seed, and mesh
counts. Without `--geometry-scale-search`, its `scale_search` entry is
`{"enabled": false, "probes": [], "stop_reason": "disabled"}`. With it, the entry
records probes, scale bounds, selected padding, convergence, baseline fallback,
and the stopping reason; `dense_probe_calls` counts main-model probes and
`uses_dense_check_model` is false. The standalone geometry command accepts the
same option as `--scale-search`.
Native receipts identify `Pixal3D native geometry cascade` and record
`final_sparse_structure_resolution = 32` and final `resolution = 1024`.
In the complete frontend these diagnostics live under `geometry/`;
the final mesh and matching camera are copied to the main input directory.

## Scale and coordinate contract

With `--geometry-scale-search`, both backends use the paper scale-update formula
with the adapter's bounded selection policy. Without it, they generate directly at
the initial scale. The prepared MoGe-2 camera owns both FOV and distance.
By default, scale starts at 1.0 and the frontend computes
`distance = 0.5 / tan(FOV / 2)`.
The native backend uses this camera directly for 64-cell occupancy probes,
then generates a fresh 32-cell sparse structure and runs the 512/1024 shape
cascade. Its decoded mesh is already canonical and needs no rotation or scale
conversion.

The paper backend retains the original outer inference path. The source first
computes its own distance using initial scale and its image-padding convention;
before any probe, the hook replaces that shared distance tensor with the
prepared value. Its receipt retains both
`upstream_initial_generation_camera` and `initial_generation_camera` and records
the substitution. Both backends keep FOV and distance fixed throughout
calibration and final generation and record
`camera_distance_policy = preserve_prepared_demo_camera`.

Calibration aims for boundary padding in a 64-cubed occupancy grid. The default
initial scale is 1.0 and target padding is 3 cells; padding from 2 through 4 cells
is accepted. The baseline is probed first, followed by at most two scale updates
(three probes total). `--geometry-max-iterations` must be positive and can lower
the update budget; values above 2 still allow only two updates. Target padding
must be an integer from 1 through 30. Proposals are bounded to half through twice
the initial scale. Clipped candidates continue shrinking within those bounds.

Only nonempty, finite, integer occupancy coordinates inside the 64-cell grid
are valid. A candidate is safe when its padding reaches the lower acceptance
bound (2 cells by default). The search retains the safe candidate closest to
the target, even if it never reaches the acceptance range. A safe baseline is
retained unless a later safe candidate is closer; an unsafe baseline yields
to any safe candidate, even if that candidate has a larger target error. An
invalid later probe retains the best candidate already found. If no safe
candidate exists, calibration falls back to the baseline. This optimizes
occupancy padding, not FOV, pose, or image silhouette IoU.

The paper source exports its final mesh after rotating `(x,y,z)` to `(x,-z,y)` and
scaling by `0.5 / optimized_mesh_scale`. The adapter reverses the rotation
and multiplies by the selected scale, recovering the canonical mesh in
`[-0.5,0.5]`. The final camera retains the prepared generation distance and
selected `mesh_scale`; SUPRA-Tex projection and DDA both use effective distance
`distance * mesh_scale`. This paired conversion preserves the source projection.
The original MoGe camera is retained under `moge_initial_camera` for provenance.
Falling back to scale 1.0 restores the prepared baseline camera parameters.
The paper and native models and image preprocessing differ, so identical
geometry across the two backends is not promised. A safer occupancy margin also
does not guarantee better shape, alignment, or texture quality.

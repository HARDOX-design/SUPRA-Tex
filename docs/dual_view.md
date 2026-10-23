# Optional dual-view inference

Dual-view inference adds a generated back observation to the single-view
pipeline. Both views use the same geometry, material backend, model artifacts,
and material seed 42. Front observations take priority where they overlap.

> **Note:** Automatic back-view generation runs GPT-6 Astra with its built-in
> image generation tool through the external Codex CLI (see
> [Requirements](#requirements)), which this repository does not provide. As
> foundation image models improve, dual-view results can improve with them.
> Without that access, use single-view inference; for small visible text, add
> `--refine` (see [texturing](texturing.md#texture-refinement)), which needs no
> external service.

## Requirements

Use the existing [CUDA environment](installation.md) and prepared models.
Automatic back-image generation additionally invokes the `codex` executable
with the model and image tool configured in the
[generation harness](../supratex/backview/harness/README.md): the current code
requires `gpt-6-astra`, `ultra` reasoning, and built-in `image_gen.imagegen` access.
These capabilities are separate from the Python dependencies and are required
by the implementation; they are not supplied by this repository.

`--local-files-only` controls model loading and does not make image generation
offline. Reusing a prepared back package avoids a new generation request.

Dual-view always refines both views: it projects each observed albedo into the
atlas and repairs the UV gutters from the observations, as
[`--refine`](texturing.md#texture-refinement) does for one view. It therefore
needs the 4096 PNG atlas; any face target works, and `--gutter` uses the
observation-guided gutter. If refinement fails, it is skipped with a warning and
the completed front/back result is written unrefined (`refine: skipped` in
`receipt.json`). It requires a new output GLB and a fresh work directory.

## Run automatically

With the generation environment available, `--dual-view` without `--input` uses
the cinema signage building from the project page (`assets/cinema_signage.png`):

```bash
python infer.py --pipeline supra3d/supra-tex --dual-view \
  --output outputs/cinema_signage_dual_view.glb \
  --work-dir outputs/cinema_signage_dual_view_work
```

The image is opaque RGB, so `--pipeline` also prepares RMBG-2 for foreground
extraction (see [model preparation](quickstart.md#2-prepare-the-models)).

The command prepares the front geometry and material maps, renders a back normal
pack, generates the back image, predicts matching material maps, and runs
completion followed by front/back texture reprojection. It records the front
material metadata automatically.

`--back-view-dir /path/to/prepared-back` reuses an existing prepared package.
Its geometry, camera, condition images, and metadata must match this run's front
inputs. To reuse retained front inputs as well, use the staged workflow below.

## Run the stages separately

### 1. Prepare front inputs

Generate geometry and reference material maps with provenance recording enabled:

```bash
TORCH_HOME="$PWD/ckpts/ready/torch" python -m supratex.frontend.pipeline \
  --input assets/cinema_signage.png --output work/front \
  --rmbg-model ckpts/ready/external/rmbg2 \
  --moge-model ckpts/ready/external/moge2 \
  --geometry-package deployment/runtime/geometry \
  --geometry-model ckpts/ready/geometry \
  --material-base ckpts/ready/external/qwen-base \
  --material-model ckpts/ready/project/material \
  --record-material-provenance
```

The example uses the default [prepared installation](quickstart.md). If you
selected local overrides or another output directory, use the model paths and
NAF cache recorded in its `models.json`.
Prepare RMBG-2 with `--with-rmbg` (or BiRefNet with `--rmbg`) first. For an
RGBA input with foreground alpha, replace `--rmbg-model` with
`--use-input-alpha`. The front directory must contain:

```text
front/
  geometry.pt
  camera.json
  geometry_receipt.json
  condition_rgba.png
  condition_rgb.png
  albedo.png
  mra.png
  material_receipt.json
```

Keep the model files unchanged between front and back prediction. The receipts
bind the images, geometry, camera, backend, and model selection. The geometry
receipt must come from the current frontend. If using the optional FireRed
backend, custom `--firered-manifest` inputs are not accepted when recording
material provenance.

### 2. Render the back normal pack

```bash
python -m supratex.backview render \
  --input-dir work/front --output-dir work/back_pack
python -m supratex.backview validate \
  --pack-dir work/back_pack --input-dir work/front
```

The pack contains the front reference, `back_normal.png`, `back_mask.png`,
`cameras.json`, and an HTML preview. The renderer chooses a back camera around
the opposite view while matching foreground occupancy. Normal and mask images
share the same camera and rasterization. The accepted camera protocol is
`supratex.back_view_camera.v4`; regenerate older packs from the original inputs.

### 3. Generate and review the back image

First prepare the input snapshot and inspect the plan:

```bash
python -m supratex.backview generate \
  --pack-dir work/back_pack --run-root work/back_generation
```

Without `--execute`, this command makes no generation request. To execute:

```bash
python -m supratex.backview generate \
  --pack-dir work/back_pack --run-root work/back_generation \
  --execute
python -m supratex.backview.harness audit --run-root work/back_generation
python -m supratex.backview.harness report --run-root work/back_generation
```

Inspect `work/back_generation/report/index.html`. A result needs successful
worker execution, all provenance and output checks, and a current `acceptable`
or `needs_review` visual review for the selected image. `needs_review` permits
the next stage without changing the review label or removing its defects.
`needs_regeneration`, missing or stale reviews, and hard-check failures remain
blocked. An image file alone is insufficient. The default case ID is `input`; if using
`--case`, keep it consistent across generation and material preparation.

### 4. Prepare back material maps

```bash
python -m supratex.backview materials \
  --front-input-dir work/front --pack-dir work/back_pack \
  --run-root work/back_generation --output-dir work/back_input
python -m supratex.backview validate \
  --back-view-dir work/back_input --input-dir work/front
```

This stage selects the material backend and model artifacts recorded for the
front. It preserves the generated image, applies the rendered silhouette,
and writes matching condition images, material maps, and validation metadata.
The `back_view_review` metadata retains the original visual grade, review flag,
defects, and review source/hash for follow-up iterations.
Material preparation keeps its historical review snapshot. A new completion run
revalidates the current review and records that snapshot without overwriting the
preparation record. If the review changes during completion, final publication
stops; a later review change makes a newly recorded completion snapshot stale.
Legacy records without a snapshot use the current review and are identified as
not having a recorded review snapshot.

### 5. Complete and validate the mesh

```bash
python texturing.py --pipeline supra3d/supra-tex \
  --input-dir work/front --back-view-dir work/back_input --dual-view \
  --output outputs/object_dual_view.glb --work-dir work/dual_view
python -m supratex.backview validate --work-dir work/dual_view
```

Front observations win overlap. Back-only observations extend the visible
material set, and completion runs once. Texture reprojection then processes
the front and back views while preserving front-protected texels.
`receipt.json` records the final result and its validation status separately from
the retained `back_view_review` snapshot. A successful result can still carry
`needs_review`; execution success is not a clean visual rating. Failed checks
leave intermediate files and stop final output publication.

## Resume an interrupted run

For interrupted generation, reuse its run directory:

```bash
python -m supratex.backview.harness run \
  --run-root work/back_generation --execute
```

Add `--retry-review` to request a new attempt for either `needs_review` or
`needs_regeneration`, including eligible results retained for a later iteration.
Otherwise recovery can finish a missing review without requesting a new image.
After generation succeeds, run material preparation into a new output directory
and texturing into a fresh work directory. Full `infer.py` does not resume a
nonempty work directory.

Changing the geometry, camera, condition images, or selected model artifacts
requires matching back-view inputs. Do not reuse a package from another case.
The validation commands check that these inputs match.

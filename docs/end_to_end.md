# Image to textured mesh

`infer.py` takes one RGB/RGBA image and runs foreground extraction, camera
estimation, geometry generation, reference-view material prediction, 3D material
completion, and GLB export. Install the runtime with the
[setup guide](quickstart.md) before running these commands from the repository root.

## Run from a model release

```bash
python infer.py --pipeline supra3d/supra-tex --output outputs/spiny_whelk_shell.glb
```

`--pipeline` accepts a model repository ID/URL, a local model repository directory,
or a local `pipeline.json`. It selects the Qwen-Image material backend, prepares
all required models under `ckpts/ready/`, and uses the manifest's inference
defaults. Use `--pipeline-revision` to select a Hub snapshot and `--models-dir`
to select another prepared-model directory.

Without `--input`, `infer.py` uses the bundled `assets/spiny_whelk_shell.png`
(`assets/cinema_signage.png` with `--dual-view`); pass `--input path/to/image.png`
for your own image.
The bundled example has foreground alpha, which is reused automatically. For
an opaque RGB image, the pipeline also prepares RMBG-2. DINOv3 and RMBG-2 require
upstream access approval; see [model preparation](quickstart.md#2-prepare-the-models).
`--use-input-alpha` explicitly requires a non-opaque RGBA input.

Use a new `.glb` output path. The default work directory is
`<output stem>_work` beside it; `--work-dir` selects another empty or absent
directory. Existing outputs are never overwritten.

## Run with models prepared in advance

After the [preparation step](quickstart.md#2-prepare-the-models):

```bash
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/spiny_whelk_shell.glb
```

The runner supplies local model/runtime paths, configures the NAF cache, and
runs with Hub access disabled. Add `--print-command` to inspect its invocation
without starting inference. For an opaque RGB input, prepare RMBG with
`--with-rmbg` first.

The runner accepts inference controls such as `--seed`, `--texture-size`,
`--attention`, `--low-vram`, and `--work-dir`. Model paths, revisions, the
material backend, and offline settings come from `--config` and cannot be
overridden through this launcher. Use `infer.py` directly for those changes.

## Inference controls

- `--seed` defaults to 42 for geometry and completion. Material prediction uses
  its fixed seed 42.
- `--attention auto` selects a compatible installed attention backend.
  `--attention sdpa` avoids the FlashAttention dependency.
- `--low-vram` enables CPU offload; `--no-low-vram` disables it. Automatic
  selection enables offload on GPUs with at most 24 GiB. See
  [GPU compatibility](gpu_compatibility.md) for memory and validation limits.
- Export defaults are `--texture-size 4096`, `--decimation-target 500000`, and
  `--texture-format png`; `webp` is also supported. Explicit values override
  manifest defaults.
- `--gutter` adds one texel of base-color padding outside UV charts. It is
  disabled by default and preserves covered texels, alpha, metallic/roughness,
  geometry, and UVs.
- The depth pre-filter is on by default for single-view runs. It rejects
  unreliable observations near depth edges before completion to reduce texture
  leakage. `--no-filter` (alias `--no-image-depth`) disables it. Dual-view runs
  omit it automatically. See the [comparison and usage](texturing.md#depth-pre-filter).
- Floater removal is on by default: before completion, disconnected geometry
  pieces that project mostly outside the image foreground are dropped, and a
  `Floaters:` line reports the count. `--no-remove-floaters` keeps them.
  Pieces hidden behind the object and extra geometry attached to it are kept;
  see [texturing](texturing.md#floating-geometry). The saved `geometry.pt` stays
  unchanged, and the dual-view back view is rendered from it.
- `--refine` further refines the visible texture after export and aligns it
  with the input image, keeping very small text and intricate detail sharp. It is
  off by default because it adds about two minutes on an RTX 3090. It needs the
  default 4096 PNG atlas, works with any face target and with `--gutter` (which
  then uses the observation-guided gutter). `--dual-view` always refines both
  views, so it implies `--refine`. If refinement fails, it is skipped with a
  warning and the unrefined result is written. Its receipts are under
  `<work-dir>/refine`.
  See [texturing](texturing.md#texture-refinement).

The Qwen material predictor uses 1024-square inputs, Torch attention, and a
single endpoint prediction. It produces albedo and metallic/roughness maps;
coverage in the MRA handoff comes from input alpha. Geometry keeps Pixal3D's
fixed mesh scale (`--geometry-initial-mesh-scale 1.0`) by default;
`--geometry-scale-search` enables bounded scale calibration. See
[geometry](geometry.md) for the calibration procedure and
[frontend](frontend.md) for material formats. `--parallel-frontend` runs the
geometry and Qwen material stages concurrently on the first two visible GPUs.

## Explicit model paths

To use the default prepared layout without `--pipeline` or the launcher:

```bash
TORCH_HOME="$PWD/ckpts/ready/torch" python infer.py \
  --output outputs/spiny_whelk_shell.glb \
  --moge-model ckpts/ready/external/moge2 \
  --geometry-package deployment/runtime/geometry \
  --geometry-model ckpts/ready/geometry \
  --material-base ckpts/ready/external/qwen-base \
  --material-model ckpts/ready/project/material \
  --pretrained ckpts/ready/project/completion \
  --trellis-root ckpts/ready/external/vae
```

Without `--pipeline`, `--moge-model`, `--geometry-package`, and
`--geometry-model` are required. The geometry directory must have its prepared
`pipeline.json`, model files, and local DINOv3 directory. Configure its local
NAF cache as shown above. If preparation used local overrides or another output
directory, use the corresponding paths in `models.json`.
For opaque RGB input, add `--input` and `--rmbg-model /path/to/RMBG-2.0`.

Completion and material bundles can be local directories or Hub repository
IDs. `--revision`, `--geometry-revision`, and `--material-revision` select their
respective snapshots. `--cache-dir` selects a shared Hub cache and
`--local-files-only` requires local or cached weights.
With `--pipeline`, external-model overrides must be existing local paths;
completion and material overrides may also use Hub IDs.

To select individual completion artifacts, replace `--pretrained` with
`--model-config` and `--checkpoint`, keeping `--trellis-root`. See
[weights](weights.md) for the supported safetensors and legacy layouts.
The bundled Qwen runtime is selected automatically; `--qwen-runtime` and
`--qwen-deps` allow an alternate compatible runtime and dependency overlay.

FireRed is an optional backend for the explicit-path invocation. Set
`--material-backend firered` and provide its corresponding `--material-base`
and `--material-model`. The default release manifest and prepared-model runner
use Qwen-Image. FireRed does not support the CPU-offload path; use
`--no-low-vram` on devices where offload would otherwise be selected.
See [frontend](frontend.md) for FireRed's checkpoint format and explicit
weight/contract arguments.

## Outputs

```text
outputs/
  object.glb
  object_work/
    run.json
    input/
      condition_rgb.png
      condition_rgba.png
      geometry.pt
      camera.json
      albedo.png
      mra.png
      geometry/
      material/
```

`run.json` records progress, selected paths, and failures. A completed run has
`status: complete` and `stage: complete`; inspect the GLB separately for visual
quality. Final decoding/export can change mesh topology and UV layout.

A failed run keeps intermediate files for inspection. Preflight errors can
occur before `run.json` is created. Full inference does not resume a nonempty
work directory: choose a fresh directory, or reuse valid intermediate inputs
with the [frontend](frontend.md) and [texturing](texturing.md) entrypoints.

## Optional dual-view inference

`--dual-view` adds a generated back observation and requires a separate image
generation environment. It is disabled by default. See
[dual-view inference](dual_view.md) for prerequisites, staged commands, and
recovery.

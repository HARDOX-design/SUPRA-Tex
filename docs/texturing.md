# Texture an existing mesh

`texturing.py` completes PBR materials for a supplied mesh using reference-view
material maps. Prepare the maps with the [material frontend](frontend.md) or
supply compatible maps yourself. Keep the mesh and its matching camera together;
the [geometry frontend](geometry.md) can produce them.

## Inputs

The input directory must contain:

- `geometry.pt`: a dictionary with `verts` (`float[N,3]`) and `faces` (`int[M,3]`).
  A compressed `geometry.npz` with the same `verts` and `faces` arrays is also
  accepted for single-view texturing; dual-view inputs use `geometry.pt`.
- `camera.json`: canonical `camera_angle_x` in radians and `distance`, with
  optional `mesh_scale`. The effective camera distance is `distance * mesh_scale`.
- `condition_rgba.png`: a square observation with foreground alpha, aligned
  with the mesh and camera.
- `albedo.png` and `mra.png`: RGB images at 1024 by 1024. MRA stores metallic,
  roughness, and alpha in R, G, and B respectively.

## Run

From the repository root after [installation](quickstart.md):

```bash
python texturing.py --pipeline supra3d/supra-tex \
  --input-dir inputs/object \
  --output outputs/textured_object.glb
```

Without `--input-dir`, `texturing.py` uses the bundled
[pitcher plant example](../assets/examples/pitcher_plant): the project-page mesh
with its camera, foreground condition, and reference-view material maps.

`--pipeline` accepts a model repository ID/URL, a local model repository directory,
or a local manifest file. This entrypoint loads only the completion model and
TRELLIS.2 VAEs. It does not generate the input geometry or reference material maps.
No separate `texturing.json` is needed.

The manifest supplies the seed, texture size, face target, texture format, and
attention defaults. Explicit CLI values take precedence. The bundled defaults
are seed 42, a 4096-pixel PNG atlas, a 500000-face target, and automatic attention.
Use `--low-vram` for CPU offload or `--attention sdpa` to avoid FlashAttention;
see [GPU compatibility](gpu_compatibility.md).

`--pipeline-revision` selects a Hub release snapshot. `--cache-dir` selects
the download cache; `--local-files-only` requires local or cached weights.
For a local release and separately stored VAEs:

```bash
python texturing.py \
  --pipeline /path/to/model-release/pipeline.json \
  --trellis-root /path/to/TRELLIS.2-4B --local-files-only \
  --input-dir inputs/object --output outputs/textured_object.glb
```

A local manifest's completion path is relative to that manifest. Its external
VAE source still applies unless overridden with `--trellis-root`.

## Alternate checkpoint layouts

Use `--pretrained /path/to/completion` and `--trellis-root /path/to/TRELLIS.2-4B`
in place of `--pipeline` for a standalone completion bundle. `--pretrained` also
accepts a Hub repository ID. `--revision` selects that bundle's revision;
`--pipeline-revision` selects the enclosing pipeline release when used together.

Individual legacy artifacts can be selected explicitly:

```bash
python texturing.py \
  --input-dir inputs/object --output outputs/textured_object.glb \
  --model-config configs/inference_model.json \
  --checkpoint /path/to/material_completion.pt \
  --trellis-root /path/to/sparse_encoders
```

The checked-in legacy config resolves decoder stems under `weights/scvae/`
and normalization relative to the config file. Populate those paths or update
the config to match your files. For the published safetensors layout, use the
release's `config.json` and all four external VAEs instead. See
[weights](weights.md) for the exact layouts.

## Completion and export

Completion uses a fixed 25-step Euler sampler with `rescale_t=3` and
`sigma_min=1e-5`. Visible observations use trusted alpha at least 254,
a 5-by-5 erosion at 1024 resolution, and a depth-discontinuity filter before
encoding material features. These filters leave the saved RGBA image unchanged.

Each run prints an `Observation:` line that compares the projected geometry
silhouette with the image foreground (IoU, the fraction of the geometry inside
the image, and the fraction of the image covered by the geometry) and counts the
admitted and rejected trusted pixels. An IoU below 0.9 also prints a warning:
in mismatched regions the texture relies on completion rather than the image.

The output must be a new `.glb` path. The mesh stays fixed while material
completion runs, but final decoding and export can change vertices, faces,
and UV layout. Input vertex arrays and UVs are not preserved by this workflow.

Optional `--gutter` adds one texel of base-color padding outside UV charts.
It is disabled by default and preserves covered texels, alpha, metallic/roughness,
geometry, and UVs. See the [image-to-3D guide](end_to_end.md) for export controls.

## Texture refinement

`--refine` further refines the texture that the reference view sees and aligns
it with the input image. It is off by default because it adds time. After
completion and export, it writes the observed albedo into the atlas texels visible from the reference
camera, then repairs the one-texel UV gutters from the observed albedo gradient.
The two steps always run together; they are the front half of dual-view texturing
and pass the same nineteen-phase render gates. Only base color changes;
metallic, roughness, geometry, UVs, and unseen regions stay as completed. Fine
print such as table entries and warning labels, which can blur through the
material latent, stays legible.

The atlas must be the default 4096 PNG; any `--decimation-target` works.
`--gutter` may be combined with it: refinement then supplies the
observation-guided gutter instead of the export-time nearest-color padding,
which would only copy pre-refinement colors. `--dual-view` always applies the
same refinement to both views, so `--refine` is implied there. Intermediate meshes and
receipts go to `--work-dir` (default `<output stem>_refine` beside the GLB).
If refinement fails a render gate or errors, it is skipped with a warning: the
unrefined completion is written instead (with the export-time gutter when
`--gutter` is set), and the receipt records `refine: skipped` and the reason.
On an RTX 3090 it adds about two minutes.

```bash
python texturing.py --pipeline supra3d/supra-tex --refine \
  --output outputs/pitcher_plant_refined.glb
```

## Floating geometry

Generated geometry can be misaligned with the image or contain extra pieces.
Those pieces receive completed texture that the image never showed. Floater
removal drops them before completion when they can be identified from the image
alone. Image-to-3D inference (`infer.py`) applies it by default
(`--no-remove-floaters` turns it off). `texturing.py` applies it only with
`--remove-floaters`, because a supplied mesh may intentionally extend beyond the
image:

- The mesh is split into connected components; duplicate vertex positions, such
  as UV seams in an exported mesh, are welded first.
- Every component except the largest is projected into the reference view. A
  component is removed when fewer than half of its faces land inside the image
  foreground (alpha at least 128, dilated by 16 pixels at 1024 resolution to
  tolerate ordinary misalignment).
- Each run prints a `Floaters:` line with the removed components and faces. The
  input files are not modified. The check runs on the GPU and takes under a
  second for a 10-million-face mesh on an RTX 3090.

Because the test uses only the reference view, it keeps everything the
foreground covers. It does not remove pieces hidden behind the object, which
cannot be told apart from genuine unseen parts such as the back of a plant, and
it does not affect extra geometry attached to the object. For those cases,
generate the geometry again with another `--seed`, or clean the mesh in a mesh
editor and texture it with `texturing.py`. On our test cases it removed nothing
from correctly generated geometry.

```bash
python texturing.py --pipeline supra3d/supra-tex --remove-floaters \
  --input-dir inputs/object --output outputs/object_cleaned.glb
```

## Optional dual-view texturing

A prepared back-view package can add observations of the unseen side:

```bash
python texturing.py --pipeline supra3d/supra-tex \
  --input-dir work/front --back-view-dir work/back_input --dual-view \
  --output outputs/object_dual_view.glb --work-dir work/dual_view

python -m supratex.backview validate --work-dir work/dual_view
```

This workflow requires matching geometry/material metadata for both
views; a raw normal image is insufficient. It always refines both views (see
[texture refinement](#texture-refinement)), so it needs the 4096 PNG atlas;
any face target works, `--gutter` uses the observation-guided gutter, and the
work directory must be new. Front observations take priority where views overlap. See [dual-view inference](dual_view.md)
for preparation and validation commands. In `texturing.py`, `--work-dir` applies
only to `--dual-view` and `--refine`.

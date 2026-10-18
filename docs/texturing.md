# Texture an existing mesh

`texturing.py` completes PBR materials for a supplied mesh using reference-view
material maps. Prepare the maps with the [material frontend](frontend.md) or
supply compatible maps yourself. Keep the mesh and its matching camera together;
the [geometry frontend](geometry.md) can produce them.

## Inputs

The input directory must contain:

- `geometry.pt`: a dictionary with `verts` (`float[N,3]`) and `faces` (`int[M,3]`).
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

The output must be a new `.glb` path. The mesh stays fixed while material
completion runs, but final decoding and export can change vertices, faces,
and UV layout. Input vertex arrays and UVs are not preserved by this workflow.

Optional `--gutter` adds one texel of base-color padding outside UV charts.
It is disabled by default and preserves covered texels, alpha, metallic/roughness,
geometry, and UVs. See the [image-to-3D guide](end_to_end.md) for export controls.

## Optional dual-view texturing

A prepared back-view package can add observations of the unseen side:

```bash
python texturing.py --pipeline supra3d/supra-tex \
  --input-dir work/front --back-view-dir work/back_input --dual-view \
  --output outputs/object_dual_view.glb --work-dir work/dual_view

python -m supratex.backview validate --work-dir work/dual_view
```

This workflow requires matching geometry/material metadata for both
views; a raw normal image is insufficient. It uses a 4096 PNG atlas and a
500000-face target, with `--gutter` disabled and a new work directory. Front
observations take priority where views overlap. See [dual-view inference](dual_view.md)
for preparation and validation commands. `--work-dir` is used only by this
optional path in `texturing.py`.

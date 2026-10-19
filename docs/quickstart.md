# Quick start

This source directory contains the project runtime, including the native
geometry, Qwen PBR, MoGe, utils3d and NAF Python implementations. The steps below
install their dependencies, prepare model weights, and run an example.

## 1. Install the software

Use Linux x86_64, Conda, Git, a C++ compiler, an NVIDIA GPU/driver,
and a CUDA 12.4 toolkit with `nvcc`. From the source root:

```bash
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python scripts/install_runtime.py --conda --dry-run
python scripts/install_runtime.py --conda
```

The installer uses the active Conda environment. It installs
PyTorch 2.6.0/cu124, torchvision 0.21.0/cu124, the pinned Python requirements,
bundled MoGe/utils3d, NATTEN, and fixed public revisions of CuMesh, FlexGEMM,
nvdiffrast and TRELLIS.2's o-voxel extension. It then checks package dependencies,
CUDA access and actual runtime imports. Native compilation may take some time.
FlashAttention is not installed by default and inference then uses SDPA;
FlashAttention-2/3 is faster and is selected automatically once installed (see
[FlashAttention](installation.md#flashattention-recommended-for-speed)).

Use a dedicated Conda environment rather than `base`. If you prefer `venv`,
run `python3.10 scripts/install_runtime.py --venv /path/to/new/environment`
instead, then activate it with `source /path/to/new/environment/bin/activate`.
The `--python /path/to/python3.10` option selects Python for venv creation only.
It stops on incompatible existing source checkouts instead of resetting them.
It can be rerun after a failed download/build using the same managed environment.
FlexGEMM is built from a temporary patched copy, so its install-time autotune
cache is written under the environment's `.cache/flex_gemm` instead of `~/.flex_gemm`.
See [installation details](installation.md) for manual setup.

## 2. Prepare the models

The [project model repository](https://huggingface.co/supra3d/supra-tex) contains
`pipeline.json`, the completion model, and the material adapter. See
[model weights](weights.md) for component layouts and external dependencies.
It is openly available and needs no access request.

DINOv3 is required even for the default RGBA example and requires the
[publisher's access approval](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
Log in with an approved Hugging Face account before downloading, or pass an
existing local DINO directory with `--dino`:

```bash
hf auth login
```

```bash
python scripts/prepare_models.py --dry-run
python scripts/prepare_models.py
```

The preparation entrypoint fetches the project subdirectories and external
Qwen-Image-2.1, Pixal3D geometry, DINOv3, MoGe-2, TRELLIS.2 VAEs and NAF weights.
Component paths, external versions, file selections and inference defaults come
from the root `pipeline.json` at the requested project revision.
It assembles the geometry configuration, local DINO directory and NAF Torch Hub
cache, then writes `ckpts/ready/models.json`. Inference consumes these prepared
local files with Hub access disabled and uses the selected manifest's defaults.
The generated `models.json` records local runtime bindings; component architecture
and normalization are maintained only in their existing component configs.

For opaque RGB inputs, also request access to
[RMBG-2](https://huggingface.co/briaai/RMBG-2.0) and include it in preparation:

```bash
python scripts/prepare_models.py --with-rmbg
```

To avoid RMBG-2's access request, use the public
[BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) instead. RMBG-2 is built on
it and both load through the same foreground adapter; on our examples their
foreground masks agree closely (IoU above 0.99). Download it and pass its directory:

```bash
hf download ZhengPeng7/BiRefNet --local-dir /path/to/BiRefNet
python scripts/prepare_models.py --rmbg /path/to/BiRefNet
```

Either option also adds the foreground model to an existing preparation in the
same output directory.
Keep the original model sources and options; the script updates its generated
configuration without replacing user edits. Authentication is not stored in
`models.json`.

Use `--revision` to select a project snapshot, or `--repo /path/to/supra-tex`
for a local model repository root. `--pipeline /path/to/pipeline.json` selects
an explicit local manifest. `--completion` and `--material`
override the project subdirectories with existing local model directories or
separate Hub repositories. External local overrides are available as
`--qwen-base`, `--geometry`, `--dino`, `--vae`, `--moge`, `--naf-weights` and
`--rmbg`; run `python scripts/prepare_models.py --help` for their accepted layouts.
Local overrides avoid downloading those models but remain external file
dependencies. `--cache-dir` uses a shared Hugging Face cache and links the
prepared files to it; keep that cache available. It reuses models already in
that cache, so pass the hub cache directory (`HF_HUB_CACHE`, by default
`~/.cache/huggingface/hub`), not `HF_HOME`. Without an explicit cache,
downloads go into the prepared model directory and do not reuse an existing
Hugging Face cache. Use `--local-files-only` for
local files and cached downloads; NAF weights must already be prepared or
supplied through `--naf-weights`.

The default models occupy much more space than the source or adapter alone:
Qwen's complete foundation components, geometry networks and VAEs are required.
`--dry-run` previews selected sources and destinations without network access.
For a Hub repository it uses the bundled [configuration](../configs/pipeline.json); supply a local repository or
`--pipeline` to preview another manifest. A normal preparation run obtains the
manifest from the requested Hub revision.
Preparation reuses completed downloads after interruption. Retry using the
same arguments. If changing model sources, choose a new `--output-dir` and pass
its `models.json` to the checker and runner with `--config`. Existing local
models can be reused with the overrides above.

## 3. Check and run

```bash
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/spiny_whelk_shell.glb
```

The default bundled example has foreground alpha. For an opaque RGB image, such
as the bundled candle jar, prepare RMBG as above and run:

```bash
python scripts/run_example.py --input assets/metalwork_candle_jar.png \
  --output outputs/metalwork_candle_jar.glb
```

Replace `--input` with your own image.

`--print-command` shows the exact inference command without running it.
`--config` selects a different prepared `models.json`. The launcher uses this
checkout's runtime, reads manifest defaults and sets `TORCH_HOME` to its prepared
NAF cache. Explicit inference controls take precedence over the defaults;
model paths come from `models.json`. Use a new
output filename/work directory for every inference; existing results are kept.
The underlying [inference CLI](end_to_end.md) exposes further controls, including
CPU offload and the optional dual-view workflow.

Alternatively, prepare and run directly from one pipeline source:

```bash
python infer.py --pipeline supra3d/supra-tex \
  --input path/to/image.png --output outputs/object.glb
```

`--pipeline` also accepts a local model repository root or manifest file.
`--pipeline-revision` selects a Hub snapshot; `--models-dir` selects the prepared
model directory. The same upstream access requirements apply.

A successful full run produces a nonempty GLB and a sibling work directory
whose `run.json` has `status: complete` and `stage: complete`. Inspect the GLB
for visual quality separately. `check_runtime.py` checks model paths/headers,
actual Qwen and geometry imports, native extensions and a small CUDA operation;
passing it alone is not a full model inference result. `--imports-only` needs
no model weights; `--files-only` skips the CUDA/import probes.

The optional automatic dual-view mode needs its separately documented image
generation capability. The default single-view example does not require it.

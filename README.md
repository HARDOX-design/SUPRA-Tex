<h1 align="center">SUPRA-Tex: 3D Generation with Super-Aligned Texture</h1>

<p align="center">
  <strong>English</strong> | <a href="README_zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://supra-3d.github.io/supratex/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2610.22352"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv paper"></a>
  <a href="https://huggingface.co/spaces/UniLat3D/Supra-Tex"><img src="https://img.shields.io/badge/Demo-Hugging_Face-blueviolet" alt="Online demo"></a>
  <a href="https://huggingface.co/supra3d/supra-tex"><img src="https://img.shields.io/badge/Models-Hugging_Face-yellow" alt="Model weights"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT%20%2B%20third--party-green" alt="Source license"></a>
</p>

<p align="center">
  <a href="https://chensjtu.github.io/">Chen Yang</a><sup>1*</sup>,
  <a href="https://jaminfong.cn/">Jiemin Fang</a><sup>1*†✉</sup>,
  <a href="https://mfp0610.github.io/">Fanpeng Meng</a><sup>1*</sup>,
  <a href="https://inso-13.github.io/">Zihao Huang</a><sup>1</sup>,<br>
  <a href="https://supra-3d.github.io/supratex/#paper">Jichen Pang</a><sup>2</sup>,
  <a href="https://wuguanjun0627.github.io/">Guanjun Wu</a><sup>3</sup>,
  <a href="https://supra-3d.github.io/supratex/#paper">Zhikuan Bao</a><sup>1</sup>,
  <a href="https://www.qitian1987.com/index.html">Qi Tian</a><sup>1✉</sup>
</p>

<p align="center">
  <sup>1</sup>Huawei Inc. &nbsp;&nbsp;
  <sup>2</sup>Shanghai Jiao Tong University &nbsp;&nbsp;
  <sup>3</sup>Huazhong University of Science and Technology<br>
  <sup>*</sup>Equal contribution &nbsp;&nbsp;
  <sup>†</sup>Project lead &nbsp;&nbsp;
  <sup>✉</sup>Corresponding authors
</p>

https://github.com/user-attachments/assets/ecc9e9ec-e91f-4eba-927d-86322796459a

**SUPRA-Tex** generates textured 3D assets from a single image, preserving details
such as lettering, patterns, and material boundaries. It predicts PBR materials
in the reference view, encodes them as surface-aligned 3D observations, and
completes unseen regions with an **Interleaved Flow Transformer**. The output is
a relightable GLB with base color, metallic, and roughness textures.

- **Image-aligned textures.** Use visible surface observations to guide 3D
  material generation and retain the details of the input image.
- **Complete PBR assets.** Generate materials for unseen surfaces and export
  textures that support rendering under new lighting.
- **Image-to-3D and mesh texturing.** Generate an asset from an image, or texture
  an existing mesh with matching camera and material observations.

Learn more on the [project page](https://supra-3d.github.io/supratex/), or try the
[online demo](https://huggingface.co/spaces/UniLat3D/Supra-Tex).

## Release Notes

- 🔥 **[2026-10-19]** Inference code released for [image-to-3D](docs/end_to_end.md) and [mesh texturing](docs/texturing.md), along with a [training overview](docs/train.md).
- 🎨 **[2026-10-18]** The [SUPRA-Tex Studio](https://supra-3d.github.io/supratex/studio/studio.html) is open for a tour.
- 📄 **[2026-10-16]** [Project page](https://supra-3d.github.io/supratex/) and [paper](https://arxiv.org/abs/2610.22352) released.

## Pretrained Models

Our models are openly available on [Hugging Face](https://huggingface.co/supra3d/supra-tex).

| Model | Parameters | Download |
| --- | --- | --- |
| SUPRA-Tex 3D Completion | 2.1B | [completion](https://huggingface.co/supra3d/supra-tex/tree/main/completion) |
| SUPRA-Tex Material Adapter | 33.6M trainable | [material](https://huggingface.co/supra3d/supra-tex/tree/main/material) |

## Installation

SUPRA-Tex requires **Linux** with an **NVIDIA GPU**, a **CUDA 12.4** toolkit with
`nvcc`, Git, and a C++ compiler. The installer sets up a dedicated Python 3.10
Conda environment with PyTorch 2.6.0 and the native CUDA extensions:

```bash
git clone https://github.com/Supra-3D/SUPRA-Tex.git
cd SUPRA-Tex
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python scripts/install_runtime.py --conda
```

For faster attention, also install
[FlashAttention](https://github.com/Dao-AILab/flash-attention): FlashAttention-2
on Ampere, Ada, or Hopper GPUs, and FlashAttention-3 on Hopper. Without it,
inference uses PyTorch SDPA; once installed, it is selected automatically. See the
[installation guide](docs/installation.md#flashattention-recommended-for-speed)
for the commands.

The full image-to-3D pipeline has been tested on an NVIDIA RTX 3090 (24 GB), where
CPU offload is enabled automatically. See the [installation guide](docs/installation.md)
for `venv` setup and manual installation, and
[GPU compatibility](docs/gpu_compatibility.md) for memory options.

## Quick Start

**1. Prepare the models.** The [SUPRA-Tex models](https://huggingface.co/supra3d/supra-tex)
need no access request, but the image-to-3D pipeline also uses Meta's
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), which
requires access approval. Request access on its model page, log in with an
approved account, then download and assemble all models into `ckpts/ready`:

```bash
hf auth login
python scripts/prepare_models.py
```

Add `--with-rmbg` to prepare [RMBG-2](https://huggingface.co/briaai/RMBG-2.0)
for opaque RGB inputs; images with a foreground alpha channel do not need it.
RMBG-2 also requires access approval. To avoid it, use the public
[BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet), which RMBG-2 is built on
and which loads the same way: download it and pass its directory with
`--rmbg /path/to/BiRefNet` instead of `--with-rmbg`.

**2. Check the runtime and run the bundled example** (the spiny whelk shell in
[`assets/`](assets)):

```bash
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/spiny_whelk_shell.glb
```

**3. Generate a textured asset from your own image:**

```bash
python scripts/run_example.py --input assets/metalwork_candle_jar.png \
  --output outputs/metalwork_candle_jar.glb
```

This example uses the opaque RGB candle jar in [`assets/`](assets), so it needs
the RMBG-2 or BiRefNet model from step 1; replace `--input` with your own image.
The output is a GLB with base color, metallic, and roughness textures. A sibling
work directory keeps intermediate files and a `run.json` receipt.

**Texture an existing mesh.** Given a mesh, its camera, a foreground image, and
reference-view material maps, complete the PBR textures directly. The repository
bundles the project-page pitcher plant mesh with its inputs in
[`assets/examples/pitcher_plant`](assets/examples/pitcher_plant), the default
input of `texturing.py`:

```bash
python texturing.py --pipeline supra3d/supra-tex --output outputs/pitcher_plant_textured.glb
```

Pass `--input-dir` to texture your own mesh; see [texturing](docs/texturing.md)
for the input format.

## Improving Results

The defaults suit most inputs. When a result falls short, these options help;
for run time, see [Inference Speed](#inference-speed).

- **Refine fine detail.** For very small, intricate text and textures, add
  `--refine`. After export, it further refines the visible texture and aligns it
  with the input image. Refinement adds about two minutes on an RTX 3090, so it is
  off by default. In this example, the wiring table on the industrial disconnect
  switch packs a specification table, circuit symbols, and fine caution print into
  one label:

  ```bash
  python scripts/run_example.py --input assets/industrial_disconnect_switch.png \
    --output outputs/wiring_table.glb --refine
  ```

- **Start from a clean input.** Use a sharp image of one object that is fully in
  frame and fills most of it: the image is reduced to 1024 pixels on its longest
  side before cropping to the object, so a small object in a large frame loses
  detail. Prefer an RGBA image with a clean foreground alpha to automatic
  background removal. Only pixels with alpha of at least 254 serve as texture
  observations, so soft or noisy edges leave more to completion.
- **Check the alignment report.** Each run prints an `Observation: silhouette IoU`
  line that compares the geometry with the image foreground, with a warning
  below 0.9. Mismatched regions take their texture from completion rather than
  the image. Add `--geometry-scale-search` to calibrate the mesh scale, or try
  another `--seed` to resample geometry and completion.
- **Handle extra geometry.** Pixal3D geometry is sometimes misaligned with the
  image or carries extra pieces, which then receive texture the image never
  showed. Image-to-3D runs remove disconnected pieces that project mostly
  outside the image foreground before completion (`--no-remove-floaters` keeps
  them; `texturing.py` applies it with `--remove-floaters`). It keeps everything
  the foreground covers, so it does not remove pieces hidden behind the object or
  extra geometry attached to it. For those, try another `--seed`, or clean the
  mesh yourself and texture it with `texturing.py`.
- **Keep more geometric detail.** `--decimation-target` sets the exported face
  count (500,000 by default); raise it for intricate shapes.

## Make It Yours

Single-view inference observes only what the reference image shows; everything
else comes from completion. Dual-view texturing adds a second, generated
observation of the back so that the unseen side also follows an image. It works
in five stages, each a separate command in
[dual-view texturing](docs/dual_view.md):

1. **Front inputs.** The image-to-3D frontend produces the mesh, camera, and
   reference-view material maps, recording which models produced them.
2. **Back normal pack.** The mesh is rendered from a back camera chosen around
   the opposite view, as a normal map and silhouette mask that share the same
   rasterization.
3. **Back image.** An image generator paints a back view that follows the normal
   pack, and a visual review accepts or rejects it. Our implementation runs
   GPT-6 Astra with its built-in image generation tool through the external Codex
   CLI; this is the only stage outside this repository.
4. **Back materials.** The accepted image is cut to the rendered silhouette, and
   the same material adapter predicts its albedo and metallic/roughness maps.
5. **Completion and refinement.** Both views become surface observations, with
   the front taking priority where they overlap, and a single completion run
   fills the rest. Refinement then projects each view into the texture atlas and
   repairs the UV gutters.

The back image is the creative part: it decides what the unseen side looks like.
Stages 1, 2, 4, and 5 run locally, so the pipeline is ready for another back-view
generator. As foundation image models improve, dual-view results can improve
with them. See [dual-view texturing](docs/dual_view.md) for requirements and the
staged commands.

## Documentation

| Guide | Contents |
| --- | --- |
| [Quick start](docs/quickstart.md) | Installation, model preparation, and the first run in detail |
| [Method overview](docs/method.md) | Pipeline stages and where each is implemented |
| [Training overview](docs/train.md) | Data preparation and training workflow |
| [End-to-end inference](docs/end_to_end.md) | All `infer.py` options |
| [Material frontend](docs/frontend.md) | Foreground, camera, and reference-view material maps |
| [Geometry](docs/geometry.md) | Pixal3D geometry generation and optional scale search |
| [Texturing](docs/texturing.md) | `texturing.py` inputs and options |
| [Dual-view texturing](docs/dual_view.md) | Front/back observation fusion with a generated back view; back-view generation uses GPT-6 Astra through the external Codex CLI |
| [Model weights](docs/weights.md) | Model layouts and external dependencies |
| [GPU compatibility](docs/gpu_compatibility.md) | Attention backends, CPU offload, and tested GPUs |

## Inference Speed

Most inference time on modest hardware goes to reading weights and moving them
between CPU and GPU, not to model compute. These settings help:

- **Install FlashAttention.** FlashAttention-2, or FlashAttention-3 on Hopper, is
  faster than the PyTorch SDPA fallback and is selected automatically once
  installed ([commands](docs/installation.md#flashattention-recommended-for-speed)).

- **Keep weights on an SSD.** The geometry and Qwen stages read tens of
  gigabytes of weights per run. On a hard disk, loading dominates both stages; an
  SSD cuts their load time by more than half.
- **Use a GPU with at least 48 GB.** GPUs with up to 24 GiB enable CPU offload
  automatically, which moves the Qwen model block by block during inference.
  Larger GPUs keep models resident, and the Qwen stage becomes much faster.
- **Use two GPUs.** The geometry and Qwen material stages are independent.
  `--parallel-frontend` runs them concurrently on the first two visible GPUs;
  `CUDA_VISIBLE_DEVICES` selects which ones. Both stages then hold their models
  in CPU memory at the same time, so check host RAM when offload is enabled.
- **Store geometry weights in their runtime precision.** The geometry flow
  checkpoints store float32 for blocks that run in bfloat16. Writing cast copies
  once halves the bytes read on every run and gives identical results:

  ```bash
  python scripts/convert_geometry_weights.py
  ```

  The copies (about 8 GB) are saved beside the originals and used automatically.
- **Geometry scale search is optional.** By default, geometry follows Pixal3D
  with a fixed mesh scale. `--geometry-scale-search` enables a bounded scale
  calibration that adds one sampling pass per probe (up to three).
- **Pre-tuned sparse-convolution kernels apply only to H800.** On H800 with
  PyTorch 2.6 (CUDA 12.4), Triton 3.2, and FlexGEMM 1.0, a measured kernel
  configuration is loaded. Other GPUs autotune these kernels on first use.

## Acknowledgments

Our implementation builds on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
and [Pixal3D](https://github.com/TencentARC/Pixal3D). We also thank the authors of
Qwen-Image, FireRed-Image-Edit, MoGe, NAF, CuMesh, FlexGEMM, and nvdiffrast for
making their work available.

## License

Original SUPRA-Tex source code and the 3D completion weights are released
under the [MIT License](LICENSE). The repository also includes third-party code
under MIT, Apache-2.0, and the DINOv3 License;
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists each component and its
terms. **Built with Qwen:** the material adapter uses the Qwen Research License
for non-commercial research and evaluation. External models retain their
respective licenses.

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{yang2026supratex,
  title   = {{SUPRA-Tex}: 3D Generation with Super-Aligned Texture},
  author  = {Yang, Chen and Fang, Jiemin and Meng, Fanpeng and Huang, Zihao
             and Pang, Jichen and Wu, Guanjun and Bao, Zhikuan and Tian, Qi},
  journal = {arXiv preprint arXiv:2610.22352},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.22352}
}
```

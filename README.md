<h1 align="center">SUPRA-Tex: 3D Generation with Super-Aligned Texture</h1>

<p align="center">
  Chen Yang · Jiemin Fang · Fanpeng Meng · Zihao Huang<br>
  Jichen Pang · Guanjun Wu · Zhikuan Bao · Qi Tian
</p>

<p align="center">
  <a href="https://supra-3d.github.io/supratex/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <a href="https://huggingface.co/spaces/UniLat3D/Supra-Tex"><img src="https://img.shields.io/badge/Demo-Hugging_Face-blueviolet" alt="Online demo"></a>
  <a href="https://huggingface.co/supra3d/supra-tex"><img src="https://img.shields.io/badge/Models-Hugging_Face-yellow" alt="Model weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="Source license"></a>
</p>

https://github.com/user-attachments/assets/208ede7e-6db7-4f23-8deb-61d2ac5541f5

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

Learn more in the [method overview](docs/method.md), or try the
[online demo](https://huggingface.co/spaces/UniLat3D/Supra-Tex).

## Installation

Use **Linux** with an **NVIDIA GPU** and **CUDA 12.4**. The setup below creates a
Python 3.10 environment and installs PyTorch 2.6.0 and the required CUDA extensions.
See [GPU compatibility](docs/gpu_compatibility.md) for tested hardware and memory options.

### 1. Clone the repository

```bash
git clone https://github.com/Supra-3D/SUPRA-Tex.git
cd SUPRA-Tex
```

### 2. Create the environment and install dependencies

```bash
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python scripts/install_runtime.py --conda
```

For CUDA setup and manual installation, see the [installation guide](docs/installation.md).

## Pretrained Models

Our checkpoints are available on [Hugging Face](https://huggingface.co/supra3d/supra-tex).

| Model | Parameters | Download |
| --- | --- | --- |
| SUPRA-Tex 3D Completion | 2.1B | [completion](https://huggingface.co/supra3d/supra-tex/tree/main/completion) |
| SUPRA-Tex Material Adapter | 33.6M trainable | [material](https://huggingface.co/supra3d/supra-tex/tree/main/material) |

The image-to-3D pipeline also uses Meta's
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
To download this upstream dependency, request access on its model page and log in:

```bash
hf auth login
```

The inference command below downloads the required models on first use and
reuses them on later runs. Both inference entrypoints use the release's
[`pipeline.json`](pipeline.json) for model configuration.
See [model weights](docs/weights.md) for the complete dependency list and licenses.

## Usage

### Image to 3D

Generate a textured asset from the bundled [example image](assets/pitcher_plant.png):

```bash
python infer.py --pipeline supra3d/supra-tex \
  --input assets/pitcher_plant.png \
  --output outputs/pitcher_plant.glb
```

Replace `--input` with your image and choose a new `--output` path. The command
exports the GLB and saves intermediate material maps in a sibling work directory.
More [example images](assets/README.md) are included in `assets/`.

Images with foreground alpha, including the bundled example, use that alpha
automatically. For images without foreground alpha, the pipeline downloads
[RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) for background removal;
request access to that model before running on these images.

Add `--low-vram` to enable CPU offload. See the
[inference guide](docs/end_to_end.md) for further options, or the
[setup guide](docs/quickstart.md) to download models ahead of time and run offline.

### Texture an existing mesh

Prepare a mesh, its camera, and reference-view material maps as described in
[the texturing guide](docs/texturing.md), then run:

```bash
python texturing.py --pipeline supra3d/supra-tex \
  --input-dir inputs/object \
  --output outputs/textured_object.glb
```

This entrypoint loads the completion model and TRELLIS.2 VAEs. The
[material frontend](docs/frontend.md) can prepare the reference-view material maps.
For an additional back-view observation, see [dual-view texturing](docs/dual_view.md).

## Training

We provide training code for the 3D completion model and the Qwen material adapter.

- **3D completion:** Follow the [training guide](docs/training.md) to prepare
  aligned shape, material, and partial-observation latents and configure a run.
- **Material adapter:** Follow the [material training guide](docs/training.md#material-adapter-training)
  to encode paired-image data and launch the Qwen-Image-2.1 adapter trainer.

## Acknowledgments

Our implementation builds on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
and [Pixal3D](https://github.com/TencentARC/Pixal3D). We also thank the authors of
Qwen-Image, FireRed-Image-Edit, MoGe, NAF, CuMesh, FlexGEMM, and nvdiffrast for
making their work available.

## License

Original SUPRA-Tex source and the 3D completion weights are released under the
[MIT License](LICENSE). **Built with Qwen:** the material adapter uses the
Qwen Research License for non-commercial research and evaluation. Third-party
code and external models retain their respective licenses; see
[third-party notices](LICENSE) and [model licenses](docs/weights.md).

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{yang2026supratex,
  title   = {{SUPRA-Tex}: 3D Generation with Super-Aligned Texture},
  author  = {Yang, Chen and Fang, Jiemin and Meng, Fanpeng and Huang, Zihao
             and Pang, Jichen and Wu, Guanjun and Bao, Zhikuan and Tian, Qi},
  journal = {Tech report},
  year    = {2026}
}
```

<h1 align="center">Supra-Tex: super-aligned 3D texture generation</h1>
<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#models">Models</a> ·
  <a href="docs/method.md">Method</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <b>Trained only on open-source data, Supra-Tex achieves performance<br>
  comparable to or better than commercial systems.</b>
</p>

https://github.com/user-attachments/assets/208ede7e-6db7-4f23-8deb-61d2ac5541f5

<p align="center"><a href="assets/supratex_fighter_typography.mp4">▶ Watch the Supra-Tex demo</a></p>

**Supra-Tex** turns a single RGB image into a textured 3D mesh by combining
geometry generation, material prediction, and surface-aligned texture
completion. The output is a relightable GLB with base color, metallic, and
roughness maps.

## Highlights

- **Trained only on open-source data.** Achieves performance comparable to or
  better than commercial systems.
- **RGB to textured mesh.** Generate geometry and PBR materials from a single
  reference image with the complete image-to-3D pipeline.
- **Surface alignment.** Project reference-view materials onto the mesh and
  encode them in the same 3D latent space used for generation.
- **Observation-guided completion.** Keep observed material latents as fixed
  anchors while the Interleaved Flow Transformer completes the surface.
- **Relightable assets.** Generate PBR materials that respond to new lighting.

See the [method overview](docs/method.md) for the three-stage pipeline.

## Quick Start

Use Linux x86_64, an NVIDIA CUDA GPU, Python 3.10, a CUDA 12.4 toolkit (`nvcc`),
Git and a C++ compiler. All project runtime source is included in this
checkout. The installer creates a separate environment and obtains the
pinned public Python/CUDA dependencies; no sibling project checkout is needed.

### RGB to textured mesh

Run from this source directory. The project models live under `completion/`
and `material/` in [supra3d/supra-tex](https://huggingface.co/supra3d/supra-tex).
The preparation script also obtains the required external base models and
assembles the geometry/DINO and NAF layouts automatically:

```bash
python3.10 scripts/install_runtime.py
source .venv/bin/activate

python scripts/prepare_models.py --repo supra3d/supra-tex
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/pitcher_plant.glb --attention sdpa
```

The default [pitcher plant](assets/pitcher_plant.png) includes foreground alpha.
DINOv3 is a required upstream model with publisher-controlled access; arrange
access in your own Hugging Face account before preparation, or supply an
existing local copy with `--dino`. For an opaque RGB input, prepare with
`--with-rmbg` (RMBG-2 also requires upstream access), then pass
`--input path/to/image.png` to `run_example.py`. The runner detects foreground
alpha and requires RMBG for an opaque image.

Both installation and preparation support `--dry-run`. See the
[source setup guide](docs/quickstart.md) for existing local weights, download
recovery, and what each validation step establishes. The underlying
[infer.py interface](docs/end_to_end.md) remains available for individual paths
and advanced options.

Attention is selected automatically for A100, RTX, and Hopper GPUs. GPUs with
24 GB or less enable CPU offload by default; use `--low-vram` to enable it
explicitly. See [GPU compatibility](docs/gpu_compatibility.md) for the memory
tradeoffs and current validation status.

<details>
<summary><b>Texture an existing mesh</b></summary>

For an existing mesh, prepare its camera and material observations following
[3D texturing](docs/texturing.md), then run:

```bash
python texturing.py \
  --input-dir inputs/object \
  --output outputs/object.glb \
  --pretrained /path/to/completion \
  --trellis-root /path/to/TRELLIS.2-4B
```

</details>

## Models

| Model | Contents | License |
| --- | --- | --- |
| [Supra-Tex 3D Completion](https://huggingface.co/supra3d/supra-tex/tree/main/completion) | Full 2.1B-parameter completion model | [MIT](LICENSE) |
| [Supra-Tex Material Adapter](https://huggingface.co/supra3d/supra-tex/tree/main/material) | 33.6M trainable parameters for Qwen-Image-2.1 | [Qwen Research License](LICENSE) |

See [Weights](docs/weights.md) for the repository layout and required external
models. The material adapter requires
the full Qwen-Image-2.1 base; completion requires the pretrained sparse VAEs.

**Built with Qwen.** The material adapter is released for non-commercial
research and evaluation under the Qwen agreement.

## Documentation

| Task | Guide |
| --- | --- |
| Set up the environment | [Installation](docs/installation.md) |
| Configure A100 or 24 GB GPUs | [GPU compatibility](docs/gpu_compatibility.md) |
| Generate a textured mesh from an image | [RGB-to-textured-mesh inference](docs/end_to_end.md) |
| Texture an existing mesh | [3D texturing](docs/texturing.md) |
| Predict reference-view materials | [Material frontend](docs/frontend.md) |
| Add a generated back observation | [Dual-view inference](docs/dual_view.md) |
| Train the completion model | [Training](docs/training.md) |

## Citation

Temporary placeholder: [GaussianObject](https://github.com/chensjtu/GaussianObject).
The citation for **Supra-Tex: super-aligned 3D texture generation** will replace this entry.

```bibtex
@article{yang2024gaussianobject,
  title   = {GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting},
  author  = {Chen Yang and Sikuang Li and Jiemin Fang and Ruofan Liang and
             Lingxi Xie and Xiaopeng Zhang and Wei Shen and Qi Tian},
  journal = {ACM Transactions on Graphics},
  year    = {2024}
}
```

## License and Acknowledgments

Original Supra-Tex source and the 3D completion weights use the
[MIT License](LICENSE). The material adapter uses the Qwen Research License
listed above. Incorporated third-party code and external models retain their
own terms; see [Third-party notices](LICENSE).

We thank the contributors to TRELLIS/TRELLIS.2, Qwen, FireRed-Image-Edit,
MoGe, and the geometry and rendering libraries used by this project.

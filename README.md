<h1 align="center">SUPRA-Tex: 3D Generation with Super-Aligned Texture</h1>

<p align="center">
  <a href="https://supra-3d.github.io/supratex/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2610.22352"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv paper"></a>
  <a href="https://huggingface.co/spaces/UniLat3D/Supra-Tex"><img src="https://img.shields.io/badge/Demo-Hugging_Face-blueviolet" alt="Online demo"></a>
  <a href="https://huggingface.co/supra3d/supra-tex"><img src="https://img.shields.io/badge/Models-Hugging_Face-yellow" alt="Model weights"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green" alt="Source license"></a>
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

- **[2026-10-16]** Initial release of the [project page](https://supra-3d.github.io/supratex/) and README. Inference code will follow.

## Pretrained Models

Our models are available on [Hugging Face](https://huggingface.co/supra3d/supra-tex).

> **Note:** We are actively working to resolve model deployment issues and will make the model available for use as soon as they are resolved; results from the current version may be inconsistent and do not reflect the final model quality.

| Model | Parameters | Download |
| --- | --- | --- |
| SUPRA-Tex 3D Completion | 2.1B | [completion](https://huggingface.co/supra3d/supra-tex/tree/main/completion) |
| SUPRA-Tex Material Adapter | 33.6M trainable | [material](https://huggingface.co/supra3d/supra-tex/tree/main/material) |

## Inference Speed

Most inference time on modest hardware goes to reading weights and moving them
between CPU and GPU, not to model compute. These settings help:

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
  python scripts/convert_geometry_weights.py --config ckpts/ready/models.json
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

The 3D completion weights are released under the
[MIT License](https://opensource.org/license/mit). Original SUPRA-Tex source
will be released under the same license. **Built with Qwen:** the material
adapter uses the Qwen Research License for non-commercial research and
evaluation. Third-party code and external models retain their respective
licenses.

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

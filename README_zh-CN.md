<h1 align="center">SUPRA-Tex: 3D Generation with Super-Aligned Texture</h1>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://supra-3d.github.io/supratex/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="项目主页"></a>
  <a href="https://arxiv.org/abs/2610.22352"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv 论文"></a>
  <a href="https://huggingface.co/spaces/UniLat3D/Supra-Tex"><img src="https://img.shields.io/badge/Demo-Hugging_Face-blueviolet" alt="在线演示"></a>
  <a href="https://huggingface.co/supra3d/supra-tex"><img src="https://img.shields.io/badge/Models-Hugging_Face-yellow" alt="模型权重"></a>
  <a href="#许可证"><img src="https://img.shields.io/badge/License-MIT%20%2B%20third--party-green" alt="源码许可证"></a>
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

**SUPRA-Tex** 从单张图像生成带纹理的 3D 资产，保留文字、图案和材质边界等细节。
它先预测参考视角下的 PBR 材质，将其编码为与三维表面对齐的观测，再通过
**Interleaved Flow Transformer（交错式流 Transformer）** 补全不可见区域。
最终输出包含基础颜色、金属度和粗糙度贴图的 GLB，可在不同光照下重新渲染。

- **与图像对齐的纹理。** 利用可见表面的观测引导三维材质生成，保留输入图像中的细节。
- **完整的 PBR 资产。** 为不可见表面生成材质，并导出支持不同光照条件下渲染的贴图。
- **图像生成 3D 与已有网格贴纹理。** 从图像生成资产，或结合匹配的相机和材质观测，为已有网格生成纹理。

更多介绍请参阅[项目主页](https://supra-3d.github.io/supratex/)，也可以体验
[在线演示](https://huggingface.co/spaces/UniLat3D/Supra-Tex)。

## 更新日志

- **[2026-10-19]** 发布推理代码，支持从图像生成 3D
  （[`infer.py`](infer.py)）和为已有网格生成纹理（[`texturing.py`](texturing.py)）。
- **[2026-10-19]** 发布[训练参考文档](docs/train.md)，可以参考该文档训练自己的模型。
- **[2026-10-18]** [SUPRA-Tex 工作室](https://supra-3d.github.io/supratex/studio/studio.html)
  已开放，欢迎参观。
- **[2026-10-16]** 首次发布[项目主页](https://supra-3d.github.io/supratex/)、README 和
  [arXiv 论文](https://arxiv.org/abs/2610.22352)。

## 预训练模型

模型已发布至 [Hugging Face](https://huggingface.co/supra3d/supra-tex)。

> **说明：** 我们正在解决模型部署问题。在问题解决前，模型暂时需要申请访问权限：请先在[模型页面](https://huggingface.co/supra3d/supra-tex)申请访问，并通过 `hf auth login` 登录后再下载。问题解决后，模型将直接开放，无需申请访问权限。当前版本的生成结果可能不稳定，不代表最终模型质量。

| 模型 | 参数量 | 下载 |
| --- | --- | --- |
| SUPRA-Tex 三维材质补全模型 | 2.1B | [completion](https://huggingface.co/supra3d/supra-tex/tree/main/completion) |
| SUPRA-Tex 材质适配器 | 33.6M 可训练参数 | [material](https://huggingface.co/supra3d/supra-tex/tree/main/material) |

## 安装

SUPRA-Tex 需要 **Linux**、**NVIDIA GPU**、包含 `nvcc` 的 **CUDA 12.4** 工具包、
Git 和 C++ 编译器。安装脚本会在专用的 Python 3.10 Conda 环境中配置
PyTorch 2.6.0 和原生 CUDA 扩展：

```bash
git clone https://github.com/Supra-3D/SUPRA-Tex.git
cd SUPRA-Tex
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python scripts/install_runtime.py --conda
```

如需更快的 attention，请另外安装 [FlashAttention](https://github.com/Dao-AILab/flash-attention)：
Ampere、Ada 或 Hopper GPU 安装 FlashAttention-2，Hopper GPU 可安装 FlashAttention-3。
未安装时推理使用 PyTorch SDPA；安装后会自动选用。安装命令见
[安装指南](docs/installation.md#flashattention-recommended-for-speed)。

完整的图像生成 3D 流程已在 NVIDIA RTX 3090（24 GB）上测试，该显卡会自动启用 CPU 卸载。
`venv` 环境配置和手动安装请参阅[安装指南](docs/installation.md)；
显存相关选项请参阅 [GPU 兼容性说明](docs/gpu_compatibility.md)。

## 快速开始

**1. 准备模型。** [SUPRA-Tex 模型](https://huggingface.co/supra3d/supra-tex)
暂时需要申请访问权限（见[上方说明](#预训练模型)），图像生成 3D 流程还使用 Meta 的
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)。
请在这两个模型页面申请访问权限，使用已获批准的账号登录，然后将所有模型下载并整理到 `ckpts/ready`：

```bash
hf auth login
python scripts/prepare_models.py
```

对于不透明的 RGB 输入，添加 `--with-rmbg` 以准备
[RMBG-2](https://huggingface.co/briaai/RMBG-2.0)；带前景 alpha 通道的图像无需此模型。
RMBG-2 同样需要申请访问权限。也可以使用公开的
[BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)：RMBG-2 基于该模型构建，二者的加载方式相同。
下载 BiRefNet 后，通过 `--rmbg /path/to/BiRefNet` 指定其目录，替代 `--with-rmbg`。

**2. 检查运行环境并运行仓库附带的示例**（[`assets/`](assets) 中的多刺海螺）：

```bash
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/spiny_whelk_shell.glb
```

**3. 使用自己的图像生成带纹理的 3D 资产：**

```bash
python scripts/run_example.py --input assets/metalwork_candle_jar.png \
  --output outputs/metalwork_candle_jar.glb
```

该示例使用 [`assets/`](assets) 中不带 alpha 的 RGB 金属香薰蜡烛罐图像，因此需要第 1 步中的
RMBG-2 或 BiRefNet 模型；将 `--input` 换成自己的图像即可。
输出为包含基础颜色、金属度和粗糙度贴图的 GLB。
输出文件旁的工作目录会保留中间文件和 `run.json` 运行记录。

**为已有网格生成纹理。** 给定网格、对应相机、前景图像和参考视角的材质图，即可直接补全 PBR 纹理。
仓库在 [`assets/examples/pitcher_plant`](assets/examples/pitcher_plant) 中附带了项目主页上的
猪笼草网格及其输入文件，也是 `texturing.py` 的默认输入：

```bash
python texturing.py --pipeline supra3d/supra-tex --output outputs/pitcher_plant_textured.glb
```

通过 `--input-dir` 为自己的网格指定输入目录；输入格式请参阅[纹理生成指南](docs/texturing.md)。

## 提升生成效果

默认设置适用于大多数输入。如果结果不够理想，可以尝试以下方法；运行速度相关的设置见[推理速度](#推理速度)。

- **进一步细化纹理。** 对于非常细小、精密的文字和纹理，可以添加 `--refine`：
  导出后，它会进一步细化可见区域的纹理，并使其与输入图像对齐。
  细化会增加耗时（RTX 3090 上约两分钟），因此默认关闭。
  下面的示例中，工业隔离开关的接线表在一张标签上集中了规格表、电路符号和细小的警示文字：

  ```bash
  python scripts/run_example.py --input assets/industrial_disconnect_switch.png \
    --output outputs/wiring_table.glb --refine
  ```

- **使用干净的输入图像。** 使用清晰的单个物体图像，物体完整入镜且占据画面的大部分：
  图像会先把长边缩小到 1024 像素再按物体裁剪，物体在大画面中占比过小会损失细节。
  相比自动去背景，优先使用带干净前景 alpha 的 RGBA 图像。只有 alpha 不低于 254 的像素
  会作为纹理观测，边缘柔化或有噪声时，更多区域需要依赖补全。
- **查看对齐报告。** 每次运行都会打印一行 `Observation: silhouette IoU`，比较几何与图像前景的
  轮廓，低于 0.9 时会给出警告。不匹配的区域由补全生成纹理，而不是取自图像。
  可以添加 `--geometry-scale-search` 校准网格尺度，或换一个 `--seed` 重新采样几何和补全。
- **处理多余几何。** Pixal3D 生成的几何有时会和图像对不齐，或者多出额外的部件，
  这些部件会被赋予图像中并不存在的纹理。从图像生成 3D 时，默认会在补全前删除与主体不相连、
  且投影大部分落在图像前景之外的部件（`--no-remove-floaters` 可保留；`texturing.py`
  需加 `--remove-floaters` 才会启用）。前景覆盖范围内的几何都会保留，因此被物体挡住的碎片、
  以及连在主体上的多余几何不会被删除。遇到这两类情况，可以换一个 `--seed` 重新生成，
  或者自行清理网格后用 `texturing.py` 上色。
- **保留更多几何细节。** `--decimation-target` 设置导出的面数（默认 500,000），
  复杂形状可以适当调高。

## Make It Yours

单视角推理只能观测到参考图中可见的部分，其余区域都由补全生成。dual-view
额外生成一张背面观测，让看不到的一侧也能参照图像生成纹理。它分为五个阶段，
每个阶段都是 [dual-view 文档](docs/dual_view.md)中的一条单独命令：

1. **正面输入。** 图像生成 3D 的前端生成网格、相机和参考视角的材质图，并记录所用的模型。
2. **背面法线包。** 从相反方向附近选定的背面相机渲染网格，得到共用同一次光栅化的
   法线图和轮廓 mask。
3. **背面图像。** 图像生成器按法线包绘制背面图，再由视觉审核决定是否采用。
   我们的实现通过外部的 Codex CLI 调用 GPT-6 Astra 及其内置的图像生成工具，这是唯一不在本仓库内的阶段。
4. **背面材质。** 采用的图像按渲染轮廓裁切，再用同一个材质 adapter 预测 albedo 和金属度/粗糙度图。
5. **补全与细化。** 两个视角都转为表面观测，重叠处以正面为准，只运行一次补全来生成其余区域；
   随后细化把每个视角投影进纹理图集，并修复 UV 边缘留白。

背面图像是最有创造性的部分，它决定了看不到的一侧是什么样子。第 1、2、4、5
阶段都在本地运行，因此可以接入其他背面生成器。随着图像基础模型的发展，dual-view 的效果
也会随之提升。环境要求和分步命令见 [dual-view 文档](docs/dual_view.md)。

## 文档

| 指南 | 内容 |
| --- | --- |
| [快速开始](docs/quickstart.md) | 安装、模型准备和首次运行的详细步骤 |
| [方法概述](docs/method.md) | 流程各阶段及对应的代码实现 |
| [训练流程概述](docs/train.md) | 数据准备与训练流程；训练代码暂未开源 |
| [端到端推理](docs/end_to_end.md) | `infer.py` 的全部选项 |
| [材质前端](docs/frontend.md) | 前景处理、相机和参考视角材质图 |
| [几何生成](docs/geometry.md) | Pixal3D 几何生成和可选的尺度搜索 |
| [纹理生成](docs/texturing.md) | `texturing.py` 的输入格式与选项 |
| [双视角纹理生成](docs/dual_view.md) | 结合生成背面视角的前后视角观测融合；背面视角通过外部 Codex CLI 调用 GPT-6 Astra 生成 |
| [模型权重](docs/weights.md) | 模型目录结构和外部依赖 |
| [GPU 兼容性](docs/gpu_compatibility.md) | Attention 后端、CPU 卸载和已测试的 GPU |

## 推理速度

在硬件资源有限的情况下，推理耗时主要来自读取权重，以及在 CPU 和 GPU 之间搬运权重，
模型计算本身并非主要耗时。以下设置有助于加速：

- **安装 FlashAttention。** FlashAttention-2（Hopper 上可用 FlashAttention-3）比默认回退的
  PyTorch SDPA 更快，安装后会自动选用（[安装命令](docs/installation.md#flashattention-recommended-for-speed)）。

- **将权重放在 SSD 上。** 几何和 Qwen 阶段每次运行都需要读取数十 GB 的权重。
  使用机械硬盘时，加载会占据这两个阶段的大部分耗时；使用 SSD 可将加载时间缩短一半以上。
- **使用至少 48 GB 显存的 GPU。** 显存不超过 24 GiB 的 GPU 会自动启用 CPU 卸载，
  在推理时逐块搬运 Qwen 模型。更大显存的 GPU 可以让模型常驻显存，显著加快 Qwen 阶段。
- **使用两张 GPU。** 几何生成和 Qwen 材质预测相互独立。
  `--parallel-frontend` 会在前两张可见 GPU 上并行运行这两个阶段；
  可通过 `CUDA_VISIBLE_DEVICES` 指定 GPU。此时两个阶段会同时在 CPU 内存中保留模型，
  因此启用卸载时需留意主机内存容量。
- **按运行时精度存储几何权重。** 几何 flow 模型中以 bfloat16 运行的模块，
  其 checkpoint 仍以 float32 存储。预先保存一次转换后的副本，可以将每次运行的读取量减半，
  同时保持输出结果一致：

  ```bash
  python scripts/convert_geometry_weights.py
  ```

  转换后的副本约为 8 GB，保存在原文件旁，并会被自动使用。
- **几何尺度搜索为可选项。** 默认按照 Pixal3D 的方式使用固定网格尺度。
  `--geometry-scale-search` 会启用有次数上限的尺度校准，每次探测增加一轮采样，最多三轮。
- **预调优的稀疏卷积内核配置仅适用于 H800。** 在 H800、PyTorch 2.6（CUDA 12.4）、
  Triton 3.2 和 FlexGEMM 1.0 的组合下，会加载经过实测的内核配置。
  其他 GPU 会在首次使用时自动调优这些内核。

## 致谢

本项目基于 [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
和 [Pixal3D](https://github.com/TencentARC/Pixal3D) 构建。
同时感谢 Qwen-Image、FireRed-Image-Edit、MoGe、NAF、CuMesh、FlexGEMM
和 nvdiffrast 的作者公开其工作。

## 许可证

SUPRA-Tex 原创源码和三维材质补全模型权重采用 [MIT 许可证](LICENSE)。
仓库还包含采用 MIT、Apache-2.0 和 DINOv3 License 的第三方代码；
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 列出了各组件及其条款。
**Built with Qwen：** 材质适配器采用 Qwen Research License，供非商业研究和评估使用。
外部模型仍遵循各自的许可证。

## 引用

如果本项目对您的研究有帮助，请引用：

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

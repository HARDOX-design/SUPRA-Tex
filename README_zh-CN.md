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
  <sup>1</sup>华为 &nbsp;&nbsp;
  <sup>2</sup>上海交通大学 &nbsp;&nbsp;
  <sup>3</sup>华中科技大学<br>
  <sup>*</sup>共同第一作者 &nbsp;&nbsp;
  <sup>†</sup>项目负责人 &nbsp;&nbsp;
  <sup>✉</sup>通讯作者
</p>

https://github.com/user-attachments/assets/ecc9e9ec-e91f-4eba-927d-86322796459a

**SUPRA-Tex** 只需一张图像，就能生成带纹理的 3D 资产，并保留文字、图案、材质分界等细节。
它先在参考视角下预测 PBR 材质，再把这些材质作为观测贴合到三维表面上，最后由
**Interleaved Flow Transformer** 补全看不到的区域。输出的 GLB 包含基础色、金属度和粗糙度贴图，
可以在任意光照下重新渲染。

- **纹理与图像对齐**：以可见表面的观测引导三维材质生成，忠实保留输入图像中的细节。
- **完整的 PBR 资产**：看不到的表面同样会生成材质，导出的贴图支持重新打光。
- **既能生成，也能上色**：可以从图像直接生成 3D 资产，也可以为已有网格生成纹理（需提供对应的相机和材质观测）。

更多介绍见[项目主页](https://supra-3d.github.io/supratex/)，也欢迎体验[在线演示](https://huggingface.co/spaces/UniLat3D/Supra-Tex)。

## 更新日志

- 🔥 **[2026-10-19]** 发布推理代码，支持[图像生成 3D](docs/end_to_end.md)和[网格纹理生成](docs/texturing.md)，并公开[训练流程概述](docs/train.md)。
- 🎨 **[2026-10-18]** [SUPRA-Tex 工作室](https://supra-3d.github.io/supratex/studio/studio.html)开放参观。
- 📄 **[2026-10-16]** 发布[项目主页](https://supra-3d.github.io/supratex/)和[论文](https://arxiv.org/abs/2610.22352)。

## 预训练模型

模型已在 [Hugging Face](https://huggingface.co/supra3d/supra-tex) 公开，可直接下载。

| 模型 | 参数量 | 下载 |
| --- | --- | --- |
| SUPRA-Tex 三维材质补全模型 | 2.1B | [completion](https://huggingface.co/supra3d/supra-tex/tree/main/completion) |
| SUPRA-Tex 材质适配器 | 33.6M（可训练参数） | [material](https://huggingface.co/supra3d/supra-tex/tree/main/material) |

## 安装

运行环境需要 **Linux**、**NVIDIA GPU**、带 `nvcc` 的 **CUDA 12.4** 工具包，以及 Git 和 C++ 编译器。
安装脚本会在独立的 Python 3.10 Conda 环境中安装 PyTorch 2.6.0 和所需的 CUDA 扩展：

```bash
git clone https://github.com/Supra-3D/SUPRA-Tex.git
cd SUPRA-Tex
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python scripts/install_runtime.py --conda
```

建议再装上 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 来加速注意力计算：
Ampere、Ada、Hopper 架构的显卡可用 FlashAttention-2，Hopper 还可以用 FlashAttention-3。
没装时会退回 PyTorch SDPA，装好后无需任何设置即可自动启用。安装命令见
[安装指南](docs/installation.md#flashattention-recommended-for-speed)。

完整的图像生成 3D 流程已在 NVIDIA RTX 3090（24 GB）上测试通过，这类显卡会自动开启 CPU 卸载。
`venv` 环境和手动安装方法见[安装指南](docs/installation.md)，显存相关选项见
[GPU 兼容性说明](docs/gpu_compatibility.md)。

## 快速开始

**1. 准备模型。** [SUPRA-Tex 模型](https://huggingface.co/supra3d/supra-tex)无需申请即可下载，
但图像生成 3D 还会用到 Meta 的 [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)，
需要先在它的模型页面申请访问。申请通过后，用该账号登录，再下载并整理全部模型到 `ckpts/ready`：

```bash
hf auth login
python scripts/prepare_models.py
```

默认情况下，模型会下载到 `ckpts/ready` 中，这个目录完全自包含。如果 Hugging Face 缓存里已经有其中一部分模型，
可以加上 `--cache-dir ~/.cache/huggingface/hub` 直接复用，不必重新下载。注意要指向 hub 缓存目录
（`HF_HUB_CACHE`），而不是 `HF_HOME`；这样 `ckpts/ready` 会链接到该缓存，因此不要删除它。

如果输入是不带 alpha 通道的 RGB 图像，需要加上 `--with-rmbg` 来准备抠图模型
[RMBG-2](https://huggingface.co/briaai/RMBG-2.0)；带前景 alpha 的图像不需要。RMBG-2 同样需要申请访问。
不想申请的话，可以改用公开的 [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)：RMBG-2 正是基于它构建的，
两者的加载方式相同。下载 BiRefNet 后，用 `--rmbg /path/to/BiRefNet` 指定目录，代替 `--with-rmbg`。

**2. 检查环境，运行内置示例**（[`assets/`](assets) 中的多刺海螺）：

```bash
python scripts/check_runtime.py
python scripts/run_example.py --output outputs/spiny_whelk_shell.glb
```

**3. 用你自己的图像生成带纹理的 3D 资产：**

```bash
python scripts/run_example.py --input assets/metalwork_candle_jar.png \
  --output outputs/metalwork_candle_jar.glb
```

这个示例用的是 [`assets/`](assets) 中的金属香薰蜡烛罐，它是不带 alpha 的 RGB 图像，因此需要第 1 步中的
RMBG-2 或 BiRefNet。换成你自己的图像，只需修改 `--input`。
输出是一个包含基础色、金属度和粗糙度贴图的 GLB，旁边的工作目录会保留中间文件和运行记录 `run.json`。

**为已有网格生成纹理。** 如果已经有网格，再提供对应的相机、前景图像和参考视角的材质图，就能直接补全 PBR 纹理。
仓库在 [`assets/examples/pitcher_plant`](assets/examples/pitcher_plant) 中附带了项目主页上的猪笼草网格及其输入，
这也是 `texturing.py` 的默认输入：

```bash
python texturing.py --pipeline supra3d/supra-tex --output outputs/pitcher_plant_textured.glb
```

要为自己的网格生成纹理，用 `--input-dir` 指定输入目录；输入格式见[纹理生成指南](docs/texturing.md)。

## 提升生成效果

大多数输入用默认设置即可。如果效果不够理想，可以试试下面的方法；想提升运行速度，请看[推理速度](#推理速度)。

- **细化纹理**：遇到非常细小、精密的文字和纹理时，可以加上 `--refine`。它会在导出后进一步细化可见区域的纹理，
  并与输入图像对齐。由于会多花一些时间（RTX 3090 上约两分钟），默认不开启。
  下面的例子里，工业隔离开关的接线表在一张标签上挤满了规格表、电路符号和细小的警示文字：

  ```bash
  python scripts/run_example.py --input assets/industrial_disconnect_switch.png \
    --output outputs/wiring_table.glb --refine
  ```

- **用干净的输入图像**：尽量使用清晰、只有一个物体的图像，物体完整入镜并占据画面的大部分。
  图像会先把长边缩到 1024 像素再按物体裁剪，物体在画面中太小就会丢失细节。
  如果手上有前景 alpha 干净的 RGBA 图像，优先用它，而不是依赖自动抠图：只有 alpha 不低于 254 的像素
  才会被当作纹理观测，边缘发虚或有噪点时，更多区域就只能靠补全。
- **看一眼对齐报告**：每次运行都会打印一行 `Observation: silhouette IoU`，比较几何与图像前景的轮廓，
  低于 0.9 时会给出警告。对不上的区域，纹理来自补全而不是图像。这时可以加上 `--geometry-scale-search`
  校准网格尺度，或者换一个 `--seed` 重新采样几何和补全。
- **处理多余的几何**：Pixal3D 生成的几何有时会和图像对不齐，或者多出一些零碎部件，
  这些部件会被贴上图像里根本不存在的纹理。从图像生成 3D 时，默认会在补全前删掉与主体不相连、
  且投影大部分落在图像前景之外的部件（加 `--no-remove-floaters` 可保留；`texturing.py` 默认不删，
  需要时加 `--remove-floaters`）。前景范围内的几何都会保留，所以藏在物体背后的碎片、
  以及和主体连在一起的多余几何不会被删掉。遇到这两种情况，可以换一个 `--seed` 重新生成，
  或者自己清理网格后再用 `texturing.py` 生成纹理。
- **保留更多几何细节**：`--decimation-target` 控制导出的面数（默认 500,000），形状复杂时可以调高。

## 随心定制

单视角推理只能看到参考图里露出来的部分，其余区域全靠补全。dual-view 会额外生成一张背面图作为观测，
让看不到的那一侧也有图可依。整个过程分五步，每一步都对应 [dual-view 文档](docs/dual_view.md)里的一条命令：

1. **正面输入**：图像生成 3D 的前端生成网格、相机和参考视角的材质图，并记录所用的模型。
2. **背面法线图**：从大致相反方向选一个背面相机渲染网格，得到同一次光栅化的法线图和轮廓 mask。
3. **背面图像**：图像生成模型参照法线图画出背面，再经过视觉审核决定是否采用。
   我们通过外部的 Codex CLI 调用 GPT-6 Astra 及其内置的图像生成工具来完成这一步，这也是唯一不在本仓库里的环节。
4. **背面材质**：按渲染出的轮廓裁切采用的背面图，再用同一个材质适配器预测 albedo 和金属度、粗糙度。
5. **补全与细化**：正反两面都转成表面观测，重叠处以正面为准，只需一次补全就能生成其余区域；
   之后细化步骤把两个视角分别投影到纹理图集上，并修补 UV 岛边缘的留白。

背面图是最有发挥空间的一环，它决定了看不到的那一侧长什么样。第 1、2、4、5 步都在本地运行，
所以完全可以换成你喜欢的背面生成器。图像基础模型越强，dual-view 的效果也会越好。
环境要求和分步命令见 [dual-view 文档](docs/dual_view.md)。

## 文档

| 指南 | 内容 |
| --- | --- |
| [快速开始](docs/quickstart.md) | 安装、准备模型和首次运行的详细步骤 |
| [方法概述](docs/method.md) | 各个阶段的原理及对应代码 |
| [训练流程概述](docs/train.md) | 数据准备与训练流程 |
| [端到端推理](docs/end_to_end.md) | `infer.py` 的全部选项 |
| [材质前端](docs/frontend.md) | 前景处理、相机估计和参考视角材质图 |
| [几何生成](docs/geometry.md) | Pixal3D 几何生成，以及可选的尺度搜索 |
| [纹理生成](docs/texturing.md) | `texturing.py` 的输入格式与选项 |
| [双视角纹理生成](docs/dual_view.md) | 结合生成的背面图，融合正反两面的观测；背面图通过外部 Codex CLI 调用 GPT-6 Astra 生成 |
| [模型权重](docs/weights.md) | 模型目录结构和外部依赖 |
| [GPU 兼容性](docs/gpu_compatibility.md) | 注意力后端、CPU 卸载和已测试的显卡 |

## 推理速度

在硬件条件一般的机器上，大部分时间花在读取权重以及在 CPU 和 GPU 之间来回搬运权重上，模型计算反而不是瓶颈。
下面这些做法可以明显提速：

- **安装 FlashAttention**：FlashAttention-2（Hopper 上可用 FlashAttention-3）比默认的 PyTorch SDPA 更快，
  装好后会自动启用（[安装命令](docs/installation.md#flashattention-recommended-for-speed)）。
- **把权重放在 SSD 上**：几何和 Qwen 阶段每次运行都要读取几十 GB 的权重。放在机械硬盘上时，
  这两个阶段的大部分时间都耗在加载上；换成 SSD，加载时间能缩短一半以上。
- **用 48 GB 以上显存的显卡**：显存不超过 24 GiB 的显卡会自动开启 CPU 卸载，推理时把 Qwen 模型逐块搬到显卡上。
  显存更大时模型可以常驻显存，Qwen 阶段会快很多。
- **用两张显卡**：几何生成和 Qwen 材质预测互不依赖，`--parallel-frontend` 会在前两张可见的显卡上并行运行它们，
  可以用 `CUDA_VISIBLE_DEVICES` 指定显卡。这时两个阶段的模型会同时占用内存，开启 CPU 卸载时请留意主机内存是否够用。
- **按运行精度保存几何权重**：几何 flow 模型中以 bfloat16 运行的模块，checkpoint 里仍是 float32。
  提前转换并保存一份副本，每次运行读取的数据量就能减半，输出结果完全一致：

  ```bash
  python scripts/convert_geometry_weights.py
  ```

  转换后的副本约 8 GB，保存在原文件旁边，之后会自动使用。
- **几何尺度搜索是可选的**：默认沿用 Pixal3D 的固定网格尺度。加上 `--geometry-scale-search` 会做有限次数的
  尺度校准，每次探测多一轮采样，最多三轮。
- **预调优的稀疏卷积内核只针对 H800**：在 H800 + PyTorch 2.6（CUDA 12.4）+ Triton 3.2 + FlexGEMM 1.0
  的环境下，会直接加载实测好的内核配置；其他显卡会在第一次运行时自动调优。

## 致谢

本项目基于 [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 和 [Pixal3D](https://github.com/TencentARC/Pixal3D) 构建。
也感谢 Qwen-Image、FireRed-Image-Edit、MoGe、NAF、CuMesh、FlexGEMM 和 nvdiffrast 的作者开源他们的工作。

## 许可证

SUPRA-Tex 的原创源码和三维材质补全模型权重采用 [MIT 许可证](LICENSE)。
仓库中还包含采用 MIT、Apache-2.0 和 DINOv3 License 的第三方代码，各组件及其条款见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
**Built with Qwen：** 材质适配器采用 Qwen Research License，仅限非商业的研究和评估用途。
外部模型遵循各自的许可证。

## 引用

如果 SUPRA-Tex 对你的研究有帮助，欢迎引用：

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

# Runtime dependencies and model acquisition

Follow the [source setup guide](../../docs/quickstart.md). Run the commands below
from the repository root; bundled runtime code is under `deployment/runtime/`.

This source release contains inference code, with no model weights.
Use `scripts/prepare_models.py` to download or bind the project and external
models, then run `scripts/check_runtime.py` to check the installation and files.
The inventory below documents the model sources and prepared directory layout.
DINOv3 and RMBG-2 require their publishers' access approval or an existing
authorized local copy. Foreground RGBA input can avoid RMBG-2; DINOv3 remains
required.

## Runtime installation (only for missing software)

For automatic installation, use `python scripts/install_runtime.py --conda` in
the active Python 3.10 environment. For manual installation, first install the
Python packages and NATTEN from the [installation guide](../../docs/installation.md),
then build the extensions below. These commands require Linux x86_64,
PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA 12.4 with `nvcc`, a C++
compiler, and Ninja. Use a fresh `external-src/` directory for the manual clones.

FlashAttention is optional. Its build must match the GPU and Torch/CUDA
versions; do not install a build that replaces the pinned Torch version.
Qwen material prediction uses Torch attention. Geometry and completion use the
selected attention backend and FlexGEMM; see
[GPU compatibility](../../docs/gpu_compatibility.md) for selection and overrides.

For A100 or RTX 3090/4090, optionally install FA2 for the selected Torch/CUDA ABI:

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation --no-deps
python -c "import flash_attn"
```

`--attention sdpa` requires neither FA2 nor FA3. It still needs the other CUDA
extensions below. The default Qwen material path supports CPU offload through
`--low-vram`; see [GPU compatibility](../../docs/gpu_compatibility.md).

FlashAttention-3 is a separate package from FlashAttention-2. For H100/H800
(CUDA >= 12.3), install its FA3 interface from the `hopper/` build at upstream
revision `72c7ba484d33ca43711897de44e5bb8e0589a7f4`:

```bash
mkdir -p external-src
git clone https://github.com/Dao-AILab/flash-attention.git external-src/flash-attention
git -C external-src/flash-attention checkout 72c7ba484d33ca43711897de44e5bb8e0589a7f4
git -C external-src/flash-attention submodule update --init --recursive
python -m pip install external-src/flash-attention/hopper --no-build-isolation --no-deps
python -c "import flash_attn_interface"
```

Source instructions are retained in the official [FlashAttention README](https://github.com/Dao-AILab/flash-attention/blob/72c7ba484d33ca43711897de44e5bb8e0589a7f4/README.md).
On a non-Hopper device, use `--attention auto`, `--attention flash_attn`, or
`--attention sdpa`. Clear incompatible `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND`
environment overrides when using `auto`; FA3 is not an A100 backend.

The other CUDA extensions are source dependencies. These commands pin the
upstream revisions and build on the destination machine; this source release does not
redistribute installed binary extensions:

```bash
mkdir -p external-src
git clone https://github.com/JeffreyXiang/CuMesh.git external-src/CuMesh
git -C external-src/CuMesh checkout ef40f3ae4612ea3ec934b4151e019dc95c5d276b
git -C external-src/CuMesh submodule update --init --recursive
NVCC_APPEND_FLAGS="--extended-lambda" \
  python -m pip install external-src/CuMesh --no-build-isolation --no-deps
git clone https://github.com/JeffreyXiang/FlexGEMM.git external-src/FlexGEMM
git -C external-src/FlexGEMM checkout b1965ad0ae4bc8488d13b5280afb54439ce27359
git -C external-src/FlexGEMM submodule update --init --recursive
python -m pip install external-src/FlexGEMM --no-build-isolation --no-deps
git clone https://github.com/NVlabs/nvdiffrast.git external-src/nvdiffrast
git -C external-src/nvdiffrast checkout 253ac4fcea7de5f396371124af597e6cc957bfae
python -m pip install external-src/nvdiffrast --no-build-isolation --no-deps
git clone https://github.com/microsoft/TRELLIS.2.git external-src/TRELLIS.2
git -C external-src/TRELLIS.2 checkout 5565d240c4a494caaf9ece7a554542b76ffa36d3
git -C external-src/TRELLIS.2 submodule update --init --recursive
NVCC_APPEND_FLAGS="--extended-lambda -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__" \
  python -m pip install external-src/TRELLIS.2/o-voxel --no-build-isolation --no-deps
```

The CuMesh/cubvh revision requires CUDA extended lambdas. The o-voxel build
also needs the half-operator flags shown above. These flags are appended by
the source installer without modifying the upstream source.

After building, run `python -m pip check` and
`python scripts/check_runtime.py --imports-only`. Once the model files are
prepared, run `python scripts/check_runtime.py` to check them as well.

Primary installation references: [TRELLIS.2 source](https://github.com/microsoft/TRELLIS.2),
[CuMesh](https://github.com/JeffreyXiang/CuMesh),
[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM),
[nvdiffrast](https://github.com/NVlabs/nvdiffrast), and
[NAF](https://github.com/valeoai/NAF).

## External model inventory

The source configuration [`configs/pipeline.json`](../../configs/pipeline.json) records the external repository
revisions and required file patterns. The table shows the default prepared
paths relative to `ckpts/ready/`; `--output-dir` changes that root, and local
overrides can point outside it. `models.json` records the paths actually used.

| Model | Source | Default prepared path |
|---|---|---|
| Native single-view geometry | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | `external/geometry/`, bound under `geometry/` |
| Geometry DINOv3 conditioner | [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `external/dino/`, linked at `geometry/dino/` |
| Qwen-Image-2.1 base | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | `external/qwen-base/` |
| MoGe-2 camera estimator | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | `external/moge2/model.pt` |
| RGB foreground segmentation | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | `external/rmbg2/` when requested |
| Sparse shape/PBR VAEs | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | `external/vae/ckpts/` |
| NAF feature upsampler | [Official NAF release](https://github.com/valeoai/NAF/releases/download/model/naf_release.pth) | `external/naf/naf_release.pth`, linked under `torch/hub/checkpoints/` |

Keep the upstream license and model-card files alongside downloaded models.
Qwen Image 2.1 has a Qwen Research License; DINOv3 and RMBG access is controlled
by their publishers. Accept upstream terms and supply authorized access through
your own local tooling.

Native geometry uses the generated `geometry/pipeline.json` and these five
stems, each with both `.json` and `.safetensors`, under `geometry/ckpts/`:

```text
ss_flow_img_dit_1_3B_64_bf16
ss_dec_conv3d_16l8_fp16
slat_flow_img2shape_dit_1_3B_512_bf16
slat_flow_img2shape_dit_1_3B_1024_bf16
shape_dec_next_dc_f16c32_fp16
```

Do not substitute the similarly named TRELLIS.2 geometry flow weights: the
Pixal3D networks use projection conditioning. The supplied native adapter
loads only these five geometry networks. Texture model entries retained in
the geometry pipeline are not loaded. The DINO folder needs `config.json`
and `model.safetensors` or a safetensors index with all its shards. The bundled
feature extractors normalize images directly. Retain the upstream license files.

The Qwen foundation directory needs **all files** under `processor/`, `vae/`,
`text_encoder/`, and `transformer/`, including every safetensors shard named
by each index. The selected project checkpoint contains the trainable PBR
state, not these foundation components. Retain the 2.1 model architecture;
Qwen-Image, Qwen-Image-Edit, and FireRed bases are not replacements.

RMBG needs `config.json`, `model.safetensors`, `BiRefNet_config.py`,
`birefnet.py`, and `preprocessor_config.json`. Its config loads its local
model implementation with Transformers remote-code support. Those Python
files must be acquired with the model. RGB input requires it; a meaningful
RGBA foreground alpha can instead use `--use-input-alpha`.

The TRELLIS.2 directory must contain these four VAE stems with both `.json`
and `.safetensors` under `ckpts/`:

```text
shape_enc_next_dc_f16c32_fp16
tex_enc_next_dc_f16c32_fp16
shape_dec_next_dc_f16c32_fp16
tex_dec_next_dc_f16c32_fp16
```

They stay in the upstream layout; preparation does not copy them into the
completion model directory. The `--vae` local override also accepts a directory
containing these pairs directly.

NAF source is bundled at `deployment/runtime/naf/`. The model preparation script links
this source and the existing `naf_release.pth` into its generated Torch Hub
cache. The launcher sets `TORCH_HOME` to that cache automatically. Both source and weights
must already exist before geometry launches: its adapter explicitly rejects
an incomplete NAF cache and replaces the upstream Torch Hub call with local
source loading. `--local-files-only` alone does not construct this cache.

## Bundled source provenance and scope

`deployment/runtime/SOURCE_PROVENANCE.json` describes the bundled components and their
upstream sources. The geometry runtime includes the `pixal3d/` Python package,
its model loaders, and conditioner utilities. These runtime trees contain
source code; obtain model weights separately.

The Qwen runtime includes `diffsynth/` plus `examples/qwen_image_21/pbr/`.
The PBR entrypoint uses `infer.py`, `inference_utils.py`, and
`inference_offload.py`. A generic current DiffSynth installation is not an
equivalent replacement. Use the bundled files together. `modelscope==1.34.0`
is required by a loader import, even though inference uses explicit local
models. Install it in the active environment; `--qwen-deps` can optionally
select an existing dependency overlay.

The bundled MoGe 2.0.0 package records upstream commit
`07444410f1e33f402353b99d6ccd26bd31e469e8`.
The matching utils3d 0.0.2 Python source is supplied separately.
Install both bundled source packages to avoid a different utils3d API.
MoGe's local DINOv2 architecture source is included; loading the local MoGe
checkpoint does not require separate DINOv2 weights.

On a new machine, check native-extension imports and complete an image-to-GLB
run before relying on the installation. See [GPU compatibility](../../docs/gpu_compatibility.md)
for validated hardware and remaining limitations.

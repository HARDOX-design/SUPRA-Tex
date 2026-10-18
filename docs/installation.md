# Installation

For a new installation, use the [source setup guide](quickstart.md):

```bash
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python scripts/install_runtime.py --conda
```

The installer uses fixed public Python/CUDA dependencies and the bundled
project runtimes, then checks actual imports. `--dry-run` prints its plan.
It installs into the active Conda environment; model weights are prepared with
`scripts/prepare_models.py`. The commands below document manual installation.

The supported installation uses Linux x86_64, Python 3.10, PyTorch 2.6.0+cu124,
torchvision 0.21.0+cu124, and the CUDA 12.4 toolkit with `nvcc`. Run the following
commands from the repository root:

```bash
conda create -n supra-tex python=3.10 -y
conda activate supra-tex
python -m pip install pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 packaging==25.0
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r deployment/runtime/requirements-python.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install --no-build-isolation --no-deps ./deployment/runtime/moge ./deployment/runtime/utils3d
python -m pip install --no-index --no-deps \
  'https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.5/natten-0.17.5%2Btorch260cu124-cp310-cp310-linux_x86_64.whl'
```

The [Python requirements](../deployment/runtime/requirements-python.txt) pin the
runtime packages, including Qwen's `peft` and `modelscope` imports. PyTorch and
compiled CUDA extensions are installed separately. The `frontend` extra alone
does not install all Qwen runtime dependencies. Use the bundled MoGe and utils3d
packages to preserve the APIs required by the frontend.

The native geometry/material encoder-decoder and GLB export use:

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2): pretrained shape and PBR
  VAE weights and the `o-voxel` extension from that repository.
- [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM): sparse convolution kernels.
- [CuMesh](https://github.com/JeffreyXiang/CuMesh): GPU mesh processing.
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast): differentiable rasterization.
- [utils3d](https://github.com/EasternJournalist/utils3d): installed from the
  bundled `deployment/runtime/utils3d` source by the command above.

Build the CUDA extensions using the pinned commands in the
[dependency reference](../deployment/docs/DEPENDENCIES.md#runtime-installation-only-for-missing-software)
from the same environment and repository root. Then check the installation:

```bash
python -m pip check
python scripts/check_runtime.py --imports-only
```

## FlashAttention (recommended for speed)

The installer does not install FlashAttention by default, so inference falls back
to PyTorch SDPA. [FlashAttention](https://github.com/Dao-AILab/flash-attention)
is faster: FlashAttention-2 supports Ampere, Ada, and Hopper GPUs (A100,
RTX 30/40 series, H100/H800), and FlashAttention-3 is faster still on Hopper.
Install them into the same environment:

```bash
# FlashAttention-2 (or rerun the installer with --flash-attn)
pip install flash-attn==2.8.3 --no-build-isolation

# FlashAttention-3, Hopper only; builds from source
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install
```

Inference defaults to `--attention auto`, which picks the fastest installed
backend: FlashAttention-3 on Hopper, then FlashAttention-2, then SDPA. No flag is
needed after installation. Use `--attention sdpa` if an installed FlashAttention
build does not match your GPU or CUDA/PyTorch version.
Selected backends are printed at startup. Explicit backend requests are
validated and do not silently fall back.
The prepared-model runner and `infer.py --pipeline` clear inherited
`ATTN_BACKEND` and `SPARSE_ATTN_BACKEND` overrides when using `auto`. Direct
component commands respect them; see [GPU compatibility](gpu_compatibility.md)
for override precedence.

## Runtime options

GPUs with at most 24 GiB enable CPU offload automatically; `--low-vram` and
`--no-low-vram` override this policy. See [GPU compatibility](gpu_compatibility.md)
for the supported paths, memory tradeoffs, and hardware validation status.

The default Qwen-Image material backend uses torch attention and the
DiffSynth runtime bundled at `deployment/runtime/qwen`, selected automatically.
Its subprocess uses the active Python environment and the dependencies installed
above. Packages installed only with `pip --user` are excluded in the subprocess.

To use another compatible runtime, pass `--qwen-runtime /path/to/qwen-runtime`.
An optional `--qwen-deps /path/to/dependencies` adds an existing dependency
overlay; no overlay directory is needed for a standard installation. Preserve
the whole runtime, including its package and inference helpers. See
[Frontend](frontend.md) and [Weights](weights.md) for the base model and adapter
formats.

The selectable FireRed material backend uses the bundled model source. Its
material Transformer uses FA3 on supported Hopper setups and native Torch
attention otherwise; its Qwen text encoder selects FA2 when available or
SDPA. FireRed does not have the Qwen material backend's CPU-offload path.

After installation, [prepare the model weights](quickstart.md#2-prepare-the-models)
and run an example. The installer checks software dependencies without
downloading weights or running full inference.

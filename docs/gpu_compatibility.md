# GPU compatibility and low-VRAM inference

The inference entrypoints select an attention backend for the visible GPU.
The default Qwen image-to-textured-mesh pipeline also supports CPU offload.
Full-weight single-view inference has been validated with PyTorch SDPA on
H800 80 GB and RTX 3090 24 GB. The RTX 3090 run used CPU offload and the
default 4096-pixel texture atlas; see the measured configuration below.

## Device policy

| Device | Default attention, without environment overrides | Default CPU offload |
| --- | --- | --- |
| A100 40 / 80 GB | FA2 if available, otherwise PyTorch SDPA | Off |
| H100 / H800 80 GB | FA3 if available, then FA2, then SDPA | Off |
| RTX 3090 / 4090 24 GB | FA2 if available, otherwise SDPA | On |
| Other CUDA GPUs | Recognized supported FlashAttention backend, otherwise SDPA | On when total memory is at most 24 GiB |

FA3 requires Hopper hardware. The native CUDA extensions must also support
the selected GPU; attention selection alone does not establish support for
every device. Qwen material prediction keeps its native Torch attention.
The selected attention backend and offload policy are printed at startup;
the complete image-to-mesh run records them in `run.json`.

## Usage

Keep the model and input arguments from your existing command and add:

```bash
# Automatic device policy (the default).
--attention auto

# Explicit CPU offload, including on a larger GPU.
--low-vram

# No FlashAttention installation required; other CUDA extensions still apply.
--attention sdpa --low-vram

# Keep model weights resident when sufficient VRAM is available.
--no-low-vram
```

These options are accepted by `infer.py`, `texturing.py`, and the material
frontend CLI. `scripts/check_runtime.py` checks the installed runtime and model
files; it does not run full inference or measure whether the models fit in VRAM.

An explicit `--attention` value other than `auto` overrides both
`ATTN_BACKEND` and `SPARSE_ATTN_BACKEND`. With `auto`, `scripts/run_example.py`
and `infer.py --pipeline` clear these inherited values and select by hardware.
Direct `infer.py` without `--pipeline`, `texturing.py`, and the material
frontend respect them. For those direct commands, clear old FA3 overrides
before using an A100 or RTX GPU:

```bash
unset ATTN_BACKEND SPARSE_ATTN_BACKEND
```

`SUPRATEX_LOW_VRAM=1` or `0` overrides automatic memory selection; the explicit
`--low-vram` / `--no-low-vram` CLI options take precedence. Automatic selection
uses total device capacity, not currently free memory. It does not retry a
failed run or change quality settings after an out-of-memory error. Use one
request at a time when evaluating a device's capacity.

## What CPU offload changes

- **Geometry:** the native geometry pipeline already stages its models by
  inference phase.
- **Qwen materials:** base components load on CPU. VAE encoding, text encoding,
  DiT prediction, and VAE decoding run separately; language and DiT blocks
  move onto the GPU only for their forward pass, including attached adapters.
- **3D completion:** shape/material encoders run sequentially; the completion
  model releases GPU memory before decoding. Decoder layers are staged
  individually. Required sparse topology maps survive CPU handoffs, while
  recomputable caches and the optional observation cache are released.
- **Texture export:** attribute volumes upload when baking begins, and
  interpolation queries run in bounded chunks.

Weights, precision policy, input resolution, sampling parameters, and output
texture resolution are preserved. CPU offload does not quantize the model or
reduce the default 4096-pixel texture atlas. Different attention kernels can
introduce normal floating-point differences; identical output across GPU
families or backends is not promised.

System RAM must hold the offloaded models plus loading and inference workspaces.
Synchronous CPU-to-GPU transfers increase latency. Sparse activation size
depends on the input surface, and rendering/mesh processing also uses VRAM;
weight offload alone does not bound every stage's peak. The measurements below
describe the tested example, rather than a minimum system-RAM requirement.

The optional FireRed material backend has compatible attention selection, but
does not implement this CPU-offload path. Image-to-mesh inference rejects
FireRed with low-VRAM mode; use the default `qwen-image` backend for that mode.

## Validation status

The recorded full-pipeline validation on 2026-10-14 used H800 80 GB,
Python 3.10, PyTorch 2.6.0/cu124, SDPA, and CPU offload disabled. Two bundled
images, `pitcher_plant.png` (RGBA) and `metalwork_candle_jar.png` (opaque RGB),
completed image-to-GLB inference with 4096-square textures. The RGB example
used RMBG-2. This run used local copies of the release and upstream weights;
it did not test anonymous model downloads.

On 2026-10-16, one RTX 3090 24 GB completed the bundled RGBA example
`pitcher_plant.png` using Python 3.10, PyTorch 2.6.0/cu124, CUDA toolkit 12.4,
and `--attention sdpa --low-vram`. Full weights and the default 4096-square
textures were preserved. The first run, including kernel compilation, took
665 seconds. Sampling every five seconds recorded a peak device-memory use
of 12,593 MiB and a peak process-tree RSS sum of 32.4 GiB; shared pages can be
counted more than once in the RSS sum. The exported GLB passed mesh, UV, and
embedded-texture checks. This RTX 3090 run covered single-view RGBA input.

On 2026-10-19, RTX 3090 GPUs with the same software and automatic CPU offload
ran the release defaults, including the `denoiser_step0030000.safetensors`
completion checkpoint: image-to-GLB inference on three project-page cases (a
metal candle tin, multimeter dial markings, and the disconnect-switch wiring
table), `--refine` on the wiring table, and `texturing.py` on the bundled
pitcher plant example. Dual-view inference was not part of these runs.

When evaluating another device or input, check the complete pipeline and its
output, GPU memory, system RAM, and elapsed time. Qwen's `REPORT.json` records
only the material stage's peak allocated/reserved GPU memory.

See the [FlashAttention documentation](https://github.com/Dao-AILab/flash-attention)
for attention hardware and build requirements.

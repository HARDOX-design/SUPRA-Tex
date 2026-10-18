# Qwen-Image-2.1 material inference

This runtime loads the SUPRA-Tex material adapter with the complete
Qwen-Image-2.1 base model to predict albedo, metallic, and roughness from an
RGB or RGBA image. The adapter includes attention LoRA weights and learned
modality embeddings.

For the complete SUPRA-Tex handoff, use the
[material frontend](../../../../../../docs/frontend.md). It prepares the
foreground and camera, then converts the native material outputs into the
canonical albedo/MRA maps used by 3D completion.

## Models and runtime

The base model directory must contain the complete `transformer/`,
`text_encoder/`, `vae/`, and `processor/` components, including their configs,
safetensors shards and indexes, tokenizer, and processor files. The adapter
requires `adapter_model.safetensors` and its sibling `adapter_config.json`.
See [model weights](../../../../../../docs/weights.md) for download instructions.

Keep `infer.py`, `inference_utils.py`, and `inference_offload.py` together with
this runtime's `diffsynth/` package. Install the dependencies using the
[repository installation guide](../../../../../../docs/installation.md).
The inference process loads local model files and requires a CUDA device.

## Run material inference directly

From the SUPRA-Tex repository root:

```bash
PYTHONPATH="$PWD/deployment/runtime/qwen${PYTHONPATH:+:$PYTHONPATH}" \
python deployment/runtime/qwen/examples/qwen_image_21/pbr/infer.py \
  --checkpoint /path/to/material/adapter_model.safetensors \
  --model-root /path/to/Qwen-Image-2.1 \
  --input /path/to/condition_rgba.png \
  --output-dir outputs/material
```

Use a new output directory. Add `--low-vram` for CPU offload, or
`--restore-input-size` to restore the original image dimensions after prediction.
The released adapter uses a 1024-square canvas, seed 42, Torch attention, and
one endpoint-prediction step. Input colors should be straight RGB with coverage
in the alpha channel; do not premultiply RGB by alpha.

The output directory contains `albedo.png`, `mra.png`, and `REPORT.json`.
Albedo is sRGB. In the native MRA output, red is linear metallic, green is
linear roughness, blue is a constant-one placeholder, and alpha preserves input
coverage. The SUPRA-Tex frontend replaces the blue placeholder with coverage
when producing its canonical RGB MRA map. Input alpha describes coverage;
these outputs do not estimate material transparency.

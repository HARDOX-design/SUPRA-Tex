# Local web app

`app.py` runs the image-to-3D pipeline through a local Gradio interface on one
NVIDIA GPU. It uses the same prepared models as `scripts/run_example.py`.

## Setup and launch

Complete [installation](installation.md) and [model preparation](quickstart.md)
first, then run these commands from the repository root in the inference
environment:

```bash
pip install -e '.[app]'
python app.py
```

The `app` extra installs Gradio 6.10.0. Open
[http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. The default model
configuration is `ckpts/ready/models.json`.

Upload an RGB image or an RGBA image with foreground alpha. Opaque RGB inputs
need RMBG-2 or BiRefNet included in the prepared configuration; see
[foreground model preparation](quickstart.md). Choose a seed (default `42`),
then generate, preview the GLB, and download it.

The **Depth pre-filter** checkbox is enabled by default. It applies the image/geometry
depth pre-filter before material completion. **Refine** is optional and off by
default; enabling it adds reference-view texture refinement after completion.
See [the filter description](frontend.md) and [refinement](end_to_end.md) for
details.

The app processes one generation at a time. Low-memory mode is enabled by
default for a single RTX 3090 with 24 GB VRAM. Each run creates a new directory
under `outputs/gradio`; results stay on the machine running the app. The app
does not automatically remove old run directories, so clean them up manually
when they are no longer needed.

## Launch options

- `--config PATH`: use another prepared `models.json`.
- `--host ADDRESS`: bind address; default `127.0.0.1`.
- `--port PORT`: listening port; default `7860`.
- `--output-dir PATH`: parent directory for new runs; default `outputs/gradio`.
- `--no-filter`: start with the Depth pre-filter checkbox off.
- `--no-low-vram`: disable the app's default low-memory mode.

For example:

```bash
python app.py --config /path/to/models.json --port 7861 \
  --output-dir /path/to/results
```

For command-line inference, both `infer.py` and `scripts/run_example.py` accept
`--no-filter` to disable the depth pre-filter; `--no-image-depth` is an alias:

```bash
python scripts/run_example.py --input /path/to/image.png \
  --output outputs/example.glb --no-filter
```

## Remote GPU access

To use a remote GPU through SSH, start the app on that machine with its default
loopback binding:

```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```

On your own computer, forward the port:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-host
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) on your own computer.
`CUDA_VISIBLE_DEVICES=0` selects one physical GPU for the app.

For direct access over a reachable network, bind the server to all interfaces:

```bash
CUDA_VISIBLE_DEVICES=0 python app.py --host 0.0.0.0 --port 7860
```

Then open `http://<gpu-host-address>:7860`. Model execution and output files
remain on the GPU host.

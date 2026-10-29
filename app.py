#!/usr/bin/env python3
"""Serve SUPRA-Tex locally with Gradio and the prepared-model inference pipeline."""
from __future__ import annotations

import argparse
import atexit
from contextlib import closing
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

from scripts.run_example import ROOT, build_command, read_config


class LocalRunner:
    """One GPU job at a time, with a fresh subprocess and directory per request."""

    def __init__(self, config: Path, output_dir: Path, *, low_vram: bool = True):
        if not config.is_file():
            raise FileNotFoundError(f"Prepared models are missing: {config}. Run scripts/prepare_models.py first.")
        self.config = read_config(config)
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.low_vram = low_vram
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def close(self) -> None:
        """Stop only this app's inference process and its frontend workers."""
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()

    @staticmethod
    def _receipt(work: Path) -> dict:
        path = work / "run.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    def generate(self, image_path: str, seed: int, use_filter: bool = True, refine: bool = False):
        """Yield progress and completed files; closing the iterator cancels the job."""
        if not image_path or not Path(image_path).is_file():
            raise ValueError("Upload a reference image first.")
        if isinstance(seed, bool) or not isinstance(seed, (int, float)) or not math.isfinite(seed) or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")
        # Gradio can release its queue slot before a cancelled iterator has
        # finished closing. Wait for our GPU owner instead of dropping that job.
        while not self._lock.acquire(timeout=.5):
            yield {"status": "Waiting for the previous task to finish…"}
        try:
            from PIL import Image, ImageOps

            job = Path(tempfile.mkdtemp(prefix="run-", dir=self.output_dir))
            source, output, work, log = job / "input.png", job / "mesh.glb", job / "work", job / "inference.log"
            # Own a copy of the upload; Gradio can expire its temporary file.
            with Image.open(image_path) as uploaded:
                oriented = ImageOps.exif_transpose(uploaded)
                mode = "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
                oriented.convert(mode).save(source)
            args = SimpleNamespace(input=source, output=output, attention=None,
                                   low_vram=self.low_vram, use_input_alpha=False)
            extra = ["--work-dir", str(work), "--seed", str(int(seed)),
                     "--texture-size", "4096", "--texture-format", "png"]
            if not use_filter:
                extra.append("--no-filter")
            if refine:
                extra.append("--refine")
            command, environment = build_command(args, extra, self.config)
            request = {"seed": int(seed), "image_depth_filter": bool(use_filter),
                       "refine": bool(refine), "low_vram": self.low_vram}
            (job / "request.json").write_text(json.dumps(request, indent=2) + "\n")
            started = time.monotonic()
            yield {"status": "Preparing your image…"}
            with log.open("x", encoding="utf-8") as stream:
                self._process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
                while True:
                    try:
                        returncode = self._process.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        stage = self._receipt(work).get("stage")
                        phase = "Generating the PBR model" if stage == "completion" else "Preparing image, geometry and materials"
                        elapsed = int(time.monotonic() - started)
                        yield {"status": f"{phase}… {elapsed // 60}m {elapsed % 60:02d}s"}
            receipt = self._receipt(work)
            if returncode or receipt.get("status") != "complete" or not output.is_file():
                yield {"status": "Generation failed. See the run log for details.", "log": str(log)}
                raise RuntimeError(receipt.get("error") or f"Inference failed (exit code {returncode}). See {log}")
            with output.open("rb") as mesh:
                if mesh.read(4) != b"glTF":
                    raise RuntimeError(f"Inference did not produce a GLB: {output}")
            elapsed = int(time.monotonic() - started)
            yield {"status": f"Ready · seed {int(seed)} · {elapsed // 60}m {elapsed % 60:02d}s",
                   "model": str(output), "log": str(log),
                   "foreground": str(work / "input/condition_rgba.png"),
                   "albedo": str(work / "input/albedo.png")}
        finally:
            self.close()
            self._process = None
            self._lock.release()


def build_ui(runner: LocalRunner, *, use_filter: bool = True):
    import gradio as gr

    with gr.Blocks(title="SUPRA-Tex · Local", analytics_enabled=False) as demo:
        gr.Markdown("# SUPRA-Tex\n3D Generation with Super-Aligned Texture")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                image = gr.Image(label="Reference image", type="filepath", image_mode="RGBA", format="png", height=400)
                with gr.Accordion("Generation settings", open=True):
                    seed = gr.Number(value=42, label="Seed", precision=0, minimum=0, maximum=2**32 - 1)
                    depth_filter = gr.Checkbox(value=use_filter, label="Depth pre-filter",
                                               info="Reduce texture bleed near depth boundaries.")
                    refine = gr.Checkbox(value=False, label="Refine visible textures",
                                         info="Sharpen reference-view details after generation; takes extra time.")
                with gr.Row():
                    generate = gr.Button("Generate 3D", variant="primary")
                    stop = gr.Button("Stop")
                examples = [str(ROOT / "assets" / name) for name in
                            ("spiny_whelk_shell.png", "metalwork_candle_jar.png", "cinema_signage.png")
                            if (ROOT / "assets" / name).is_file()]
                if examples:
                    gr.Examples(examples=examples, inputs=image, label="Examples")
            with gr.Column(scale=2, min_width=400):
                model = gr.Model3D(label="PBR model", interactive=False, height=540,
                                   clear_color=(.08, .08, .08, 1))
                download = gr.DownloadButton("Download GLB", value=None, interactive=False)
                status = gr.Markdown("Upload an image to begin. Jobs run one at a time.")
        with gr.Accordion("Reference materials and run log", open=False):
            with gr.Row():
                foreground = gr.Image(label="Foreground", interactive=False)
                albedo = gr.Image(label="Albedo", interactive=False)
            log = gr.File(label="Run log", interactive=False)

        def generate_mesh(image_path, seed_value, filter_enabled, refine_enabled):
            try:
                with closing(runner.generate(image_path, seed_value, filter_enabled, refine_enabled)) as updates:
                    for update in updates:
                        result = update.get("model")
                        yield (result, gr.update(value=result, interactive=result is not None), update["status"],
                               update.get("foreground"), update.get("albedo"), update.get("log"))
            except (OSError, ValueError, RuntimeError) as error:
                raise gr.Error(str(error)) from error

        event = generate.click(generate_mesh, [image, seed, depth_filter, refine],
                               [model, download, status, foreground, albedo, log],
                               api_name="generate", concurrency_limit=1, concurrency_id="gpu", trigger_mode="once")
        stop.click(lambda: "Stopped.", outputs=status, cancels=[event], queue=False, api_visibility="private")
    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "ckpts/ready/models.json", help="prepared models.json")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 for LAN access")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gradio", help="persistent per-run files")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True,
                        help="CPU offload for a single 24 GB GPU (default: on)")
    parser.add_argument("--no-filter", "--no-image-depth", action="store_true",
                        help="start with the depth pre-filter unchecked (default: enabled)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> None:
    args = parse_args()
    try:
        import gradio as gr
    except ImportError as error:
        raise SystemExit('Install the web UI first: pip install -e ".[app]"') from error
    try:
        runner = LocalRunner(args.config.expanduser(), args.output_dir, low_vram=args.low_vram)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    atexit.register(runner.close)
    def shutdown(signum, _frame):
        runner.close()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, shutdown)
    demo = build_ui(runner, use_filter=not args.no_filter)
    # Keep the Space's dark/gold presentation without its cloud account services.
    theme = gr.themes.Base(font=["Arial", "sans-serif"]).set(
        body_background_fill="#101010", body_background_fill_dark="#101010",
        body_text_color="#f4f0e7", body_text_color_dark="#f4f0e7",
        block_background_fill="#191919", block_background_fill_dark="#191919",
        input_background_fill="#121212", input_background_fill_dark="#121212",
        button_primary_background_fill="#f2cd70", button_primary_background_fill_dark="#f2cd70",
        button_primary_text_color="#19150e", button_primary_text_color_dark="#19150e")
    demo.launch(server_name=args.host, server_port=args.port, share=False, ssr_mode=False,
                theme=theme, css=".gradio-container {max-width: 1200px !important; margin: auto;}",
                allowed_paths=[str(runner.output_dir), str(ROOT / "assets")])


if __name__ == "__main__":
    main()

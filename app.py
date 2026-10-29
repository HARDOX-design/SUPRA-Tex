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


UI_CSS = """
.gradio-container {
    width: 100% !important; max-width: 1320px !important;
    margin: 0 auto !important; padding: 24px 28px 32px !important;
}
.gradio-container .app {padding: 0 !important;}
#app-header {margin-bottom: 12px;}
#app-header .html-container {padding: 0;}
.app-brand {display: flex; align-items: center; gap: 14px; padding-bottom: 22px; border-bottom: 1px solid #e5e8ed;}
.brand-icon {display: grid; place-items: center; width: 42px; height: 42px; border: 1px solid #dce2e9; border-radius: 12px; color: #34465c;}
.app-brand h1 {margin: 0; font-size: 25px; line-height: 1.2; font-weight: 700; letter-spacing: -.7px; color: #172536;}
.app-brand p {margin: 5px 0 0; font-size: 13px; color: #6b7786;}
.local-label {margin-left: auto; font-size: 12px; color: #6b7786;}
#workspace {gap: 28px !important; align-items: flex-start;}
#reference-panel {flex: 0 0 320px !important; min-width: 280px !important; gap: 16px;}
#preview-panel {flex: 1 1 0 !important; min-width: 0 !important; gap: 14px;}
.section-heading h3 {margin: 0 !important; font-size: 16px !important; font-weight: 600; color: #253449;}
#reference-heading {min-height: 34px; display: flex; align-items: center;}
#reference-image {border: 1px dashed #c6cfd9 !important; border-radius: 12px; background: #fafbfc; overflow: hidden;}
#examples {padding: 0; border: 0; background: transparent;}
#examples .gallery {gap: 8px;}
#examples button {border-radius: 8px;}
#examples .gallery-item .container {width: 64px; max-width: 64px; height: 64px;}
#generate-actions {gap: 8px;}
#generate-button {min-height: 44px; font-weight: 600;}
#stop-button {min-height: 44px;}
#generation-options {border-radius: 10px;}
#preview-toolbar {align-items: center; min-height: 34px; gap: 12px;}
#preview-toolbar .section-heading {flex: 1;}
#download-button {min-height: 34px;}
#model-preview {border: 1px solid #e2e7ed !important; border-radius: 12px; overflow: hidden; background: #f4f6f8;}
#model-preview .empty {display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px; background: #f4f6f8;}
#model-preview .empty svg {width: 42px; height: 42px; color: #a0adbb;}
#model-preview .empty::after {content: "Your 3D preview\\aGenerate a model to rotate and explore it"; white-space: pre-line; text-align: center; font-size: 14px; line-height: 1.9; color: #738295;}
#generation-status {padding: 11px 14px; border: 1px solid #e5e9ef; border-radius: 8px; background: #fafbfc; color: #536276; min-height: 43px;}
#generation-status p {margin: 0; font-size: 13px;}
#reference-details {border-radius: 10px;}
@media (max-width: 860px) {
    .gradio-container {padding: 18px 18px 28px !important;}
    #workspace {flex-direction: column; gap: 24px !important;}
    #reference-panel, #preview-panel {flex: 1 1 auto !important; width: 100% !important; min-width: 0 !important;}
    #model-preview {height: 420px !important;}
}
@media (max-width: 480px) {
    .local-label {display: none;}
    .app-brand h1 {font-size: 23px;}
    .app-brand p {font-size: 12px;}
}
"""


def ui_theme():
    import gradio as gr

    theme = gr.themes.Base(primary_hue="slate", secondary_hue="slate", neutral_hue="slate",
                           font=["Arial", "sans-serif"], radius_size="lg").set(
        body_background_fill="#ffffff", body_text_color="#253449",
        body_text_color_subdued="#6b7786", background_fill_primary="#ffffff",
        background_fill_secondary="#f4f6f8", block_background_fill="#ffffff",
        block_border_color="#e2e7ed", block_label_background_fill="#ffffff",
        block_label_text_color="#536276", block_title_text_color="#253449",
        block_info_text_color="#738295", input_background_fill="#f8fafc",
        input_border_color="#dce2e9", input_border_color_focus="#71849b",
        input_placeholder_color="#8793a1", button_primary_background_fill="#263b52",
        button_primary_background_fill_hover="#1c2f43", button_primary_border_color="#263b52",
        button_primary_text_color="#ffffff", button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f1f4f8", button_secondary_border_color="#dce2e9",
        button_secondary_text_color="#536276", checkbox_background_color_selected="#263b52",
        checkbox_border_color_selected="#263b52", block_shadow="none",
        input_shadow="none", button_primary_shadow="none", button_secondary_shadow="none")
    # Keep this light workspace consistent when the browser prefers dark mode.
    styles = theme.to_dict()["theme"]
    return theme.set(**{name: styles[name[:-5]] for name in styles
                        if name.endswith("_dark") and name[:-5] in styles})


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

    with gr.Blocks(title="SUPRA-Tex · Local", analytics_enabled=False, fill_width=True) as demo:
        gr.HTML('''<header class="app-brand">
            <span class="brand-icon" aria-hidden="true"><svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M12 2 21 7v10l-9 5-9-5V7Z M3 7l9 5 9-5 M12 12v10 M7.5 4.5l9 5"/></svg></span>
            <div><h1>SUPRA-Tex</h1><p>3D Generation with Super-Aligned Texture</p></div>
            <span class="local-label">Local workspace</span>
        </header>''', elem_id="app-header")
        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=1, min_width=280, elem_id="reference-panel"):
                gr.Markdown("### Reference image", elem_id="reference-heading", elem_classes="section-heading")
                image = gr.Image(label="Reference image", show_label=False, type="filepath", image_mode="RGBA",
                                 format="png", sources=["upload", "clipboard"], height=280, elem_id="reference-image")
                examples = [str(ROOT / "assets" / name) for name in
                            ("pumpkin_wizard.png", "mechanical_beetle.png", "sourdough_bread.png",
                             "rocket_backpack.png", "peony_teapot.png", "oolong_tea_jar.png",
                             "spiny_whelk_shell.png", "metalwork_candle_jar.png")
                            if (ROOT / "assets" / name).is_file()]
                if examples:
                    gr.Examples(examples=examples, inputs=image, label="Try an example",
                                examples_per_page=8, elem_id="examples")
                with gr.Row(elem_id="generate-actions"):
                    generate = gr.Button("Generate 3D", variant="primary", scale=4, min_width=180, elem_id="generate-button")
                    stop = gr.Button("Stop", scale=1, min_width=64, elem_id="stop-button")
                with gr.Accordion("Generation options", open=False, elem_id="generation-options"):
                    seed = gr.Number(value=42, label="Seed", precision=0, minimum=0, maximum=2**32 - 1)
                    depth_filter = gr.Checkbox(value=use_filter, label="Depth pre-filter",
                                               info="Reduce texture bleed near depth boundaries.")
                    refine = gr.Checkbox(value=False, label="Refine visible textures",
                                         info="Sharpen reference-view details after generation; takes extra time.")
            with gr.Column(scale=2, min_width=360, elem_id="preview-panel"):
                with gr.Row(elem_id="preview-toolbar"):
                    gr.Markdown("### 3D preview", elem_classes="section-heading")
                    download = gr.DownloadButton("Download GLB", value=None, interactive=False,
                                                 size="sm", scale=0, min_width=132, elem_id="download-button")
                model = gr.Model3D(label="PBR model", show_label=False, interactive=False, height=460,
                                   clear_color=(.957, .965, .973, 1), elem_id="model-preview")
                status = gr.Markdown("Upload an image, then select **Generate 3D**.", elem_id="generation-status")
                with gr.Accordion("Reference materials and run log", open=False, elem_id="reference-details"):
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
    demo.launch(server_name=args.host, server_port=args.port, share=False, ssr_mode=False,
                theme=ui_theme(), css=UI_CSS, footer_links=["api"],
                allowed_paths=[str(runner.output_dir), str(ROOT / "assets")])


if __name__ == "__main__":
    main()

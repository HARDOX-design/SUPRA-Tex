#!/usr/bin/env python3
"""Check the source release's real runtime imports and prepared model files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "deployment/runtime"


def runtime_environment(config: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               FLEX_GEMM_USE_AUTOTUNE_CACHE="0", FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0",
               PYTHONPATH=os.pathsep.join(str(p) for p in
                   (ROOT, RUNTIME / "moge", RUNTIME / "utils3d")))
    if config and config.get("torch_home"):
        env["TORCH_HOME"] = config["torch_home"]
    return env


def run_import_check(code: str, *, config: dict | None = None) -> str:
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env=runtime_environment(config), capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-5000:].strip())
    return result.stdout.strip()


def import_checks(attention: str) -> list[tuple[str, str]]:
    bootstrap = ("from supratex.inference.runtime import configure_inference_runtime; "
                 f"configure_inference_runtime(attention={attention!r}, material_backend='qwen-image'); ")
    return [
        ("Python, PyTorch and CUDA", """
import sys, torch, torchvision
if sys.version_info[:2] != (3, 10):
    raise RuntimeError('The supported installation uses Python 3.10')
if torch.__version__.split('+')[0] != '2.6.0' or torch.version.cuda != '12.4':
    raise RuntimeError('Expected PyTorch 2.6.0 with CUDA 12.4; got ' + str((torch.__version__, torch.version.cuda)))
if torchvision.__version__.split('+')[0] != '0.21.0':
    raise RuntimeError('Expected torchvision 0.21.0')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; check the NVIDIA driver and GPU access')
x = torch.tensor([1., 2.], device='cuda')
assert (x * x).sum().item() == 5.
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
"""),
        ("Native extensions, MoGe and utils3d", """
import torch, natten, natten.functional
import o_voxel, flex_gemm, cumesh, nvdiffrast.torch
import moge.model.v2, utils3d
if not (hasattr(natten.functional, 'na2d') or
        all(hasattr(natten.functional, n) for n in ('na2d_av', 'na2d_qk'))):
    raise RuntimeError('NATTEN does not expose the NAF attention API')
assert hasattr(utils3d, 'pt'), 'Bundled utils3d API is required'
print('Native modules and bundled camera utilities imported')
"""),
        ("Geometry and NAF implementation", bootstrap +
         f"import sys; sys.path.insert(0, {str(RUNTIME / 'geometry')!r}); "
         f"sys.path.insert(0, {str(RUNTIME / 'naf')!r}); "
         "from pixal3d.pipelines import Pixal3DImageTo3DPipeline; "
         "from pixal3d.modules.projected_image_feature_extractor import DinoV3ProjFeatureExtractor; "
         "import hubconf; print('Geometry, projection conditioner and NAF imported')"),
        ("Qwen models, PBR pipeline and inference helpers",
         f"import sys; sys.path.insert(0, {str(RUNTIME / 'qwen')!r}); "
         f"sys.path.insert(0, {str(RUNTIME / 'qwen/examples/qwen_image_21/pbr')!r}); "
         "import peft; "
         "import diffsynth.models.qwen_image_21_dit; "
         "import diffsynth.models.qwen_image_21_text_encoder; "
         "import diffsynth.models.qwen_image_21_vae; "
         "import diffsynth.pipelines.qwen_image_21; "
         "import diffsynth.pipelines.qwen_image_21_pbr; "
         "from inference_utils import build_model, decode_pair, endpoint_noise, checkpoint_component, explicit_load, PROMPT; "
         "import inference_offload; print('Actual Qwen model and helper imports passed')"),
        ("Completion implementation", bootstrap +
         "from supratex.inference import run_completion; "
         "from supratex.models import InterleavedFlowTransformer; "
         "print('Completion implementation imported')"),
    ]


def require_nonempty(*paths: Path) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required nonempty file is missing: {path}")


def file_checks(config: dict) -> list[tuple[str, object]]:
    sys.path.insert(0, str(ROOT))
    from supratex.frontend.geometry import validate_source, validate_model
    from supratex.frontend.qwen_image import validate
    from supratex.weights import resolve_completion_artifacts

    def completion():
        resolve_completion_artifacts(Path(config["pretrained"]),
                                     model_config=config.get("model_config"),
                                     checkpoint=config.get("checkpoint"),
                                     trellis_root=Path(config["trellis_root"]))

    moge = Path(config["moge_model"])
    checks = [
        ("Bundled geometry source", lambda: validate_source(Path(config["geometry_package"]))),
        ("Geometry and DINO model files", lambda: validate_model(
            Path(config["geometry_package"]), Path(config["geometry_model"]))),
        ("Qwen base and released material adapter", lambda: validate(
            Path(config["qwen_runtime"]), config["material_base"], config["material_model"],
            deps=Path(config["qwen_deps"]) if config.get("qwen_deps") else None, local_files_only=True)),
        ("Released completion and four sparse VAEs", completion),
        ("MoGe weights", lambda: require_nonempty(moge / "model.pt" if moge.is_dir() else moge)),
        ("NAF local source and weights", lambda: require_nonempty(
            Path(config["torch_home"]) / "hub/valeoai_NAF_main/hubconf.py",
            Path(config["torch_home"]) / "hub/checkpoints/naf_release.pth")),
    ]
    if config.get("rmbg_model"):
        checks.append(("RGB foreground model", lambda: require_nonempty(*(
            Path(config["rmbg_model"]) / name for name in
            ("config.json", "model.safetensors", "BiRefNet_config.py", "birefnet.py")))))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ckpts/ready/models.json")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--imports-only", action="store_true", help="check installed software without model weights")
    scope.add_argument("--files-only", action="store_true", help="validate model layout and headers without CUDA checks")
    parser.add_argument("--attention", choices=("auto", "sdpa", "flash_attn", "flash_attn_3", "xformers"), default="sdpa")
    args = parser.parse_args(argv)
    results = []

    def check(name, operation):
        print(f"[check] {name}", file=sys.stderr, flush=True)
        try:
            detail = operation()
            entry = {"check": name, "status": "passed"}
            if isinstance(detail, str) and detail:
                entry["detail"] = detail
            results.append(entry)
        except Exception as error:
            results.append({"check": name, "status": "failed", "error": str(error)})

    config = None
    if not args.imports_only:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from run_example import read_config
            config = read_config(args.config)
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            os.environ["TORCH_HOME"] = config["torch_home"]
            checks = file_checks(config)
        except Exception as error:
            results.append({"check": "Prepared model configuration", "status": "failed", "error": str(error)})
        else:
            for name, operation in checks:
                check(name, operation)
    if not args.files_only:
        for name, code in import_checks(args.attention):
            check(name, lambda code=code: run_import_check(code, config=config))
    passed = bool(results) and all(entry["status"] == "passed" for entry in results)
    print(json.dumps({"status": "passed" if passed else "failed", "checks": results,
                      "scope": "Model files and actual runtime imports; full image-to-GLB inference is a separate check"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

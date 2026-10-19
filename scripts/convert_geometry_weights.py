#!/usr/bin/env python3
"""Write runtime-dtype copies of the prepared native geometry weights.

The native flow checkpoints store float32 for blocks that run in bfloat16.
A cast copy halves the bytes read on every load (faster on slow disks) and
loads bit-identical tensors. Copies are written beside the originals as
``<stem>.runtime.safetensors`` (about 8 GB) and are used automatically.
Run it in the inference environment with a GPU visible; no GPU memory is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ckpts/ready/models.json",
                        help="prepared models.json (default: ckpts/ready/models.json)")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    package = (config_path.parent / config["geometry_package"]).resolve()
    model_root = (config_path.parent / config["geometry_model"]).resolve()
    sys.path[:0] = [str(package), str(ROOT)]
    from supratex.frontend.geometry import NATIVE_GEOMETRY_MODELS
    from supratex.frontend.geometry_loading import write_runtime_weights

    stems = json.loads((model_root / "pipeline.json").read_text(encoding="utf-8"))["args"]["models"]
    for name in NATIVE_GEOMETRY_MODELS:
        written = write_runtime_weights(model_root / stems[name])
        print(f"{name}: {written or 'already compact; skipped'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Public Image-to-Material Translation entrypoint for the bundled model code."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--items-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="External factorized joint-transformer safetensors file.")
    parser.add_argument("--spec", type=Path, required=True, help="External joint-contract JSON file.")
    parser.add_argument("--checkpoint-manifest", type=Path, required=True, help="External checkpoint completion manifest JSON file.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    for path in (args.weights, args.spec, args.checkpoint_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.base_model = args.base_model.resolve()
    args.items_jsonl = args.items_jsonl.resolve()
    args.out_dir = args.out_dir.resolve()
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="supra_tex_material_") as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "diffusion_pytorch_model.safetensors").symlink_to(args.weights.resolve())
        shutil.copy2(args.spec, checkpoint / "joint_contract.json")
        shutil.copy2(args.checkpoint_manifest, checkpoint / "checkpoint_complete.json")
        # The checkpoint v2/v3 contract records its resolution beside the
        # checkpoint. The public spec is the source of that explicit value.
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        resolution = spec.get("resolution", 1024)
        (checkpoint.parent / "RUN_CONTRACT.json").write_text(json.dumps({"resolution": resolution}) + "\n", encoding="utf-8")
        command = [sys.executable, str(here / "runtime/pbr/predict_joint_pbr.py"),
                   "--items-jsonl", str(args.items_jsonl), "--out-dir", str(args.out_dir),
                   "--base-model", str(args.base_model), "--joint-checkpoint", str(checkpoint),
                   "--transformer-path", str(checkpoint / "diffusion_pytorch_model.safetensors"),
                   "--route", "aligned", "--input-size", str(resolution), "--steps", "1", "--cfg", "1",
                   "--seed", "42", "--num-shards", str(args.num_shards), "--shard-index", str(args.shard_index)]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        raise SystemExit(subprocess.call(command, cwd=here / "runtime"))


if __name__ == "__main__":
    main()

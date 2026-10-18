"""Prepare and validate optional Supra-Tex back-view inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="render a v4 normal/mask/camera package")
    render.add_argument("--input-dir", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--device", default="cuda")
    generate = commands.add_parser("generate", help="prepare a snapshot and plan or run Astra workers")
    generate.add_argument("--pack-dir", type=Path, required=True)
    generate.add_argument("--run-root", type=Path, required=True)
    generate.add_argument("--case", default="input")
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--execute", action="store_true")
    material = commands.add_parser("materials", help="prepare a generated back and predict matching material maps")
    material.add_argument("--front-input-dir", type=Path, required=True)
    material.add_argument("--pack-dir", type=Path, required=True)
    material.add_argument("--run-root", type=Path, required=True)
    material.add_argument("--output-dir", type=Path, required=True)
    material.add_argument("--case", default="input")
    material.add_argument("--device", default="cuda")
    material.add_argument("--cache-dir", type=Path)
    material.add_argument("--local-files-only", action="store_true")
    validate = commands.add_parser("validate", help="verify a pack, prepared input, or completed result")
    group = validate.add_mutually_exclusive_group(required=True)
    group.add_argument("--pack-dir", type=Path)
    group.add_argument("--back-view-dir", type=Path)
    group.add_argument("--work-dir", type=Path)
    validate.add_argument("--input-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            from .pack import render_pack
            result = {"pack": str(render_pack(args.input_dir, args.output_dir, device=args.device))}
        elif args.command == "generate":
            from .harness import prepare_run, run_generation
            prepare_run(args.pack_dir, args.run_root, case_id=args.case)
            result = run_generation(args.run_root, workers=args.workers, execute=args.execute)
            print(json.dumps(result, indent=2))
            return 0 if not args.execute or result.get("success") else 1
        elif args.command == "materials":
            from .workflow import prepare_back_material
            result = {"back_view_dir": str(prepare_back_material(
                args.front_input_dir, args.pack_dir, args.run_root, args.output_dir,
                case_id=args.case, device=args.device, cache_dir=args.cache_dir,
                local_files_only=args.local_files_only))}
        elif args.pack_dir is not None:
            from .pack import validate_pack
            cameras = validate_pack(args.pack_dir, input_dir=args.input_dir)
            result = {"valid": True, "camera_protocol": cameras["protocol"]}
        elif args.back_view_dir is not None:
            if args.input_dir is None:
                parser.error("--back-view-dir validation requires --input-dir")
            from supratex.inference.dual_view import load_back_case
            loaded = load_back_case(args.input_dir, args.back_view_dir)
            result = {"valid": True, "camera_protocol": loaded["cameras"]["protocol"]}
        else:
            from supratex.inference.dual_view import validate_result
            result = validate_result(args.work_dir)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

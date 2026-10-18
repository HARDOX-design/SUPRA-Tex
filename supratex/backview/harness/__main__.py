"""Command-line helpers used by the Astra worker and offline reviewers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import get_task, load_tasks, prepare_run, register, validate_inputs
from .postprocess import process
from .report import audit_run, build_report
from .run_batch import run_generation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "check", "prompt", "register", "postprocess", "audit", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        if name in ("prepare", "register", "postprocess"):
            sub.add_argument("--case", default="input" if name != "postprocess" else None)
        if name == "prepare":
            sub.add_argument("--pack-dir", type=Path, required=True)
        elif name == "run":
            sub.add_argument("--execute", action="store_true")
            sub.add_argument("--workers", type=int, default=1)
            sub.add_argument("--timeout", type=int, default=1800)
            sub.add_argument("--codex", default="codex")
            sub.add_argument("--case", action="append")
            sub.add_argument("--retry-review", action="store_true")
        elif name == "prompt":
            sub.add_argument("--refine", action="store_true")
        elif name == "register":
            sub.add_argument("--raw", type=Path, required=True)
            sub.add_argument("--tool-result", type=Path, required=True)
            sub.add_argument("--prompt-file", type=Path, required=True)
            sub.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    try:
        if args.command == "prepare":
            value = {"run_root": str(prepare_run(args.pack_dir, root, case_id=args.case))}
        elif args.command == "run":
            value = run_generation(root, workers=args.workers, execute=args.execute, timeout=args.timeout,
                                   codex=args.codex, case_ids=args.case, retry_review=args.retry_review)
        elif args.command == "check":
            validate_inputs(root)
            value = {"inputs_unchanged": True, "case_count": len(load_tasks(root))}
        elif args.command == "prompt":
            validate_inputs(root)
            content = (root / "prompts/backview_universal.txt").read_text(encoding="utf-8")
            if args.refine:
                content += "\n" + (root / "prompts/refinement_universal.txt").read_text(encoding="utf-8")
            print(content, end="")
            return 0
        elif args.command == "register":
            value = register(root, args.case, args.raw, args.tool_result, args.prompt_file, args.attempt)
        elif args.command == "postprocess":
            validate_inputs(root)
            tasks = [get_task(root, args.case)] if args.case else load_tasks(root)
            value = {"cases": [process(task, root) for task in tasks]}
        elif args.command == "audit":
            value = audit_run(root)
        else:
            value = {"report": str(build_report(root))}
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if args.command == "run" and args.execute:
            return 0 if value["success"] else 1
        if args.command == "audit":
            return 0 if value["all_accepted"] else 1
        if args.command == "postprocess":
            return 0 if all(row["status"] == "needs_visual_review" for row in value["cases"]) else 1
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Run resumable GPT-6 Astra workers without substituting models or images."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .core import (
    MODEL, EFFORT, checked_path, current_review, generation_current, load_tasks,
    read_json, sha, validate_inputs, validate_result, write_json,
)
from .postprocess import process

TERMINAL_FAILURES = {"terminal_policy_refusal", "policy_refusal", "refused", "moderation_blocked",
                     "missing_required_capability"}


def run_generation(run_root: Path, *, workers: int = 1, execute: bool = False,
                   codex: str = "codex", timeout: int = 1800,
                   case_ids: list[str] | None = None, retry_review: bool = False) -> dict:
    """Plan by default; execution requires Codex with Astra and built-in image_gen.

    A completed needs_review observation may continue while retaining its label;
    needs_regeneration remains blocked. Corrections to either require
    retry_review=True; resume never silently spends another generation.
    """
    root = Path(run_root).resolve()
    if not 1 <= workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    validate_inputs(root)
    tasks = load_tasks(root)
    if case_ids:
        unknown = set(case_ids) - {task["case_id"] for task in tasks}
        if unknown:
            raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
        tasks = [task for task in tasks if task["case_id"] in case_ids]
    base = [codex, "exec", "--skip-git-repo-check", "--cd", str(root), "--model", MODEL,
            "-c", 'model_reasoning_effort="ultra"', "--sandbox", "workspace-write", "--json"]
    pending, blocked = [], []
    for task in tasks:
        output = checked_path(root, task["output_dir"])
        failure_path = output / "failure.json"
        if failure_path.exists() and read_json(failure_path).get("status") in TERMINAL_FAILURES:
            blocked.append({"case_id": task["case_id"], "failure": read_json(failure_path)})
            continue
        current = generation_current(root, task)
        review = current_review(root, task) if current else None
        status_path = output / "worker_status.json"
        completed = read_json(status_path).get("run_success", False) if status_path.exists() else False
        if not current or review is None or not completed or (retry_review and review["status"] != "acceptable"):
            pending.append(task)
    summary = {"mode": "execute" if execute else "plan", "run_root": str(root),
               "worker_model": MODEL, "worker_reasoning_effort": EFFORT,
               "required_capability": "Codex built-in image_gen", "command_prefix": base,
               "pending_cases": [task["case_id"] for task in pending], "blocked": blocked,
               "statuses": [], "results": []}
    if not execute:
        return summary
    if pending and shutil.which(codex) is None:
        raise RuntimeError("Codex CLI is unavailable; generation requires GPT-6 Astra with built-in image_gen")
    lock = root / ".generation.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Run is locked: {lock}. Remove a stale lock only after confirming no worker is running") from exc
    os.close(descriptor)
    try:
        worker_prompt = (root / "prompts/worker_universal.txt").read_text(encoding="utf-8")
        env = os.environ.copy()
        import_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = import_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        def run(task):
            output = checked_path(root, task["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            attempts = []
            for pattern, prefix in (("raw_attempt*.png", "raw_attempt"), ("prompt_attempt*.txt", "prompt_attempt"),
                                    ("tool_result_attempt*.json", "tool_result_attempt")):
                attempts += [int(path.stem.removeprefix(prefix)) for path in output.glob(pattern)
                             if path.stem.removeprefix(prefix).isdigit()]
            next_attempt = max(attempts, default=0) + 1
            previous_runs = [int(path.stem.removeprefix("worker_invocation"))
                             for path in output.glob("worker_invocation*.json")
                             if path.stem.removeprefix("worker_invocation").isdigit()]
            run_number = max(previous_runs, default=0) + 1
            invocation_path = output / f"worker_invocation{run_number}.json"
            events_path = output / f"worker_events{run_number}.jsonl"
            stderr_path = output / f"worker_stderr{run_number}.log"
            final_path = output / f"worker_final{run_number}.txt"
            current = generation_current(root, task)
            review_only = current and not retry_review
            assignment = {"HARNESS_ROOT": str(root), "CASE_IDS": [task["case_id"]],
                          "PYTHON_EXECUTABLE": sys.executable, "START_ATTEMPT": next_attempt}
            prompt = worker_prompt + "\nASSIGNMENT\n" + json.dumps(assignment, ensure_ascii=False) + "\n"
            # Also applies to review recovery using older immutable prompt snapshots.
            prompt += ("CURRENT REVIEW CONTRACT: after postprocessing, inspect back_rgba.png, back_rgb.png, "
                       "overlay.png and alignment.json. Only then set postprocessing_reviewed=true, copy "
                       "validation.json's processed_file_sha256 to reviewed_processed_file_sha256 and its "
                       "postprocessing_policy_sha256 unchanged into visual_review.json. Each defect must have "
                       "severity=minor|major and a concrete description; acceptable has no recorded defects, "
                       "minor defects require needs_review and any major defect requires needs_regeneration. "
                       "Never bind an existing review to changed outputs without inspecting them again.\n")
            if review_only:
                prompt += ("REVIEW-ONLY RECOVERY: do not call image_gen, do not register a new candidate, "
                           "and do not change the selected raw. Inspect the existing inputs and raw, run "
                           "postprocessing, and restore the full visual_review.json bound to its SHA-256.\n")
            elif current and retry_review:
                prompt += "CORRECTION: begin with the fixed universal correction prompt and preserve earlier attempts.\n"
            command = base + ["--output-last-message", str(final_path), "-"]
            write_json(invocation_path, {"command": command, "case_id": task["case_id"],
                       "required_capability": "built-in image_gen", "automatic_model_fallback": False,
                       "mode": "review_only_recovery" if review_only else "generation",
                       "worker_prompt_sha256": sha(root / "prompts/worker_universal.txt"),
                       "input_sha256": task["input_sha256"], "start_attempt": next_attempt})
            status = {"case_id": task["case_id"], "mode": "review_only_recovery" if review_only else "generation",
                      "exit_code": None, "invocation_path": invocation_path.relative_to(root).as_posix(),
                      "events_path": events_path.relative_to(root).as_posix()}
            try:
                with events_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    worker = subprocess.run(command, input=prompt, text=True, encoding="utf-8", stdout=stdout,
                                            stderr=stderr, timeout=timeout, cwd=root, env=env)
                status["exit_code"] = worker.returncode
            except (subprocess.TimeoutExpired, OSError) as exc:
                status["error"] = f"{type(exc).__name__}: {exc}"
            status["registered"] = generation_current(root, task)
            status["review_current"] = current_review(root, task) is not None
            failure_path = output / "failure.json"
            status["terminal_failure"] = (failure_path.exists()
                                          and read_json(failure_path).get("status") in TERMINAL_FAILURES)
            provenance_path = output / "provenance.json"
            provenance = read_json(provenance_path) if provenance_path.exists() else {}
            status["new_candidate_registered"] = any(a.get("attempt", 0) >= next_attempt for a in provenance.get("attempts", []))
            status["run_success"] = (status["exit_code"] == 0 and not status["terminal_failure"]
                                     and status["registered"] and status["review_current"]
                                     and (review_only or status["new_candidate_registered"]))
            status["selected_raw_sha256"] = sha(output / "raw.png") if (output / "raw.png").is_file() else None
            status["invocation_sha256"] = sha(invocation_path)
            status["events_sha256"] = sha(events_path) if events_path.exists() else None
            if not status["run_success"] and "error" not in status:
                status["error"] = ("Astra did not produce a registered image and complete review. Inspect worker events, "
                                   "stderr, and failure.json; built-in image_gen and the requested model are required")
            write_json(output / f"worker_status{run_number}.json", status)
            write_json(output / "worker_status.json", status)
            return status

        with ThreadPoolExecutor(max_workers=workers) as pool:
            summary["statuses"] = list(pool.map(run, pending))
        for task in tasks:
            # Do not rewrite artifacts already inspected and bound by the worker.
            if current_review(root, task) is None:
                process(task, root)
        summary["results"] = [validate_result(root, task["case_id"]) for task in tasks]
        summary["success"] = (not blocked and all(s["run_success"] for s in summary["statuses"])
                              and all(r["valid"] for r in summary["results"]))
        write_json(root / "cli_batch_status.json", summary)
        from .report import build_report
        build_report(root)
        return summary
    finally:
        lock.unlink(missing_ok=True)

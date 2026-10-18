"""Immutable inputs, generation provenance, and the back-view acceptance gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image

MODEL = "gpt-6-astra"
EFFORT = "ultra"
GENERATION_TOOL = "image_gen.imagegen"
SCHEMA = "supratex.backview.harness.v1"
INPUT_FILES = {
    "front_rgb": "front_rgb.png",
    "back_normal": "back_normal.png",
    "back_mask": "back_mask.png",
    "cameras": "cameras.json",
}
PROMPT_FILES = ("backview_universal.txt", "refinement_universal.txt", "worker_universal.txt")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Atomically replace a record so concurrent readers never see partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_path(root: Path, relative: str) -> Path:
    """Reject manifest paths that escape the run, including symlink escapes."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path must remain inside the run: {relative}")
    return path


def check_case_id(case_id: str) -> None:
    if not case_id or case_id in (".", "..") or any(c in case_id for c in "/\\\x00\n\r"):
        raise ValueError("case_id must be a nonempty directory name without path separators")


def copy_checked(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha(source) != sha(destination):
            raise ValueError(f"Refusing to overwrite different run data: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def approved_prompt_hashes(root: Path) -> dict:
    base = (root / "prompts/backview_universal.txt").read_text(encoding="utf-8")
    correction = (root / "prompts/refinement_universal.txt").read_text(encoding="utf-8")
    return {"base": hashlib.sha256(base.encode()).hexdigest(),
            "universal_refinement": hashlib.sha256((base + "\n" + correction).encode()).hexdigest()}


def prepare_run(pack_dir: Path, run_root: Path, *, case_id: str = "input") -> Path:
    """Snapshot a validated camera pack without altering any source image or camera."""
    from supratex.backview.pack import validate_pack

    pack_dir, root = Path(pack_dir).resolve(), Path(run_root).resolve()
    check_case_id(case_id)
    validate_pack(pack_dir)
    if (root / "run_config.json").exists():
        validate_inputs(root)
    sources = {key: pack_dir / name for key, name in INPUT_FILES.items()}
    source_prompts = Path(__file__).with_name("prompts")
    task = {"case_id": case_id, "output_dir": f"results/{case_id}",
            "source_pack": str(pack_dir), "input_sha256": {k: sha(p) for k, p in sources.items()}}
    for key, name in INPUT_FILES.items():
        task[key] = f"inputs/{case_id}/{name}"
    tasks_path = root / "tasks.json"
    tasks = read_json(tasks_path) if tasks_path.exists() else []
    existing = next((row for row in tasks if row["case_id"] == case_id), None)
    if existing is not None and any(existing.get(k) != task[k] for k in task if k != "source_pack"):
        raise ValueError(f"Run already contains different inputs for {case_id}; choose a new run root")
    # Check every collision before writing any snapshots.
    pairs = [(sources[k], checked_path(root, task[k])) for k in INPUT_FILES]
    pairs += [(source_prompts / name, checked_path(root, "prompts/" + name)) for name in PROMPT_FILES]
    for source, target in pairs:
        if target.exists() and (not target.is_file() or sha(source) != sha(target)):
            raise ValueError(f"Run snapshot differs: {target}; choose a new run root")
    config_path = root / "run_config.json"
    config = read_json(config_path) if config_path.exists() else {
        "schema": SCHEMA, "worker_model": MODEL, "worker_reasoning_effort": EFFORT,
        "generation_tool": GENERATION_TOOL, "image_synthesis_model": "not disclosed by built-in tool",
        "automatic_model_fallback": False, "input_order": ["back_normal", "front_rgb"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if config.get("schema") != SCHEMA or config.get("worker_model") != MODEL or config.get("worker_reasoning_effort") != EFFORT:
        raise ValueError("Existing run has incompatible harness or model settings")
    for source, target in pairs:
        copy_checked(source, target)
    if existing is None:
        tasks.append(task)
    config.update(case_count=len(tasks), approved_prompt_sha256=approved_prompt_hashes(root),
                  prompt_sha256={name: sha(root / "prompts" / name) for name in PROMPT_FILES})
    write_json(tasks_path, tasks)
    config["tasks_sha256"] = sha(tasks_path)
    write_json(config_path, config)
    return root


def load_tasks(root: Path) -> list[dict]:
    tasks = read_json(root / "tasks.json")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks.json must contain at least one case")
    ids = [task["case_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in tasks.json")
    for task in tasks:
        check_case_id(task["case_id"])
        if task["output_dir"] != f"results/{task['case_id']}":
            raise ValueError("Unexpected case output directory")
        for key, name in INPUT_FILES.items():
            if task.get(key) != f"inputs/{task['case_id']}/{name}":
                raise ValueError(f"Unexpected input path for {task['case_id']}: {key}")
            checked_path(root, task[key])
        checked_path(root, task["output_dir"])
    return tasks


def get_task(root: Path, case_id: str) -> dict:
    check_case_id(case_id)
    for task in load_tasks(root):
        if task["case_id"] == case_id:
            return task
    raise ValueError(f"Unknown case_id: {case_id}")


def validate_inputs(root: Path, task: dict | None = None) -> None:
    config = read_json(root / "run_config.json")
    if (config.get("schema") != SCHEMA or config.get("worker_model") != MODEL
            or config.get("worker_reasoning_effort") != EFFORT
            or config.get("generation_tool") != GENERATION_TOOL
            or config.get("automatic_model_fallback") is not False):
        raise ValueError("Run requires GPT-6 Astra, ultra, and built-in image_gen without fallback")
    if sha(root / "tasks.json") != config.get("tasks_sha256"):
        raise ValueError("Task manifest is missing or changed")
    for name in PROMPT_FILES:
        path = root / "prompts" / name
        if not path.is_file() or sha(path) != config.get("prompt_sha256", {}).get(name):
            raise ValueError(f"Prompt snapshot missing or changed: {name}")
    if config.get("approved_prompt_sha256") != approved_prompt_hashes(root):
        raise ValueError("Approved generation prompt hashes do not match the snapshots")
    for row in ([task] if task is not None else load_tasks(root)):
        for key in INPUT_FILES:
            path = checked_path(root, row[key])
            if not path.is_file() or sha(path) != row["input_sha256"].get(key):
                raise ValueError(f"Input missing or changed: {row['case_id']}/{key}")


def tool_return_contains_path(value, source_path: str) -> bool:
    """Match returned paths exactly, including paths embedded in JSON content."""
    expected = str(source_path)
    if not expected:
        return False
    windows = bool(re.match(r"^[A-Za-z]:[\\/]", expected) or expected.startswith("\\\\"))
    normalize = lambda s: s.replace("\\", "/").casefold() if windows else s
    pattern = re.compile(r"(?<![\w./\\:-])" + re.escape(normalize(expected)) + r"(?![\w./\\:-])")
    decoder = json.JSONDecoder()

    def visit(item, depth=0):
        if depth > 12:
            return False
        if isinstance(item, dict):
            return any(visit(v, depth + 1) for v in item.values())
        if isinstance(item, (tuple, list)):
            return any(visit(v, depth + 1) for v in item)
        if not isinstance(item, str):
            return False
        if pattern.search(normalize(item)):
            return True
        index = 0
        while index < len(item):
            match = re.search(r'[\[{\"]', item[index:])
            if match is None:
                break
            index += match.start()
            try:
                decoded, end = decoder.raw_decode(item, index)
            except json.JSONDecodeError:
                index += 1
                continue
            if visit(decoded, depth + 1):
                return True
            index = end
        return False

    return visit(value)


def register(root: Path, case_id: str, raw: Path, tool_result: Path,
             prompt_file: Path, attempt: int) -> dict:
    """Preserve the actual returned image and tool record as immutable attempts."""
    root, raw = Path(root).resolve(), Path(raw).resolve()
    tool_result, prompt_file = Path(tool_result).resolve(), Path(prompt_file).resolve()
    task = get_task(root, case_id)
    validate_inputs(root, task)
    if attempt < 1:
        raise ValueError("Attempt numbers must be positive")
    if not tool_return_contains_path(read_json(tool_result), str(raw)):
        raise ValueError("Raw image path is absent from the saved tool return")
    if sha(prompt_file) not in approved_prompt_hashes(root).values():
        raise ValueError("Generation must use an unchanged universal prompt")
    if sha(raw) in task["input_sha256"].values():
        raise ValueError("A generation input cannot be registered as an output")
    with Image.open(raw) as im:
        im.verify()
    with Image.open(raw) as im:
        size, mode = list(im.size), im.mode
    output = checked_path(root, task["output_dir"])
    files = {"raw_path": (raw, output / f"raw_attempt{attempt}.png"),
             "prompt_file": (prompt_file, output / f"prompt_attempt{attempt}.txt"),
             "tool_result_path": (tool_result, output / f"tool_result_attempt{attempt}.json")}
    for source, target in files.values():
        copy_checked(source, target)
    record = {"attempt": attempt, "source_path": str(raw), "native_size": size,
              "native_mode": mode, "registered_at": datetime.now(timezone.utc).isoformat()}
    for key, (_, path) in files.items():
        record[key] = path.relative_to(root).as_posix()
    record.update(raw_sha256=sha(files["raw_path"][1]), prompt_sha256=sha(files["prompt_file"][1]),
                  tool_result_sha256=sha(files["tool_result_path"][1]))
    provenance_path = output / "provenance.json"
    old = read_json(provenance_path) if provenance_path.exists() else {}
    attempts = [a for a in old.get("attempts", []) if a["attempt"] != attempt] + [record]
    shutil.copy2(files["raw_path"][1], output / "raw.png")
    provenance = {"schema": SCHEMA, "case_id": case_id, "worker_model": MODEL,
                  "worker_reasoning_effort": EFFORT, "generation_tool": GENERATION_TOOL,
                  "image_synthesis_model": "not disclosed by built-in tool",
                  "input_sha256": task["input_sha256"], "selected_attempt": attempt,
                  "raw_sha256": record["raw_sha256"], "prompt_sha256": record["prompt_sha256"],
                  "attempts": attempts, "interior_pixel_alignment_proven": False}
    write_json(provenance_path, provenance)
    return record


def generation_current(root: Path, task: dict) -> bool:
    output = checked_path(root, task["output_dir"])
    try:
        provenance = read_json(output / "provenance.json")
        return (provenance.get("input_sha256") == task["input_sha256"]
                and provenance.get("prompt_sha256") in approved_prompt_hashes(root).values()
                and provenance.get("raw_sha256") == sha(output / "raw.png"))
    except (OSError, ValueError, KeyError):
        return False


def current_review(root: Path, task: dict) -> dict | None:
    output = checked_path(root, task["output_dir"])
    try:
        review = read_json(output / "visual_review.json")
        required = ("rear_view", "geometry", "materials", "text_and_patterns", "lighting_and_brightness")
        if (review.get("reviewed_raw_sha256") == sha(output / "raw.png")
                and review.get("case_id") == task["case_id"]
                and review.get("status") in ("acceptable", "needs_review", "needs_regeneration")
                and all(review.get(k) for k in required)
                and isinstance(review.get("defects"), list)
                and review.get("interior_pixel_alignment_proven") is False):
            return review
    except (OSError, ValueError, KeyError):
        pass
    return None


def validate_result(run_root: Path, case_id: str = "input") -> dict:
    """Accept only current, auditable, visually accepted 1024-square observations."""
    root = Path(run_root).resolve()
    result = {"case_id": case_id, "valid": False, "records_valid": False,
              "status": "invalid", "errors": [], "warnings": [], "back_rgba": None,
              "interior_pixel_alignment_proven": False}
    errors = result["errors"]
    try:
        task = get_task(root, case_id)
        validate_inputs(root, task)
        output = checked_path(root, task["output_dir"])
        provenance = read_json(output / "provenance.json")
        if (provenance.get("worker_model") != MODEL or provenance.get("worker_reasoning_effort") != EFFORT
                or provenance.get("generation_tool") != GENERATION_TOOL):
            errors.append("Generation provenance does not identify the required model, effort, and tool")
        if provenance.get("input_sha256") != task["input_sha256"]:
            errors.append("Generation inputs do not match the current input snapshot")
        raw_hash = sha(output / "raw.png")
        attempts = provenance.get("attempts", [])
        numbers = [a["attempt"] for a in attempts]
        if len(numbers) != len(set(numbers)):
            errors.append("Duplicate attempt numbers")
        selected = next((a for a in attempts if a["attempt"] == provenance.get("selected_attempt")), None)
        if selected is None:
            raise ValueError("Missing selected generation attempt")
        for attempt in attempts:
            for key, hash_key in (("raw_path", "raw_sha256"), ("prompt_file", "prompt_sha256"),
                                  ("tool_result_path", "tool_result_sha256")):
                path = checked_path(root, attempt[key])
                if sha(path) != attempt.get(hash_key):
                    errors.append(f"Attempt {attempt['attempt']} changed: {key}")
            returned = read_json(checked_path(root, attempt["tool_result_path"]))
            if not tool_return_contains_path(returned, attempt.get("source_path", "")):
                errors.append(f"Attempt {attempt['attempt']} lacks the original tool image path")
            if attempt.get("prompt_sha256") not in approved_prompt_hashes(root).values():
                errors.append(f"Attempt {attempt['attempt']} used an unapproved prompt")
        if raw_hash != provenance.get("raw_sha256") or raw_hash != selected.get("raw_sha256"):
            errors.append("Selected raw image is missing or changed")
        if raw_hash in task["input_sha256"].values():
            errors.append("The raw output is a copy of a generation input")
        if provenance.get("prompt_sha256") != selected.get("prompt_sha256"):
            errors.append("Selected prompt does not match provenance")
        alignment = read_json(output / "alignment.json")
        processing = read_json(output / "validation.json")
        if alignment.get("raw_sha256") != raw_hash or processing.get("raw_sha256") != raw_hash:
            errors.append("Postprocessing is stale")
        if processing.get("status") != "needs_visual_review":
            errors.append("Postprocessing did not complete successfully")
        for key in ("front_rgb", "back_normal", "back_mask"):
            if alignment.get(key + "_sha256") != task["input_sha256"][key]:
                errors.append(f"Postprocessing input changed: {key}")
        for name in ("back_rgba.png", "back_rgb.png", "overlay.png"):
            if sha(output / name) != alignment.get("output_sha256", {}).get(name):
                errors.append(f"Processed output missing or changed: {name}")
        with Image.open(output / "back_rgba.png") as image:
            if image.mode != "RGBA" or image.size != (1024, 1024):
                errors.append("Back observation must be RGBA at 1024 x 1024")
            alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
        with Image.open(checked_path(root, task["back_mask"])) as image:
            mask = np.asarray(image.convert("L"))
        if alpha.shape != mask.shape or not np.array_equal(alpha, mask):
            errors.append("Final alpha must equal the original renderer mask exactly")
        review = current_review(root, task)
        if review is None:
            errors.append("Visual review is missing, incomplete, or stale")
        else:
            result["status"] = review["status"]
            if review["status"] != "acceptable":
                result["warnings"].append("Visual review requires attention; the observation is not accepted for inference")
        status = read_json(output / "worker_status.json")
        if status.get("run_success") is not True or status.get("exit_code") != 0:
            errors.append("The required Astra worker did not complete successfully")
        if status.get("selected_raw_sha256") != raw_hash:
            errors.append("Worker completion record is stale")
        invocation_path = checked_path(root, status["invocation_path"])
        events_path = checked_path(root, status["events_path"])
        if sha(invocation_path) != status.get("invocation_sha256") or sha(events_path) != status.get("events_sha256"):
            errors.append("Worker invocation or event log changed")
        if events_path.stat().st_size == 0:
            errors.append("Worker event log is empty")
        invocation = read_json(invocation_path)
        command = invocation["command"]
        if (command[command.index("--model") + 1] != MODEL
                or 'model_reasoning_effort="ultra"' not in command
                or invocation.get("automatic_model_fallback") is not False
                or invocation.get("required_capability") != "built-in image_gen"):
            errors.append("Worker invocation does not enforce the required model and tool")
        if (invocation.get("case_id") != case_id or invocation.get("input_sha256") != task["input_sha256"]
                or invocation.get("worker_prompt_sha256") != sha(root / "prompts/worker_universal.txt")):
            errors.append("Worker invocation is not bound to the current case inputs and prompt")
        result["back_rgba"] = str(output / "back_rgba.png")
        result["records_valid"] = not errors
        result["valid"] = not errors and result["status"] == "acceptable"
        if result["records_valid"]:
            result.update(worker_model=provenance["worker_model"],
                          worker_reasoning_effort=provenance["worker_reasoning_effort"],
                          generation_tool=provenance["generation_tool"],
                          cameras=str(checked_path(root, task["cameras"])),
                          cameras_sha256=task["input_sha256"]["cameras"],
                          back_rgba_sha256=sha(output / "back_rgba.png"), raw_sha256=raw_hash,
                          provenance=str(output / "provenance.json"),
                          invocation=str(invocation_path), events=str(events_path),
                          input_sha256=task["input_sha256"])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return result

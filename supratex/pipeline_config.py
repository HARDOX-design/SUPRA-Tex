"""Read the SUPRA-Tex release manifest without importing model runtimes."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def runtime_root(root: Path) -> Path:
    """Locate the single runtime tree in a source checkout or legacy bundle."""
    source = root / "deployment/runtime"
    bundled = root / "runtime"
    return bundled if not source.is_dir() and bundled.is_dir() else source


def relative_name(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"expected a relative path without traversal: {value!r}")
    return value


def model_source(value: str | os.PathLike, revision: str | None = None) -> dict:
    """Normalize local directories and public-form HF IDs/URLs without networking."""
    local_only = isinstance(value, os.PathLike)
    value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise ValueError("model source must be a nonempty path or namespace/repo HF ID")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("model revision must be a nonempty string")
    local = Path(value).expanduser()
    if local.exists():
        if not local.is_dir():
            raise NotADirectoryError(f"model source must be a directory: {local}")
        return {"kind": "local", "path": str(local.resolve())}
    if local_only or value.startswith(("/", "./", "../", "~")) or value in (".", ".."):
        raise FileNotFoundError(f"model directory does not exist: {local}")
    if value.startswith("https://"):
        url = urlsplit(value)
        if url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("use a public-form https://huggingface.co/namespace/repo URL without credentials or query")
        parts = unquote(url.path).strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree":
            if revision is not None and revision != parts[3]:
                raise ValueError("HF URL revision conflicts with --revision")
            revision = parts[3]
            parts = parts[:2]
        if len(parts) != 2:
            raise ValueError("HF URL must select a repository root or /tree/<revision>")
        value = "/".join(parts)
    value = value.removeprefix("hf://")
    if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", value)
            or ".." in value or "--" in value or value.endswith((".", "-"))):
        raise ValueError("model source must be an existing directory or a namespace/repo HF ID or URL")
    return {"kind": "hub", "repo_id": value, "revision": revision}


def _keys(value: object, required: set[str], name: str, optional: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"pipeline {name} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise ValueError(f"pipeline {name} has missing keys {sorted(missing)} or unsupported keys {sorted(unknown)}")


def load_pipeline(path: str | os.PathLike) -> dict:
    """Load and validate the data-only schema; component configs stay separate."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"pipeline manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _keys(manifest, {"format", "schema_version", "models", "external_models", "defaults"}, "root")
    if (manifest["format"] != "supratex_pipeline" or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1):
        raise ValueError("pipeline requires format 'supratex_pipeline' and schema_version 1")
    _keys(manifest["models"], {"completion", "material"}, "models")
    for value in manifest["models"].values():
        relative_name(value)
    external = manifest["external_models"]
    _keys(external, {"qwen-base", "geometry", "moge2", "dino", "rmbg2", "vae", "naf"}, "external_models")
    for name, entry in external.items():
        if name == "naf":
            _keys(entry, {"url", "filename"}, "external_models.naf")
            filename = relative_name(entry["filename"])
            if "/" in filename:
                raise ValueError("pipeline NAF filename must be a single filename")
            if not isinstance(entry["url"], str):
                raise ValueError("pipeline NAF url must be an HTTPS URL")
            url = urlsplit(entry["url"])
            if (url.scheme != "https" or not url.hostname or url.username or url.password
                    or url.query or url.fragment or not url.path
                    or any(char.isspace() for char in entry["url"])):
                raise ValueError("pipeline NAF url must be an HTTPS URL without credentials or query")
            continue
        _keys(entry, {"repo_id", "revision", "allow_patterns"}, f"external_models.{name}", {"gated"})
        repo_id = entry["repo_id"]
        if not isinstance(repo_id, str):
            raise ValueError(f"pipeline {name} repo_id must be a namespace/repo HF ID")
        source = model_source("hf://" + repo_id, entry["revision"])
        if source["kind"] != "hub" or source["repo_id"] != repo_id or entry["revision"] is None:
            raise ValueError(f"pipeline {name} requires an HF repo_id and a nonempty revision")
        patterns = entry["allow_patterns"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"pipeline {name} allow_patterns must be a nonempty list")
        for pattern in patterns:
            relative_name(pattern)
        if "gated" in entry and type(entry["gated"]) is not bool:
            raise ValueError(f"pipeline {name} gated must be a boolean")
    validate_defaults(manifest["defaults"])
    return manifest


def validate_defaults(defaults: dict) -> dict:
    """Validate a complete defaults object, including resolved runtime configs."""
    _keys(defaults, {"seed", "texture_size", "decimation_target", "texture_format", "attention"}, "defaults")
    for name in ("seed", "texture_size", "decimation_target"):
        if type(defaults[name]) is not int or defaults[name] < (0 if name == "seed" else 1):
            raise ValueError(f"pipeline default {name} must be {'nonnegative' if name == 'seed' else 'positive'} integer")
    if defaults["texture_format"] not in ("png", "webp"):
        raise ValueError("pipeline default texture_format must be png or webp")
    if defaults["attention"] not in ("auto", "flash_attn_3", "flash_attn", "sdpa", "xformers"):
        raise ValueError("pipeline default attention is unsupported")
    return defaults


# Refinement reads and writes a 4096-square PNG base-color atlas.
REFINE_ATLAS = {"texture_size": 4096, "texture_format": "png"}


def validate_export_options(args) -> None:
    """Check export options shared by infer.py and texturing.py; None means not yet resolved."""
    # Dual-view always refines both views, so it implies --refine.
    if (args.dual_view or getattr(args, "refine", False)) and any(
            getattr(args, key) not in (None, value) for key, value in REFINE_ATLAS.items()):
        raise ValueError("--refine and --dual-view require the 4096 PNG atlas")
    if ((args.seed is not None and args.seed < 0)
            or (args.texture_size is not None and args.texture_size <= 0)
            or (args.decimation_target is not None and args.decimation_target <= 0)):
        raise ValueError("seed must be nonnegative; texture-size and decimation-target must be positive")


def default_pipeline() -> dict:
    """Read the repository's canonical defaults, without maintaining a copy."""
    return load_pipeline(ROOT / "configs/pipeline.json")


def resolve_pipeline(
    source: str | os.PathLike,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    local_files_only: bool = False,
) -> tuple[dict, dict]:
    """Return a manifest and its local release root or pinned Hub descriptor.

    Hub resolution downloads only ``pipeline.json``. The returned Hub revision
    follows its cache snapshot so subsequent component downloads use that same
    commit even if the requested branch changes in the meantime.
    """
    value = os.fspath(source)
    if not isinstance(value, str) or not value:
        raise ValueError("pipeline source must be a path or a namespace/repo HF ID")
    path = Path(value).expanduser()
    if path.is_file() or (not value.startswith(("https://", "hf://")) and path.suffix == ".json"):
        # Keep the logical path: an exported manifest can be a symlink, and its
        # component paths must still be relative to the selected release root.
        path = path.absolute()
        return load_pipeline(path), {"kind": "local", "path": str(path.parent)}
    descriptor = model_source(source, revision)
    if descriptor["kind"] == "local":
        return load_pipeline(Path(descriptor["path"]) / "pipeline.json"), descriptor
    from huggingface_hub import hf_hub_download

    try:
        path = Path(hf_hub_download(repo_id=descriptor["repo_id"], filename="pipeline.json",
                                  revision=descriptor["revision"], cache_dir=cache_dir,
                                  local_files_only=local_files_only))
    except Exception:
        detail = "a cached manifest is required in offline mode" if local_files_only else "check the repository, revision and your access"
        raise RuntimeError(f"Cannot load pipeline.json from {descriptor['repo_id']}; {detail}") from None
    manifest = load_pipeline(path)
    if path.parent.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", path.parent.name):
        descriptor["revision"] = path.parent.name
    return manifest, descriptor

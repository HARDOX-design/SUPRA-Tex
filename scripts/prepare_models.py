#!/usr/bin/env python3
"""Download or bind Supra-Tex models, then publish a portable models.json.

Downloads use the inherited Hugging Face configuration and your existing access.
Local overrides are linked without copying large weights and remain dependencies
of the prepared tree. Cached Hub snapshots are also linked when using an explicit
cache or preparing offline; keep that cache available. No tensor data is loaded
and no checksums are computed.
"""
from __future__ import annotations

import argparse
import copy
import filecmp
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PROJECT_REPO = "supra3d/supra-tex"
COMPLETION_PATTERNS = ["config.json", "partial_normalization.json", "*.safetensors",
                       "*.safetensors.index.json", "LICENSE*", "README.md"]
MATERIAL_PATTERNS = ["adapter_config.json", "adapter_model.safetensors", "LICENSE*", "Notice", "README.md"]
GEOMETRY_KEYS = ("sparse_structure_flow_model", "sparse_structure_decoder",
                 "shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder")
VAE_STEMS = ("shape_enc_next_dc_f16c32_fp16", "tex_enc_next_dc_f16c32_fp16",
             "shape_dec_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16")
MARKER = ".prepare-models.json"

# Keep setup/help usable without importing the GPU-facing supratex package.
_spec = importlib.util.spec_from_file_location("_supratex_pipeline_config", ROOT / "supratex/pipeline_config.py")
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
relative_name = _pipeline.relative_name
model_source = _pipeline.model_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=PROJECT_REPO, help="pipeline release directory or HF repo ID/URL (default: %(default)s)")
    parser.add_argument("--revision", help="project repository revision; defaults to its current main")
    parser.add_argument("--pipeline", type=Path, help="explicit local pipeline.json; otherwise read it from --repo")
    parser.add_argument("--completion-subdir", help="override the completion directory declared by pipeline.json")
    parser.add_argument("--material-subdir", help="override the material directory declared by pipeline.json")
    parser.add_argument("--completion", help="override with standalone completion directory, HF repo ID, or HF URL")
    parser.add_argument("--material", help="override with material adapter directory, HF repo ID, or HF URL")
    parser.add_argument("--completion-revision", help="revision for a separate --completion repository")
    parser.add_argument("--material-revision", help="revision for a separate --material repository")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ckpts/ready")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache for manifests and model snapshots; prepared files link to this cache, so keep it available")
    for name in ("qwen-base", "geometry", "dino", "vae", "moge", "rmbg", "naf-weights"):
        parser.add_argument("--" + name, type=Path, help="existing local override (no download)")
    parser.add_argument("--with-rmbg", action="store_true", help="also prepare RMBG for RGB inputs; default example uses RGBA")
    parser.add_argument("--local-files-only", action="store_true", help="use local files and cached Hub snapshots; never download")
    parser.add_argument("--dry-run", action="store_true", help="offline plan; remote repositories use the bundled pipeline preview")
    return parser.parse_args(argv)


def local_override(value: Path, *, file_allowed: bool = False) -> dict:
    path = value.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"local override does not exist: {path}")
    if not path.is_dir() and not (file_allowed and path.is_file()):
        raise ValueError(f"local override must be {'a directory or file' if file_allowed else 'a directory'}: {path}")
    return {"kind": "local", "path": str(path)}


def build_plan(args: argparse.Namespace, *, resolve_remote: bool = False) -> dict:
    """Build a plan from one pipeline; offline previews never fetch metadata."""
    output = args.output_dir.expanduser().resolve()
    project = model_source(args.repo, args.revision)
    project_needed = [name for name in ("completion", "material") if getattr(args, name) is None]
    if args.pipeline is not None:
        pipeline = _pipeline.load_pipeline(args.pipeline)
        manifest_source = "explicit local pipeline"
    elif project["kind"] == "local":
        pipeline, project = _pipeline.resolve_pipeline(args.repo)
        manifest_source = "local release"
    elif resolve_remote and (project_needed or args.repo != PROJECT_REPO or args.revision):
        pipeline, project = _pipeline.resolve_pipeline(
            args.repo, revision=args.revision, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
        manifest_source = "Hub pipeline at selected revision"
    else:
        pipeline = _pipeline.load_pipeline(ROOT / "configs/pipeline.json")
        manifest_source = "bundled preview; Hub pipeline is resolved before preparation" if project_needed else "bundled pipeline"
    manifest = pipeline["external_models"]
    operations, models = [], {}
    subdirs = {name: relative_name(getattr(args, name + "_subdir") or pipeline["models"][name])
               for name in ("completion", "material")}
    patterns = {"completion": COMPLETION_PATTERNS, "material": MATERIAL_PATTERNS}
    if project_needed and project["kind"] == "hub":
        operations.append(dict(project, name="project", destination="project",
                               allow_patterns=["pipeline.json", "README.md", ".gitattributes"] +
                               [f"{subdirs[name]}/{pattern}" for name in project_needed for pattern in patterns[name]]))
    for name in subdirs:
        value = getattr(args, name)
        if value is None:
            if project["kind"] == "local":
                models[name] = local_override(Path(project["path"]) / subdirs[name])
            else:
                models[name] = {"kind": "prepared", "path": f"project/{subdirs[name]}"}
        else:
            source = model_source(value, getattr(args, name + "_revision"))
            models[name] = source if source["kind"] == "local" else {"kind": "prepared", "path": name}
            if source["kind"] == "hub":
                operations.append(dict(source, name=name, destination=name, allow_patterns=patterns[name]))
    for key, flag in (("qwen-base", "qwen_base"), ("geometry", "geometry"), ("dino", "dino"),
                      ("vae", "vae"), ("moge2", "moge"), ("rmbg2", "rmbg")):
        override = getattr(args, flag)
        if key == "rmbg2" and override is None and not args.with_rmbg:
            continue
        if override is not None:
            models[key] = local_override(override, file_allowed=key == "moge2")
        else:
            entry = manifest[key]
            destination = f"external/{key}"
            operations.append({"name": key, "kind": "hub", "repo_id": entry["repo_id"],
                               "revision": entry["revision"], "allow_patterns": entry["allow_patterns"],
                               "gated": entry.get("gated", False), "destination": destination})
            models[key] = {"kind": "prepared", "path": destination}
    if args.naf_weights is not None:
        models["naf"] = local_override(args.naf_weights, file_allowed=True)
    else:
        entry = manifest["naf"]
        destination = "external/naf/" + entry["filename"]
        operations.append({"name": "naf", "kind": "url", "url": entry["url"], "destination": destination})
        models["naf"] = {"kind": "prepared", "path": destination}
    for operation in operations:
        if args.local_files_only:
            operation["local_files_only"] = True
        if args.cache_dir is not None and operation["kind"] == "hub":
            operation["cache_dir"] = str(args.cache_dir.expanduser().resolve())
    return {"schema_version": 1, "output_dir": str(output), "downloads": operations, "models": models,
            "cache_dir": str(args.cache_dir.expanduser().resolve()) if args.cache_dir is not None else None,
            "local_files_only": args.local_files_only,
            "pipeline": pipeline, "pipeline_source": manifest_source,
            "bindings": {"geometry": "geometry/pipeline.json + five config/weight pairs + dino",
                         "pipeline_source": "local geometry pipeline.json or deployment/runtime/geometry/pipeline.json",
                         "vae": "all four encoder/decoder config/weight pairs under upstream ckpts/",
                         "naf_source": "torch/hub/valeoai_NAF_main -> deployment/runtime/naf",
                         "naf_weights": "torch/hub/checkpoints/naf_release.pth",
                         "config": "models.json (generated paths and selected pipeline defaults)"},
            "access": "Gated repositories require your existing upstream permission and local HF authentication. HF_ENDPOINT is inherited.",
            "local_override_note": "Local overrides remain external dependencies; their absolute paths/links are intentionally machine-local."}


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required model file is missing or empty: {path}")
    return path


def check_index(path: Path) -> None:
    index = json.loads(require_file(path).read_text())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"checkpoint index requires a nonempty weight_map: {path}")
    for filename in set(mapping.values()):
        relative_name(filename)
        if not filename.endswith(".safetensors"):
            raise ValueError(f"index shard must be .safetensors: {filename}")
        require_file(path.parent / filename)


def check_safe_weights(directory: Path, single: str = "model.safetensors") -> Path:
    index = directory / (single + ".index.json")
    weights = directory / single
    if index.is_file() and weights.is_file():
        raise ValueError(f"ambiguous single and sharded weights: {directory}")
    if index.is_file():
        check_index(index)
        return index
    return require_file(weights)


def validate_inputs(paths: dict[str, Path]) -> tuple[dict, Path]:
    # Load the path-only helper directly: package __init__ imports GPU-facing
    # modules, and resolve_completion_artifacts performs deeper ABI validation.
    spec = importlib.util.spec_from_file_location("_supratex_prepare_weights", ROOT / "supratex/weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in ("completion", "material"):
        module.resolve_model_dir(paths[key], local_files_only=True)
    completion = json.loads(require_file(paths["completion"] / "config.json").read_text())
    if completion.get("format") != "supratex_completion" or completion.get("schema_version") != 1:
        raise ValueError("completion requires the standalone Supra-Tex config.json (schema_version 1)")
    require_file(paths["completion"] / "partial_normalization.json")
    checkpoint = check_safe_weights(paths["completion"])
    adapter = json.loads(require_file(paths["material"] / "adapter_config.json").read_text())
    if adapter.get("format") != "supratex_qwen_material_adapter" or adapter.get("weight_file") != "adapter_model.safetensors":
        raise ValueError("material requires the standalone Supra-Tex adapter_config.json and adapter_model.safetensors")
    require_file(paths["material"] / "adapter_model.safetensors")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        require_file(paths["qwen-base"] / "processor" / name)
    for name in ("vae", "text_encoder", "transformer"):
        component = paths["qwen-base"] / name
        require_file(component / "config.json")
        weights = list(component.glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError(f"Qwen foundation component has no weights: {component}")
        for weight in weights:
            require_file(weight)
        for index in component.glob("*.safetensors.index.json"):
            check_index(index)
    # The geometry extractors normalize images directly and load only DINO's
    # model/config; a Transformers image-processor file is not required.
    require_file(paths["dino"] / "config.json")
    check_safe_weights(paths["dino"])
    if "rmbg2" in paths:
        for name in ("config.json", "model.safetensors", "preprocessor_config.json", "BiRefNet_config.py", "birefnet.py"):
            require_file(paths["rmbg2"] / name)
    paths["moge2"] = require_file(paths["moge2"] / "model.pt" if paths["moge2"].is_dir() else paths["moge2"])
    paths["naf"] = require_file(paths["naf"] / "naf_release.pth" if paths["naf"].is_dir() else paths["naf"])
    vae = paths["vae"] / "ckpts" if (paths["vae"] / "ckpts").is_dir() else paths["vae"]
    for stem in VAE_STEMS:
        for suffix in (".json", ".safetensors"):
            require_file(vae / (stem + suffix))
    pipeline_file = paths["geometry"] / "pipeline.json"
    if not pipeline_file.is_file():
        pipeline_file = _pipeline.runtime_root(ROOT) / "geometry/pipeline.json"
    pipeline = json.loads(require_file(pipeline_file).read_text())
    for key in GEOMETRY_KEYS:
        stem = relative_name(pipeline["args"]["models"][key])
        for suffix in (".json", ".safetensors"):
            require_file(paths["geometry"] / (stem + suffix))
    return pipeline, checkpoint


def write_once(path: Path, text: str) -> None:
    """Accept identical products on retry, never overwrite an existing file."""
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text() != text:
            raise FileExistsError(f"refusing to overwrite conflicting output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2) + "\n"


def link_once(destination: Path, source: Path, *, relative: bool) -> None:
    target = os.path.relpath(source, destination.parent) if relative else str(source)
    if destination.is_symlink() and os.readlink(destination) == target:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite conflicting output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(target, target_is_directory=source.is_dir())


def bind_cached_snapshot(destination: Path, cached: Path, patterns: list[str]) -> None:
    """Fill missing cached files only after checking every existing destination."""
    if not destination.exists() and not destination.is_symlink():
        link_once(destination, cached, relative=False)
        return
    if destination.is_symlink():
        if destination.resolve() == cached.resolve():
            return
    if not destination.is_dir():
        raise FileExistsError(f"refusing to overwrite conflicting output: {destination}")
    missing = []
    for source in sorted(cached.rglob("*")):
        relative = source.relative_to(cached)
        if not source.is_file() or not any(fnmatch.fnmatch(relative.as_posix(), pattern) for pattern in patterns):
            continue
        target = destination / relative
        parent = target.parent
        while parent != destination:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise FileExistsError(f"refusing to modify conflicting output: {parent}")
            parent = parent.parent
        if target.exists() or target.is_symlink():
            if not target.is_file() or (not source.samefile(target) and not filecmp.cmp(source, target, shallow=False)):
                raise FileExistsError(f"refusing to overwrite conflicting output: {target}")
        else:
            if destination.is_symlink():
                raise FileExistsError(f"refusing to modify a conflicting cache binding: {destination}")
            missing.append((target, source))
    for target, source in missing:
        link_once(target, source, relative=False)


def download(operation: dict, output: Path) -> None:
    destination = output / operation["destination"]
    if operation["kind"] == "url":
        if destination.exists():
            require_file(destination)
            return
        if operation.get("local_files_only"):
            raise FileNotFoundError("NAF weights are unavailable offline; supply --naf-weights or prepare them online first")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pth.part")
        try:
            with urlopen(operation["url"], timeout=60) as response, temporary.open("wb") as target:
                expected_size = response.headers.get("Content-Length")
                shutil.copyfileobj(response, target)
            require_file(temporary)
            if expected_size is not None and temporary.stat().st_size != int(expected_size):
                raise ValueError("NAF response ended before its declared Content-Length")
            # A same-directory hard link publishes without replacing a file.
            os.link(temporary, destination)
        except Exception as error:
            raise RuntimeError("NAF download failed; retry or supply --naf-weights with the official naf_release.pth") from error
        finally:
            temporary.unlink(missing_ok=True)
        return
    from huggingface_hub import snapshot_download
    try:
        if operation.get("local_files_only"):
            try:
                cached = snapshot_download(repo_id=operation["repo_id"], revision=operation["revision"],
                                           allow_patterns=operation["allow_patterns"],
                                           cache_dir=operation.get("cache_dir"), local_files_only=True)
            except Exception:
                if destination.is_dir():
                    # Online local_dir downloads may have no shared cache entry.
                    # The caller validates all required files before publishing.
                    return
                raise
        elif operation.get("cache_dir") is not None:
            # local_dir bypasses the shared cache in huggingface_hub, so an
            # explicit cache must use a snapshot and bind its files afterwards.
            cached = snapshot_download(repo_id=operation["repo_id"], revision=operation["revision"],
                                       allow_patterns=operation["allow_patterns"],
                                       cache_dir=operation["cache_dir"])
        else:
            snapshot_download(repo_id=operation["repo_id"], revision=operation["revision"],
                              allow_patterns=operation["allow_patterns"], local_dir=str(destination),
                              cache_dir=operation.get("cache_dir"))
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        access = operation.get("gated") or status in (401, 403) or "GatedRepo" in type(error).__name__
        detail = ("This upstream repository requires existing access permission and your own local HF authentication. "
                  "Accept its publisher's terms and authenticate with your normal HF tooling, or provide a local override."
                  if access else "Check the repository/revision and your network or local HF authentication, then retry.")
        # Do not interpolate exception text: upstream URLs can contain secrets.
        raise RuntimeError(f"Download failed for {operation['repo_id']}. {detail}") from None
    if operation.get("local_files_only") or operation.get("cache_dir") is not None:
        # Keep local conflict errors distinct from remote access failures.
        bind_cached_snapshot(destination, Path(cached), operation["allow_patterns"])


def plan_identity(plan: dict) -> dict:
    """Transport/cache choices are not part of the prepared model identity."""
    identity = {key: copy.deepcopy(value) for key, value in plan.items()
                if key not in {"output_dir", "pipeline_source", "cache_dir", "local_files_only"}}
    identity["downloads"] = [{key: value for key, value in item.items()
                              if key not in {"local_files_only", "cache_dir"}}
                             for item in plan["downloads"]]
    return identity


def without_rmbg(identity: dict) -> dict:
    identity = copy.deepcopy(identity)
    identity["models"].pop("rmbg2", None)
    identity["downloads"] = [item for item in identity["downloads"] if item["name"] != "rmbg2"]
    return identity


def replace_owned_json(path: Path, value: dict, previous: dict, *, check_only: bool = False) -> None:
    """Atomically extend a generated file, rejecting user edits and symlinks."""
    text = json_text(value)
    if not path.exists() and not path.is_symlink():
        if not check_only:
            write_once(path, text)
        return
    if path.is_symlink() or not path.is_file() or path.read_text() not in {json_text(previous), text}:
        raise FileExistsError(f"refusing to overwrite conflicting output: {path}")
    if check_only or path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(plan: dict) -> Path:
    plan = copy.deepcopy(plan)
    output = Path(plan["output_dir"])
    marker = output / MARKER
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.is_dir() and any(output.iterdir()) and not marker.is_file():
        raise FileExistsError(f"output directory is not owned by prepare_models.py; choose an empty --output-dir: {output}")
    previous = None
    identity = plan_identity(plan)
    if marker.exists():
        if marker.is_symlink():
            raise FileExistsError(f"refusing to overwrite conflicting output: {marker}")
        previous = json.loads(marker.read_text())
        if without_rmbg(previous) != without_rmbg(identity):
            raise FileExistsError(f"refusing to overwrite conflicting output: {marker}")
        if "rmbg2" in previous["models"]:
            old_downloads = [item for item in previous["downloads"] if item["name"] == "rmbg2"]
            if "rmbg2" in identity["models"]:
                new_downloads = [item for item in identity["downloads"] if item["name"] == "rmbg2"]
                if identity["models"]["rmbg2"] != previous["models"]["rmbg2"] or old_downloads != new_downloads:
                    raise FileExistsError(f"refusing to overwrite conflicting output: {marker}")
            else:
                # Keep an existing optional dependency when switching to RGBA.
                plan["models"] = {**previous["models"], **plan["models"]}
                for operation in old_downloads:
                    operation = copy.deepcopy(operation)
                    if plan.get("local_files_only") or any(item.get("local_files_only") for item in plan["downloads"]):
                        operation["local_files_only"] = True
                    if plan.get("cache_dir") is not None:
                        operation["cache_dir"] = plan["cache_dir"]
                    position = next(index for index, item in enumerate(previous["downloads"]) if item["name"] == "rmbg2")
                    plan["downloads"].insert(position, operation)
                identity = plan_identity(plan)
    else:
        write_once(marker, json_text(identity))
    paths = {name: Path(source["path"]) if source["kind"] == "local" else output / source["path"]
             for name, source in plan["models"].items()}
    existing_inputs_complete = False
    if any(operation.get("local_files_only") for operation in plan["downloads"]):
        try:
            validate_inputs(dict(paths))
            existing_inputs_complete = True
        except (OSError, ValueError, KeyError):
            pass
    # Check publisher-controlled access before downloading the large public bases.
    for operation in sorted(plan["downloads"], key=lambda item: not item.get("gated", False)):
        if operation.get("local_files_only") and existing_inputs_complete:
            continue
        download(operation, output)
    pipeline, checkpoint = validate_inputs(paths)
    geometry = output / "geometry"
    for key in GEOMETRY_KEYS:
        stem = pipeline["args"]["models"][key]
        write_once(geometry / (stem + ".json"), (paths["geometry"] / (stem + ".json")).read_text())
        link_once(geometry / (stem + ".safetensors"), paths["geometry"] / (stem + ".safetensors"),
                  relative=plan["models"]["geometry"]["kind"] != "local")
    write_once(geometry / "pipeline.json", json_text(pipeline))
    link_once(geometry / "dino", paths["dino"], relative=plan["models"]["dino"]["kind"] != "local")
    runtime = _pipeline.runtime_root(ROOT)
    require_file(runtime / "naf/hubconf.py")
    link_once(output / "torch/hub/valeoai_NAF_main", runtime / "naf", relative=True)
    link_once(output / "torch/hub/checkpoints/naf_release.pth", paths["naf"], relative=plan["models"]["naf"]["kind"] != "local")

    def config_path(path: Path, local: bool = False) -> str:
        return str(path) if local else os.path.relpath(path, output)

    def model_path(name: str) -> str:
        return config_path(paths[name], plan["models"][name]["kind"] == "local")

    local_names = [key for key, value in plan["models"].items() if value["kind"] == "local"]
    config = {"moge_model": model_path("moge2"), "rmbg_model": model_path("rmbg2") if "rmbg2" in paths else None,
              "geometry_package": config_path(runtime / "geometry"), "geometry_model": "geometry",
              "qwen_runtime": config_path(runtime / "qwen"),
              "material_base": model_path("qwen-base"), "material_model": model_path("material"),
              "pretrained": model_path("completion"), "torch_home": "torch", "trellis_root": model_path("vae"),
              "model_config": config_path(paths["completion"] / "config.json", "completion" in local_names),
              "checkpoint": config_path(checkpoint, "completion" in local_names),
              "defaults": plan["pipeline"]["defaults"],
              "_meta": {"schema_version": 1, "local_overrides": local_names,
                        "portability": "Relative paths resolve from models.json. Move with the source package; local overrides remain external dependencies.",
                        "validation": "File presence and shard inventory only; run scripts/check_runtime.py before inference."}}
    result = output / "models.json"
    if "rmbg2" in identity["models"]:
        old_config = copy.deepcopy(config)
        old_config["rmbg_model"] = None
        old_config["_meta"]["local_overrides"] = [name for name in old_config["_meta"]["local_overrides"] if name != "rmbg2"]
        # Check for user edits before recording the extension. Recording the
        # dependency first lets either input type recover an interrupted publish.
        replace_owned_json(result, config, old_config, check_only=True)
        if previous is not None and previous != identity:
            replace_owned_json(marker, identity, previous)
        replace_owned_json(result, config, old_config)
    else:
        write_once(result, json_text(config))
    return result


def prepare_pipeline(source: str | Path, *, output_dir: str | Path | None = None,
                     revision: str | None = None, cache_dir: str | Path | None = None,
                     local_files_only: bool = False,
                     with_rmbg: bool = False, overrides: dict | None = None) -> Path:
    """Resolve a release and generate the runtime bindings used by inference."""
    source = Path(source).expanduser().absolute() if isinstance(source, Path) else source
    # A manifest filename denotes its release root as well as the selected config.
    local = Path(source).expanduser()
    manifest_file = local.is_file() or (not str(source).startswith(("https://", "hf://")) and local.suffix == ".json")
    argv = ["--repo", str(local.absolute().parent if manifest_file else source)]
    if manifest_file:
        argv.extend(("--pipeline", str(local)))
    if output_dir is not None:
        argv.extend(("--output-dir", str(output_dir)))
    if revision is not None:
        argv.extend(("--revision", revision))
    if cache_dir is not None:
        argv.extend(("--cache-dir", str(cache_dir)))
    if local_files_only:
        argv.append("--local-files-only")
    if with_rmbg:
        argv.append("--with-rmbg")
    flags = {"completion": "completion", "material": "material", "qwen-base": "qwen-base",
             "geometry": "geometry", "dino": "dino", "vae": "vae", "moge2": "moge",
             "rmbg2": "rmbg", "naf": "naf-weights",
             "completion_revision": "completion-revision", "material_revision": "material-revision"}
    for key, value in (overrides or {}).items():
        if key not in flags:
            raise ValueError(f"unknown pipeline model override: {key}")
        argv.extend(("--" + flags[key], str(value)))
    return prepare(build_plan(parse_args(argv), resolve_remote=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args, resolve_remote=not args.dry_run)
        if args.dry_run:
            print(json_text(plan), end="")
        else:
            result = prepare(plan)
            print(f"Prepared {result}")
            if any(value["kind"] == "local" for value in plan["models"].values()):
                print("Local overrides remain external dependencies; keep their original paths available.")
            print(f"Next: python scripts/check_runtime.py --config {result}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

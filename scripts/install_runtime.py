#!/usr/bin/env python3
"""Install the recorded Linux/Python 3.10/CUDA 12.4 runtime into Conda or a venv.

No model weights are downloaded. Native sources come from the recorded public
revisions in deployment/docs/DEPENDENCIES.md. Run --dry-run to inspect commands
without creating directories, invoking subprocesses, or accessing the network.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git", "ef40f3ae4612ea3ec934b4151e019dc95c5d276b", ""),
    ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", "b1965ad0ae4bc8488d13b5280afb54439ce27359", ""),
    ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "253ac4fcea7de5f396371124af597e6cc957bfae", ""),
    ("TRELLIS.2", "https://github.com/microsoft/TRELLIS.2.git", "5565d240c4a494caaf9ece7a554542b76ffa36d3", "o-voxel"),
)
O_VOXEL_FLAGS = " ".join((
    "--extended-lambda", "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
))
CONSTRAINTS = "torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\n"
BUILD_TOOLS = ("pip==25.1.1", "setuptools==80.9.0", "wheel==0.45.1", "packaging==25.0")
# Official https://whl.natten.org/ index links this exact 0.17.5 release asset.
# The former shi-labs.com wheel index has an expired TLS certificate.
NATTEN_WHEEL = (
    "https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.5/"
    "natten-0.17.5%2Btorch260cu124-cp310-cp310-linux_x86_64.whl"
)
NATTEN_VERSION = "0.17.5+torch260cu124"
PYTHON_PROBE = (
    "import json,sys; "
    "print(json.dumps({'version':list(sys.version_info[:2]),"
    "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
)
TORCH_PROBE = (
    "import torch,torchvision; "
    "assert torch.__version__ == '2.6.0+cu124', torch.__version__; "
    "assert torchvision.__version__ == '0.21.0+cu124', torchvision.__version__; "
    "assert torch.version.cuda == '12.4', torch.version.cuda; "
    "assert torch.cuda.is_available(), 'No usable NVIDIA CUDA GPU/driver'; "
    "torch.empty(1, device='cuda'); torch.cuda.synchronize(); "
    "print('Torch 2.6.0 / CUDA 12.4 GPU check passed')"
)


class InstallError(RuntimeError):
    """An actionable installation failure, with no automatic destructive repair."""


def clean_environment(venv: Path, *, conda: bool = False) -> dict[str, str]:
    """Do not inherit Python overlays or pip destinations from the invoking shell."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("PIP_") and key not in {
            "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV",
            "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
        }
    }
    env.update({
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_REQUIRE_VIRTUALENV": "false" if conda else "true",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CACHE_DIR": str(venv / ".cache" / "pip"),
        "TORCH_EXTENSIONS_DIR": str(venv / ".cache" / "torch_extensions"),
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(venv / ".cache" / "flex_gemm" / "autotune_cache.json"),
        "ATTN_BACKEND": "sdpa",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "BUILD_TARGET": "cuda",
    })
    if conda:
        env["CONDA_PREFIX"] = str(venv)
        if "CONDA_DEFAULT_ENV" in os.environ:
            env["CONDA_DEFAULT_ENV"] = os.environ["CONDA_DEFAULT_ENV"]
    return env


def capture(command: list[str], env: dict[str, str]) -> str:
    try:
        return subprocess.check_output(command, env=env, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", str(exc))
        raise InstallError(f"Command failed: {shlex.join(command)}\n{detail}") from exc


def validate_venv_destination(venv: Path) -> None:
    if venv in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        raise InstallError("--venv cannot be the invoking Python environment. Choose a separate directory.")
    if not venv.exists():
        return
    config = venv / "pyvenv.cfg"
    if not config.is_file() or not (venv / "bin" / "python").is_file():
        raise InstallError(f"Refusing to overwrite {venv}: existing destination is not a complete venv. Choose a new --venv.")
    settings = dict(
        line.lower().strip().split("=", 1)
        for line in config.read_text().splitlines() if "=" in line
    )
    settings = {key.strip(): value.strip() for key, value in settings.items()}
    if settings.get("include-system-site-packages") != "false":
        raise InstallError(f"{venv} must have include-system-site-packages = false. Choose a new --venv.")


def active_conda_environment() -> Path:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise InstallError("Activate a dedicated Conda environment first: conda activate supra-tex.")
    destination = Path(prefix).expanduser().resolve()
    conda_exe = os.environ.get("CONDA_EXE")
    if os.environ.get("CONDA_DEFAULT_ENV") == "base" or (
        conda_exe and Path(conda_exe).resolve().parent.parent == destination
    ):
        raise InstallError("Create and activate a dedicated Conda environment; do not install into base.")
    if (Path(sys.prefix).resolve() != destination
            or Path(sys.base_prefix).resolve() != destination):
        raise InstallError("Run this script with the active Conda environment's Python, without a nested venv.")
    if not (destination / "conda-meta").is_dir() or not (destination / "bin/python").is_file():
        raise InstallError(f"{destination} is not a complete Conda Python environment.")
    return destination


def validate_target_python(info: dict, destination: Path, *, conda: bool = False) -> None:
    valid = info.get("version") == [3, 10] and Path(info["prefix"]).resolve() == destination
    if conda:
        valid = valid and Path(info["base_prefix"]).resolve() == destination
    else:
        valid = valid and info["prefix"] != info["base_prefix"]
    if not valid:
        raise InstallError(f"{destination} is not the expected Python 3.10 {'Conda environment' if conda else 'isolated venv'}.")


def prerequisites(python: str, venv: Path, env: dict[str, str], *, conda: bool = False) -> dict[str, str]:
    """Validate before any directory creation, download, or package installation."""
    if platform.system() != "Linux":
        raise InstallError("This recorded native runtime requires Linux.")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise InstallError("The pinned NATTEN 0.17.5 wheel requires Linux x86_64.")
    if not conda:
        validate_venv_destination(venv)
    try:
        probe = PYTHON_PROBE if conda else "import venv,ensurepip; " + PYTHON_PROBE
        info = json.loads(capture([python, "-I", "-c", probe], env))
    except (ValueError, KeyError) as exc:
        raise InstallError("Could not determine the selected Python version.") from exc
    if info.get("version") != [3, 10]:
        hint = "Create a Conda environment with python=3.10." if conda else "Use --python /path/to/python3.10."
        raise InstallError(f"Python 3.10 is required; {python} reports {info.get('version')}. {hint}")
    if conda:
        validate_target_python(info, venv, conda=True)
    elif Path(info["prefix"]).resolve() == venv:
        raise InstallError("--venv cannot be the selected --python environment. Choose a separate directory.")
    if not shutil.which("git", path=env.get("PATH")):
        raise InstallError("git is missing from PATH; install git before rerunning.")
    compiler = shlex.split(env.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0], path=env.get("PATH")):
        raise InstallError("A C++ compiler is required; install g++/c++ or set CXX to a working compiler.")
    capture([*compiler, "--version"], env)
    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    nvcc = str(Path(cuda_home).expanduser() / "bin" / "nvcc") if cuda_home else shutil.which("nvcc", path=env.get("PATH"))
    if not nvcc or not Path(nvcc).is_file():
        raise InstallError("CUDA 12.4 nvcc is required. Install the CUDA toolkit and set CUDA_HOME or add its bin directory to PATH.")
    version = capture([nvcc, "--version"], env)
    if not re.search(r"release\s+12\.4(?:\D|$)", version):
        raise InstallError(f"CUDA toolkit 12.4 is required; {nvcc} reports:\n{version}")
    result = env.copy()
    result["CUDA_HOME"] = str(Path(cuda_home).expanduser().resolve() if cuda_home else Path(nvcc).resolve().parent.parent)
    result["PATH"] = os.pathsep.join((str(venv / "bin"), str(Path(result["CUDA_HOME"]) / "bin"), env.get("PATH", "")))
    return result


def validate_checkout(path: Path, url: str, revision: str, env: dict[str, str]) -> None:
    """Reuse only the exact, clean checkout; never reset or clean user sources."""
    if not (path / ".git").exists():
        raise InstallError(f"Refusing to overwrite {path}: not a Git checkout. Choose a fresh environment.")
    git = ["git", "-C", str(path)]
    top = capture([*git, "rev-parse", "--show-toplevel"], env)
    origin = capture([*git, "remote", "get-url", "origin"], env)
    head = capture([*git, "rev-parse", "HEAD"], env)
    dirty = capture([*git, "status", "--porcelain", "--untracked-files=no"], env)
    if Path(top).resolve() != path.resolve() or origin.rstrip("/") != url or head != revision or dirty:
        raise InstallError(
            f"Refusing to alter existing checkout {path}: expected origin {url}, HEAD {revision}, "
            "and no tracked changes (including submodules). Repair it manually or choose a fresh environment."
        )


def run(command: list[str], env: dict[str, str], dry_run: bool, extra_env: dict[str, str] | None = None) -> None:
    display = ["env", *(f"{key}={value}" for key, value in extra_env.items()), *command] if extra_env else command
    print("$ " + shlex.join(display), flush=True)
    if not dry_run:
        child_env = dict(env, **(extra_env or {}))
        try:
            subprocess.run(command, env=child_env, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise InstallError(f"Installation stopped at: {shlex.join(command)}\n{exc}\nExisting files were kept; no rollback or deletion was attempted.") from exc


def install_natten(pip: list[str], env: dict[str, str], dry_run: bool) -> None:
    # A direct wheel URL otherwise downloads again on each resumed installation.
    # Query only the target environment; the final import checks still verify its ABI.
    if dry_run:
        print(f"# Skip NATTEN download only if this environment already has distribution version {NATTEN_VERSION}; final import checks still run.")
    else:
        probe = (
            "from importlib.metadata import version, PackageNotFoundError\n"
            "try:\n    print(version('natten'))\n"
            "except PackageNotFoundError:\n    print('')\n"
        )
        installed = capture([pip[0], "-I", "-c", probe], env)
        if installed == NATTEN_VERSION:
            print(f"# NATTEN {NATTEN_VERSION} already installed in target environment; skipping download.")
            return
    run([*pip, "--no-deps", "--no-index", "--only-binary=:all:", NATTEN_WHEEL], env, dry_run)


def install_flex_gemm(pip: list[str], source: Path, managed: Path,
                      env: dict[str, str], dry_run: bool) -> None:
    """Redirect the pinned setup.py cache write without altering its checkout."""
    print(f"# FlexGEMM: build a temporary source copy under {managed}; "
          f"install and runtime cache: {env['FLEX_GEMM_AUTOTUNE_CACHE_PATH']}.")
    command = [*pip, "--no-build-isolation", "--no-deps"]
    if dry_run:
        run([*command, str(managed / "flex-gemm-build-<temporary>" / "FlexGEMM")], env, True)
        return
    old = '''os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)
shutil.copyfile(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
    os.path.expanduser('~/.flex_gemm/autotune_cache.json'),
)'''
    new = '''cache_path = os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"]
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
shutil.copyfile(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
    cache_path,
)'''
    setup = (source / "setup.py").read_text()
    if setup.count(old) != 1:
        raise InstallError("FlexGEMM setup.py does not match the pinned cache-write block; refusing an unverified patch.")
    with tempfile.TemporaryDirectory(prefix="flex-gemm-build-", dir=managed) as temporary:
        build_source = Path(temporary) / "FlexGEMM"
        shutil.copytree(source, build_source, ignore=shutil.ignore_patterns(".git"))
        (build_source / "setup.py").write_text(setup.replace(old, new))
        run([*command, str(build_source)], env, False)


def install(args: argparse.Namespace) -> None:
    venv = active_conda_environment() if args.conda else Path(args.venv).expanduser().resolve()
    selected_python = args.python or sys.executable
    python = shutil.which(selected_python) or str(Path(selected_python).expanduser().absolute())
    env = clean_environment(venv, conda=args.conda)
    managed = venv / "supra-tex-runtime"
    sources = managed / "sources"
    constraints = managed / "constraints.txt"
    target_python = str(venv / "bin" / "python")
    if args.conda:
        python = target_python
    pip = [target_python, "-I", "-m", "pip", "install", "--no-input"]
    dry = args.dry_run
    print(f"{'DRY RUN: no subprocesses, downloads, or writes.' if dry else 'Installing runtime.'}\nSource: {ROOT}\n{'Conda environment' if args.conda else 'Virtual environment'}: {venv}")
    print("Prerequisites: Linux x86_64; Python 3.10; git; C++ compiler; CUDA toolkit 12.4 nvcc; usable NVIDIA GPU/driver.")
    print("Child processes ignore inherited Python overlays and pip configuration/destinations; caches stay in the target environment. Default attention is SDPA; FlashAttention is optional.")
    print("Native source pins identify upstream inputs, not verified equivalent binaries. FlexGEMM's install-time cache path is patched in a temporary copy; its original checkout stays unchanged.")
    if not dry:
        for required in (ROOT / "deployment/runtime/requirements-python.txt", ROOT / "deployment/runtime/moge/pyproject.toml", ROOT / "deployment/runtime/utils3d/pyproject.toml", ROOT / "scripts/check_runtime.py"):
            if not required.is_file():
                raise InstallError(f"Incomplete source release: missing {required}")
        env = prerequisites(python, venv, env, conda=args.conda)
        # Check all existing source trees before changing this environment.
        for name, url, revision, _ in SOURCES:
            path = sources / name
            if path.exists():
                validate_checkout(path, url, revision, env)
    if args.conda:
        print("# Install into the active Conda environment.")
    elif not venv.exists():
        run([python, "-I", "-m", "venv", str(venv)], env, dry)
    else:
        print("# Reuse only an isolated Python 3.10 venv; never clear an existing directory.")
    if not dry:
        info = json.loads(capture([target_python, "-I", "-c", PYTHON_PROBE], env))
        validate_target_python(info, venv, conda=args.conda)
        managed.mkdir(parents=True, exist_ok=True)
        sources.mkdir(exist_ok=True)
        constraints.write_text(CONSTRAINTS)
    print(f"# Write {constraints}: torch==2.6.0+cu124; torchvision==0.21.0+cu124")
    run([*pip, *BUILD_TOOLS], env, dry)
    run([*pip, "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124"], env, dry)
    run([*pip, "--constraint", str(constraints), "-r", str(ROOT / "deployment/runtime/requirements-python.txt")], env, dry)
    run([*pip, "--no-build-isolation", "--no-deps", str(ROOT / "deployment/runtime/moge"), str(ROOT / "deployment/runtime/utils3d")], env, dry)
    run([*pip, "--no-build-isolation", "--no-deps", "-e", str(ROOT)], env, dry)
    install_natten(pip, env, dry)
    run([target_python, "-I", "-c", TORCH_PROBE], env, dry)
    for name, url, revision, subdir in SOURCES:
        path = sources / name
        if not path.exists():
            run(["git", "clone", "--no-checkout", url, str(path)], env, dry)
            run(["git", "-C", str(path), "checkout", "--detach", revision], env, dry)
        else:
            print(f"# Verify origin={url}, HEAD={revision}, and clean tracked files: {path}; no reset/clean.")
        run(["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], env, dry)
        flags = None
        if name == "CuMesh":
            print("# CuMesh/cubvh: append --extended-lambda for its CUDA device lambdas without patching source.")
            flags = {"NVCC_APPEND_FLAGS": " ".join(filter(None, (env.get("NVCC_APPEND_FLAGS"), "--extended-lambda")))}
        elif subdir == "o-voxel":
            print("# o-voxel: NVCC_APPEND_FLAGS appends the recorded extended-lambda/half flags without patching source. This build still requires destination validation.")
            flags = {"NVCC_APPEND_FLAGS": " ".join(filter(None, (env.get("NVCC_APPEND_FLAGS"), O_VOXEL_FLAGS)))}
        if name == "FlexGEMM":
            install_flex_gemm(pip, path, managed, env, dry)
        else:
            run([*pip, "--no-build-isolation", "--no-deps", str(path / subdir)], env, dry, flags)
    if args.flash_attn:
        run([*pip, "--no-build-isolation", "--no-deps", "flash-attn==2.8.3"], env, dry)
        run([target_python, "-I", "-c", "import flash_attn; print(flash_attn.__version__)"], env, dry)
    run([target_python, "-I", "-m", "pip", "check"], env, dry)
    run([target_python, "-I", "-c", TORCH_PROBE], env, dry)
    run([target_python, "-I", str(ROOT / "scripts/check_runtime.py"), "--imports-only"], env, dry)
    if dry:
        print("Dry run complete. Prerequisites, builds, and imports have NOT been validated.")
    else:
        print(f"Runtime dependency checks passed in {venv}. Model acquisition and an image-to-GLB run are still required for inference acceptance. Use --attention sdpa for the default runtime.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--conda", action="store_true", help="Install into the active Conda environment (recommended; activate it first)")
    destination.add_argument("--venv", default=".venv", help="Create/reuse an isolated venv instead (default without --conda: .venv)")
    parser.add_argument("--python", help="Python 3.10 executable used to create a venv (default: invoking Python; unavailable with --conda)")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete plan without subprocesses, network, or filesystem writes")
    parser.add_argument("--flash-attn", action="store_true", help="Also build optional FlashAttention 2.8.3; default SDPA needs neither FA2 nor FA3")
    args = parser.parse_args(argv)
    if args.conda and args.python:
        parser.error("--python cannot be combined with --conda; use the active Conda environment's Python.")
    try:
        install(args)
    except (InstallError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

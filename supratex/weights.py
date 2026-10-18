"""Resolve model directories from local storage or the Hugging Face Hub."""

from pathlib import Path


# Default completion weights inside a standalone release (RAW step 30000). The
# release also keeps the earlier sharded model, which is used only as a fallback.
DEFAULT_COMPLETION_CHECKPOINT = "denoiser_step0030000.safetensors"
COMPLETION_ALLOW_PATTERNS = [
    "config.json", "partial_normalization.json", DEFAULT_COMPLETION_CHECKPOINT,
    "configs/*.json", "weights/material_completion.pt", "weights/sparse_encoders/*.json",
    "weights/scvae/*.json", "vae/*.json", "vae/ckpts/*.json",
]


def resolve_completion_artifacts(
    root: str | Path | None = None,
    *,
    model_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    trellis_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve and preflight a local completion release plus its sparse VAEs.

    Standalone releases contain root config/normalization and safe weights;
    their separately obtained VAEs live at ``trellis_root`` (or ``root/vae``).
    The original configs/weights bundle layout remains supported unchanged.
    """
    if root is not None:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"completion bundle is not a directory: {root}")
        standalone = (root / "config.json").is_file()
        if model_config is None:
            model_config = root / ("config.json" if standalone else "configs/inference_model.json")
        if checkpoint is None:
            if standalone and (root / DEFAULT_COMPLETION_CHECKPOINT).is_file():
                checkpoint = root / DEFAULT_COMPLETION_CHECKPOINT
            elif standalone:
                candidates = [root / name for name in ("model.safetensors", "model.safetensors.index.json")]
                existing = [path for path in candidates if path.is_file()]
                if len(existing) > 1:
                    raise ValueError("completion bundle has both single and sharded weights; select --checkpoint explicitly")
                if not existing:
                    raise FileNotFoundError(f"missing completion checkpoint: expected {DEFAULT_COMPLETION_CHECKPOINT}, "
                                            f"model.safetensors, or model.safetensors.index.json in {root}")
                checkpoint = existing[0]
            else:
                checkpoint = root / "weights/material_completion.pt"
        if trellis_root is None:
            trellis_root = root / ("vae" if standalone else "weights/sparse_encoders")
    if model_config is None or checkpoint is None or trellis_root is None:
        raise ValueError("provide a completion bundle or model_config, checkpoint, and trellis_root")
    # Snapshot files are symlinks to opaque cache blobs. Keep their logical
    # directory so config-relative metadata and index-relative shards resolve.
    paths = tuple(Path(path).expanduser().absolute() for path in (model_config, checkpoint, trellis_root))
    from .inference.pipeline import validate_completion_artifacts
    validate_completion_artifacts(*paths)
    return paths


def resolve_model_dir(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Return an existing directory for a local path or ``namespace/repo``.

    Local directories take precedence over Hub IDs. Use ``hf://namespace/repo``
    to identify a Hub source explicitly; use a ``Path`` or an explicit path
    prefix to ensure a missing local directory never triggers a download.
    """
    if not isinstance(source, (str, Path)):
        raise TypeError("Model source must be a local path or a Hugging Face repo ID")
    if source == "":
        raise ValueError("Invalid model source: expected a nonempty path or namespace/repo")

    local_path = Path(source).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise NotADirectoryError(f"Model source is not a directory: {local_path}")
        return local_path.resolve()

    if isinstance(source, Path) or local_path.is_absolute() or source.startswith(
        ("./", "../", "~/")
    ) or source in (".", "..", "~"):
        raise FileNotFoundError(f"Model directory does not exist: {local_path}")

    repo_id = source.removeprefix("hf://")
    # Keep Hub imports off the local loading path (and lightweight CLI help).
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(repo_id)
        if repo_id.count("/") != 1:
            raise ValueError("expected namespace/repo")
    except (HFValidationError, ValueError) as exc:
        raise ValueError(
            f"Invalid model source {source!r}: expected an existing local directory "
            "or a Hugging Face repo ID in namespace/repo form"
        ) from exc

    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    ).expanduser()
    if not downloaded.exists():
        raise FileNotFoundError(f"Downloaded model directory does not exist: {downloaded}")
    if not downloaded.is_dir():
        raise NotADirectoryError(f"Downloaded model source is not a directory: {downloaded}")
    return downloaded.resolve()

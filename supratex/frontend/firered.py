"""FireRed Image-to-Material Translation adapter.

The supplied package remains the sole FireRed implementation: this adapter
creates its documented JSONL item, runs its public ``infer.py``, and fixes
the decoded MR blue channel to opaque for the 3D material encoder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def item_for(condition: Path, key: str) -> dict[str, object]:
    """Build the public package record without inventing material supervision."""
    condition = condition.resolve()
    return {
        "key": key, "dataset": "user", "sha": key, "asset_id": key, "view": 0, "cond": str(condition),
        # The public package validates these schema slots but prediction only
        # consumes ``cond``.  Point them at the supplied condition so there is
        # no fabricated or private GT material file in an inference request.
        "albedo_gt": str(condition), "mra_gt": str(condition), "normal_gt": str(condition), "normal_valid": str(condition),
        "instr_albedo": "Convert this rendered image into its albedo (base color) map: remove all lighting, shading, shadows and highlights, keep only the intrinsic surface colors on a black background.",
        # Retain the checkpoint's native prompt; run() replaces its predicted
        # material alpha with the constant opaque channel used by completion.
        "instr_mra": "Convert this rendered image into its material property map: encode metallic in the red channel, roughness in the green channel, and material alpha in the blue channel, on a black background.",
        "instr_normal": "Convert this rendered image into its camera-space surface normal map: encode the normal direction at every pixel as RGB with x pointing right, y pointing up and z toward the camera, on a flat gray background.", "split": "eval",
    }


def write_manifest(condition: Path, path: Path, *, key: str = "input") -> Path:
    if not condition.is_file():
        raise FileNotFoundError(condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item_for(condition, key), sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(package: Path, base_model: str | Path, condition: Path, output: Path, manifest: Path | None = None,
        *, key: str = "input", weights: Path | None = None, spec: Path | None = None,
        checkpoint_manifest: Path | None = None, material_model: str | Path | None = None,
        revision: str | None = None, cache_dir: Path | None = None,
        local_files_only: bool = False) -> tuple[Path, Path]:
    if not key or Path(key).name != key or key in {".", ".."} or "\\" in key:
        raise ValueError("key must be a single safe filename component")
    from supratex.weights import resolve_model_dir
    if material_model is not None:
        if any(value is not None for value in (weights, spec, checkpoint_manifest)):
            raise ValueError("--material-model cannot be combined with explicit material artifact paths")
        root = resolve_model_dir(material_model, revision=revision, cache_dir=cache_dir,
                                 local_files_only=local_files_only,
                                 allow_patterns=["full_transformer.safetensors", "joint_contract.json", "checkpoint_complete.json"])
        weights, spec, checkpoint_manifest = (root / name for name in
                                              ("full_transformer.safetensors", "joint_contract.json", "checkpoint_complete.json"))
        for path in (weights, spec, checkpoint_manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
    base_model = resolve_model_dir(base_model, cache_dir=cache_dir, local_files_only=local_files_only)
    package, condition, output = (p.resolve() for p in (package, condition, output))
    if manifest is not None:
        manifest = manifest.resolve()
    entry = package / "infer.py"
    if not entry.is_file():
        raise FileNotFoundError(f"FireRed public entrypoint is missing: {entry}")
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{key}_{channel}.png" for channel in ("albedo", "mra", "normal")]
    targets += [output / name for name in ("albedo.png", "mra.png", "firered_albedo.png", "firered_mra.png")]
    if any(path.exists() for path in targets):
        raise FileExistsError("material outputs already exist; select a fresh output directory")
    manifest = manifest or output / "firered_input.jsonl"
    if manifest == output / "firered_input.jsonl":
        write_manifest(condition, manifest, key=key)
    elif not manifest.is_file():
        raise FileNotFoundError(manifest)
    command = [sys.executable, str(entry), "--base-model", str(base_model), "--items-jsonl", str(manifest), "--out-dir", str(output)]
    bundled = package.resolve() == (Path(__file__).resolve().parent / "material_model").resolve()
    if bundled:
        if not all((weights, spec, checkpoint_manifest)):
            raise ValueError("bundled material model requires --weights, --spec, and --checkpoint-manifest")
        command += ["--weights", str(weights), "--spec", str(spec), "--checkpoint-manifest", str(checkpoint_manifest)]
    subprocess.run(command, check=True)
    albedo, mra = output / f"{key}_albedo.png", output / f"{key}_mra.png"
    if not albedo.is_file() or not mra.is_file():
        raise FileNotFoundError("FireRed did not emit the expected albedo and MRA maps")
    from PIL import Image
    for generated in (albedo, mra):
        with Image.open(generated) as image:
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise ValueError(f"material prediction must be RGB 1024x1024: {generated}")
    # Completion consumes (albedo RGB, metallic, roughness, opacity). Its MR
    # placeholder is constant 1; foreground coverage comes from a separate mask.
    with Image.open(mra) as image:
        fixed_mra = Image.merge("RGB", (image.getchannel("R"), image.getchannel("G"),
                                        Image.new("L", image.size, 255)))
    fixed_mra.save(mra)
    canonical_albedo, canonical_mra = output / "firered_albedo.png", output / "firered_mra.png"
    shutil.copyfile(albedo, canonical_albedo)
    shutil.copyfile(mra, canonical_mra)
    shutil.copyfile(albedo, output / "albedo.png")
    shutil.copyfile(mra, output / "mra.png")
    return canonical_albedo, canonical_mra


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FireRed Image-to-Material Translation through its packaged public entrypoint.")
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent / "material_model",
                        help="Bundled material-model package or a compatible package containing infer.py")
    parser.add_argument("--base-model", required=True, help="local FireRed base directory or Hugging Face namespace/repo")
    parser.add_argument("--material-model", help="local material-model directory or Hugging Face namespace/repo")
    parser.add_argument("--revision", help="material-model Hugging Face revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="Optional package-contract JSONL; omitted builds a validation-only local record.")
    parser.add_argument("--key", default="input")
    parser.add_argument("--weights", type=Path, help="External factorized material-model weights for the bundled package.")
    parser.add_argument("--spec", type=Path, help="External factorized material-model checkpoint contract.")
    parser.add_argument("--checkpoint-manifest", type=Path, help="External material-model completion manifest.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    albedo, mra = run(args.package, args.base_model, args.condition, args.output, args.manifest, key=args.key,
                      weights=args.weights, spec=args.spec, checkpoint_manifest=args.checkpoint_manifest,
                      material_model=args.material_model, revision=args.revision, cache_dir=args.cache_dir,
                      local_files_only=args.local_files_only)
    print(f"albedo={albedo}\nmra={mra}")


if __name__ == "__main__":
    main()

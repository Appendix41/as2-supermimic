"""Materialize unchanged test inputs at paths required by the original tests."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_checked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise FileExistsError(f"Refusing to replace different data: {destination}")
    else:
        shutil.copyfile(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-assets", type=Path)
    args = parser.parse_args()
    mapping = {
        "single-jump-dense.npz":
            "outputs/forward-jump/validation/nominal-dense-trace.npz",
        "triple-jump-dense.npz":
            "outputs/triple-forward-jump/validation/nominal-dense-trace.npz",
        "triple-source.tar.gz": "outputs/triple-forward-jump/run-source.tar.gz",
    }
    for filename, destination in mapping.items():
        copy_checked(ROOT / "test-data/simulation" / filename,
                     ROOT / "as2-motion-imitation" / destination)
    if args.release_assets:
        source = args.release_assets.resolve() / "as2-proprio-actor-bundle"
        manifest = json.loads((source / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid bundle member: {name}")
            if hashlib.sha256((source / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bundle hash mismatch: {name}")
        destination = ROOT / "as2-sim2real/outputs/deployment/proprio-v2/bundle"
        for name in ("actor.npz", "reference.npz", "manifest.json",
                     "parity_cases.npz", "export-verification.json"):
            copy_checked(source / name, destination / name)
    print("Test inputs prepared. No source files or acceptance rules changed.")


if __name__ == "__main__":
    main()

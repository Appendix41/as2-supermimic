"""Snapshot source provenance without modifying the original project."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "as2-motion-imitation"
TREES = ("src", "scripts", "configs", "tests", "data", "baseline")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(path.relative_to(root)): digest(path)
            for tree in TREES for path in sorted((root / tree).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs/provenance"
    out.mkdir(parents=True, exist_ok=True)
    original = out / "original-files.json"
    if args.verify_source:
        before = json.loads(original.read_text())
        now = inventory(SOURCE)
        changed = [k for k, v in before.items() if now.get(k) != v]
        result = {"original_files_checked": len(before), "changed": changed,
                  "original_single_jump_sha256": digest(
                      SOURCE / "outputs/training/forward-jump-4096-v1/model_1000.pt")}
        (out / "source-verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        if changed:
            raise SystemExit(2)
    elif not original.exists():
        original.write_text(json.dumps(inventory(SOURCE), indent=2) + "\n")
        evidence = ROOT / "docs/evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        old = SOURCE / "outputs/sim2real-single-jump/deployment-v1"
        for filename in ("AS2_INTERFACE_CONTRACT.json", "DEPLOYMENT_STATUS.md"):
            shutil.copy2(old / filename, evidence / filename)
        shutil.copytree(old / "official-sources", evidence / "official-sources")
        shutil.copy2(old / "orin-runs/telemetry-v1/telemetry.jsonl",
                     evidence / "recorded-lowstate-and-sport.jsonl")
        shutil.copy2(ROOT.parent / "THIRD_PARTY_NOTICES.md", ROOT / "THIRD_PARTY_NOTICES.md")
    fixtures = {
        "single-jump-dense.npz": "outputs/forward-jump/validation/nominal-dense-trace.npz",
        "triple-jump-dense.npz": "outputs/triple-forward-jump/validation/nominal-dense-trace.npz",
        "triple-source.tar.gz": "outputs/triple-forward-jump/run-source.tar.gz",
    }
    destination = ROOT / "tests/fixtures"
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in fixtures.items():
        if not (destination / name).exists():
            shutil.copy2(SOURCE / source, destination / name)
    if args.snapshot:
        manifest = inventory(ROOT)
        for path in [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "docs/INPUT_CONTRACT.md"]:
            manifest[str(path.relative_to(ROOT))] = digest(path)
        (out / "training-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with tarfile.open(out / "training-source.tar.gz", "w:gz") as archive:
            for name in manifest:
                archive.add(ROOT / name, arcname=name, recursive=False)
        print(json.dumps({"snapshot_files": len(manifest), "source": str(SOURCE)}))


if __name__ == "__main__":
    main()

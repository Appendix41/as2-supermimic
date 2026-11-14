"""Create a self-contained source/data release; no SSH or robot access."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
def stage(bundle, output):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("schema") != "as2-proprio-single-jump-inference-v2"
            or manifest.get("obs_dim") != 347
            or manifest.get("hardware_execution_enabled") is not False):
        raise ValueError("Only the non-actuating 347D proprioceptive bundle is supported")
    names = ("actor.npz", "reference.npz", "parity_cases.npz")
    for name in names:
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Bundle hash mismatch: {name}")
    release = output / "release"
    archive_path = output / "as2-proprio-v2-deploy.tar.gz"
    if release.exists() or archive_path.exists():
        raise FileExistsError("Refusing to overwrite an existing release/archive")
    release.mkdir(parents=True)
    package = release / "src/as2_motion_imitation"
    package.mkdir(parents=True)
    shutil.copy2(ROOT / "src/as2_motion_imitation/__init__.py", package / "__init__.py")
    shutil.copytree(
        ROOT / "src/as2_motion_imitation/deployment", package / "deployment",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (release / "bundle").mkdir()
    for name in (*names, "manifest.json"):
        shutil.copy2(bundle / name, release / "bundle" / name)
    shutil.copy2(ROOT / "docs/ORIN_DEPLOYMENT.md", release / "README.md")
    (release / "docs").mkdir()
    shutil.copy2(ROOT / "docs/INPUT_CONTRACT.md", release / "docs/INPUT_CONTRACT.md")
    shutil.copy2(ROOT / "docs/CONTROLLER.md", release / "docs/CONTROLLER.md")
    (release / "configs").mkdir()
    shutil.copy2(ROOT / "configs/controller-synthetic.json", release / "configs/controller-synthetic.json")
    scripts = release / "scripts"
    scripts.mkdir()
    for source, target in (
        ("run_deploy_offline.sh", "run_offline.sh"),
        ("run_deploy_telemetry.sh", "run_telemetry.sh"),
        ("run_deploy_readiness.sh", "run_readiness.sh"),
        ("run_deploy_control_selftest.sh", "run_control_selftest.sh"),
        ("run_deploy_feedback_crc.sh", "run_feedback_crc.sh"),
        ("run_deploy_remote_stop.sh", "run_remote_stop.sh"),
        ("run_deploy_controller.sh", "run_controller.sh"),
        ("run_deploy_controller_selftest.sh", "run_controller_selftest.sh"),
        ("bootstrap_orin_controller_numpy.py", "bootstrap_controller_numpy.py"),
        ("bootstrap_orin_readonly_sdk.py", "bootstrap_sdk.py"),
    ):
        shutil.copy2(ROOT / "scripts" / source, scripts / target)
    rows = {}
    for path in sorted(release.rglob("*")):
        if path.is_file():
            rows[str(path.relative_to(release))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (release / "release-files.json").write_text(json.dumps(rows, indent=2) + "\n")
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in sorted(release.iterdir()):
            archive.add(path, arcname=path.name)
    print(json.dumps({"files": len(rows), "release": str(release),
                      "archive": str(archive_path),
                      "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()},
                     indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.bundle.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

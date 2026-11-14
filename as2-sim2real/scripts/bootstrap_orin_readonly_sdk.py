"""Copy an audited SDK/runtime into the new project, without pip or service changes."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


SOURCE = Path("/home/jrz/project/Unitree")
SDK_COMMIT = "814556d15970dd2ecf1c9984e845ca02ab07e206"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.destination
    if ROOT != Path("/home/jrz/project/as2-supermimic-proprio-v2"):
        raise ValueError("Destination must be the dedicated new proprio-v2 deployment")
    if not ROOT.is_dir() or ROOT.is_symlink() or ROOT.resolve() != ROOT:
        raise ValueError("Dedicated deployment directory must already exist and not be a link")
    sdk_source = SOURCE / "vendor/unitree_sdk2_python"
    commit = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(sdk_source), "rev-parse", "HEAD"], text=True,
    ).strip()
    changes = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(sdk_source), "status", "--short"], text=True,
    ).strip()
    if commit != SDK_COMMIT or changes:
        raise ValueError("Source SDK revision/cleanliness changed; inspect before copying")
    for name in ("runtime", ".venv-sdk", "vendor"):
        if (ROOT / name).exists():
            raise FileExistsError(f"Refusing to overwrite deployment component: {name}")
    sdk = ROOT / "vendor/unitree_sdk2_python"
    sdk.mkdir(parents=True)
    archive = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(sdk_source), "archive", SDK_COMMIT],
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(sdk, filter="data")
    packages = SOURCE / ".venv/lib/python3.10/site-packages"
    runtime = ROOT / "runtime"
    (runtime / "python310").mkdir(parents=True)
    for name in ("cyclonedds", "cyclonedds-0.10.2.dist-info"):
        shutil.copytree(packages / name, runtime / "python310" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "lib").mkdir()
    copied = []
    for name in ("libddsc.so", "libddsc.so.0", "libssl.so.1.1", "libcrypto.so.1.1"):
        source = SOURCE / "vendor/lib" / name
        target = runtime / "lib" / name
        shutil.copy2(source, target, follow_symlinks=True)
        copied.append({"source": str(source), "copy": str(target), "sha256": digest(target),
                       "source_sha256": digest(source)})
    interpreter = Path("/home/jrz/miniconda3/envs/lerobot/bin/python")
    subprocess.run(
        [str(interpreter), "-B", "-m", "venv", "--without-pip", str(ROOT / ".venv-sdk")],
        check=True, timeout=30,
    )
    receipt = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "sdk_source": str(sdk_source), "sdk_commit": commit,
        "sdk_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "cyclonedds_version": "0.10.2", "libraries": copied,
        "python_base_interpreter": str(interpreter),
        "note": "Dedicated venv still depends on the existing base Python interpreter, not its site-packages",
        "cyclonedds_loading": "CYCLONEDDS_HOME and LD_LIBRARY_PATH must point to this project's runtime",
        "no_pip_install": True, "no_service_changes": True, "robot_commands_sent": 0,
    }
    (ROOT / "runtime-copy-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

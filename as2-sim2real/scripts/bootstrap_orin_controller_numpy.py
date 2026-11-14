"""Copy existing NumPy into the dedicated controller runtime; no package install."""

import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path("/home/jrz/project/as2-supermimic-proprio-v2")
    source = Path("/home/jrz/project/Unitree/.venv/lib/python3.10/site-packages")
    target = root / "runtime/python310"
    names = ("numpy", "numpy.libs", "numpy-1.26.4.dist-info")
    receipt = root / "controller-numpy-copy.json"
    if not root.is_dir() or root.resolve() != root or not target.is_dir():
        raise ValueError("Dedicated deployment runtime is missing or symlinked")
    if receipt.exists() or any((target / name).exists() for name in names):
        raise FileExistsError("Refusing to replace existing runtime components")
    if not all((source / name).is_dir() for name in names):
        raise FileNotFoundError("Existing Python 3.10 NumPy source is unavailable")
    files = {}
    for name in names:
        shutil.copytree(source / name, target / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in sorted((target / name).rglob("*")):
            if path.is_file():
                relative = path.relative_to(target)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != hashlib.sha256((source / relative).read_bytes()).hexdigest():
                    raise ValueError(f"Copied file differs: {relative}")
                files[str(relative)] = digest
    report = {"source": str(source), "destination": str(target),
              "files": files, "package_installs": 0, "shared_environment_changes": 0,
              "robot_commands_sent": 0}
    with receipt.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"copied_files_verified": len(files), "receipt": str(receipt)}))


if __name__ == "__main__":
    main()

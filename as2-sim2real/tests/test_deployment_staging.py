"""Release isolation tests; no network or robot access."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def staging_module():
    spec = importlib.util.spec_from_file_location(
        "stage_release", ROOT / "scripts/stage_single_jump_deployment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(directory):
    directory.mkdir()
    manifest = {
        "schema": "as2-proprio-single-jump-inference-v2",
        "obs_dim": 347, "hardware_execution_enabled": False, "files": {},
    }
    for name in ("actor.npz", "reference.npz", "parity_cases.npz"):
        payload = name.encode("ascii")
        (directory / name).write_bytes(payload)
        manifest["files"][name] = hashlib.sha256(payload).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_stage_preserves_bundle_and_refuses_overwrite(tmp_path):
    source, output = tmp_path / "bundle", tmp_path / "output"
    make_bundle(source)
    module = staging_module()
    module.stage(source, output)
    release = output / "release"
    for name in ("actor.npz", "reference.npz", "parity_cases.npz", "manifest.json"):
        assert (release / "bundle" / name).read_bytes() == (source / name).read_bytes()
    manifest = json.loads((release / "release-files.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == expected
    with tarfile.open(output / "as2-proprio-v2-deploy.tar.gz") as archive:
        names = archive.getnames()
        assert "bundle/actor.npz" in names
        assert all(not n.startswith("/") and ".." not in Path(n).parts for n in names)
    with pytest.raises(FileExistsError):
        module.stage(source, output)


@pytest.mark.parametrize("change", ["schema", "obs_dim", "enabled", "tamper"])
def test_stage_rejects_wrong_bundle_before_creating_release(tmp_path, change):
    source, output = tmp_path / "bundle", tmp_path / "output"
    manifest = make_bundle(source)
    if change == "schema":
        manifest["schema"] = "old-v1"
    elif change == "obs_dim":
        manifest["obs_dim"] = 359
    elif change == "enabled":
        manifest["hardware_execution_enabled"] = True
    else:
        (source / "actor.npz").write_bytes(b"changed")
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        staging_module().stage(source, output)
    assert not output.exists()

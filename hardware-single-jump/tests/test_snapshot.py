"""Offline checks only: never import the robot SDK or execute hardware control."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SNAPSHOT = json.loads((ROOT / "snapshot.json").read_text())


@pytest.fixture(scope="module")
def bundle_path():
    candidates = [
        REPO / "checkpoints/as2-proprio-actor-bundle",
        REPO.parent / "release-assets/as2-proprio-actor-bundle",
    ]
    explicit = os.environ.get("AS2_INFERENCE_BUNDLE")
    if explicit:
        candidates = [Path(explicit).resolve()]
    for path in candidates:
        if (path / "manifest.json").is_file():
            return path
    pytest.skip("Download the as2-proprio-actor-bundle attachment first")


@pytest.fixture(scope="module")
def runtime():
    path = ROOT / "src/as2_motion_imitation/deployment/runtime.py"
    spec = importlib.util.spec_from_file_location("hardware_snapshot_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("row", SNAPSHOT["files"], ids=lambda row: row["file"])
def test_imported_file_preserved(row):
    assert hashlib.sha256((ROOT / row["file"]).read_bytes()).hexdigest() == row["sha256"]


def test_selected_record_and_provenance():
    report = json.loads((ROOT / SNAPSHOT["selected_run"]).read_text())
    assert report["status"] == "completed"
    assert report["steps_done"] == report["steps"] == 400
    assert report["dt"] == 0.02
    assert report["owner_after_restore"] == report["recovery_owner"] == "ai"
    assert SNAPSHOT["source_snapshot_postdates_selected_run"] is True
    assert SNAPSHOT["historical_run_embeds_source_or_actor_hashes"] is False


def test_joint_mapping_matches_snapshot():
    tree = ast.parse((ROOT / "as2_jump.py").read_text())
    slots = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(name, ast.Name) and name.id == "SLOTS"
                         for name in node.targets))
    assert ast.literal_eval(slots) == (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)


@pytest.mark.parametrize("filename", sorted(SNAPSHOT["bundle_files"]))
def test_bundle_matches_agx_snapshot(bundle_path, filename):
    actual = hashlib.sha256((bundle_path / filename).read_bytes()).hexdigest()
    assert actual == SNAPSHOT["bundle_files"][filename]


def test_runtime_contract_and_historical_flag(runtime, bundle_path):
    bundle = runtime.InferenceBundle(bundle_path)
    assert (bundle.steps, bundle.dt) == (400, 0.02)
    assert bundle.manifest["hardware_execution_enabled"] is False
    assert bundle.manifest["checkpoint_sha256"] == (
        SNAPSHOT["checkpoint_sha256_from_bundle_manifest"])
    assert runtime.OBS_DIM == 347


def test_recorded_input_parity(runtime, bundle_path, record_property):
    bundle = runtime.InferenceBundle(bundle_path)
    maxima = np.zeros(3)
    history = None
    with np.load(bundle_path / "parity_cases.npz", allow_pickle=False) as archive:
        cases = {name: archive[name] for name in archive.files}
    assert len(cases["step"]) == 8000
    for index, step in enumerate(cases["step"]):
        sensor = cases["sensor"][index]
        history = (runtime.init_history(sensor) if step == 0
                   else runtime.push_history(history, sensor))
        observation = bundle.observe(history, int(step))
        action = bundle.act(observation)
        target = bundle.target(action)
        errors = [np.max(np.abs(value - cases[key][index]))
                  for value, key in ((observation, "observation"),
                                     (action, "action"), (target, "target"))]
        maxima = np.maximum(maxima, errors)
    for name, value in zip(("observation", "action", "target_rad"), maxima):
        record_property(f"max_abs_{name}_error", float(value))
    assert np.all(maxima <= [5e-6, 3e-5, 1e-5])


def test_plan_only_without_robot_sdk(bundle_path):
    # Fail before any SDK code can load, even on machines with the SDK installed.
    code = """
import importlib.abc
import runpy
import sys
class NoRobotSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"unitree_sdk2py", "cyclonedds"}:
            raise AssertionError("Robot SDK import forbidden in offline test")
sys.meta_path.insert(0, NoRobotSDK())
sys.argv = [sys.argv[1], "--bundle", sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "as2_jump.py"), str(bundle_path)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=30)
    plan = json.loads(result.stdout)
    assert plan["status"] == "plan_only"
    assert plan["steps"] == 400
    assert plan["dt"] == 0.02

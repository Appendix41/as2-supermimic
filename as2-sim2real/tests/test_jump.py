import json
from pathlib import Path

import numpy as np
import pytest
import torch

from as2_motion_imitation.motion import MotionClip, sha256
from as2_motion_imitation.physics import DT, Physics
from as2_motion_imitation.prepare_jump import (
    ACCEPTANCE, JumpParams, jump_target, longest_run, summarize,
)
from as2_motion_imitation.task import TrackingTask

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "tests/fixtures"


@pytest.fixture(scope="module")
def trace():
    with np.load(VALIDATION / "single-jump-dense.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def test_longest_airborne_interval():
    assert longest_run([False, False], 0.02) == (0.0, 0, 0)
    assert longest_run([True, True, False, True], 0.02) == (0.04, 0, 2)
    assert longest_run([False, True, True, True], 0.02) == (0.06, 1, 4)


def test_script_targets_preserve_joint_and_action_limits():
    cfg = JumpParams(**json.loads((ROOT / "configs/forward-jump-script.json").read_text()))
    p = Physics()
    q = np.array([jump_target(t, cfg) for t in np.linspace(0, 8, 4001)])
    limits = p.model.jnt_range[p.jids]
    assert ((q >= limits[:, 0]) & (q <= limits[:, 1])).all()
    assert np.abs((q - p.default_q) / 0.25).max() <= 4
    np.testing.assert_allclose(q[0], p.default_q)
    np.testing.assert_allclose(q[-1], p.default_q)


def test_recorded_jump_passes_dense_acceptance(trace):
    criteria = json.loads(ACCEPTANCE.read_text())
    result = summarize(trace, criteria)
    assert result["passed"]
    assert len(result["flight_intervals"]) == 1
    assert result["airborne_seconds"] > 0.30
    assert result["airborne_forward_m"] > 0.11
    assert result["final_forward_m"] > 0.24
    assert result["final_stable_seconds"] > 4
    assert trace["time"].shape == (4001,)
    np.testing.assert_allclose(np.diff(trace["time"]), 0.002, atol=1e-12)


@pytest.mark.parametrize("fault", [
    "sliding", "backwards", "no_clearance", "nonfoot", "joint_limit",
    "unstable_tail", "short_airtime", "no_height_gain",
])
def test_rejects_false_or_unsafe_jump(trace, fault):
    modified = {k: v.copy() for k, v in trace.items()}
    if fault == "sliding":
        modified["feet_contact"][:] = 4
    elif fault == "backwards":
        modified["base_pos"][:, 0] *= -1
    elif fault == "no_clearance":
        modified["foot_clearance"][:] = 0
    elif fault == "nonfoot":
        modified["nonfoot"][1200] = True
    elif fault == "joint_limit":
        modified["joint_limit"][1200] = True
    elif fault == "unstable_tail":
        modified["base_vel"][-1000:, 0] = 0.1
    elif fault == "short_airtime":
        modified["feet_contact"][:] = 4
        modified["feet_contact"][1050:1060] = 0
    elif fault == "no_height_gain":
        modified["base_pos"][:, 2] = modified["base_pos"][0, 2]
    assert not summarize(modified, json.loads(ACCEPTANCE.read_text()))["passed"]


def test_capture_reproduces_from_actions_without_pose_writes():
    p = Physics()
    with np.load(ROOT / "data/raw/as2-native-forward-jump/physics_capture.npz") as f:
        assert f["action"].shape == (400, 12)
        np.testing.assert_allclose(f["action_time"], np.arange(400) * DT)
        np.testing.assert_allclose(p.data.qpos, f["qpos"][0], atol=1e-7)
        for i, action in enumerate(f["action"]):
            p.step(action)
            np.testing.assert_allclose(p.data.qpos, f["qpos"][i + 1], atol=2e-6, rtol=1e-5)
            np.testing.assert_allclose(p.data.qvel, f["qvel"][i + 1], atol=2e-6, rtol=1e-5)
        assert not p.data.xfrc_applied.any()
        assert not p.data.qfrc_applied.any()


def test_jump_loads_in_existing_tracker():
    clip = MotionClip(ROOT / "data/processed/as2_forward_jump.npz")
    assert clip.frames == 401 and clip.duration == 8
    time = torch.arange(401) * DT
    ref = clip.sample(time)
    assert not TrackingTask(clip).failure(ref["state"], time).any()
    assert torch.isfinite(ref["state"]).all()
    provenance = clip.metadata["provenance"]
    assert provenance["generator_sha256"] == sha256(ROOT / provenance["generator"])
    assert provenance["acceptance_sha256"] == sha256(ACCEPTANCE)
    assert not provenance["pose_writes_after_reset"]
    assert not provenance["learned_policy"]

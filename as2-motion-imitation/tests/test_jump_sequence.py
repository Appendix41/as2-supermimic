import json
import hashlib
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from as2_motion_imitation.demonstration import RecordedCommands
from as2_motion_imitation.motion import MotionClip, MotionClock, sha256
from as2_motion_imitation.physics import DT, Physics
from as2_motion_imitation.prepare_jump_sequence import (
    ACCEPTANCE, load_sequence, sequence_target, summarize_sequence,
)

ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT / "data/processed/as2_triple_forward_jump.npz"
RAW = ROOT / "data/raw/as2-native-triple-forward-jump/physics_capture.npz"


@pytest.fixture(scope="module")
def trace():
    with np.load(ROOT / "outputs/triple-forward-jump/validation/nominal-dense-trace.npz") as f:
        return {k: f[k] for k in f.files}


def test_three_targets_are_continuous_at_template_switches():
    cfg, params, duration = load_sequence()
    assert duration == 13.6
    p = Physics()
    for onset in (0.6 + cfg["period_seconds"], 0.6 + 2 * cfg["period_seconds"]):
        before = sequence_target(onset - 1e-7, params, 3, cfg["period_seconds"])
        after = sequence_target(onset + 1e-7, params, 3, cfg["period_seconds"])
        np.testing.assert_allclose(before, after, atol=1e-10)
        np.testing.assert_allclose(after, p.default_q, atol=1e-10)
    targets = np.array([sequence_target(t, params, 3, cfg["period_seconds"])
                        for t in np.arange(0, duration + DT, DT)])
    assert np.abs((targets - p.default_q) / 0.25).max() <= 4
    limits = p.model.jnt_range[p.jids]
    assert ((targets >= limits[:, 0]) & (targets <= limits[:, 1])).all()


def test_dense_reference_contains_exactly_three_physical_jumps(trace):
    criteria = json.loads(ACCEPTANCE.read_text())
    result = summarize_sequence(trace, criteria)
    assert result["passed"] and result["jump_count"] == 3
    assert all(j["duration_s"] > 0.30 and j["airborne_forward_m"] > 0.11
               for j in result["jumps"])
    assert result["final_forward_m"] > 0.70
    assert result["final_stable_seconds"] > 4.0
    assert trace["time"].shape == (6801,)
    np.testing.assert_allclose(np.diff(trace["time"]), 0.002, atol=1e-12)


@pytest.mark.parametrize("fault", [
    "missing_second", "missing_third", "extra_jump", "backward_second",
    "sliding", "unstable_tail", "nonfoot", "joint_limit", "torque_saturation",
    "short_second", "uncleared_third", "long_interjump_gap", "no_final_landing",
])
def test_sequence_rejects_incomplete_or_false_success(trace, fault):
    a = {k: v.copy() for k, v in trace.items()}
    t = a["time"]
    second = (t >= 4.8) & (t < 5.13)
    third = (t >= 7.6) & (t < 7.93)
    if fault == "missing_second":
        a["feet_contact"][second] = 4
    elif fault == "missing_third":
        a["feet_contact"][third] = 4
    elif fault == "extra_jump":
        a["feet_contact"][(t >= 10) & (t < 10.2)] = 0
    elif fault == "backward_second":
        a["base_pos"][second, 0] *= -1
    elif fault == "sliding":
        a["feet_contact"][:] = 4
    elif fault == "unstable_tail":
        a["base_vel"][-1000:, 0] = 0.2
    elif fault == "nonfoot":
        a["nonfoot"][2500] = True
    elif fault == "joint_limit":
        a["joint_limit"][2500] = True
    elif fault == "torque_saturation":
        a["torque_saturation"][2500] = 1.0
    elif fault == "short_second":
        a["feet_contact"][second] = 4
        a["feet_contact"][(t >= 4.8) & (t < 4.86)] = 0
    elif fault == "uncleared_third":
        a["foot_clearance"][third] = 0
    elif fault == "long_interjump_gap":
        a["feet_contact"][third] = 4
        a["feet_contact"][(t >= 9) & (t < 9.33)] = 0
    elif fault == "no_final_landing":
        a["feet_contact"][t >= 7.6] = 0
    assert not summarize_sequence(a, json.loads(ACCEPTANCE.read_text()))["passed"]


def test_sequence_reproduces_all_controls_without_between_jump_resets():
    p = Physics()
    with np.load(RAW) as f:
        assert f["action"].shape == (680, 12)
        np.testing.assert_allclose(p.data.qpos, f["qpos"][0], atol=1e-7)
        for i, action in enumerate(f["action"]):
            p.step(action)
            np.testing.assert_allclose(p.data.qpos, f["qpos"][i + 1], atol=2e-6, rtol=1e-5)
            np.testing.assert_allclose(p.data.qvel, f["qvel"][i + 1], atol=2e-6, rtol=1e-5)
    assert not p.data.xfrc_applied.any() and not p.data.qfrc_applied.any()


def test_sequence_loader_commands_and_reference_provenance():
    from as2_motion_imitation.task import TrackingTask

    clip = MotionClip(MOTION)
    assert clip.frames == 681 and clip.duration == 13.6
    commands = RecordedCommands(RAW, clip)
    assert commands.actions.shape == (680, 12)
    times = torch.arange(681) * DT
    assert not TrackingTask(clip).failure(clip.sample(times)["state"], times).any()
    provenance = clip.metadata["provenance"]
    # Provenance identifies the generating source, not later compatible changes.
    with tarfile.open(ROOT / "outputs/triple-forward-jump/run-source.tar.gz") as source:
        for name, expected in provenance["source_sha256"].items():
            assert hashlib.sha256(source.extractfile(name).read()).hexdigest() == expected
    assert not provenance["resets_between_jumps"]
    assert not provenance["concatenated_state_clips"]
    assert not provenance["pose_writes_after_reset"]
    assert not provenance["learned_policy"]


def test_long_motion_clock_remains_aligned_with_recorded_control_ticks():
    frames = torch.arange(680)
    clock = MotionClock(SimpleNamespace(duration=13.6, device="cpu"), len(frames))
    clock.reset(frames, frames * DT)
    commands = RecordedCommands.__new__(RecordedCommands)
    commands.actions = frames[:, None].expand(-1, 12).float()
    for step in range(681):
        expected = (frames + step).clamp(max=679).float()
        torch.testing.assert_close(commands.sample(clock.time)[:, 0], expected, rtol=0, atol=0)
        clock.advance(DT)
    clock.reset(torch.tensor([0]), 0.0)
    clock.advance(DT)
    assert clock.time[0].item() == torch.tensor(DT).item()
    assert clock.time[1].item() == torch.tensor(13.6).item()


def test_phase_prioritization_covers_each_jump():
    from as2_motion_imitation.learner import TrainCfg, Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.device = "cpu"
    windows = ((1.5, 3.4), (4.3, 6.2), (7.1, 9.0))
    trainer.cfg = TrainCfg(phase_windows=windows, phase_prioritized_fraction=0.5)
    times = torch.arange(680) * DT
    torch.manual_seed(23)
    chosen = times[trainer.phase_selection(times, 256)[:128]]
    inside = torch.stack([(chosen >= low) & (chosen <= high) for low, high in windows])
    assert inside.any(0).all() and inside.any(1).all()

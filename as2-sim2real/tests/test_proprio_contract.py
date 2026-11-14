import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from as2_motion_imitation.deployment.runtime import lowstate_sensor, projected_gravity
from as2_motion_imitation.motion import MotionClip
from as2_motion_imitation.physics import Physics
from as2_motion_imitation.task import OBS_DIM, TrackingTask, sensors

ROOT = Path(__file__).resolve().parents[1]
SLOTS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def calibration():
    return {"verified_on_robot": False, "sdk_slots": SLOTS,
            "joint_signs": [1] * 12, "joint_offsets_rad": [0] * 12,
            "policy_from_imu": np.eye(3).tolist()}


def test_actor_observation_independent_of_all_privileged_state():
    clip = MotionClip(ROOT / "data/processed/as2_forward_jump.npz")
    task = TrackingTask(clip)
    history = torch.randn(8, 210)
    time = torch.linspace(0, 8, 8)
    a = task.observe(history, None, None, time, None)
    b = task.observe(history, torch.full((8, 70), float("nan")),
                     torch.full((8, 3), float("nan")), time,
                     torch.full((8, 3), float("nan")))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a.shape == (8, 347) and OBS_DIM == 347 and torch.isfinite(a).all()


def test_privileged_reward_still_sees_position_error():
    clip = MotionClip(ROOT / "data/processed/as2_forward_jump.npz")
    task = TrackingTask(clip)
    s, pose = clip.states[100:101], clip.poses[100:101]
    time, origin, last = torch.tensor([2.0]), clip.poses[0:1], torch.zeros(1, 12)
    _, first = task.reward(s, pose, time, origin, last, last)
    _, shifted = task.reward(s, pose + torch.tensor([[1., 0, 0]]), time, origin, last, last)
    assert shifted["root"].item() > first["root"].item() + 0.5


def test_sensor_slices_exclude_privileged_fields_and_preserve_last_action():
    s = torch.randn(10, 70)
    s[:, 1:4] = torch.nn.functional.normalize(s[:, 1:4], dim=-1)
    last = torch.rand(10, 12)
    expected = sensors(s, last)
    s[:, [0, 4, 5, 6]] = float("nan")
    s[:, 34:] = float("nan")
    torch.testing.assert_close(expected, sensors(s, last), atol=0, rtol=0)
    noisy = sensors(s, last, (0.01, 0.003, 0.002, 0.03))
    assert torch.isfinite(noisy).all()
    torch.testing.assert_close(noisy[:, 30:], last, atol=0, rtol=0)
    torch.testing.assert_close(noisy[:, 3:6].norm(dim=-1), torch.ones(10))


def test_simulated_lowstate_reconstructs_training_sensor():
    physics = Physics()
    with np.load(ROOT / "data/processed/as2_forward_jump.npz") as trace:
        for index in (0, 80, 100, 110, 120, 150, 400):
            physics.data.qpos[:], physics.data.qvel[:] = trace["qpos"][index], trace["qvel"][index]
            mujoco.mj_forward(physics.model, physics.data)
            state, _ = physics.state()
            motors = [{"slot": i, "q": 0.0, "dq": 0.0} for i in range(35)]
            for j, slot in enumerate(SLOTS):
                motors[slot]["q"] = float(physics.data.qpos[physics.qadr[j]])
                motors[slot]["dq"] = float(physics.data.qvel[physics.vadr[j]])
            row = {"topic": "rt/lowstate", "received_monotonic_ns": 100_000_000,
                   "quaternion_raw": physics.data.xquat[physics.base_id].tolist(),
                   "gyroscope_raw": state[7:10].tolist(), "motor_slots": motors}
            last = np.linspace(-1, 1, 12, dtype=np.float32)
            actual = lowstate_sensor(row, physics.default_q, last, calibration(),
                                     105_000_000, allow_unverified=True)
            expected = sensors(torch.tensor(state)[None], torch.tensor(last)[None])[0].numpy()
            np.testing.assert_allclose(actual, expected, atol=3e-7)
            np.testing.assert_allclose(projected_gravity(row["quaternion_raw"]),
                                       state[1:4], atol=3e-7)


def test_recorded_real_lowstate_input_is_computable_but_not_calibrated():
    default_q = Physics().default_q
    count = 0
    path = ROOT / "docs/evidence/recorded-lowstate-and-sport.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["topic"] != "rt/lowstate":
            continue
        actual = lowstate_sensor(row, default_q, np.zeros(12), calibration(),
                                 row["received_monotonic_ns"], allow_unverified=True)
        assert actual.shape == (42,) and np.isfinite(actual).all()
        count += 1
        if count == 1:
            with pytest.raises(ValueError, match="not been verified"):
                lowstate_sensor(row, default_q, np.zeros(12), calibration(),
                                row["received_monotonic_ns"])
            with pytest.raises(ValueError, match="Stale"):
                lowstate_sensor(row, default_q, np.zeros(12), calibration(),
                                row["received_monotonic_ns"] + 100_000_000,
                                allow_unverified=True)
    assert count > 100


def test_calibrated_joint_and_imu_transform_are_not_implicit_identity():
    c = calibration()
    c["verified_on_robot"] = True
    c["joint_signs"] = [1, -1] * 6
    c["joint_offsets_rad"] = np.linspace(-0.1, 0.1, 12).tolist()
    c["policy_from_imu"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    row = {
        "topic": "rt/lowstate", "received_monotonic_ns": 0,
        "quaternion_raw": [1, 0, 0, 0], "gyroscope_raw": [1, 2, 3],
        "motor_slots": [{"slot": i, "q": i / 10, "dq": i / 20} for i in range(35)],
    }
    out = lowstate_sensor(row, np.zeros(12), np.ones(12), c, 0)
    np.testing.assert_allclose(out[:6], [-2, 1, 3, 0, 0, -1])
    np.testing.assert_allclose(out[6:18], np.array(SLOTS) / 10 * c["joint_signs"]
                               + c["joint_offsets_rad"], atol=1e-7)
    np.testing.assert_allclose(out[18:30], np.array(SLOTS) / 20 * c["joint_signs"], atol=1e-7)
    np.testing.assert_array_equal(out[30:], np.ones(12))
    c["policy_from_imu"] = np.diag([1, 1, -1]).tolist()
    with pytest.raises(ValueError, match="proper rotation"):
        lowstate_sensor(row, np.zeros(12), np.ones(12), c, 0)


def test_gravity_is_quaternion_based_during_flight_not_accelerometer_based():
    c = calibration()
    c["verified_on_robot"] = True
    row = {
        "topic": "rt/lowstate", "received_monotonic_ns": 0,
        "quaternion_raw": [np.sqrt(0.5), 0, np.sqrt(0.5), 0],
        "gyroscope_raw": [0, 0, 0], "accelerometer_raw": [0, 0, 0],
        "motor_slots": [{"slot": i, "q": 0, "dq": 0} for i in range(35)],
    }
    out = lowstate_sensor(row, np.zeros(12), np.zeros(12), c, 0)
    np.testing.assert_allclose(out[3:6], [1, 0, 0], atol=2e-7)
    row["accelerometer_raw"] = [float("nan")] * 3
    np.testing.assert_array_equal(lowstate_sensor(row, np.zeros(12), np.zeros(12), c, 0), out)

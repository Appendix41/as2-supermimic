"""Check recorded inputs against the policy observation contract."""

import json
from pathlib import Path

import numpy as np
import torch

from as2_motion_imitation.deployment.runtime import lowstate_sensor
from as2_motion_imitation.motion import MotionClip, sha256
from as2_motion_imitation.task import OBSERVATION_SCHEMA, TrackingTask

ROOT = Path(__file__).resolve().parents[1]


def main():
    clip = MotionClip(ROOT / "data/processed/as2_forward_jump.npz")
    task = TrackingTask(clip)
    history = torch.arange(210).float().repeat(400, 1) / 100
    times = torch.arange(400).float() * 0.02
    clean = task.observe(history, None, None, times, None)
    poison = task.observe(history, torch.full((400, 70), float("nan")),
                          torch.full((400, 3), float("nan")), times,
                          torch.full((400, 3), float("nan")))
    assert torch.equal(clean, poison) and torch.isfinite(clean).all()
    calibration = {
        "verified_on_robot": False, "sdk_slots": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        "joint_signs": [1] * 12, "joint_offsets_rad": [0] * 12,
        "policy_from_imu": np.eye(3).tolist(),
    }
    path = ROOT / "docs/evidence/recorded-lowstate-and-sport.jsonl"
    count = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["topic"] != "rt/lowstate":
            continue
        out = lowstate_sensor(row, clip.default_q.numpy(), np.zeros(12), calibration,
                              row["received_monotonic_ns"], allow_unverified=True)
        assert np.isfinite(out).all()
        count += 1
    result = {
        "schema": OBSERVATION_SCHEMA, "actor_dim": 347,
        "sensor_history": {"frames": 5, "per_frame_dim": 42},
        "known_reference": {"windows": 4, "per_window_dim": 34, "phase_dim": 1},
        "privileged_poison_test_ticks": 400, "privileged_poison_max_difference": 0.0,
        "recorded_real_lowstate_rows_decoded": count, "source_sha256": sha256(path),
        "measured_xy_yaw_required": False, "sportmodestate_required": False,
        "measured_height_velocity_contacts_required": False,
        "hardware_calibration_verified": False, "robot_commands_sent": 0,
        "limitations": [
            "Recorded LowState collected with built-in motion service still running",
            "Identity calibration used only as an explicit offline dimensional diagnostic",
            "Previous applied action is synthetic here; vendor-controller commands were not recorded",
            "This is input computability, not hardware policy execution or safety acceptance",
        ],
    }
    destination = ROOT / "outputs/validation/input-contract.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

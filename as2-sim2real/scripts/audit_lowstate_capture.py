"""Summarize saved LowState feedback; calibration is checked separately."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from as2_motion_imitation.deployment.runtime import finite_array, projected_gravity, sha256


SDK_SLOTS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def audit(rows):
    low = [row for row in rows if row["topic"] == "rt/lowstate"]
    sport = [row for row in rows if row["topic"] == "rt/sportmodestate"]
    if not low:
        raise ValueError("No high-rate LowState rows")
    quaternions, gravities, gyros, positions, velocities, torques = [], [], [], [], [], []
    for row in low:
        motors = {m["slot"]: m for m in row["motor_slots"]}
        if len(motors) != len(row["motor_slots"]) or not all(s in motors for s in SDK_SLOTS):
            raise ValueError("Missing or duplicate motor slots")
        quaternions.append(finite_array(row["quaternion_raw"], (4,), "quaternion"))
        gravities.append(projected_gravity(row["quaternion_raw"]))
        gyros.append(finite_array(row["gyroscope_raw"], (3,), "gyro"))
        for field, dest in (("q", positions), ("dq", velocities), ("tau_est", torques)):
            dest.append(finite_array([motors[s][field] for s in SDK_SLOTS], (12,), field))
    receipts = np.array([row["received_monotonic_ns"] for row in low], dtype=np.int64)
    gaps_ms = np.diff(receipts) / 1e6
    if len(gaps_ms) and np.any(gaps_ms <= 0):
        raise ValueError("Non-increasing saved receipt timestamps")
    ticks = np.array([row["tick"] for row in low], dtype=np.int64)
    tick_gaps = np.diff(ticks) % (2 ** 32)
    if len(tick_gaps) and np.any((tick_gaps == 0) | (tick_gaps >= 2 ** 31)):
        raise ValueError("Repeated or backward robot ticks")
    q, dq, tau = np.asarray(positions), np.asarray(velocities), np.asarray(torques)
    norms = np.linalg.norm(quaternions, axis=1)
    return {
        "scope": "Offline structure audit of saved real feedback; no actor execution",
        "saved_high_rate_lowstate_rows": len(low),
        "structural_checks_passed": True,
        "robot_commands_sent": 0,
        "previous_applied_action_available": False,
        "full_actor_observation_constructed": False,
        "hardware_calibration_verified": False,
        "hardware_execution_enabled": False,
        "mode_machine_counts": dict(Counter(str(r["mode_machine"]) for r in low)),
        "mode_pr_counts": dict(Counter(str(r["mode_pr"]) for r in low)),
        "sport_mode_counts": dict(Counter(str(r["mode"]) for r in sport)),
        "sport_error_code_counts": dict(Counter(str(r["error_code"]) for r in sport)),
        "reserved_version_values_not_firmware": sorted({tuple(r["version"]) for r in low}),
        "quaternion_norm_min_max": [float(norms.min()), float(norms.max())],
        "raw_imu_projected_gravity_mean": np.mean(gravities, axis=0).tolist(),
        "raw_gyro_abs_max": float(np.max(np.abs(gyros))),
        "saved_receipt_gap_ms_max": float(gaps_ms.max()) if len(gaps_ms) else None,
        "saved_robot_tick_gap_max": int(tick_gaps.max()) if len(tick_gaps) else None,
        "saved_span_s": float((receipts[-1] - receipts[0]) / 1e9),
        "joint_statistics": [
            {
                "sdk_slot": slot,
                "q_min_max_raw": [float(q[:, i].min()), float(q[:, i].max())],
                "dq_abs_max_raw": float(np.abs(dq[:, i]).max()),
                "tau_est_abs_max_raw": float(np.abs(tau[:, i]).max()),
                "mode_values": sorted({m["mode"] for r in low
                                       for m in r["motor_slots"] if m["slot"] == slot}),
                "status_values": sorted({m["status"] for r in low
                                         for m in r["motor_slots"] if m["slot"] == slot}),
            } for i, slot in enumerate(SDK_SLOTS)
        ],
        "limits": [
            "Stats use documented slot order only; signs/offsets/IMU axes remain unverified",
            "Saved rows are decimated; their gaps are not the all-message receive gaps",
            "Receipt timestamps do not prove sensor age; DDS/source clocks are not synchronized",
            "Motor status values are reported raw, not interpreted as safety approval",
            "Built-in controller action history is unknown and was not synthesized",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.capture.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report = audit(rows)
    report["capture_sha256"] = sha256(args.capture)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

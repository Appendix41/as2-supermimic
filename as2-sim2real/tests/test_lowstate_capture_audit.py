"""Offline telemetry audit must not grant calibration or fill missing actions."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "capture_audit", Path(__file__).resolve().parents[1] / "scripts/audit_lowstate_capture.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def packet():
    return {
        "topic": "rt/lowstate", "received_monotonic_ns": 1000000000, "tick": 100,
        "quaternion_raw": [1, 0, 0, 0], "gyroscope_raw": [0, 0, 0],
        "mode_machine": 1, "mode_pr": 0, "version": [0, 0],
        "motor_slots": [{"slot": i, "q": 0, "dq": 0, "tau_est": 0,
                         "mode": 1, "status": 0} for i in range(35)],
    }


def test_audit_is_not_calibration_or_actor_test():
    result = MODULE.audit([packet()])
    assert result["structural_checks_passed"]
    assert result["robot_commands_sent"] == 0
    for key in ("hardware_calibration_verified", "hardware_execution_enabled",
                "previous_applied_action_available", "full_actor_observation_constructed"):
        assert result[key] is False


@pytest.mark.parametrize("defect", ["quaternion", "nonfinite", "missing", "duplicate", "receipt", "tick"])
def test_malformed_capture_rejected(defect):
    first = packet()
    second = deepcopy(first)
    second["received_monotonic_ns"] += 20000000
    second["tick"] += 20
    if defect == "quaternion":
        second["quaternion_raw"] = [0, 0, 0, 0]
    elif defect == "nonfinite":
        second["motor_slots"][0]["q"] = float("nan")
    elif defect == "missing":
        second["motor_slots"].pop(0)
    elif defect == "duplicate":
        second["motor_slots"].append(second["motor_slots"][0])
    elif defect == "receipt":
        second["received_monotonic_ns"] = first["received_monotonic_ns"]
    else:
        second["tick"] = first["tick"]
    with pytest.raises(ValueError):
        MODULE.audit([first, second])


def test_no_high_rate_feedback_rejected():
    with pytest.raises(ValueError):
        MODULE.audit([])

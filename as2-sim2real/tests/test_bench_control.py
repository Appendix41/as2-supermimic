"""Synthetic guard/encoding tests; no DDS or robot access."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from as2_motion_imitation.deployment.bench_control import (
    Decision, Feedback, Health, Intent, ProbeGuard, build_sdk_frame, pulse_target,
)


T0 = 1_000_000_000


def feedback(now=T0, tick=100):
    return Feedback(now, tick, 1, 0, (0.0,) * 12, (0.0,) * 12, (0,) * 12, True)


def guard():
    return ProbeGuard(feedback(), T0, 3, -0.5, 0.5)


def next_inputs():
    now = T0 + 5_000_000
    return [now, feedback(now, 105), Health(now, 0, True), Intent(0, now, 0.0)]


def test_bounded_pulse_completes_and_stop_latches():
    g = guard()
    final = None
    for i in range(1, 601):
        now = T0 + i * 5_000_000
        target = pulse_target(0.0, i * 0.005)
        values = list(feedback(now, 100 + i * 5).q)
        values[3] = target
        final = g.evaluate(now, replace(feedback(now, 100 + i * 5), q=tuple(values)),
                           Health(now, 0, True), Intent(i, now, target))
        assert final.enabled == (i < 600)
    assert final.reason == "duration_complete"
    assert g.evaluate(*next_inputs()).reason == "duration_complete"


@pytest.mark.parametrize("defect,expected", [
    ("late_loop", "loop_deadline_or_clock"),
    ("reverse_clock", "loop_deadline_or_clock"),
    ("future_feedback", "feedback_age"),
    ("stale_feedback", "feedback_age"),
    ("bad_crc", "invalid_feedback_or_crc"),
    ("motor_error", "invalid_feedback_or_crc"),
    ("nan_q", "invalid_feedback_or_crc"),
    ("nan_dq", "invalid_feedback_or_crc"),
    ("bad_machine_type", "invalid_feedback_or_crc"),
    ("mode_change", "mode_changed"),
    ("tick_replay", "robot_tick_replayed"),
    ("board_stale", "health_or_operator_enable"),
    ("board_future", "health_or_operator_enable"),
    ("estop", "health_or_operator_enable"),
    ("operator_release", "health_or_operator_enable"),
    ("parent_gone", "invalid_or_expired_intent"),
    ("expired_intent", "invalid_or_expired_intent"),
    ("future_intent", "invalid_or_expired_intent"),
    ("nan_target", "invalid_or_expired_intent"),
    ("large_request", "joint_envelope"),
    ("too_fast", "target_rate"),
    ("overspeed", "joint_speed"),
    ("wrong_joint_motion", "unselected_joint_motion"),
])
def test_faults_latch_stop(defect, expected):
    values = next_inputs()
    now, state, health, intent = values
    if defect == "late_loop":
        values[0] = T0 + 30_000_000
    elif defect == "reverse_clock":
        values[0] = T0 - 1
    elif defect == "future_feedback":
        values[1] = replace(state, received_ns=now + 1)
    elif defect == "stale_feedback":
        values[1] = replace(state, received_ns=now - 40_000_001)
    elif defect == "bad_crc":
        values[1] = replace(state, crc_valid=False)
    elif defect == "motor_error":
        values[1] = replace(state, status=(1,) + (0,) * 11)
    elif defect == "nan_q":
        values[1] = replace(state, q=(float("nan"),) + (0.0,) * 11)
    elif defect == "nan_dq":
        values[1] = replace(state, dq=(float("nan"),) + (0.0,) * 11)
    elif defect == "bad_machine_type":
        values[1] = replace(state, mode_machine=True)
    elif defect == "mode_change":
        values[1] = replace(state, mode_machine=2)
    elif defect == "tick_replay":
        values[1] = replace(state, tick=99)
    elif defect == "board_stale":
        values[2] = replace(health, received_ns=now - 150_000_001)
    elif defect == "board_future":
        values[2] = replace(health, received_ns=now + 1)
    elif defect == "estop":
        values[2] = replace(health, mainboard_status=4096)
    elif defect == "operator_release":
        values[2] = replace(health, operator_enable=False)
    elif defect == "parent_gone":
        values[3] = None
    elif defect == "expired_intent":
        values[3] = replace(intent, generated_ns=now - 40_000_001)
    elif defect == "future_intent":
        values[3] = replace(intent, generated_ns=now + 1)
    elif defect == "nan_target":
        values[3] = replace(intent, target_rad=float("nan"))
    elif defect == "large_request":
        values[3] = replace(intent, target_rad=0.03)
    elif defect == "too_fast":
        values[3] = replace(intent, target_rad=0.005)
    elif defect == "overspeed":
        values[1] = replace(state, dq=(0.51,) + (0.0,) * 11)
    else:
        values[1] = replace(state, q=(0.06,) + (0.0,) * 11)
    g = guard()
    decision = g.evaluate(*values)
    assert not decision.enabled
    assert decision.reason == expected
    assert g.evaluate(*next_inputs()).reason == expected


def test_frozen_robot_tick_despite_fresh_receipts():
    g = guard()
    for i in range(1, 10):
        now = T0 + i * 5_000_000
        decision = g.evaluate(now, feedback(now, 100), Health(now, 0, True),
                              Intent(i, now, 0.0))
    assert decision.reason == "robot_tick_frozen"


def test_uint32_tick_wrap_is_allowed():
    initial = replace(feedback(), tick=2 ** 32 - 3)
    g = ProbeGuard(initial, T0, 3, -0.5, 0.5)
    values = next_inputs()
    values[1] = replace(values[1], tick=2)
    assert g.evaluate(*values).enabled


@pytest.mark.parametrize("mutate", ["target", "stamp", "old_sequence", "new_sequence_old_stamp"])
def test_intent_mutation_and_replay(mutate):
    g = guard()
    now, state, health, intent = next_inputs()
    intent = replace(intent, sequence=2)
    assert g.evaluate(now, state, health, intent).enabled
    now += 5_000_000
    if mutate == "target":
        bad = replace(intent, target_rad=0.0001)
    elif mutate == "stamp":
        bad = replace(intent, generated_ns=now)
    elif mutate == "old_sequence":
        bad = replace(intent, sequence=1)
    else:
        bad = replace(intent, sequence=3)
    assert g.evaluate(now, feedback(now, 110), Health(now, 0, True),
                      bad).reason == "intent_replayed_or_mutated"


def test_reuse_of_unchanged_unexpired_intent_is_allowed():
    g = guard()
    now, state, health, intent = next_inputs()
    assert g.evaluate(now, state, health, intent).enabled
    now += 5_000_000
    assert g.evaluate(now, feedback(now, 110), Health(now, 0, True), intent).enabled


def factory():
    return SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(35)])


class FakeCrc:
    def Crc(self, command):
        assert command.crc == 0
        return 12345


def test_official_encoder_adapter_does_not_enable_unused_motors():
    decision = Decision(True, "bench_candidate", 3, 0.01, 5, 1)
    command = build_sdk_frame(decision, factory, FakeCrc())
    assert command.mode_machine == 5 and command.mode_pr == 1
    assert command.crc == 12345
    for i, motor in enumerate(command.motor_cmd):
        assert motor.mode == int(i == 3)
        assert motor.dq == motor.tau == motor.reserve == 0
        if i != 3:
            assert motor.q == motor.kp == motor.kd == 0
    stopped = build_sdk_frame(replace(decision, enabled=False), factory, FakeCrc())
    assert all(m.mode == m.kp == m.kd == m.tau == 0 for m in stopped.motor_cmd)


@pytest.mark.parametrize("slot", [-1, 12, True])
def test_invalid_motor_slot_rejected(slot):
    with pytest.raises(ValueError):
        build_sdk_frame(Decision(True, "bad", slot, 0.0, 1, 0), factory, FakeCrc())


def test_no_hardware_execution_interface():
    import ast
    import inspect
    from as2_motion_imitation.deployment import bench_control
    tree = ast.parse(inspect.getsource(bench_control))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection({"DataWriter", "Publisher", "MotionSwitcherClient"})

"""Integrated controller tests with synthetic feedback and memory-only writes."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from as2_motion_imitation.deployment import controller as c
from as2_motion_imitation.deployment.controller_run import run_session
from as2_motion_imitation.deployment.runtime import InferenceBundle


ROOT = Path(__file__).resolve().parents[1]
T0 = 1_000_000_000


def config_document():
    document = json.loads((ROOT / "configs/controller-synthetic.json").read_text())
    document["safety"].update(ramp_s=0.2, warmup_s=0.1, hold_s=0.1, stop_s=0.1)
    return document


@pytest.fixture
def config():
    return c.ControllerConfig(config_document())


class Bundle:
    default_q = np.array([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8] * 2, dtype=np.float32)
    steps = 400
    manifest = {"checkpoint_sha256": "synthetic-checkpoint"}
    observe = lambda self, history, step: np.concatenate((history, np.zeros(136), [step / 400]))
    target = lambda self, action: self.default_q + 0.25 * np.asarray(action)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        self.now += 1000
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1e9)


class Transport:
    def __init__(self, config, clock=None):
        self.clock = clock or Clock()
        self.config = config
        self.q = Bundle.default_q.copy()
        self.owner = "synthetic-vendor"
        self.writes = []
        self.release_count = 0
        self.fail_write = False

    def snapshot(self, now=None, mask=0):
        now = self.clock() if now is None else now
        raw_q = np.empty(12)
        raw_q[c.SLOTS] = (self.q - self.config.offsets) / self.config.signs
        motors = [dict(slot=i, q=float(q), dq=0.0, tau_est=0.0, status=0, mode=1,
                       temperature_raw=[30, 30], voltage_raw=42.0, warning=0)
                  for i, q in enumerate(raw_q)]
        row = dict(topic="rt/lowstate", received_monotonic_ns=now,
                   mode_machine=1, mode_pr=0, tick=(now // 1_000_000) % 2 ** 32,
                   quaternion_raw=[1, 0, 0, 0], gyroscope_raw=[0, 0, 0],
                   wireless_remote_hex=(b"\x55\x51" + mask.to_bytes(2, "little") + bytes(36)).hex(),
                   motor_slots=motors)
        return c.Snapshot(row, True, now, (0,) * 6)

    def enable_watch(self):
        pass

    def monitor_owner(self):
        pass

    def check_mode(self):
        return 0, {"name": self.owner}

    def release_mode(self):
        self.release_count += 1
        self.owner = ""
        return 0, None

    def publish(self, command):
        self.writes.append(command)
        if self.fail_write:
            return False
        if command.kind == "position":
            self.q[:] = np.asarray(command.raw_q)[c.SLOTS] * self.config.signs + self.config.offsets
        return True


class Worker:
    def __init__(self):
        self.pending = None
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        self.pending = c.Reply(request.sequence, request.step,
                               tuple([0.05 * np.sin(request.step / 20)] * 12))

    def poll(self):
        result, self.pending = self.pending, None
        return result


def running(config):
    io = Transport(config)
    engine = c.Controller(Bundle(), config)
    engine.begin(T0, io.snapshot(T0))
    return engine, io


def policy_pending(config):
    engine, io = running(config)
    for i in range(1, 200):
        now = T0 + i * 2_000_000
        cmd, request = engine.advance(now, io.snapshot(now))
        assert cmd is not None, engine.reason
        engine.submitted(cmd, io.publish(cmd), now)
        if request is not None:
            return engine, io, now, request
    pytest.fail("policy never requested")


def test_full_controller_executes_exactly_400_steps_then_damping(config):
    clock = Clock()
    io, worker, events = Transport(config, clock), Worker(), []
    report = run_session(Bundle(), config, io, worker, events, clock=clock, sleep=clock.sleep)
    assert report["passed"], report
    assert report["controller_state"] == "STOPPED"
    assert [r.step for r in worker.requests] == list(range(400))
    assert io.release_count == 1
    assert io.owner == ""
    assert io.writes[-1].kind == "damping"
    assert not report["vendor_motion_automatically_restored"]
    assert not report["physical_stop_verified"]
    first = worker.requests[0].observation
    assert len(first) == 347
    np.testing.assert_allclose(np.asarray(first[:210]).reshape(5, 42)[:, 30:], 0, atol=1e-6)
    for previous, request in zip(worker.requests, worker.requests[1:]):
        expected = 0.05 * np.sin(previous.step / 20)
        np.testing.assert_allclose(request.observation[198:210], expected, atol=1e-6)


@pytest.mark.parametrize("fault,reason", [
    ("crc", "feedback_crc_or_missing"), ("stale", "feedback_age"),
    ("future", "feedback_age"), ("machine", "mode_changed"), ("mode_pr", "mode_changed"),
    ("board_stale", "mainboard_fault_or_age"), ("board_error", "mainboard_fault_or_age"),
    ("board_unknown", "mainboard_fault_or_age"), ("board_bool", "mainboard_fault_or_age"),
    ("stop", "remote_stop"), ("start", "remote_intervention"),
    ("unknown_header", "unrecognized_remote"), ("motor_error", "motor_fault"),
    ("motor_warning", "motor_fault"), ("hot", "motor_thermal_or_voltage"),
    ("voltage", "motor_thermal_or_voltage"), ("speed", "joint_speed_or_torque"),
    ("torque", "joint_speed_or_torque"), ("joint_limit", "measured_joint_limit"),
    ("tilt", "base_tilt"), ("replayed", "replayed_feedback"),
    ("transport", "latched_callback_fault"),
])
def test_feedback_fault_latches_and_start_cannot_rearm(config, fault, reason):
    engine, io = running(config)
    now = T0 + 2_000_000
    snapshot = io.snapshot(now)
    row = snapshot.low
    if fault == "crc":
        snapshot = replace(snapshot, crc_valid=False)
    elif fault == "stale":
        row["received_monotonic_ns"] = now - 40_000_001
    elif fault == "future":
        row["received_monotonic_ns"] = now + 1
    elif fault == "machine":
        row["mode_machine"] = 7
    elif fault == "mode_pr":
        row["mode_pr"] = 1
    elif fault == "board_stale":
        snapshot = replace(snapshot, board_ns=now - 150_000_001)
    elif fault == "board_error":
        snapshot = replace(snapshot, board=(4096, 0, 0, 0, 0, 0))
    elif fault == "board_unknown":
        snapshot = replace(snapshot, board=(0, 0, 0, 0, 0, 1))
    elif fault == "board_bool":
        snapshot = replace(snapshot, board=(False, 0, 0, 0, 0, 0))
    elif fault in ("stop", "start"):
        snapshot = io.snapshot(now, mask=544 if fault == "stop" else 4)
    elif fault == "unknown_header":
        row["wireless_remote_hex"] = bytes(40).hex()
    elif fault == "motor_error":
        row["motor_slots"][0]["status"] = 1
    elif fault == "motor_warning":
        row["motor_slots"][0]["warning"] = 1
    elif fault == "hot":
        row["motor_slots"][0]["temperature_raw"][0] = 100
    elif fault == "voltage":
        row["motor_slots"][0]["voltage_raw"] = 20
    elif fault == "speed":
        row["motor_slots"][0]["dq"] = 51
    elif fault == "torque":
        row["motor_slots"][0]["tau_est"] = 1001
    elif fault == "joint_limit":
        row["motor_slots"][0]["q"] = 6
    elif fault == "tilt":
        row["quaternion_raw"] = [0, 1, 0, 0]
    elif fault == "replayed":
        row["tick"] = 999
    else:
        snapshot = replace(snapshot, transport_fault="latched_callback_fault")
    command, request = engine.advance(now, snapshot)
    assert engine.reason == reason
    assert request is None
    assert command is None or command.kind == "damping"
    command, _ = engine.advance(now + 2_000_000, io.snapshot(now + 2_000_000, mask=4))
    assert engine.reason == reason
    assert command is None or command.kind == "damping"
    with pytest.raises(c.ControlFault, match="single_use"):
        engine.begin(now + 4_000_000, io.snapshot(now + 4_000_000))


def test_mode_change_uses_fresh_mode_for_passive_frame(config):
    engine, io = running(config)
    snapshot = io.snapshot(T0 + 2_000_000)
    snapshot.low["mode_machine"] = 7
    command, _ = engine.advance(T0 + 2_000_000, snapshot)
    assert command.kind == "damping" and command.mode_machine == 7


def test_frozen_tick_and_loop_deadline(config):
    engine, io = running(config)
    for i in range(1, 23):
        now = T0 + i * 2_000_000
        snapshot = io.snapshot(now)
        snapshot.low["tick"] = 1000
        command, _ = engine.advance(now, snapshot)
        if command:
            engine.submitted(command, io.publish(command), now)
    assert engine.reason == "frozen_feedback"
    engine, io = running(config)
    engine.advance(T0 + 21_000_000, io.snapshot(T0 + 21_000_000))
    assert engine.reason == "supervisor_deadline"


@pytest.mark.parametrize("fault", ["stale", "wrong_seq", "wrong_step", "error", "nan", "too_large"])
def test_bad_actor_reply_stops_without_consuming_step(config, fault):
    engine, io, now, request = policy_pending(config)
    now += 2_000_000
    reply = c.Reply(request.sequence, request.step, (0.0,) * 12)
    if fault == "stale":
        now += 14_000_000
    elif fault == "wrong_seq":
        reply = replace(reply, sequence=999)
    elif fault == "wrong_step":
        reply = replace(reply, step=999)
    elif fault == "error":
        reply = replace(reply, error="actor crashed")
    elif fault == "nan":
        reply = replace(reply, action=(float("nan"),) * 12)
    else:
        reply = replace(reply, action=(4.1,) * 12)
    cmd, _ = engine.advance(now, io.snapshot(now), reply)
    assert cmd is None or cmd.kind == "damping"
    assert engine.reason and engine.step == 0


def test_failed_submission_invalidates_ledger_and_never_advances_step(config):
    engine, io, now, request = policy_pending(config)
    now += 2_000_000
    cmd, _ = engine.advance(now, io.snapshot(now), c.Reply(0, 0, (0.25,) * 12))
    engine.submitted(cmd, False, now)
    assert engine.reason == "command_write_failed"
    assert engine.step == 0
    assert engine.ledger.action is None


def test_no_uncommitted_proposal_can_be_overwritten(config):
    engine, io = running(config)
    engine.advance(T0 + 2_000_000, io.snapshot(T0 + 2_000_000))
    engine.advance(T0 + 4_000_000, io.snapshot(T0 + 4_000_000))
    assert engine.reason == "previous_command_not_committed"


def test_target_envelope_and_rate_are_not_silently_clipped(config):
    engine, io, now, _ = policy_pending(config)
    config.target_rate[:] = 0.001
    now += 2_000_000
    cmd, _ = engine.advance(now, io.snapshot(now), c.Reply(0, 0, (4.0,) * 12))
    assert cmd.kind == "damping"
    assert engine.reason == "target_slew"
    assert engine.step == 0


def test_nonidentity_mapping_and_submitted_action_roundtrip(config):
    config.signs[::2] = -1
    config.offsets[:] = np.linspace(-0.03, 0.03, 12)
    config.calibration["joint_signs"] = config.signs.tolist()
    config.calibration["joint_offsets_rad"] = config.offsets.tolist()
    target = Bundle.default_q + np.linspace(-0.1, 0.1, 12)
    cmd = c.make_command(target, config.kp, config.kd, config)
    ledger = c.CommandLedger(Bundle.default_q, config)
    assert ledger.action is None
    ledger.commit(cmd, True, T0)
    np.testing.assert_allclose(ledger.action, (target - Bundle.default_q) / 0.25, atol=1e-6)
    assert cmd.raw_kp[8] == config.kp[11]


class Crc:
    def Crc(self, frame):
        assert frame.crc == 0
        return 1234


def factory():
    return SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(35)])


@pytest.mark.parametrize("kind", ["position", "damping", "stop"])
def test_encoder_zeros_all_unused_slots(config, kind):
    command = c.make_command(Bundle.default_q, config.kp, config.kd, config, kind)
    frame = c.encode_command(command, factory, Crc())
    assert frame.crc == 1234
    for i, motor in enumerate(frame.motor_cmd):
        assert motor.mode == int(i < 12 and kind != "stop")
        assert motor.tau == motor.dq == motor.reserve == 0
        if i >= 12 or kind == "stop":
            assert motor.q == motor.kp == motor.kd == 0
        elif kind == "damping":
            assert motor.q == motor.kp == 0
    with pytest.raises(c.ControlFault):
        c.encode_command(replace(command, mode_pr=1), factory, Crc())


@pytest.mark.parametrize("stage", ["check", "wrong_owner", "release", "not_released"])
def test_service_failures_emit_no_commands(config, stage):
    io = Transport(config)
    if stage == "check":
        io.check_mode = lambda: (1, None)
    elif stage == "wrong_owner":
        io.owner = "unexpected"
    elif stage == "release":
        io.release_mode = lambda: (1, None)
    else:
        io.release_mode = lambda: (0, None)
    with pytest.raises(c.ControlFault):
        c.release_control(io, "synthetic-vendor")
    assert io.writes == []


def test_runner_actor_death_yields_only_passive_commands_after_fault(config):
    clock, worker, events = Clock(), Worker(), []
    io = Transport(config, clock)
    worker.poll = lambda: (_ for _ in ()).throw(c.ControlFault("actor_process_dead"))
    report = run_session(Bundle(), config, io, worker, events, clock=clock, sleep=clock.sleep)
    assert report["reason"] == "actor_process_dead"
    assert not report["passed"]
    assert all(cmd.kind == "damping" for cmd in io.writes)


def test_config_hardware_rejected_before_sdk_or_network(monkeypatch, tmp_path):
    import sys
    from as2_motion_imitation.deployment import controller_run
    monkeypatch.setattr(controller_run, "InferenceBundle", lambda _: Bundle())
    monkeypatch.setattr(controller_run, "network_preflight",
                        lambda *a: pytest.fail("must not touch network"))
    monkeypatch.setattr(controller_run, "ActorWorker",
                        lambda *a: pytest.fail("must not start actor"))
    monkeypatch.setattr(sys, "argv", [
        "controller", "--mode", "hardware", "--confirm-handoff",
        "--bundle", "unused", "--config", str(ROOT / "configs/controller-synthetic.json"),
        "--output", str(tmp_path / "rejected"),
    ])
    with pytest.raises(SystemExit) as result:
        controller_run.main()
    assert result.value.code == 2
    report = json.loads((tmp_path / "rejected/summary.json").read_text())
    assert report["hardware_execution_started"] is False
    assert not report["passed"]


@pytest.mark.parametrize("path,value", [
    (("safety", "actor_ms"), 20), (("safety", "feedback_ms"), float("nan")),
    (("safety", "ramp_s"), 0), (("limits", "kp"), [-1] * 12),
    (("calibration", "joint_signs"), [0] * 12),
    (("calibration", "sdk_slots"), list(range(12))),
])
def test_invalid_configuration_rejected(path, value):
    document = config_document()
    document[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        c.ControllerConfig(document)


def test_claiming_execution_enabled_does_not_complete_other_gates():
    document = config_document()
    document.update(profile="hardware", hardware_execution_enabled=True)
    config = c.ControllerConfig(document)
    errors = config.hardware_errors(Bundle())
    assert any("lowlevel_external_stop" in error for error in errors)
    assert any("standing_takeover" in error for error in errors)
    with pytest.raises(c.ControlFault):
        c.Controller(Bundle(), config)


def test_real_actor_input_and_output_contract(config):
    bundle = InferenceBundle(ROOT / "outputs/deployment/proprio-v2/bundle")
    engine, io = c.Controller(bundle, config), Transport(config)
    io.q[:] = bundle.default_q
    engine.begin(T0, io.snapshot(T0))
    request = None
    for i in range(1, 180):
        now = T0 + i * 2_000_000
        cmd, request = engine.advance(now, io.snapshot(now))
        engine.submitted(cmd, io.publish(cmd), now)
        if request:
            break
    assert request and len(request.observation) == 347
    action = bundle.act(request.observation)
    now += 2_000_000
    cmd, _ = engine.advance(now, io.snapshot(now), c.Reply(0, 0, tuple(action)))
    engine.submitted(cmd, io.publish(cmd), now)
    assert engine.step == 1 and not engine.reason
    np.testing.assert_allclose(engine.ledger.action, action, atol=1e-6)


def test_failed_stop_writes_do_not_report_success(config):
    clock, worker, events = Clock(), Worker(), []
    io = Transport(config, clock)
    original = io.publish
    io.publish = lambda cmd: False if cmd.kind == "damping" else original(cmd)
    report = run_session(Bundle(), config, io, worker, events, clock=clock, sleep=clock.sleep)
    assert report["policy_steps_submitted"] == 400
    assert report["write_failures"] > 0
    assert not report["passed"]


def test_operator_cancel_before_handoff_never_releases_or_publishes(config):
    clock = Clock()
    io = Transport(config, clock)
    result = run_session(Bundle(), config, io, Worker(), [], lambda: True, clock, clock.sleep)
    assert io.release_count == 0 and not io.writes
    assert not result["passed"]


def test_owner_loss_stops_all_publishing_to_avoid_competing_controller(config):
    engine, io = running(config)
    now = T0 + 2_000_000
    snapshot = replace(io.snapshot(now), transport_fault="motion_owner_lost")
    assert engine.advance(now, snapshot) == (None, None)
    assert engine.state == "STOPPED"
    assert engine.reason == "motion_owner_lost"


def test_nonstationary_takeover_is_rejected(config):
    io = Transport(config)
    snapshot = io.snapshot(T0)
    snapshot.low["motor_slots"][0]["dq"] = 0.3
    with pytest.raises(c.ControlFault, match="initial_joints_not_stationary"):
        c.Controller(Bundle(), config).begin(T0, snapshot)


def test_stop_during_service_query_prevents_release(config):
    clock = Clock()
    io = Transport(config, clock)
    original = io.snapshot
    io.check_mode = lambda: (setattr(io, "snapshot", lambda: replace(
        original(), transport_fault="remote_stop")) or (0, {"name": "synthetic-vendor"}))
    result = run_session(Bundle(), config, io, Worker(), [], clock=clock, sleep=clock.sleep)
    assert not result["passed"] and not io.writes and io.release_count == 0


def test_deployment_lock_is_exclusive_and_released(tmp_path):
    from as2_motion_imitation.deployment.controller_run import ControllerLock
    path = tmp_path / "controller.lock"
    first = ControllerLock(path)
    try:
        with pytest.raises(c.ControlFault, match="another_controller"):
            ControllerLock(path)
    finally:
        first.close()
    ControllerLock(path).close()


def test_cyclic_gc_state_is_restored_on_failure(config):
    import gc
    clock = Clock()
    io = Transport(config, clock)
    before = gc.isenabled()
    run_session(Bundle(), config, io, Worker(), [], lambda: True, clock, clock.sleep)
    assert gc.isenabled() == before

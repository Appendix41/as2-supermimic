"""Transport-independent, one-shot controller. Hardware approval is separate."""

from dataclasses import dataclass
import math

import numpy as np

from .bench_control import integer
from .remote_stop import decode_remote
from .runtime import finite_array, init_history, lowstate_sensor, projected_gravity, push_history


SLOTS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
ACCEPTANCE = (
    "joint_imu_calibration", "actuator_envelope", "lowlevel_external_stop",
    "firmware_command_timeout", "control_handoff", "standing_takeover",
)


class ControlFault(RuntimeError):
    pass


class ControllerConfig:
    """Explicit limits, with no force/bypass path in hardware validation."""

    def __init__(self, document):
        import copy
        self.document = copy.deepcopy(document)
        d = self.document
        if d.get("schema") != "as2-single-jump-controller-v1":
            raise ValueError("Unsupported controller config schema")
        self.profile = d["profile"]
        if self.profile not in ("simulation-only", "hardware"):
            raise ValueError("Unknown controller profile")
        self.calibration = d["calibration"]
        c = self.calibration
        if c["sdk_slots"] != SLOTS:
            raise ValueError("Unexpected AS2 motor mapping")
        self.signs = finite_array(c["joint_signs"], (12,), "joint signs").copy()
        self.offsets = finite_array(c["joint_offsets_rad"], (12,), "joint offsets").copy()
        rotation = finite_array(c["policy_from_imu"], (3, 3), "IMU rotation")
        if (not np.all(np.abs(self.signs) == 1)
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError("Invalid calibration transform")
        for name in ("lower", "upper", "kp", "takeover_kp", "kd", "stop_kd", "speed", "target_rate",
                     "tracking_error", "estimated_torque", "predicted_torque"):
            value = finite_array(d["limits"][name], (12,), name).copy()
            if name not in ("lower", "upper") and np.any(value <= 0):
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid joint interval")
        if np.any(self.takeover_kp > self.kp):
            raise ValueError("Takeover gains must not exceed policy gains")
        for name, bounds in {
            "feedback_ms": (1, 40), "board_ms": (1, 150), "loop_ms": (1, 20),
            "actor_ms": (1, 19), "tick_lateness_ms": (1, 19),
            "ramp_s": (0.2, 10), "warmup_s": (0.1, 2),
            "hold_s": (0.1, 5), "stop_s": (0.1, 2),
            "initial_error_rad": (0.001, 0.2), "initial_speed_rad_s": (0.01, 0.5),
            "tilt_rad": (0.01, 1.2),
            "temperature_max": (1, 100), "voltage_min": (1, 100),
            "voltage_max": (1, 100),
        }.items():
            value = d["safety"][name]
            if type(value) not in (float, int) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"Invalid {name}")
            setattr(self, name, value)
        if self.voltage_min >= self.voltage_max:
            raise ValueError("Invalid voltage interval")
        self.expected_machine = d["expected_mode_machine"]
        if not integer(self.expected_machine, 0, 255):
            raise ValueError("Explicit mode_machine required")

    def hardware_errors(self, bundle):
        errors = []
        d = self.document
        if self.profile != "hardware" or d.get("hardware_execution_enabled") is not True:
            errors.append("hardware profile and execution authorization are absent")
        if self.calibration.get("verified_on_robot") is not True:
            errors.append("joint/IMU calibration is unverified")
        if d.get("checkpoint_sha256") != bundle.manifest.get("checkpoint_sha256"):
            errors.append("checkpoint binding mismatch")
        if not isinstance(d.get("robot_id"), str) or not d["robot_id"].strip():
            errors.append("robot identity is unspecified")
        if not isinstance(d.get("expected_service_name"), str) or not d["expected_service_name"].strip():
            errors.append("live-queried service name is unspecified")
        for name in ACCEPTANCE:
            record = d.get("acceptance", {}).get(name, {})
            if record.get("verified") is not True or not isinstance(record.get("evidence"), str) or not record["evidence"].strip():
                errors.append(f"{name} has no verified evidence")
        return errors

    def require_hardware(self, bundle):
        errors = self.hardware_errors(bundle)
        if errors:
            raise ControlFault("; ".join(errors))


@dataclass(frozen=True)
class Snapshot:
    low: dict
    crc_valid: bool
    board_ns: int
    board: tuple
    transport_fault: str | None = None


@dataclass(frozen=True)
class Command:
    kind: str
    raw_q: tuple
    raw_kp: tuple
    raw_kd: tuple
    mode_machine: int
    mode_pr: int = 0


@dataclass(frozen=True)
class Request:
    sequence: int
    step: int
    requested_ns: int
    observation: tuple


@dataclass(frozen=True)
class Reply:
    sequence: int
    step: int
    action: tuple
    error: str | None = None


class CommandLedger:
    """Track SDK-accepted targets; this does not confirm motor receipt."""

    def __init__(self, default_q, config):
        self.default_q = finite_array(default_q, (12,), "default q").copy()
        self.config = config
        self.action = None
        self.submissions = 0
        self.last_ns = None

    def commit(self, command, accepted, now_ns):
        if accepted is not True:
            self.action = None
            raise ControlFault("command_write_failed")
        if self.last_ns is not None and now_ns <= self.last_ns:
            self.action = None
            raise ControlFault("command_clock")
        self.last_ns = now_ns
        self.submissions += 1
        if command.kind != "position":
            self.action = None
            return
        raw = finite_array(command.raw_q, (12,), "submitted target")
        q = raw[SLOTS] * self.config.signs + self.config.offsets
        action = (q - self.default_q) / np.float32(0.25)
        if np.any(np.abs(action) > 4.00001):
            self.action = None
            raise ControlFault("submitted_target_outside_action_contract")
        self.action = np.clip(action, -4, 4)


def make_command(q, kp, kd, config, kind="position", machine=None):
    q = finite_array(q, (12,), "command q")
    kp = finite_array(kp, (12,), "command kp")
    kd = finite_array(kd, (12,), "command kd")
    if kind not in ("position", "damping", "stop") or np.any(kp < 0) or np.any(kd < 0):
        raise ControlFault("invalid_command")
    raw_q, raw_kp, raw_kd = np.zeros((3, 12), dtype=np.float32)
    if kind == "position":
        raw_q[SLOTS] = (q - config.offsets) / config.signs
        raw_kp[SLOTS] = kp
    if kind != "stop":
        raw_kd[SLOTS] = kd
    return Command(kind, tuple(float(v) for v in raw_q),
                   tuple(float(v) for v in raw_kp), tuple(float(v) for v in raw_kd),
                   config.expected_machine if machine is None else machine)


def encode_command(command, factory, crc):
    """Official HG object/CRC; all 35 slots initialized, only 12 may be active."""
    if (not isinstance(command, Command) or command.kind not in ("position", "damping", "stop")
            or not integer(command.mode_machine, 0, 255) or command.mode_pr != 0):
        raise ControlFault("invalid_command_frame")
    for name in ("raw_q", "raw_kp", "raw_kd"):
        values = finite_array(getattr(command, name), (12,), name)
        if name != "raw_q" and np.any(values < 0):
            raise ControlFault("negative_gain")
    if command.kind != "position" and (any(command.raw_q) or any(command.raw_kp)):
        raise ControlFault("nonzero_passive_command")
    if command.kind == "stop" and any(command.raw_kd):
        raise ControlFault("nonzero_stop_gain")
    frame = factory()
    if len(frame.motor_cmd) != 35:
        raise ControlFault("expected_35_motor_slots")
    frame.mode_machine, frame.mode_pr = command.mode_machine, 0
    frame.reserve = [0] * 4
    for i, motor in enumerate(frame.motor_cmd):
        motor.mode = int(i < 12 and command.kind != "stop")
        motor.q = command.raw_q[i] if i < 12 else 0.0
        motor.kp = command.raw_kp[i] if i < 12 else 0.0
        motor.kd = command.raw_kd[i] if i < 12 else 0.0
        motor.dq = motor.tau = 0.0
        motor.reserve = 0
    frame.crc = 0
    frame.crc = crc.Crc(frame)
    return frame


class Controller:
    """One-use supervisor, called by the sole motor publishing process."""

    PERIOD_NS = 20_000_000

    def __init__(self, bundle, config):
        self.bundle, self.config = bundle, config
        if config.profile == "hardware":
            config.require_hardware(bundle)
        self.ledger = CommandLedger(bundle.default_q, config)
        self.state, self.reason = "DISARMED", None
        self.last_loop_ns = self.start_ns = self.stop_ns = None
        self.last_receipt = self.last_tick = self.last_tick_change = None
        self.history = None
        self.last_history_ns = None
        self.pending = self.ready_reply = self.proposal = None
        self.policy_start = None
        self.step = 0
        self.target = None
        self.last_target = None
        self.last_target_ns = None
        self.snapshot = None
        self.write_failures = 0
        self.stop_write_successes = 0

    def stop(self, reason, now_ns):
        if self.reason is None:
            self.reason, self.stop_ns = str(reason), now_ns
        self.pending = self.ready_reply = self.proposal = None
        self.state = "STOPPING"

    def _feedback(self, snapshot, now_ns):
        c = self.config
        if not isinstance(snapshot, Snapshot) or snapshot.crc_valid is not True:
            raise ControlFault("feedback_crc_or_missing")
        if snapshot.transport_fault:
            raise ControlFault(snapshot.transport_fault)
        row = snapshot.low
        receipt = row["received_monotonic_ns"]
        if not integer(receipt, 1, 2 ** 63 - 1) or not 0 <= now_ns - receipt <= c.feedback_ms * 1e6:
            raise ControlFault("feedback_age")
        if (row["mode_machine"] != c.expected_machine or row["mode_pr"] != 0
                or type(row["mode_machine"]) is not int):
            raise ControlFault("mode_changed")
        if (not integer(snapshot.board_ns, 1, 2 ** 63 - 1)
                or not 0 <= now_ns - snapshot.board_ns <= c.board_ms * 1e6
                or len(snapshot.board) != 6 or any(type(x) is not int or x != 0 for x in snapshot.board)):
            raise ControlFault("mainboard_fault_or_age")
        remote = decode_remote(bytes.fromhex(row["wireless_remote_hex"]))
        if not remote["header_matches_recorded_fixture"]:
            raise ControlFault("unrecognized_remote")
        # Never rely on mainboard bit 12: the real L2+B trial left it clear.
        if remote["l2_b"]:
            raise ControlFault("remote_stop")
        if remote["button_mask"]:
            raise ControlFault("remote_intervention")
        tick = row["tick"]
        if not integer(tick, 0, 2 ** 32 - 1):
            raise ControlFault("invalid_robot_tick")
        if self.last_tick is not None:
            advance = (tick - self.last_tick) % 2 ** 32
            if receipt < self.last_receipt or advance >= 2 ** 31:
                raise ControlFault("replayed_feedback")
            if not advance and now_ns - self.last_tick_change > c.feedback_ms * 1e6:
                raise ControlFault("frozen_feedback")
        if self.last_tick is None or tick != self.last_tick:
            self.last_tick_change = now_ns
        self.last_tick, self.last_receipt = tick, receipt
        motors = {m["slot"]: m for m in row["motor_slots"]}
        if len(motors) != len(row["motor_slots"]) or not all(i in motors for i in SLOTS):
            raise ControlFault("motor_slots")
        for i in SLOTS:
            motor = motors[i]
            if (type(motor["status"]) is not int or motor["status"] != 0
                    or type(motor["warning"]) is not int or motor["warning"] != 0):
                raise ControlFault("motor_fault")
            temperature = finite_array(motor["temperature_raw"], (2,), "temperature")
            voltage = motor["voltage_raw"]
            if (np.any(temperature > c.temperature_max) or not math.isfinite(voltage)
                    or not c.voltage_min <= voltage <= c.voltage_max):
                raise ControlFault("motor_thermal_or_voltage")
        q = finite_array([motors[i]["q"] for i in SLOTS], (12,), "joint q") * c.signs + c.offsets
        dq = finite_array([motors[i]["dq"] for i in SLOTS], (12,), "joint dq") * c.signs
        tau = finite_array([motors[i]["tau_est"] for i in SLOTS], (12,), "estimated torque")
        if np.any(q < c.lower) or np.any(q > c.upper):
            raise ControlFault("measured_joint_limit")
        if np.any(np.abs(dq) > c.speed) or np.any(np.abs(tau) > c.estimated_torque):
            raise ControlFault("joint_speed_or_torque")
        gravity = np.asarray(c.calibration["policy_from_imu"]) @ projected_gravity(row["quaternion_raw"])
        finite_array(row["gyroscope_raw"], (3,), "gyro")
        if -gravity[2] < math.cos(c.tilt_rad):
            raise ControlFault("base_tilt")
        self.snapshot = snapshot
        return q, dq

    def begin(self, now_ns, snapshot):
        if self.state != "DISARMED" or self.reason is not None:
            raise ControlFault("controller_is_single_use")
        if not integer(now_ns, 1, 2 ** 63 - 1):
            raise ControlFault("invalid_clock")
        q, dq = self._feedback(snapshot, now_ns)
        if np.any(np.abs(q - self.bundle.default_q) > self.config.initial_error_rad):
            raise ControlFault("initial_pose_not_standing")
        if np.any(np.abs(dq) > self.config.initial_speed_rad_s):
            raise ControlFault("initial_joints_not_stationary")
        self.initial_q = q.copy()
        self.start_ns = self.last_loop_ns = now_ns
        self.state = "SETTLE"

    def _sensor(self, now_ns):
        if self.ledger.action is None:
            raise ControlFault("no_submitted_action_history")
        return lowstate_sensor(
            self.snapshot.low, self.bundle.default_q, self.ledger.action,
            self.config.calibration, now_ns, self.config.feedback_ms,
            allow_unverified=self.config.profile == "simulation-only",
        )

    def _position(self, q, dq, target, kp, now_ns):
        c = self.config
        target = finite_array(target, (12,), "target")
        if (np.any(target < c.lower) or np.any(target > c.upper)
                or np.any(np.abs((target - self.bundle.default_q) / 0.25) > 4.00001)):
            raise ControlFault("target_joint_limit")
        if np.any(np.abs(target - q) > c.tracking_error):
            raise ControlFault("tracking_error")
        if np.any(np.abs(kp * (target - q) - c.kd * dq) > c.predicted_torque):
            raise ControlFault("predicted_torque")
        if self.last_target is not None:
            dt = min((now_ns - self.last_target_ns) / 1e9, self.PERIOD_NS / 1e9)
            if np.any(np.abs(target - self.last_target) > c.target_rate * dt + 1e-6):
                raise ControlFault("target_slew")
        return make_command(target, kp, c.kd, c)

    def _stopping(self, now_ns):
        # A fresh CRC-valid mode is needed even for a best-effort passive frame.
        s = self.snapshot
        if isinstance(s, Snapshot) and s.transport_fault and s.transport_fault.startswith("motion_owner_"):
            self.state = "STOPPED"
            return None, None
        row = s.low if isinstance(s, Snapshot) and isinstance(s.low, dict) else {}
        receipt = row.get("received_monotonic_ns")
        if (s is None or s.crc_valid is not True
                or not integer(receipt, 1, 2 ** 63 - 1)
                or not 0 <= now_ns - receipt <= self.config.feedback_ms * 1e6
                or row.get("mode_pr") != 0):
            self.state = "STOPPED"
            return None, None
        if now_ns - self.stop_ns >= self.config.stop_s * 1e9:
            self.state = "STOPPED"
            return None, None
        machine = row.get("mode_machine")
        if not integer(machine, 0, 255):
            self.state = "STOPPED"
            return None, None
        return make_command(np.zeros(12), np.zeros(12), self.config.stop_kd,
                            self.config, kind="damping", machine=machine), None

    def advance(self, now_ns, snapshot, reply=None):
        if self.state in ("DISARMED", "STOPPED"):
            return None, None
        if not integer(now_ns, 1, 2 ** 63 - 1):
            raise ControlFault("invalid_clock")
        if self.state == "STOPPING":
            # Do not reuse a stale mode after a mode change or dropped feedback.
            self.snapshot = snapshot if isinstance(snapshot, Snapshot) else None
            return self._stopping(now_ns)
        try:
            if self.proposal is not None:
                raise ControlFault("previous_command_not_committed")
            if not 0 < now_ns - self.last_loop_ns <= self.config.loop_ms * 1e6:
                raise ControlFault("supervisor_deadline")
            self.last_loop_ns = now_ns
            q, dq = self._feedback(snapshot, now_ns)
            c = self.config
            request = None
            if self.state == "SETTLE":
                elapsed = (now_ns - self.start_ns) / 1e9
                blend = min(elapsed / c.ramp_s, 1.0)
                target = self.initial_q * (1 - blend) + self.bundle.default_q * blend
                kp = c.takeover_kp * (1 - blend) + c.kp * blend
                if self.ledger.action is not None and (self.last_history_ns is None
                        or now_ns - self.last_history_ns >= self.PERIOD_NS):
                    sensor = self._sensor(now_ns)
                    self.history = init_history(sensor) if self.history is None else push_history(self.history, sensor)
                    self.last_history_ns = now_ns
                if elapsed >= c.ramp_s + c.warmup_s:
                    self.state = "POLICY"
                    self.policy_start = now_ns
                    self.target = self.bundle.default_q.copy()
            if self.state == "POLICY":
                target, kp = self.target, c.kp
                if self.pending is not None:
                    if now_ns - self.pending.requested_ns > c.actor_ms * 1e6:
                        raise ControlFault("actor_timeout")
                    if reply is not None:
                        if (not isinstance(reply, Reply) or reply.error
                                or type(reply.sequence) is not int or type(reply.step) is not int
                                or (reply.sequence, reply.step) != (self.pending.sequence, self.step)):
                            raise ControlFault("actor_reply_mismatch_or_error")
                        action = finite_array(reply.action, (12,), "actor action")
                        if np.any(np.abs(action) > 4):
                            raise ControlFault("actor_action_bounds")
                        self.ready_reply = reply
                        target = self.bundle.target(action)
                elif reply is not None:
                    raise ControlFault("unsolicited_actor_reply")
                due = self.policy_start + self.step * self.PERIOD_NS
                if self.pending is None and now_ns >= due and self.step < self.bundle.steps:
                    if now_ns - due > c.tick_lateness_ms * 1e6:
                        raise ControlFault("policy_tick_missed")
                    sensor = self._sensor(now_ns)
                    if self.last_history_ns != now_ns:
                        self.history = push_history(self.history, sensor)
                        self.last_history_ns = now_ns
                    request = Request(self.step, self.step, now_ns,
                                      tuple(float(v) for v in self.bundle.observe(self.history, self.step)))
                    self.pending = request
                if self.step == self.bundle.steps and now_ns >= due:
                    self.state = "HOLD"
                    self.hold_start = now_ns
            if self.state == "HOLD":
                target, kp = self.target, c.kp
                if now_ns - self.hold_start >= c.hold_s * 1e9:
                    self.stop("one_shot_complete", now_ns)
                    return self._stopping(now_ns)
            command = self._position(q, dq, target, kp, now_ns)
            self.proposal = command
            return command, request
        except (ControlFault, ValueError, KeyError, TypeError, OverflowError) as error:
            self.snapshot = snapshot if isinstance(snapshot, Snapshot) else None
            self.stop(str(error), now_ns)
            return self._stopping(now_ns)

    def submitted(self, command, accepted, now_ns):
        try:
            if accepted is not True:
                self.write_failures += 1
            if command.kind == "position" and command != self.proposal:
                raise ControlFault("unmatched_command_commit")
            self.ledger.commit(command, accepted, now_ns)
            if command.kind == "damping":
                self.stop_write_successes += 1
            if command.kind == "position":
                target = np.asarray(command.raw_q)[SLOTS] * self.config.signs + self.config.offsets
                # Measure slew from the previous changed target, not retransmissions.
                if self.last_target is None or not np.array_equal(target, self.last_target):
                    self.last_target, self.last_target_ns = target, now_ns
                if self.ready_reply is not None:
                    self.target = target
                    self.step += 1
                    self.pending = self.ready_reply = None
            self.proposal = None
        except ControlFault as error:
            self.stop(str(error), now_ns)


def release_control(transport, expected_name, before_release=lambda: None):
    """Allow begin() after release succeeds and the reported owner is empty."""
    code, mode = transport.check_mode()
    if code != 0 or not isinstance(mode, dict) or mode.get("name") != expected_name:
        raise ControlFault("unexpected_motion_owner")
    if not expected_name:
        raise ControlFault("explicit_motion_owner_required")
    before_release()
    code, _ = transport.release_mode()
    if code != 0:
        raise ControlFault("release_mode_failed")
    code, mode = transport.check_mode()
    if code != 0 or not isinstance(mode, dict) or mode.get("name") != "":
        raise ControlFault("motion_release_not_confirmed")

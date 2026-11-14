"""Offline bring-up logic for one supported joint, not a hardware controller."""

from dataclasses import dataclass
import math


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def vector(values, size):
    return (isinstance(values, tuple) and len(values) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in values))


@dataclass(frozen=True)
class Feedback:
    received_ns: int
    tick: int
    mode_machine: int
    mode_pr: int
    q: tuple
    dq: tuple
    status: tuple
    crc_valid: bool


@dataclass(frozen=True)
class Health:
    received_ns: int
    mainboard_status: int
    operator_enable: bool


@dataclass(frozen=True)
class Intent:
    sequence: int
    generated_ns: int
    target_rad: float


@dataclass(frozen=True)
class Decision:
    enabled: bool
    reason: str
    slot: int
    target_rad: float
    mode_machine: int
    mode_pr: int


class ProbeGuard:
    """Single-use, latched guard with provisional bench bounds only.

    Bounds below are software test assumptions, not AS2 hardware ratings.
    The guard cannot authorize hardware actuation, establish physical
    restraint, verify an E-stop or provide protection against its own death.
    """

    FEEDBACK_NS = 40_000_000
    HEALTH_NS = 150_000_000
    INTENT_NS = 40_000_000
    LOOP_NS = 20_000_000
    DURATION_NS = 3_000_000_000
    EXCURSION_RAD = 0.02
    TARGET_RATE_RAD_S = 0.05
    ACTUAL_SPEED_RAD_S = 0.5
    TRACKING_RAD = 0.05

    def __init__(self, initial, now_ns, slot, lower_rad, upper_rad):
        if (not integer(slot, 0, 11) or not integer(now_ns, 1, 2 ** 63 - 1)
                or not self.feedback_valid(initial)
                or not 0 <= now_ns - initial.received_ns <= self.FEEDBACK_NS
                or not all(type(x) in (int, float) and math.isfinite(x)
                           for x in (lower_rad, upper_rad))
                or not lower_rad < initial.q[slot] < upper_rad):
            raise ValueError("Invalid initial probe state or explicit raw joint bounds")
        self.slot = slot
        self.lower, self.upper = lower_rad, upper_rad
        self.initial_q = initial.q
        self.machine, self.mode_pr = initial.mode_machine, initial.mode_pr
        self.start_ns = self.last_loop_ns = self.last_tick_change_ns = now_ns
        self.last_tick = initial.tick
        self.last_receipt_ns = initial.received_ns
        self.last_intent = None
        self.last_target = initial.q[slot]
        self.reason = None

    @staticmethod
    def feedback_valid(state):
        return (isinstance(state, Feedback)
                and integer(state.received_ns, 1, 2 ** 63 - 1)
                and integer(state.tick, 0, 2 ** 32 - 1)
                and integer(state.mode_machine, 0, 255)
                and integer(state.mode_pr, 0, 1)
                and vector(state.q, 12) and vector(state.dq, 12)
                and isinstance(state.status, tuple) and len(state.status) == 12
                and all(integer(v, 0, 2 ** 32 - 1) and v == 0 for v in state.status)
                and state.crc_valid is True)

    def stop(self, reason):
        if self.reason is None:
            self.reason = reason
        return Decision(False, self.reason, self.slot, 0.0, self.machine, self.mode_pr)

    def evaluate(self, now_ns, feedback, health, intent):
        if self.reason is not None:
            return self.stop(self.reason)
        if not integer(now_ns, 1, 2 ** 63 - 1):
            return self.stop("invalid_clock")
        elapsed_ns = now_ns - self.last_loop_ns
        if not 0 < elapsed_ns <= self.LOOP_NS:
            return self.stop("loop_deadline_or_clock")
        if now_ns - self.start_ns >= self.DURATION_NS:
            return self.stop("duration_complete")
        self.last_loop_ns = now_ns
        if not self.feedback_valid(feedback):
            return self.stop("invalid_feedback_or_crc")
        if not 0 <= now_ns - feedback.received_ns <= self.FEEDBACK_NS:
            return self.stop("feedback_age")
        if (feedback.mode_machine, feedback.mode_pr) != (self.machine, self.mode_pr):
            return self.stop("mode_changed")
        if feedback.received_ns < self.last_receipt_ns:
            return self.stop("feedback_receipt_replayed")
        self.last_receipt_ns = feedback.received_ns
        advance = (feedback.tick - self.last_tick) % (2 ** 32)
        if advance >= 2 ** 31:
            return self.stop("robot_tick_replayed")
        if advance:
            self.last_tick = feedback.tick
            self.last_tick_change_ns = now_ns
        elif now_ns - self.last_tick_change_ns > self.FEEDBACK_NS:
            return self.stop("robot_tick_frozen")
        if (not isinstance(health, Health)
                or not integer(health.received_ns, 1, 2 ** 63 - 1)
                or not 0 <= now_ns - health.received_ns <= self.HEALTH_NS
                or type(health.mainboard_status) is not int
                or health.mainboard_status != 0
                or health.operator_enable is not True):
            return self.stop("health_or_operator_enable")
        if (not isinstance(intent, Intent)
                or not integer(intent.sequence, 0, 2 ** 63 - 1)
                or not integer(intent.generated_ns, 1, 2 ** 63 - 1)
                or type(intent.target_rad) not in (float, int)
                or not math.isfinite(intent.target_rad)
                or not 0 <= now_ns - intent.generated_ns <= self.INTENT_NS):
            return self.stop("invalid_or_expired_intent")
        if self.last_intent is not None:
            previous = self.last_intent
            if (intent.sequence < previous.sequence
                    or (intent.sequence == previous.sequence and intent != previous)
                    or (intent.sequence > previous.sequence
                        and intent.generated_ns <= previous.generated_ns)):
                return self.stop("intent_replayed_or_mutated")
        q = feedback.q[self.slot]
        target = intent.target_rad
        if (not self.lower <= target <= self.upper or not self.lower <= q <= self.upper
                or abs(target - self.initial_q[self.slot]) > self.EXCURSION_RAD + 1e-9
                or abs(q - target) > self.TRACKING_RAD):
            return self.stop("joint_envelope")
        if any(abs(v) > self.ACTUAL_SPEED_RAD_S for v in feedback.dq):
            return self.stop("joint_speed")
        if any(abs(qi - initial) > self.TRACKING_RAD
               for i, (qi, initial) in enumerate(zip(feedback.q, self.initial_q))
               if i != self.slot):
            return self.stop("unselected_joint_motion")
        allowed_step = self.TARGET_RATE_RAD_S * elapsed_ns / 1e9
        if abs(target - self.last_target) > allowed_step + 1e-9:
            return self.stop("target_rate")
        self.last_intent, self.last_target = intent, target
        return Decision(True, "bench_candidate", self.slot, target, self.machine, self.mode_pr)


def pulse_target(initial_rad, elapsed_s, amplitude_rad=0.015):
    """Slow out-and-back bench request; never a jump or a reference replay."""
    if (not all(type(v) in (float, int) and math.isfinite(v)
                for v in (initial_rad, elapsed_s, amplitude_rad))
            or not 0 < abs(amplitude_rad) <= 0.02 or elapsed_s < 0):
        raise ValueError("Invalid bounded pulse")
    if elapsed_s < 0.5:
        weight = elapsed_s / 0.5
    elif elapsed_s < 0.75:
        weight = 1.0
    elif elapsed_s < 1.25:
        weight = 1.0 - (elapsed_s - 0.75) / 0.5
    else:
        weight = 0.0
    return initial_rad + amplitude_rad * weight


def build_sdk_frame(decision, factory, crc):
    """Encode with the official SDK; return an object, never publish it.

    The disabled frame has all motors in STOP mode. That is only a protocol
    representation, not evidence of safe braking or physical stop delivery.
    """
    if (not isinstance(decision, Decision) or type(decision.enabled) is not bool
            or not integer(decision.slot, 0, 11)
            or not integer(decision.mode_machine, 0, 255)
            or not integer(decision.mode_pr, 0, 1)
            or type(decision.target_rad) not in (int, float)
            or not math.isfinite(decision.target_rad)):
        raise ValueError("Invalid command decision")
    command = factory()
    if len(command.motor_cmd) != 35:
        raise ValueError("Expected HG 35-slot command")
    command.mode_machine, command.mode_pr = decision.mode_machine, decision.mode_pr
    command.reserve = [0] * 4
    for motor in command.motor_cmd:
        motor.mode = 0
        motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0
        motor.reserve = 0
    if decision.enabled:
        motor = command.motor_cmd[decision.slot]
        motor.mode, motor.q = 1, decision.target_rad
        motor.kp, motor.kd = 2.0, 0.2
    command.crc = 0
    command.crc = crc.Crc(command)
    return command

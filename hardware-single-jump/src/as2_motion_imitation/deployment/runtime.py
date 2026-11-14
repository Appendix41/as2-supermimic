"""NumPy actor inference and observation assembly; no robot transport."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "as2-proprio-single-jump-inference-v2"
SENSOR_DIM = 42
HISTORY = 5
OBS_DIM = 347
ACTION_DIM = 12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite shape {shape}")
    return result


def projected_gravity(quaternion):
    q = finite_array(quaternion, (4,), "wxyz quaternion")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=0.01):
        raise ValueError("IMU quaternion must be unit length")
    w, x, y, z = q / norm
    return np.array([2 * (w * y - x * z), -2 * (y * z + w * x),
                     2 * (x * x + y * y) - 1], dtype=np.float32)


def lowstate_sensor(row, default_q, last_applied_action, calibration, now_ns,
                    max_age_ms=40.0, allow_unverified=False):
    """Decode a LowState record without opening a robot connection."""
    if row.get("topic") != "rt/lowstate":
        raise ValueError("Only high-rate LowState is a policy input")
    if calibration.get("verified_on_robot") is not True and not allow_unverified:
        raise ValueError("Hardware joint/IMU calibration has not been verified")
    if not np.isfinite(max_age_ms) or not 0 < max_age_ms <= 100:
        raise ValueError("Invalid feedback age limit")
    age_ms = (now_ns - row["received_monotonic_ns"]) / 1e6
    if not 0 <= age_ms <= max_age_ms:
        raise ValueError("Stale or future feedback")
    slots = calibration["sdk_slots"]
    if slots != [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]:
        raise ValueError("Unexpected documented AS2 slot mapping")
    signs = finite_array(calibration["joint_signs"], (12,), "joint signs")
    offsets = finite_array(calibration["joint_offsets_rad"], (12,), "joint offsets")
    rotation = finite_array(calibration["policy_from_imu"], (3, 3), "IMU frame")
    if not np.all(np.abs(signs) == 1):
        raise ValueError("Joint signs must be +/-1")
    if not (np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("IMU frame must be a proper rotation")
    motors = {m["slot"]: m for m in row["motor_slots"]}
    if len(motors) != len(row["motor_slots"]) or not all(slot in motors for slot in slots):
        raise ValueError("Missing or duplicate motor slots")
    q = finite_array([motors[i]["q"] for i in slots], (12,), "raw q") * signs + offsets
    qd = finite_array([motors[i]["dq"] for i in slots], (12,), "raw dq") * signs
    omega = rotation @ finite_array(row["gyroscope_raw"], (3,), "raw gyro")
    gravity = rotation @ projected_gravity(row["quaternion_raw"])
    return sensor_frame(omega, gravity, q, qd, last_applied_action, default_q)


def sensor_frame(body_omega, gravity, q, qd, last_action, default_q):
    omega = finite_array(body_omega, (3,), "body omega")
    gravity = finite_array(gravity, (3,), "gravity")
    if not np.isclose(np.linalg.norm(gravity), 1.0, atol=0.01):
        raise ValueError("Projected gravity must be a unit vector")
    q = finite_array(q, (12,), "joint position")
    qd = finite_array(qd, (12,), "joint velocity")
    last = finite_array(last_action, (12,), "last applied action")
    default_q = finite_array(default_q, (12,), "default q")
    if np.any(np.abs(last) > 4):
        raise ValueError("Previous action is outside the trained action contract")
    return np.concatenate((omega, gravity, q - default_q, qd, last))


def init_history(sensor_frame):
    return np.tile(finite_array(sensor_frame, (SENSOR_DIM,), "sensor"), HISTORY)


def push_history(history, sensor_frame):
    history = finite_array(history, (SENSOR_DIM * HISTORY,), "history")
    sensor_frame = finite_array(sensor_frame, (SENSOR_DIM,), "sensor")
    return np.concatenate((history[SENSOR_DIM:], sensor_frame))


class InferenceBundle:
    """Load the actor and reference; inputs must use policy joint order."""

    def __init__(self, directory):
        directory = Path(directory).resolve()
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("hardware_execution_enabled") is not False:
            raise ValueError("Unsupported or unsafe inference bundle")
        if m.get("state_frame") != "forward-axis-v1":
            raise ValueError("This bundle supports the original single forward jump only")
        if (m.get("obs_dim"), m.get("action_dim"), m.get("history")) != (OBS_DIM, 12, 5):
            raise ValueError("Observation contract mismatch")
        if m.get("required_pose") is not False:
            raise ValueError("Pose-dependent bundles are not accepted")
        self.steps = m["steps"]
        self.dt = m["dt"]
        if type(self.steps) is not int or self.steps != 400 or self.dt != 0.02:
            raise ValueError("Unexpected single-jump timing")
        for filename in ("actor.npz", "reference.npz"):
            if sha256(directory / filename) != m["files"][filename]:
                raise ValueError(f"Hash mismatch: {filename}")
        with np.load(directory / "actor.npz", allow_pickle=False) as data:
            self.mean = finite_array(data["mean"], (OBS_DIM,), "mean").copy()
            var = finite_array(data["var"], (OBS_DIM,), "variance")
            if np.any(var < 0):
                raise ValueError("Negative normalization variance")
            self.std = np.sqrt(var) + np.float32(1e-6)
            dimensions = (OBS_DIM, 512, 256, 128, 12)
            self.layers = []
            for index, (incoming, outgoing) in enumerate(zip(dimensions, dimensions[1:])):
                weight = finite_array(data[f"weight_{index}"], (outgoing, incoming), "weight")
                bias = finite_array(data[f"bias_{index}"], (outgoing,), "bias")
                self.layers.append((np.ascontiguousarray(weight), bias.copy()))
        with np.load(directory / "reference.npz", allow_pickle=False) as data:
            self.reference_states = finite_array(
                data["states"], (self.steps, 4, 34), "reference states").copy()
            self.phases = finite_array(data["phases"], (self.steps,), "phases").copy()
            self.default_q = finite_array(data["default_q"], (12,), "default q").copy()
        for array in (self.mean, self.std, self.reference_states,
                      self.phases, self.default_q):
            array.setflags(write=False)
        for weight, bias in self.layers:
            weight.setflags(write=False)
            bias.setflags(write=False)

    def sensor(self, body_omega, projected_gravity, q, qd, last_applied_action):
        """Build a sensor frame using the last applied normalized action."""
        return sensor_frame(body_omega, projected_gravity, q, qd,
                            last_applied_action, self.default_q)

    def observe(self, history, step):
        if type(step) is not int or not 0 <= step < self.steps:
            raise ValueError("Step outside the one-shot clip; no wrapping or end extrapolation")
        history = finite_array(history, (210,), "history")
        goals = self.reference_states[step].reshape(-1)
        return np.concatenate((history, goals, self.phases[step:step + 1]))

    def act(self, observation):
        x = finite_array(observation, (OBS_DIM,), "observation")
        x = np.clip((x - self.mean) / self.std, -10, 10)
        with np.errstate(over="raise", invalid="raise"):
            for index, (weight, bias) in enumerate(self.layers):
                x = weight @ x + bias
                if index != len(self.layers) - 1:
                    negative = x < 0
                    x[negative] = np.expm1(x[negative])
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite actor output")
        return np.clip(x, -4, 4)

    def target(self, action):
        action = finite_array(action, (12,), "action")
        return self.default_q + np.float32(0.25) * np.clip(action, -4, 4)

"""NumPy actor inference and observation assembly; no robot transport."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "as2-single-jump-inference-v1"
SENSOR_DIM = 42
HISTORY = 5
OBS_DIM = 359
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


def pose_delta(origin, pose):
    origin = np.asarray(origin, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    c, s = np.cos(origin[..., 2]), np.sin(origin[..., 2])
    dx, dy = pose[..., 0] - origin[..., 0], pose[..., 1] - origin[..., 1]
    angle = pose[..., 2] - origin[..., 2]
    return np.stack((c * dx + s * dy, -s * dx + c * dy,
                     np.arctan2(np.sin(angle), np.cos(angle))), axis=-1)


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
        if (m.get("obs_dim"), m.get("action_dim"), m.get("history")) != (359, 12, 5):
            raise ValueError("Observation contract mismatch")
        self.steps = m["steps"]
        self.dt = m["dt"]
        if type(self.steps) is not int or self.steps != 400 or self.dt != 0.02:
            raise ValueError("Unexpected single-jump timing")
        for filename in ("actor.npz", "reference.npz"):
            if sha256(directory / filename) != m["files"][filename]:
                raise ValueError(f"Hash mismatch: {filename}")
        with np.load(directory / "actor.npz", allow_pickle=False) as data:
            self.mean = finite_array(data["mean"], (359,), "mean").copy()
            var = finite_array(data["var"], (359,), "variance")
            if np.any(var < 0):
                raise ValueError("Negative normalization variance")
            self.std = np.sqrt(var) + np.float32(1e-6)
            dimensions = (359, 512, 256, 128, 12)
            self.layers = []
            for index, (incoming, outgoing) in enumerate(zip(dimensions, dimensions[1:])):
                weight = finite_array(data[f"weight_{index}"], (outgoing, incoming), "weight")
                bias = finite_array(data[f"bias_{index}"], (outgoing,), "bias")
                self.layers.append((np.ascontiguousarray(weight), bias.copy()))
        with np.load(directory / "reference.npz", allow_pickle=False) as data:
            self.reference_states = finite_array(
                data["states"], (self.steps, 4, 34), "reference states").copy()
            self.reference_poses = finite_array(
                data["poses"], (self.steps, 4, 3), "reference poses").copy()
            self.phases = finite_array(data["phases"], (self.steps,), "phases").copy()
            self.default_q = finite_array(data["default_q"], (12,), "default q").copy()
        for array in (self.mean, self.std, self.reference_states, self.reference_poses,
                      self.phases, self.default_q):
            array.setflags(write=False)
        for weight, bias in self.layers:
            weight.setflags(write=False)
            bias.setflags(write=False)

    def sensor(self, body_omega, projected_gravity, q, qd, last_applied_action):
        """Build a sensor frame using the last applied normalized action."""
        omega = finite_array(body_omega, (3,), "body omega")
        gravity = finite_array(projected_gravity, (3,), "gravity")
        if not np.isclose(np.linalg.norm(gravity), 1.0, atol=0.01):
            raise ValueError("Projected gravity must be a unit vector")
        q = finite_array(q, (12,), "joint position")
        qd = finite_array(qd, (12,), "joint velocity")
        last = finite_array(last_applied_action, (12,), "last applied action")
        if np.any(np.abs(last) > 4):
            raise ValueError("Previous action is outside the trained action contract")
        return np.concatenate((omega, gravity, q - self.default_q, qd, last))

    def observe(self, history, pose, origin, step):
        if type(step) is not int or not 0 <= step < self.steps:
            raise ValueError("Step outside the one-shot clip; no wrapping or end extrapolation")
        history = finite_array(history, (210,), "history")
        pose = finite_array(pose, (3,), "measured pose")
        origin = finite_array(origin, (3,), "episode origin")
        relative = pose_delta(origin, pose)
        delta = pose_delta(relative, self.reference_poses[step])
        goals = np.concatenate((self.reference_states[step], delta), axis=-1).reshape(-1)
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

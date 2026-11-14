"""Load, validate and interpolate reference motion clips."""

import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco
import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import quat_apply_inverse

from as2_motion_imitation import SCHEMA_VERSION
from as2_motion_imitation.physics import Physics
from as2_supermimic.checkpoints import robot_signature
from as2_supermimic.robot.morphology import JOINT_NAMES, ROOT_BODY

FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)


def slerp(q0, q1, alpha):
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0, 1)
    angle = torch.acos(dot.clamp(max=1 - 1e-7))
    w = alpha[..., None]
    spherical = (torch.sin((1 - w) * angle) * q0 + torch.sin(w * angle) * q1) / angle.sin()
    linear = (1 - w) * q0 + w * q1
    result = torch.where(dot > 0.9995, linear, spherical)
    return torch.nn.functional.normalize(result, dim=-1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_clip(path, arrays, physics, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **arrays, fps=np.array(50.0),
        joint_names=np.array(JOINT_NAMES), body_names=np.array(physics.body_names),
    )
    meta = {
        "schema": SCHEMA_VERSION, "robot": robot_signature(),
        "state_frame": physics.state_frame,
        "units": {"length": "m", "angle": "rad", "time": "s"},
        "axes": "x-forward-y-left-z-up", "quaternion": "wxyz",
        "sha256": sha256(path), "provenance": provenance,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


class MotionClip:
    def __init__(self, path, device="cpu"):
        self.path = Path(path)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        m = self.metadata
        same_robot = json.dumps(m.get("robot"), sort_keys=True) == json.dumps(robot_signature(), sort_keys=True)
        if m.get("schema") != SCHEMA_VERSION or not same_robot:
            raise ValueError("incompatible motion schema or robot signature")
        if m.get("units") != {"length": "m", "angle": "rad", "time": "s"}:
            raise ValueError("motion units must be SI")
        if m.get("quaternion") != "wxyz" or m.get("axes") != "x-forward-y-left-z-up":
            raise ValueError("incompatible quaternion/coordinate convention")
        if m.get("sha256") != sha256(self.path):
            raise ValueError("motion SHA256 mismatch")
        with np.load(self.path, allow_pickle=False) as f:
            raw = {k: f[k] for k in f.files}
        self.fps = float(raw["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        names = raw["joint_names"].tolist()
        if len(names) != len(set(names)) or set(names) != set(JOINT_NAMES):
            raise ValueError("joint names must match AS2 exactly")
        self.joint_indexes = [names.index(n) for n in JOINT_NAMES]
        self.state_frame = m.get("state_frame", "forward-axis-v1")
        self.physics = Physics(state_frame=self.state_frame)
        bodies = raw["body_names"].tolist()
        if bodies != list(self.physics.body_names):
            raise ValueError("body names/order must match the AS2 model")
        self.frames = len(raw["joint_pos"])
        self.duration = (self.frames - 1) / self.fps
        if self.frames < 2:
            raise ValueError("motion requires at least two frames")
        b = len(bodies)
        shapes = ((self.frames, 12), (self.frames, 12), (self.frames, b, 3),
                  (self.frames, b, 4), (self.frames, b, 3), (self.frames, b, 3))
        for key, shape in zip(FIELDS, shapes):
            if raw[key].shape != shape or not np.isfinite(raw[key]).all():
                raise ValueError(f"invalid shape or nonfinite values: {key}")
        norms = np.linalg.norm(raw["body_quat_w"], axis=-1)
        if not np.allclose(norms, 1, atol=1e-4):
            raise ValueError("quaternions must be unit length")
        q = raw["joint_pos"][:, self.joint_indexes]
        limits = self.physics.model.jnt_range[self.physics.jids]
        if np.any(q < limits[:, 0] - 1e-3) or np.any(q > limits[:, 1] + 1e-3):
            raise ValueError("reference violates joint limits")
        if np.abs(raw["joint_vel"]).max() > 30:
            raise ValueError("reference exceeds conservative 30 rad/s screening limit")
        self.loader = MotionLoader(str(path), torch.arange(b, device=device), device)
        self.arrays = {key: getattr(self.loader, key) for key in FIELDS}
        for key in ("joint_pos", "joint_vel"):
            self.arrays[key] = self.arrays[key][:, self.joint_indexes]
        quat = self.arrays["body_quat_w"].clone()
        for i in range(1, self.frames):
            quat[i] *= torch.where((quat[i - 1] * quat[i]).sum(-1, keepdim=True) < 0, -1, 1)
        self.arrays["body_quat_w"] = quat
        self.root = bodies.index(ROOT_BODY)
        # Derive limb positions from the model's named joints.
        states, poses, feet = [], [], []
        for i in range(self.frames):
            p = self.physics
            p.data.qpos[:3] = raw["body_pos_w"][i, self.root]
            p.data.qpos[3:7] = raw["body_quat_w"][i, self.root]
            p.data.qpos[p.qadr] = q[i]
            p.data.qvel[:] = 0
            mujoco.mj_forward(p.model, p.data)
            if np.max(np.abs(p.data.xpos[1:] - raw["body_pos_w"][i])) > 1e-4:
                raise ValueError("reference body positions disagree with AS2 FK")
            agreement = np.abs(np.sum(p.data.xquat[1:] * raw["body_quat_w"][i], axis=-1))
            if np.any(agreement < 1 - 1e-5):
                raise ValueError("reference body orientations disagree with AS2 FK")
            s, pose = p.state()
            states.append(s)
            poses.append(pose)
            feet.append(p.data.site_xpos[p.foot_ids].copy())
        self.states = torch.tensor(np.array(states), device=device)
        self.poses = torch.tensor(np.array(poses), device=device)
        self.feet = torch.tensor(np.array(feet), dtype=torch.float32, device=device)
        self.default_q = torch.tensor(self.physics.default_q, dtype=torch.float32, device=device)
        # Velocities belong to the recorded physics, not the zero-velocity FK pass.
        yaw = self.poses[:, 2]
        lin = self.arrays["body_lin_vel_w"][:, self.root]
        self.states[:, 4:7] = torch.stack([
            yaw.cos() * lin[:, 0] + yaw.sin() * lin[:, 1],
            -yaw.sin() * lin[:, 0] + yaw.cos() * lin[:, 1], lin[:, 2],
        ], -1)
        self.states[:, 7:10] = quat_apply_inverse(
            quat[:, self.root], self.arrays["body_ang_vel_w"][:, self.root]
        )
        self.states[:, 22:34] = self.arrays["joint_vel"]
        self.device = device

    def sample(self, times):
        t = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        if not torch.isfinite(t).all():
            raise ValueError("nonfinite reference time")
        x = t.clamp(0, self.duration) * self.fps
        lo = x.floor().long().clamp(max=self.frames - 1)
        hi = (lo + 1).clamp(max=self.frames - 1)
        alpha = x - lo

        def lerp(a):
            w = alpha.reshape(*alpha.shape, *([1] * (a.ndim - 1)))
            return a[lo] * (1 - w) + a[hi] * w

        out = {k: lerp(v) for k, v in self.arrays.items() if k != "body_quat_w"}
        q = self.arrays["body_quat_w"]
        shape = q[lo].shape
        w = alpha[..., None].expand(*alpha.shape, shape[-2]).reshape(-1)
        out["body_quat_w"] = slerp(q[lo].reshape(-1, 4), q[hi].reshape(-1, 4), w).reshape(shape)
        out["state"], out["pose"] = lerp(self.states), lerp(self.poses)
        # Heading interpolation must cross +/- pi along the shorter arc.
        dyaw = torch.atan2((self.poses[hi, 2] - self.poses[lo, 2]).sin(),
                          (self.poses[hi, 2] - self.poses[lo, 2]).cos())
        out["pose"][..., 2] = self.poses[lo, 2] + alpha * dyaw
        out["state"][..., 1:4] = quat_apply_inverse(
            out["body_quat_w"][..., self.root, :],
            torch.tensor([0., 0., -1.], device=self.device).expand(*t.shape, 3),
        )
        out["phase"] = t.clamp(0, self.duration) / self.duration
        out["ended"] = t >= self.duration
        return out


class MotionClock:
    """Track each environment's clip time without wrapping at the end."""

    def __init__(self, clip, n):
        self.clip = clip
        self.time = torch.zeros(n, device=clip.device)
        self._elapsed = torch.zeros(n, dtype=torch.float64, device=clip.device)
        self.clip_id = torch.zeros(n, dtype=torch.long, device=clip.device)
        self.origin = torch.zeros(n, 3, device=clip.device)

    def reset(self, ids, times=0.0, origins=None):
        times = torch.as_tensor(times, device=self.time.device)
        if not torch.isfinite(times).all() or (times < 0).any() or (times >= self.clip.duration).any():
            raise ValueError("reset reference time must be inside the clip")
        self.time[ids] = times
        self._elapsed[ids] = times.to(torch.float64)
        if origins is not None:
            self.origin[ids] = origins

    def advance(self, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        # Accumulate in float64 so long clips do not drift across command ticks.
        self._elapsed = (self._elapsed + dt).clamp(max=self.clip.duration)
        self.time = self._elapsed.float()
        return self.time >= self.clip.duration - 1e-6

    def packed(self):
        return torch.cat([self.clip_id[:, None], self.time[:, None], self.origin], -1)

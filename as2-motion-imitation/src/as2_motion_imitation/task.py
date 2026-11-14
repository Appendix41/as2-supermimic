"""Reference goals, observations and rewards for motion tracking."""

import numpy as np
import torch

from as2_motion_imitation.motion import MotionClock
from as2_motion_imitation.physics import DT, Physics, make_model
from as2_supermimic.supermimic import state as st

HISTORY = 5
SENSOR_DIM = 42
OFFSETS = (0.0, 0.1, 0.25, 0.5)
GOAL_DIM = len(OFFSETS) * 37 + 1
OBS_DIM = SENSOR_DIM * HISTORY + GOAL_DIM
CRITIC_DIM = 70 + GOAL_DIM + 4
REPLAY_SCHEMA = "as2-motion-replay-v1:clip_id,time,origin_xy_yaw"


def local_pose(pose, origin):
    return st.pose_delta(origin, pose)


def sensors(state, last_action):
    return torch.cat([state[:, st.W_B], state[:, st.GRAV], state[:, st.Q],
                      state[:, st.QD], last_action], -1)


def push_history(history, state, action):
    return torch.cat([history[:, SENSOR_DIM:], sensors(state, action)], -1)


class TrackingTask:
    def __init__(self, clip):
        self.clip = clip

    def goal(self, state, pose, time, origin):
        ref = self.clip.sample(time[:, None] + torch.tensor(OFFSETS, device=time.device))
        relative = local_pose(pose, origin)
        flat_pose = ref["pose"].reshape(-1, 3)
        ref_pose = st.pose_delta(self.clip.poses[0].expand_as(flat_pose), flat_pose).reshape_as(ref["pose"])
        delta = st.pose_delta(relative[:, None].expand(-1, len(OFFSETS), -1).reshape(-1, 3),
                              ref_pose.reshape(-1, 3)).reshape(-1, len(OFFSETS), 3)
        # Append root-pose error to each reference sample, then add clip phase.
        content = torch.cat([ref["state"][..., :34], delta], -1)
        return torch.cat([content.flatten(1), (time / self.clip.duration).clamp(0, 1)[:, None]], -1)

    def observe(self, history, state, pose, time, origin):
        return torch.cat([history, self.goal(state, pose, time, origin)], -1)

    def critic_input(self, state, pose, time, origin, xi):
        return torch.cat([state, self.goal(state, pose, time, origin), xi], -1)

    def reward(self, state, pose, time, origin, action, last_action):
        ref = self.clip.sample(time)
        target = ref["state"]
        ref_pose = st.pose_delta(self.clip.poses[0].expand_as(ref["pose"]), ref["pose"])
        pose_err = st.pose_delta(local_pose(pose, origin), ref_pose)
        errors = {
            "joint": (state[:, st.Q] - target[:, st.Q]).square().mean(-1),
            "height": (state[:, 0] - target[:, 0]).square(),
            "gravity": (state[:, st.GRAV] - target[:, st.GRAV]).square().sum(-1),
            "feet": (state[:, st.FEET] - target[:, st.FEET]).square().reshape(-1, 4, 3).sum(-1).mean(-1),
            "velocity": (state[:, st.V_YAW] - target[:, st.V_YAW]).square().sum(-1),
            "root": pose_err.square().sum(-1),
        }
        scales = {"joint": 0.2, "height": 0.04, "gravity": 0.2,
                  "feet": 0.05, "velocity": 0.5, "root": 0.15}
        weights = {"joint": 0.30, "height": 0.25, "gravity": 0.10,
                   "feet": 0.15, "velocity": 0.10, "root": 0.10}
        score = sum(weights[k] * torch.exp(-v / scales[k] ** 2) for k, v in errors.items())
        regularizer = 0.001 * (action - last_action).square().mean(-1)
        return (score - regularizer) * DT, errors

    def failure(self, state, time, soft=False):
        ref = self.clip.sample(time)["state"]
        height = (state[:, 0] - ref[:, 0]).abs()
        gravity_error = (state[:, st.GRAV] - ref[:, st.GRAV]).square().sum(-1)
        # Use the same height and tilt limits for hard termination and soft survival.
        if not soft:
            return (height > 0.12) | (gravity_error > 0.58) | ~torch.isfinite(state).all(-1)
        height_failure = torch.sigmoid((height - 0.12) / 0.01)
        tilt_failure = torch.sigmoid((gravity_error - 0.58) / 0.03)
        return 1 - (1 - height_failure) * (1 - tilt_failure)


class TrackingEnv:
    """CPU tracking environment with explicit resets after terminal states."""

    def __init__(self, clip, num_envs=8, seed=0):
        self.clip, self.task = clip, TrackingTask(clip)
        self.n = num_envs
        model = make_model()
        self.robots = [Physics(model, state_frame=clip.state_frame) for _ in range(num_envs)]
        self.clock = MotionClock(clip, num_envs)
        self.history = torch.zeros(num_envs, SENSOR_DIM * HISTORY)
        self.last_action = torch.zeros(num_envs, 12)
        self.xi = torch.tensor([[0.6, 0, 0, 0]]).repeat(num_envs, 1)
        self.counter = seed * 10000
        self.reset(torch.arange(num_envs))

    def read(self):
        values = [p.state() for p in self.robots]
        return torch.tensor(np.stack([v[0] for v in values])), torch.tensor(np.stack([v[1] for v in values]))

    def reset(self, ids, stagger=False):
        for i in ids.tolist():
            self.robots[i].reset(self.counter + i, perturb=0.01)
        self.counter += self.n
        state, pose = self.read()
        time = torch.rand(len(ids)) * (self.clip.duration - DT) if stagger else 0.0
        self.clock.reset(ids, time, pose[ids])
        self.last_action[ids] = 0
        self.history[ids] = sensors(state[ids], self.last_action[ids]).repeat(1, HISTORY)
        return self.observe()

    def observe(self):
        state, pose = self.read()
        return self.task.observe(self.history, state, pose, self.clock.time, self.clock.origin)

    def step(self, action):
        if action.shape != (self.n, 12) or not torch.isfinite(action).all():
            raise ValueError("invalid action batch")
        for p, a in zip(self.robots, action.detach().cpu().numpy()):
            p.step(a)
        self.clock.advance(DT)
        state, pose = self.read()
        reward, errors = self.task.reward(state, pose, self.clock.time, self.clock.origin,
                                         action, self.last_action)
        failed = self.task.failure(state, self.clock.time)
        ended = self.clock.time >= self.clip.duration - 1e-5
        self.history = push_history(self.history, state, action)
        self.last_action = action.clone()
        return self.observe(), reward, failed, ended, errors

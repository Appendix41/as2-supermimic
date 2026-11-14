"""Batched MJWarp physics with the same model, gains and 70D state as CPU."""

import numpy as np
import torch

from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.sim import Simulation, SimulationCfg
from mjlab.sim.sim import MujocoCfg

from as2_motion_imitation.physics import DT, Physics, STATE_FRAMES, make_model
from as2_motion_imitation.motion import MotionClock
from as2_motion_imitation.task import (
    HISTORY, SENSOR_DIM, TrackingTask, push_history, sensors,
)
from as2_supermimic.robot.morphology import JOINT_NAMES, LIMB_LEGS, ROOT_BODY


class GpuPhysics:
    def __init__(self, num_envs, device="cuda:0", state_frame="forward-axis-v1"):
        if state_frame not in STATE_FRAMES:
            raise ValueError("unknown state frame")
        self.state_frame = state_frame
        self.n, self.device = num_envs, device
        model = make_model()
        # Keep the reference integrator, friction cone and solver tolerances.
        cfg = SimulationCfg(
            nconmax=64, njmax=256,
            mujoco=MujocoCfg(
                timestep=model.opt.timestep, integrator="implicitfast",
                cone="elliptic", impratio=model.opt.impratio,
                iterations=model.opt.iterations, tolerance=model.opt.tolerance,
                ls_iterations=model.opt.ls_iterations, ls_tolerance=model.opt.ls_tolerance,
            ),
        )
        self.sim = Simulation(num_envs, cfg, model=model, device=device)
        self.model, self.data = model, self.sim.data
        self.jids = np.array([model.joint(n).id for n in JOINT_NAMES])
        self.qadr = torch.tensor(model.jnt_qposadr[self.jids], device=device)
        self.vadr = torch.tensor(model.jnt_dofadr[self.jids], device=device)
        self.base_id = model.body(ROOT_BODY).id
        self.hips = [model.body(f"{leg}_thigh").id for leg in LIMB_LEGS]
        self.knees = [model.body(f"{leg}_calf").id for leg in LIMB_LEGS]
        self.feet = [model.site(leg).id for leg in LIMB_LEGS]
        self.foot_geoms = [model.geom(f"{leg}_foot_collision").id for leg in LIMB_LEGS]
        self.default_q = torch.tensor(model.key(0).qpos[model.jnt_qposadr[self.jids]],
                                      device=device, dtype=torch.float32)
        self.act_order = torch.tensor(
            [list(self.jids).index(j) for j in model.actuator_trnid[:, 0]], device=device,
        )
        self.all_ids = torch.arange(num_envs, device=device)
        self.limits = torch.tensor(model.jnt_range[self.jids], device=device)
        self.reset_keyframe()
        for _ in range(150):
            self.step(torch.zeros(num_envs, 12, device=device))
        self.standing_qpos = self.data.qpos.clone()
        self.standing_qvel = self.data.qvel.clone()

    def reset_keyframe(self):
        self.sim.reset()
        self.data.qpos[:] = torch.tensor(self.model.key(0).qpos, device=self.device,
                                         dtype=torch.float32)
        self.data.qvel.zero_()
        self.data.ctrl[:] = self.default_q[self.act_order]
        self.sim.forward()

    def reset_states(self, ids, qpos, qvel):
        """Training resets/diagnostics only, never called mid-evaluation."""
        self.sim.reset(ids)
        self.data.qpos[ids] = qpos
        self.data.qvel[ids] = qvel
        self.data.ctrl[ids] = self.default_q[self.act_order]
        self.sim.forward()

    def step(self, action):
        if action.shape != (self.n, 12):
            raise ValueError("invalid GPU action batch")
        self.data.ctrl[:] = (self.default_q + 0.25 * action)[:, self.act_order]
        for _ in range(round(DT / self.model.opt.timestep)):
            self.sim.step()
        self.sim.forward()

    def state(self):
        d = self.data
        rotation = d.xmat[:, self.base_id].reshape(self.n, 3, 3)
        if self.state_frame == "lateral-axis-v2":
            yaw = torch.atan2(-rotation[:, 0, 1], rotation[:, 1, 1])
        else:
            yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        cy, sy = yaw.cos(), yaw.sin()

        def yaw_inverse(v):
            return torch.stack((cy * v[:, 0] + sy * v[:, 1],
                                -sy * v[:, 0] + cy * v[:, 1], v[:, 2]), -1)

        base = d.xpos[:, self.base_id]
        vel = compute_velocity_from_cvel(
            base, d.subtree_com[:, self.base_id], d.cvel[:, self.base_id],
        )
        parts = []
        for p in (d.xpos[:, self.hips], d.xpos[:, self.knees], d.site_xpos[:, self.feet]):
            rel = p - base[:, None]
            parts.append(torch.stack((
                cy[:, None] * rel[..., 0] + sy[:, None] * rel[..., 1],
                -sy[:, None] * rel[..., 0] + cy[:, None] * rel[..., 1],
                rel[..., 2],
            ), -1).flatten(1))
        state = torch.cat((
            base[:, 2:3], -rotation[:, 2, :], yaw_inverse(vel[:, :3]),
            torch.bmm(rotation.transpose(1, 2), vel[:, 3:, None]).squeeze(-1),
            d.qpos[:, self.qadr] - self.default_q, d.qvel[:, self.vadr], *parts,
        ), -1)
        return state, torch.cat((base[:, :2], yaw[:, None]), -1)

    def capacity_status(self):
        return {
            "max_constraints": int(self.data.nefc.max().item()),
            "constraint_capacity_per_env": self.sim.wp_data.njmax,
            "contacts_total": int(self.data.nacon.item()),
            "contacts_capacity_total": self.sim.wp_data.naconmax,
        }


class GpuTrackingEnv:
    """Reference resets are training-only; policy evaluation uses CPU physics."""

    def __init__(self, clip, num_envs=4096, seed=7, reference_reset=True):
        self.clip, self.task, self.n = clip, TrackingTask(clip), num_envs
        self.device = clip.device
        self.physics = GpuPhysics(num_envs, self.device, state_frame=clip.state_frame)
        self.clock = MotionClock(clip, num_envs)
        self.history = torch.zeros(num_envs, SENSOR_DIM * HISTORY, device=self.device)
        self.last_action = torch.zeros(num_envs, 12, device=self.device)
        self.xi = torch.tensor([[0.6, 0, 0, 0]], device=self.device).repeat(num_envs, 1)
        self.reference_reset = reference_reset
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        with np.load(clip.path, allow_pickle=False) as f:
            self.ref_qpos = torch.tensor(f["qpos"], dtype=torch.float32, device=self.device)
            self.ref_qvel = torch.tensor(f["qvel"], dtype=torch.float32, device=self.device)
        self.reset(torch.arange(num_envs, device=self.device), stagger=reference_reset)

    def read(self):
        return self.physics.state()

    def observe(self):
        state, pose = self.read()
        return self.task.observe(self.history, state, pose, self.clock.time, self.clock.origin)

    def reset(self, ids, stagger=False):
        ids = ids.to(self.device)
        if stagger and self.reference_reset:
            frames = torch.randint(0, self.clip.frames - 1, (len(ids),),
                                   device=self.device, generator=self.rng)
            qpos, qvel = self.ref_qpos[frames].clone(), self.ref_qvel[frames].clone()
            # Small joint variation; physical state and motion phase stay paired.
            qpos[:, self.physics.qadr] += (
                torch.rand(len(ids), 12, device=self.device, generator=self.rng) * 2 - 1
            ) * 0.005
            self.physics.reset_states(ids, qpos, qvel)
            self.clock.reset(ids, frames * DT, self.clip.poses[0].expand(len(ids), 3))
        else:
            self.physics.reset_states(ids, self.physics.standing_qpos[ids],
                                      self.physics.standing_qvel[ids])
            _, pose = self.read()
            self.clock.reset(ids, origins=pose[ids])
        self.last_action[ids] = 0
        state, _ = self.read()
        self.history[ids] = sensors(state[ids], self.last_action[ids]).repeat(1, HISTORY)
        return self.observe()

    def step(self, action):
        self.physics.step(action)
        self.clock.advance(DT)
        state, pose = self.read()
        reward, errors = self.task.reward(state, pose, self.clock.time, self.clock.origin,
                                         action, self.last_action)
        failed = self.task.failure(state, self.clock.time)
        ended = self.clock.time >= self.clip.duration - 1e-5
        self.history = push_history(self.history, state, action)
        self.last_action = action.clone()
        return self.observe(), reward, failed, ended, errors

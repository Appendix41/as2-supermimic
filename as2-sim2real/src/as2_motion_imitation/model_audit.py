"""Eight-step action sensitivity probes across jump phases, separate from policy eval."""

import json

import mujoco
import numpy as np
import torch

from as2_motion_imitation.physics import DT, Physics
from as2_supermimic.supermimic import state as st
from as2_supermimic.target.terrain import SCAN_POINTS


class ResponseAudit:
    def __init__(self, clip, commands, output, phases=(0.4, 1.6, 1.84, 2.1, 2.32, 2.5)):
        if commands is None:
            raise ValueError("jump response audit requires paired recorded commands")
        self.device, self.output = clip.device, output / "model-audit"
        self.output.mkdir()
        self.phases = tuple(phases)
        self.epsilon, self.horizon = 0.2, 8
        if not self.phases or any(
            round(t / DT) < 1 or round(t / DT) + self.horizon >= clip.frames
            for t in self.phases
        ):
            raise ValueError("response probes must fit inside the reference")
        p = Physics(state_frame=clip.state_frame)
        with np.load(clip.path) as f:
            qpos, qvel = f["qpos"], f["qvel"]
        source = commands.actions.cpu().numpy()
        starts, previous, actions, targets = [], [], [], []
        for phase in self.phases:
            frame = round(phase / DT)
            for j in range(12):
                for sign in (-1, 1):
                    mujoco.mj_resetData(p.model, p.data)
                    p.data.qpos[:], p.data.qvel[:] = qpos[frame], qvel[frame]
                    mujoco.mj_forward(p.model, p.data)
                    starts.append(p.state()[0])
                    previous.append(source[frame - 1])
                    commanded = source[frame:frame + self.horizon].copy()
                    commanded[:, j] += sign * self.epsilon
                    rollout = []
                    for a in commanded:
                        p.step(a)
                        rollout.append(p.state()[0])
                    actions.append(commanded)
                    targets.append(rollout)
        arrays = {"initial": np.array(starts), "previous": np.array(previous),
                  "actions": np.array(actions), "targets": np.array(targets)}
        np.savez_compressed(self.output / "physical-probes.npz", **arrays,
                            phases=np.array(self.phases), epsilon=self.epsilon)
        for name, value in arrays.items():
            setattr(self, name, torch.tensor(value, dtype=torch.float32, device=self.device))
        self.xi = torch.tensor([[0.6, 0, 0, 0]], device=self.device).repeat(len(starts), 1)
        self.relief = torch.zeros(len(starts), SCAN_POINTS, device=self.device)

    @torch.no_grad()
    def evaluate(self, wm, iteration):
        s, previous = self.initial.clone(), self.previous
        predictions = []
        for k in range(self.horizon):
            a = self.actions[:, k]
            s, _ = wm(s, a, previous, self.xi, self.relief)
            predictions.append(s)
            previous = a
        pred = torch.stack(predictions, 1)
        real = self.targets[:, -1].reshape(len(self.phases), 12, 2, 70)
        estimated = pred[:, -1].reshape_as(real)
        r = (real[:, :, 1] - real[:, :, 0]) / (2 * self.epsilon)
        e = (estimated[:, :, 1] - estimated[:, :, 0]) / (2 * self.epsilon)
        terms = {}
        for name, sl in (("height", st.Z), ("joint", st.Q), ("feet", st.FEET),
                         ("gravity", st.GRAV), ("angular", st.W_B)):
            a, b = r[..., sl].flatten(), e[..., sl].flatten()
            terms[f"{name}_response_cosine"] = float(
                torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12)
            )
            terms[f"{name}_response_relative_error"] = float((a - b).norm() / a.norm().clamp_min(1e-12))
        result = {
            "iteration": iteration, "horizon": self.horizon, "phases": self.phases,
            "scope": "Diagnostic restored-state finite differences, not policy success",
            "state_rmse_mixed_units": float((pred - self.targets).square().mean().sqrt()),
            "height_rmse_m": float((pred[..., st.Z] - self.targets[..., st.Z]).square().mean().sqrt()),
            "joint_rmse_rad": float((pred[..., st.Q] - self.targets[..., st.Q]).square().mean().sqrt()),
            "gravity_rmse": float((pred[..., st.GRAV] - self.targets[..., st.GRAV]).square().mean().sqrt()),
            "angular_rmse_rad_s": float((pred[..., st.W_B] - self.targets[..., st.W_B]).square().mean().sqrt()),
            **terms,
        }
        (self.output / f"iteration-{iteration}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

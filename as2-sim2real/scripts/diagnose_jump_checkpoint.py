"""Compare model imagination with saved actor-only physical trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2_motion_imitation.learner import Imagination, TrainCfg, load_actor
from as2_motion_imitation.demonstration import RecordedCommands
from as2_motion_imitation.motion import MotionClip, sha256
from as2_motion_imitation.physics import DT
from as2_motion_imitation.task import CRITIC_DIM, HISTORY, TrackingTask, sensors
from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.train import Critic
from as2_supermimic.supermimic.world_model import WorldModel
from as2_supermimic.target.terrain import SCAN_POINTS

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    output = args.evaluation.resolve()
    output.relative_to(ROOT / "outputs")
    torch.set_num_threads(2)
    evaluation = json.loads((output / "metrics.json").read_text())
    if evaluation["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("evaluation belongs to a different checkpoint")
    clip = MotionClip(evaluation.get("motion", ROOT / "data/processed/as2_forward_jump.npz"))
    actor, ckpt = load_actor(args.checkpoint, clip)
    cfg = TrainCfg(**ckpt["cfg"])
    wm, critic = WorldModel(tuple(cfg.wm_hidden)), Critic(CRITIC_DIM, tuple(cfg.critic_hidden))
    wm.load_state_dict(ckpt["world_model"])
    critic.load_state_dict(ckpt["critic"])
    task, imag = TrackingTask(clip), Imagination(TrackingTask(clip), cfg)
    commands = RecordedCommands(ROOT / cfg.recorded_commands, clip)
    with np.load(output / "seed-100.npz") as f:
        states, poses, actions = (torch.tensor(f[k]) for k in ("state", "pose", "action"))
        rewards = torch.tensor(f["reward"]) * DT
    xi, relief = torch.tensor([[0.6, 0, 0, 0]]), torch.zeros(1, SCAN_POINTS)
    results = []
    for start in (0, 50, 75, 90, 105, 120, 200, 300):
        previous = torch.zeros(1, 12) if start == 0 else actions[start - 1:start]
        row = {"time_s": start * DT}
        for horizon in (8, cfg.horizon):
            state, pose, prev = states[start:start + 1].clone(), poses[start:start + 1].clone(), previous
            predicted = []
            for step in range(start, start + horizon):
                a = actions[step:step + 1]
                state, delta = wm(state, a, prev, xi, relief)
                pose, prev = st.apply_pose_delta(pose, delta), a
                predicted.append(state[0])
            predicted = torch.stack(predicted)
            target = states[start + 1:start + horizon + 1]
            row[f"{horizon}_step"] = {
                "state_rmse_mixed_units": float((predicted - target).square().mean().sqrt()),
                "height_rmse_m": float((predicted[:, st.Z] - target[:, st.Z]).square().mean().sqrt()),
                "joint_rmse_rad": float((predicted[:, st.Q] - target[:, st.Q]).square().mean().sqrt()),
                "gravity_rmse": float((predicted[:, st.GRAV] - target[:, st.GRAV]).square().mean().sqrt()),
                "angular_rmse_rad_s": float((predicted[:, st.W_B] - target[:, st.W_B]).square().mean().sqrt()),
                "final_pose_error": float((pose[0] - poses[start + horizon]).norm()),
            }
        state = clip.states[start:start + 1].clone()
        prev = torch.zeros(1, 12) if start == 0 else commands.actions[start - 1:start]
        reference_predictions = []
        for step in range(start, start + cfg.horizon):
            a = commands.actions[step:step + 1]
            state, _ = wm(state, a, prev, xi, relief)
            reference_predictions.append(state[0])
            prev = a
        predicted = torch.stack(reference_predictions)
        target = clip.states[start + 1:start + cfg.horizon + 1]
        row["reference_commands_32_step_diagnostic"] = {
            "height_rmse_m": float((predicted[:, st.Z] - target[:, st.Z]).square().mean().sqrt()),
            "joint_rmse_rad": float((predicted[:, st.Q] - target[:, st.Q]).square().mean().sqrt()),
            "gravity_rmse": float((predicted[:, st.GRAV] - target[:, st.GRAV]).square().mean().sqrt()),
            "angular_rmse_rad_s": float((predicted[:, st.W_B] - target[:, st.W_B]).square().mean().sqrt()),
            "not_held_out": True,
        }
        history = []
        for k in range(start - HISTORY + 1, start + 1):
            state_index = max(k, 0)
            last = torch.zeros(1, 12) if state_index == 0 else actions[state_index - 1:state_index]
            history.append(sensors(states[state_index:state_index + 1], last))
        _, imagined = imag.rollout(wm, actor, critic, {
            "s": states[start:start + 1], "pose": poses[start:start + 1],
            "a_prev": previous, "history": torch.cat(history, -1),
            "time": torch.tensor([start * DT]), "origin": poses[0:1], "xi": xi,
        })
        discount = torch.tensor([cfg.gamma**k for k in range(cfg.horizon)])
        raw_return = float((rewards[start:start + cfg.horizon] * discount).sum())
        survival, real_return = 1.0, 0.0
        for k in range(cfg.horizon):
            real_return += float(rewards[start + k] * discount[k]) * survival
            next_time = torch.tensor([(start + k + 1) * DT])
            survival *= float(1 - task.failure(
                states[start + k + 1:start + k + 2], next_time, soft=True,
            ).item()) * float(next_time.item() < clip.duration - 1e-5)
        imagined_return = float((imagined["rewards"][0] * imagined["surv"][0, :-1] * discount).sum())
        row.update({"physical_discounted_reward_soft_survival": real_return,
                    "physical_discounted_reward_without_termination": raw_return,
                    "imagined_discounted_reward_no_bootstrap": imagined_return,
                    "return_absolute_gap": abs(real_return - imagined_return)})
        results.append(row)
    report = {
        "scope": "Seed100 phase-conditioned model diagnosis, not an additional policy trial",
        "checkpoint_sha256": sha256(args.checkpoint), "iteration": ckpt["iteration"],
        "maximum_actual_action": float(actions.abs().max()),
        "phases": results,
    }
    (output / "model-diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

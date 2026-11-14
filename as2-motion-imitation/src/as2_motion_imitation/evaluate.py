"""Evaluate policy actions in physics, retaining every held-out rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2_motion_imitation.learner import ROOT, Imagination, TrainCfg, load_actor
from as2_motion_imitation.motion import MotionClip, sha256
from as2_motion_imitation.physics import DT, Video
from as2_motion_imitation.task import HISTORY, SENSOR_DIM, TrackingEnv, sensors
from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.train import Critic
from as2_supermimic.supermimic.world_model import WorldModel
from as2_supermimic.target.terrain import SCAN_POINTS


@torch.no_grad()
def evaluate(checkpoint, output):
    torch.set_num_threads(2)
    output = Path(output).resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    criteria = json.loads((ROOT / "configs/crouch-acceptance.json").read_text())
    clip = MotionClip(ROOT / "data/processed/as2_crouch.npz")
    actor, ckpt = load_actor(checkpoint, clip)
    seeds = criteria["held_out_seeds"]
    env = TrackingEnv(clip, len(seeds))
    for i, seed in enumerate(seeds):
        env.robots[i].reset(seed, perturb=0.01)
    s0, pose0 = env.read()
    env.clock.reset(torch.arange(len(seeds)), origins=pose0)
    env.history = sensors(s0, env.last_action).repeat(1, HISTORY)
    label = "initial zero-output policy" if ckpt["iteration"] == 0 else "learned-policy physics"
    video = Video(env.robots[0], output / "policy-seed100.mp4",
                  f"AS2 {label} | iteration {ckpt['iteration']} | seed 100")
    traces = {"state": [], "pose": [], "action": [], "reward": [], "failure": []}
    errors, diagnostics = [], []
    states = [s0.numpy()]
    poses = [pose0.numpy()]
    for step in range(round(clip.duration / DT)):
        action = actor(env.observe()).clamp(-4, 4)
        _, reward, failed, _, err = env.step(action)
        s, pose = env.read()
        if step % 2 == 0:
            video.capture((step + 1) * DT)
        states.append(s.numpy())
        poses.append(pose.numpy())
        traces["action"].append(action.numpy())
        traces["reward"].append(reward.numpy())
        traces["failure"].append(failed.numpy())
        errors.append({k: v.numpy() for k, v in err.items()})
        diagnostics.append([p.diagnostics() for p in env.robots])
    video.close()
    traces["state"], traces["pose"] = np.array(states), np.array(poses)
    traces = {k: np.array(v) for k, v in traces.items()}
    np.savez_compressed(output / "all-rollouts.npz", **traces, seeds=np.array(seeds))
    metrics = []
    lo, hi = [round(v / DT) for v in criteria["crouch_window_seconds"]]
    tail = round(criteria["final_stable_seconds"] / DT)
    for i, seed in enumerate(seeds):
        s = traces["state"][:, i]
        diag = [d[i] for d in diagnostics]
        err = {k: float(np.sqrt(np.mean([e[k][i] for e in errors]))) for k in errors[0]}
        depth = float(s[0, 0] - s[lo:hi, 0].min())
        height_recovered = abs(s[-1, 0] - s[0, 0]) <= criteria["final_height_tolerance_m"]
        stable = [
            d["speed"] <= criteria["final_max_speed_m_s"]
            and d["tilt"] <= criteria["final_max_tilt_rad"]
            and d["feet_contact"] == criteria["required_final_contact_feet"]
            and abs(d["height"] - s[0, 0]) <= criteria["final_height_tolerance_m"]
            for d in diag
        ]
        stable_steps = 0
        for ok in reversed(stable):
            if not ok:
                break
            stable_steps += 1
        bad_contact = any(d["nonfoot_contact"] for d in diag)
        bad_limit = any(d["joint_limit"] for d in diag)
        fallen = bool(traces["failure"][:, i].any())
        success = (
            depth >= criteria["minimum_crouch_depth_m"] and height_recovered
            and all(stable[-tail:]) and not bad_contact and not bad_limit and not fallen
            and err["height"] <= criteria["max_height_rmse_m"]
            and err["joint"] <= criteria["max_joint_rmse_rad"]
        )
        metrics.append({
            "seed": seed, "success": bool(success), "crouch_depth_m": depth,
            "height_start_m": float(s[0, 0]), "height_end_m": float(s[-1, 0]),
            "height_recovered": bool(height_recovered), "stable_tail_seconds": stable_steps * DT,
            "tracking_score": float(traces["reward"][:, i].mean() / DT),
            "tracking_rmse": err, "fell": fallen, "nonfoot_contact": bad_contact,
            "joint_limit": bad_limit, "mean_torque_saturation": float(np.mean([d["torque_saturation"] for d in diag])),
        })
    wm = WorldModel(tuple(ckpt["cfg"]["wm_hidden"]))
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    prediction_error = {}
    for horizon in (1, 8):
        values = []
        for start in (0, 100, 200, 300):
            s = torch.tensor(traces["state"][start])
            for step in range(start, start + horizon):
                a = torch.tensor(traces["action"][step])
                prev = torch.zeros_like(a) if step == 0 else torch.tensor(traces["action"][step - 1])
                s, _ = wm(s, a, prev, env.xi, torch.zeros(env.n, SCAN_POINTS))
            values.append((s - torch.tensor(traces["state"][start + horizon])).square().mean().item())
        prediction_error[f"{horizon}_step_state_rmse_mixed_units"] = float(np.sqrt(np.mean(values)))
    # Compare a zero-bootstrap imagined reward sum against real reward over H steps.
    cfg = TrainCfg(**ckpt["cfg"])
    from as2_motion_imitation.task import CRITIC_DIM

    critic = Critic(CRITIC_DIM, tuple(cfg.critic_hidden))
    critic.load_state_dict(ckpt["critic"])
    imagined, data = Imagination(env.task, cfg).rollout(wm, actor, critic, {
        "s": s0, "pose": pose0, "a_prev": torch.zeros(env.n, 12),
        "history": sensors(s0, torch.zeros(env.n, 12)).repeat(1, HISTORY),
        "time": torch.zeros(env.n), "origin": pose0, "xi": env.xi,
    })
    discount = torch.tensor([cfg.gamma**k for k in range(cfg.horizon)])
    predicted_return = (data["rewards"] * data["surv"][:, :-1] * discount).sum(-1)
    actual_return = (torch.tensor(traces["reward"][:cfg.horizon].T) * discount).sum(-1)
    result = {
        "scope": "Policy-only CPU MuJoCo physics from normally settled standing",
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
        "motion_sha256": sha256(clip.path), "criteria": criteria, "iteration": ckpt["iteration"],
        "trials": len(seeds), "successes": sum(m["success"] for m in metrics),
        "failures": sum(not m["success"] for m in metrics), "trials_detail": metrics,
        "mean_tracking_score": float(np.mean([m["tracking_score"] for m in metrics])),
        "world_model_heldout": prediction_error,
        "first_horizon_real_return_mean": actual_return.mean().item(),
        "first_horizon_imagined_reward_return_mean": predicted_return.mean().item(),
        "first_horizon_return_absolute_gap_mean": (predicted_return - actual_return).abs().mean().item(),
        "no_reference_pose_writes": True, "no_reference_action_driver": True,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("criteria", "trials_detail")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

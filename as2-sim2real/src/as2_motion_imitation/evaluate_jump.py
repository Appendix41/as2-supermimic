"""Actor-only forward-jump evaluation from standing, with 500 Hz contact checks."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from as2_motion_imitation.learner import ROOT, load_actor
from as2_motion_imitation.motion import MotionClip, sha256
from as2_motion_imitation.physics import DT, Physics, make_model
from as2_motion_imitation.prepare_jump import ACCEPTANCE, JumpVideo, summarize
from as2_motion_imitation.task import HISTORY, TrackingTask, push_history, sensors


class JumpRecorder:
    def __init__(self, physics):
        self.p = physics
        self.rows = []
        self.foot_geoms = np.array([
            physics.model.geom(f"{leg}_foot_collision").id for leg in ("FL", "FR", "RL", "RR")
        ])
        self.foot_set = set(self.foot_geoms)
        self.floor = physics.model.geom("floor").id
        self.limits = physics.model.jnt_range[physics.jids]
        self.standing_q = physics.data.qpos[physics.qadr].copy()

    def record(self, time):
        p = self.p
        touching, nonfoot = set(), False
        for con in p.data.contact:
            if con.dist <= 0 and self.floor in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == self.floor else con.geom1
                touching.add(other)
                nonfoot |= other not in self.foot_set
        lin, ang = p.velocity(p.base_id)
        q = p.data.qpos[p.qadr]
        self.rows.append({
            "time": time, "base_pos": p.data.xpos[p.base_id].copy(),
            "base_vel": lin, "base_ang_vel": ang,
            "feet_contact": len(touching & self.foot_set), "nonfoot": nonfoot,
            "foot_clearance": np.min(
                p.data.geom_xpos[self.foot_geoms, 2] - p.model.geom_size[self.foot_geoms, 0]),
            "tilt": np.arccos(np.clip(p.data.xmat[p.base_id].reshape(3, 3)[2, 2], -1, 1)),
            "joint_limit": np.any((q < self.limits[:, 0]) | (q > self.limits[:, 1])),
            "joint_speed": np.abs(p.data.qvel[p.vadr]).max(),
            "standing_joint_error": np.abs(q - self.standing_q).max(),
            "torque_saturation": np.mean(
                np.abs(p.data.actuator_force) >= p.model.actuator_forcerange[:, 1] * 0.99),
            "peak_torque": np.abs(p.data.actuator_force).copy(),
            "qpos": p.data.qpos.copy(), "qvel": p.data.qvel.copy(),
        })

    def arrays(self):
        return {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}


@torch.no_grad()
def evaluate(checkpoint, output, seeds=range(100, 120), video=True,
             motion=None, acceptance=None, sensor_noise_std=(), perturb=0.01,
             variation=None, action_delay_steps=0):
    torch.set_num_threads(2)
    output = Path(output).resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    if not len(seeds):
        raise ValueError("evaluation requires at least one trial")
    clip = MotionClip(motion or ROOT / "data/processed/as2_forward_jump.npz")
    actor, ckpt = load_actor(checkpoint, clip)
    task = TrackingTask(clip)
    criteria = json.loads(Path(acceptance or ACCEPTANCE).read_text())
    summarize_fn, video_class, recorder_class = summarize, JumpVideo, JumpRecorder
    if "minimum_total_back_rotation_rad" in criteria:
        from as2_motion_imitation.backflip_audit import BackflipVideo, Recorder
        from as2_motion_imitation.backflip_audit import summarize as summarize_backflip

        if clip.state_frame != "lateral-axis-v2":
            raise ValueError("backflip evaluation requires the continuous sagittal state frame")
        summarize_fn, video_class, recorder_class = summarize_backflip, BackflipVideo, Recorder
    elif criteria.get("expected_jumps", 1) > 1:
        from as2_motion_imitation.prepare_jump_sequence import SequenceVideo, summarize_sequence

        summarize_fn, video_class = summarize_sequence, SequenceVideo
    results = []
    if action_delay_steps not in (0, 1, 2):
        raise ValueError("diagnostic action delay must be 0, 1 or 2 ticks")
    for seed in seeds:
        torch.manual_seed(seed)
        model = make_model()
        if variation:
            mass = variation.get("mass_scale", 1.0)
            gains = variation.get("gain_scale", 1.0)
            model.body_mass[1:] *= mass
            model.body_inertia[1:] *= mass
            model.actuator_gainprm[:, 0] *= gains
            model.actuator_biasprm[:, 1:3] *= gains
            if "friction" in variation:
                model.geom_friction[:, 0] = variation["friction"]
            mujoco.mj_setConst(model, mujoco.MjData(model))
        p = Physics(model=model, state_frame=clip.state_frame)
        p.reset(seed=seed, perturb=perturb)
        s, pose = (torch.tensor(a)[None] for a in p.state())
        origin = pose.clone()
        last = torch.zeros(1, 12)
        history = sensors(s, last, sensor_noise_std).repeat(1, HISTORY)
        queue = [torch.zeros(1, 12) for _ in range(action_delay_steps)]
        recorder = recorder_class(p)
        recorder.record(0.0)
        movie = None
        if video and seed == seeds[0]:
            movie = video_class(p, output / f"policy-seed{seed}.mp4")
            movie.label = f"AS2 actor ONLY | iteration {ckpt['iteration']} | seed {seed} | simulation"
            movie.capture(0.0)
        states, poses, actions, rewards, failures, errors = [s[0].numpy()], [pose[0].numpy()], [], [], [], []
        for step in range(round(clip.duration / DT)):
            time = torch.tensor([step * DT])
            obs = task.observe(history, s, pose, time, origin)
            action = actor(obs).clamp(-4, 4)
            if not torch.isfinite(action).all():
                raise FloatingPointError("nonfinite policy action")
            if action_delay_steps:
                queue.append(action)
                action = queue.pop(0)
            p.data.ctrl[:] = (p.default_q + 0.25 * action[0].numpy())[p.act_order]
            for substep in range(round(DT / p.model.opt.timestep)):
                mujoco.mj_step(p.model, p.data)
                mujoco.mj_forward(p.model, p.data)
                recorder.record(step * DT + (substep + 1) * p.model.opt.timestep)
            s, pose = (torch.tensor(a)[None] for a in p.state())
            reward, err = task.reward(s, pose, time + DT, origin, action, last)
            failures.append(bool(task.failure(s, time + DT).item()))
            rewards.append(reward.item() / DT)
            errors.append({k: float(v.item()) for k, v in err.items()})
            states.append(s[0].numpy())
            poses.append(pose[0].numpy())
            actions.append(action[0].numpy())
            history = push_history(history, s, action, sensor_noise_std)
            last = action
            if movie and (step + 1) % 2 == 0:
                movie.capture((step + 1) * DT)
        if movie:
            movie.close()
        trace = recorder.arrays()
        physical = summarize_fn(trace, criteria)
        result = {
            "seed": seed, **physical,
            "success": physical["passed"] and not any(failures),
            "tracking_failure": any(failures), "mean_tracking_score": float(np.mean(rewards)),
            "tracking_rmse": {k: float(np.sqrt(np.mean([e[k] for e in errors]))) for k in errors[0]},
        }
        results.append(result)
        np.savez_compressed(output / f"seed-{seed}.npz", **trace, state=np.array(states),
                            pose=np.array(poses), action=np.array(actions), reward=np.array(rewards),
                            tracking_failure=np.array(failures))
        print(json.dumps({"seed": seed, "success": result["success"],
                          "airborne_seconds": result["airborne_seconds"],
                          "forward_m": result["final_forward_m"]}), flush=True)
    summary = {
        "scope": "Actor-only CPU physical evaluation, ordinary standing, 500 Hz audit",
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
        "motion_sha256": sha256(clip.path), "iteration": ckpt["iteration"],
        "state_frame": clip.state_frame,
        "motion": str(clip.path.resolve()),
        "acceptance_sha256": sha256(acceptance or ACCEPTANCE),
        "actor_updates": ckpt.get("actor_updates"),
        "successes": sum(r["success"] for r in results), "trials": len(results),
        "trials_detail": results, "criteria": criteria,
        "no_reference_action_driver": True, "no_reference_pose_writes": True,
        "no_external_force": True, "reset_perturbation_rad": perturb,
        "sensor_noise_std": list(sensor_noise_std),
        "physics_variation": variation or {}, "action_delay_steps": action_delay_steps,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trials_detail", "criteria")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--sensor-noise", action="store_true")
    parser.add_argument("--perturb", type=float, default=0.01)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.6)
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--action-delay", type=int, default=0)
    args = parser.parse_args()
    if not (0.5 <= args.mass_scale <= 1.5 and 0.2 <= args.friction <= 1.2
            and 0.5 <= args.gain_scale <= 1.5 and 0 <= args.perturb <= 0.1):
        parser.error("diagnostic physical variations are out of bounds")
    variation = {"mass_scale": args.mass_scale, "friction": args.friction,
                 "gain_scale": args.gain_scale}
    if variation == {"mass_scale": 1.0, "friction": 0.6, "gain_scale": 1.0}:
        variation = {}
    evaluate(args.checkpoint, args.output, range(args.seed_start, args.seed_start + args.trials),
             not args.no_video, args.motion, args.acceptance,
             (0.01, 0.003, 0.002, 0.03) if args.sensor_noise else (), args.perturb,
             variation, args.action_delay)


if __name__ == "__main__":
    main()

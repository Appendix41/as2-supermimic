"""500 Hz physical backflip acceptance, independent of search proxies."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from as2_motion_imitation.physics import DT, Physics, Video
from as2_motion_imitation.motion import MotionClip, save_clip, sha256
from as2_motion_imitation.prepare_backflip import NAMES, ROOT, orientation, targets
from as2_motion_imitation.prepare_jump import longest_run


ACCEPTANCE = ROOT / "configs/backflip-acceptance.json"


class BackflipVideo(Video):
    def __init__(self, physics, path, learned=False):
        label = "AS2 actor-only backflip" if learned else "AS2 scripted PD backflip | NOT learned"
        super().__init__(physics, path, label + " | simulation only")
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 90, -12, 2.7

class Recorder:
    def __init__(self, physics):
        self.p = physics
        self.feet = {physics.model.geom(f"{leg}_foot_collision").id
                     for leg in ("FL", "FR", "RL", "RR")}
        self.floor = physics.model.geom("floor").id
        self.limits = physics.model.jnt_range[physics.jids]
        self.standing = physics.data.qpos[physics.qadr].copy()
        self.records = []

    def record(self, time):
        p = self.p
        touching = set()
        for contact in p.data.contact:
            if contact.dist <= 0 and self.floor in (contact.geom1, contact.geom2):
                touching.add(contact.geom2 if contact.geom1 == self.floor else contact.geom1)
        velocity, angular = p.velocity(p.base_id)
        q = p.data.qpos[p.qadr]
        self.records.append({
            "time": time, "base_pos": p.data.xpos[p.base_id].copy(),
            "base_quat": p.data.xquat[p.base_id].copy(),
            "base_vel": velocity, "base_ang_vel": angular,
            "feet_contact": len(touching & self.feet),
            "nonfoot": bool(touching - self.feet),
            "joint_limit": bool(np.any((q < self.limits[:, 0]) | (q > self.limits[:, 1]))),
            "joint_speed": float(np.abs(p.data.qvel[p.vadr]).max()),
            "standing_joint_error": float(np.abs(q - self.standing).max()),
            "torque": p.data.actuator_force.copy(),
            "torque_saturation": float(np.mean(np.abs(p.data.actuator_force) >=
                                               p.model.actuator_forcerange[:, 1] * .99)),
            "qpos": p.data.qpos.copy(), "qvel": p.data.qvel.copy(),
        })

    def arrays(self):
        return {key: np.array([row[key] for row in self.records]) for key in self.records[0]}


def summarize(trace, criteria):
    t, base = trace["time"], trace["base_pos"]
    dt = float(np.median(np.diff(t)))
    back, lateral, upright = orientation(trace["base_quat"])
    back -= back[0]
    flight = (trace["feet_contact"] == 0) & ~trace["nonfoot"]
    duration, start, end = longest_run(flight, dt)
    rotation = float(back[end - 1] - back[start]) if end > start else 0.0
    stable = (
        (trace["feet_contact"] == 4) & ~trace["nonfoot"] & ~trace["joint_limit"]
        & (np.linalg.norm(trace["base_vel"], axis=-1) <= criteria["final_max_speed_m_s"])
        & (np.linalg.norm(trace["base_ang_vel"], axis=-1) <= criteria["final_max_angular_speed_rad_s"])
        & (np.abs(base[:, 2] - base[0, 2]) <= criteria["final_max_height_error_m"])
        & (trace["standing_joint_error"] <= criteria["final_max_joint_error_rad"])
        & (np.arccos(np.clip(upright, -1, 1)) <= criteria["maximum_final_tilt_rad"])
    )
    bad = np.flatnonzero(~stable)
    tail = float(t[-1] - t[bad[-1]]) if len(bad) else float(t[-1] - t[0])
    checks = {
        "airborne": duration >= criteria["minimum_airborne_seconds"],
        "airborne_rotation": rotation >= criteria["minimum_airborne_back_rotation_rad"],
        "total_rotation": criteria["minimum_total_back_rotation_rad"] <= back[-1]
                          <= criteria["maximum_total_back_rotation_rad"],
        "lateral": np.max(np.abs(base[:, 1] - base[0, 1])) <= criteria["maximum_lateral_displacement_m"],
        "rotation_axis": lateral.min() >= criteria["minimum_lateral_axis_alignment"],
        "standing": tail >= criteria["final_stable_seconds"],
        "contacts": not trace["nonfoot"].any(),
        "limits": not trace["joint_limit"].any(),
        "speed": trace["joint_speed"].max() <= criteria["maximum_joint_speed_rad_s"],
        "finite": all(np.isfinite(value).all() for value in trace.values()),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "airborne_seconds": duration, "airborne_back_rotation_rad": rotation,
        "total_back_rotation_rad": float(back[-1]),
        "peak_height_m": float(base[:, 2].max()),
        "final_forward_m": float(base[-1, 0] - base[0, 0]),
        "maximum_lateral_m": float(np.max(np.abs(base[:, 1] - base[0, 1]))),
        "minimum_lateral_axis_alignment": float(lateral.min()),
        "final_stable_seconds": tail,
        "nonfoot_contact_samples": int(trace["nonfoot"].sum()),
        "joint_limit_samples": int(trace["joint_limit"].sum()),
        "max_joint_speed_rad_s": float(trace["joint_speed"].max()),
        "peak_abs_torque_nm": float(np.abs(trace["torque"]).max()),
        "torque_saturation_fraction": float(trace["torque_saturation"].mean()),
        "takeoff_s": float(t[start]) if end > start else None,
        "landing_s": float(t[end]) if end < len(t) and end > start else None,
    }


def rollout(p, parameters, seed=0, perturb=0.0, duration=8., capture=False, video=None):
    p.reset(seed, perturb)
    recorder = Recorder(p)
    recorder.record(0.0)
    commands = targets(parameters, np.arange(round(duration / DT)) * DT)[0]
    limits = p.model.jnt_range[p.jids]
    if ((commands < limits[:, 0]) | (commands > limits[:, 1])).any():
        raise ValueError("commanded targets exceed physical joint limits")
    actions = (commands - p.default_q) / .25
    if np.abs(actions).max() > 4 + 1e-9:
        raise ValueError("commanded targets exceed the original action range")
    snapshots = []
    for i in range(len(commands) + 1):
        if capture:
            snapshots.append(p.snapshot())
        if video is not None and i % 2 == 0:
            video.capture(i * DT)
        if i == len(commands):
            break
        p.data.ctrl[:] = commands[i, p.act_order]
        for substep in range(10):
            mujoco.mj_step(p.model, p.data)
            mujoco.mj_forward(p.model, p.data)
            recorder.record(i * DT + (substep + 1) * p.model.opt.timestep)
    arrays = {key: np.array([row[key] for row in snapshots],
                           dtype=np.float64 if key in ("qpos", "qvel") else np.float32)
              for key in snapshots[0]} if capture else None
    return recorder.arrays(), arrays, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    raw = ROOT / "data/raw/as2-native-backflip"
    if args.export:
        args.export = args.export.resolve()
        args.export.relative_to(ROOT / "data/processed")
        if args.trials < 20:
            raise ValueError("export requires at least 20 perturbation trials")
        if args.export.exists() or args.export.with_suffix(".json").exists() or raw.exists():
            raise ValueError("refusing to overwrite an existing reference")
    output.mkdir(parents=True, exist_ok=False)
    values = json.loads(args.config.read_text())
    params = np.array([values[name] for name in NAMES])
    criteria = json.loads(ACCEPTANCE.read_text())
    p = Physics(state_frame="lateral-axis-v2")
    movie = BackflipVideo(p, output / "scripted-backflip.mp4") if args.video else None
    trace, arrays, actions = rollout(p, params, capture=True, video=movie)
    if movie:
        movie.close()
    metrics = summarize(trace, criteria)
    np.savez_compressed(output / "nominal-dense-trace.npz", **trace)
    np.savez_compressed(output / "nominal-control-capture.npz", **arrays, action=actions,
                        action_time=np.arange(len(actions)) * DT)
    (output / "parameters.json").write_text(json.dumps(values, indent=2) + "\n")
    (output / "nominal-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    results = []
    for seed in range(100, 100 + args.trials):
        trace, _, _ = rollout(p, params, seed=seed, perturb=.01)
        result = {"seed": seed, **summarize(trace, criteria)}
        np.savez_compressed(output / f"seed-{seed}.npz", **trace)
        results.append(result)
        print(json.dumps({"seed": seed, "passed": result["passed"],
                          "failed": [name for name, okay in result["checks"].items() if not okay]}), flush=True)
    summary = {
        "scope": "Scripted physical control, not learned actor",
        "passed": sum(row["passed"] for row in results), "total": len(results),
        "perturbation_rad": .01, "results": results,
    }
    (output / "perturbation-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = [Path(__file__), ROOT / "src/as2_motion_imitation/prepare_backflip.py",
               ROOT / "src/as2_motion_imitation/physics.py",
               ROOT / "src/as2_motion_imitation/motion.py", args.config.resolve(), ACCEPTANCE]
    provenance = {
        "kind": "locally-generated-scripted-PD-physics", "robot": "AS2",
        "motion": "standing-backflip-land-stand", "upstream_motion": False,
        "learned_policy": False, "parameters": values,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "seed": 0, "settling_seconds": 3.0, "duration_seconds": 8.0,
        "physics_dt": p.model.opt.timestep, "control_dt": DT, "diagnostic_hz": 500,
        "state_frame": p.state_frame,
        "pose_writes_after_reset": False, "external_forces": False,
        "original_gains_mass_and_torque_limits": True, "self_collision_enabled": False,
        "physics_backend": "CPU MuJoCo", "nominal_metrics": metrics,
        "perturbation_successes": summary["passed"], "perturbation_trials": args.trials,
        "max_abs_normalized_action": float(np.abs(actions).max()),
        "actuator_force_limits_nm": p.model.actuator_forcerange.tolist(),
        "license_note": "Locally generated simulation data; model notices remain in baseline.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.export:
        if not metrics["passed"] or summary["passed"] != args.trials:
            raise RuntimeError("reference validation failed; no clip was exported")
        raw.mkdir(parents=True)
        np.savez_compressed(raw / "physics_capture.npz", **arrays, action=actions,
                            action_time=np.arange(len(actions)) * DT)
        (raw / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        save_clip(args.export, arrays, Physics(state_frame="lateral-axis-v2"), provenance)
        clip = MotionClip(args.export)
        print(json.dumps({"exported": str(args.export), "frames": clip.frames,
                          "duration_s": clip.duration, "state_frame": clip.state_frame}), flush=True)


if __name__ == "__main__":
    main()

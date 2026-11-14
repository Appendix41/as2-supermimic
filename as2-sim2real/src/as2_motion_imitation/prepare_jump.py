"""Search and capture scripted AS2 forward jumps using unchanged physical PD."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from as2_motion_imitation.physics import DT, Physics, Video

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/forward-jump"
ACCEPTANCE = ROOT / "configs/forward-jump-acceptance.json"


@dataclass
class JumpParams:
    crouch: float = 1.18
    crouch_lean: float = 0.0
    crouch_front: float = 0.0
    extend: float = 0.46
    push_lean: float = 0.20
    push_front: float = 0.0
    thrust_seconds: float = 0.10
    extended_seconds: float = 0.08
    flight_bend: float = 0.95
    flight_lean: float = -0.05
    flight_front: float = 0.0
    flight_seconds: float = 0.22


BOUNDS = {
    "crouch": (1.02, 1.32),
    "crouch_lean": (-0.15, 0.15),
    "crouch_front": (-0.12, 0.12),
    "extend": (0.42, 0.65),
    "push_lean": (-0.05, 0.40),
    "push_front": (-0.18, 0.18),
    "thrust_seconds": (0.04, 0.16),
    "extended_seconds": (0.02, 0.16),
    "flight_bend": (0.78, 1.12),
    "flight_lean": (-0.20, 0.20),
    "flight_front": (-0.15, 0.15),
    "flight_seconds": (0.12, 0.40),
}


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def leg_pose(bend, lean, front):
    q = np.array([[-0.1, bend + lean + front, -2 * bend],
                  [0.1, bend + lean + front, -2 * bend],
                  [-0.1, bend + lean - front, -2 * bend],
                  [0.1, bend + lean - front, -2 * bend]])
    return q.ravel()


def jump_target(t, cfg):
    stand = leg_pose(0.9, 0.0, 0.0)
    crouch = leg_pose(cfg.crouch, cfg.crouch_lean, cfg.crouch_front)
    extended = leg_pose(cfg.extend, cfg.push_lean, cfg.push_front)
    flight = leg_pose(cfg.flight_bend, cfg.flight_lean, cfg.flight_front)
    push_end = 1.8 + cfg.thrust_seconds
    tuck_start = push_end + cfg.extended_seconds
    tuck_end = tuck_start + 0.10
    recover_start = tuck_end + cfg.flight_seconds
    times = [0.0, 0.6, 1.5, 1.8, push_end, tuck_start,
             tuck_end, recover_start, recover_start + 0.8, 8.0]
    poses = [stand, stand, crouch, crouch, extended, extended,
             flight, flight, stand, stand]
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    w = ease((t - times[i]) / (times[i + 1] - times[i]))
    return (1 - w) * poses[i] + w * poses[i + 1]


def longest_run(mask, dt):
    padded = np.r_[False, mask, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    if not len(starts):
        return 0.0, 0, 0
    i = np.argmax(ends - starts)
    return float((ends[i] - starts[i]) * dt), int(starts[i]), int(ends[i])


def summarize(trace, criteria):
    t = trace["time"]
    dt = float(np.median(np.diff(t)))
    base, vel, ang = trace["base_pos"], trace["base_vel"], trace["base_ang_vel"]
    feet = trace["feet_contact"]
    flight = (feet == 0) & (t > 1.8) & (t < 4.0) & ~trace["nonfoot"]
    airtime, start, end = longest_run(flight, dt)
    air_dx = float(base[end - 1, 0] - base[start, 0]) if end > start else 0.0
    clearance = float(trace["foot_clearance"][start:end].max()) if end > start else 0.0
    boundaries = np.diff(np.r_[False, flight, False].astype(np.int8))
    flights = [
        {"start_s": float(t[s]), "last_airborne_s": float(t[e - 1]),
         "duration_s": float((e - s) * dt)}
        for s, e in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1))
    ]
    stable = (
        (feet == 4) & (np.linalg.norm(vel, axis=1) <= criteria["final_max_speed_m_s"])
        & (np.linalg.norm(ang, axis=1) <= criteria["final_max_angular_speed_rad_s"])
        & (np.abs(base[:, 2] - base[0, 2]) <= criteria["final_max_height_error_m"])
        & (trace["standing_joint_error"] <= criteria["final_max_joint_error_rad"])
        & ~trace["nonfoot"] & ~trace["joint_limit"]
    )
    unstable = np.flatnonzero(~stable)
    stable_tail = float(t[-1] - t[unstable[-1]]) if len(unstable) else float(t[-1] - t[0])
    final_dx = float(base[-1, 0] - base[0, 0])
    gain = float(base[:, 2].max() - base[0, 2])
    max_tilt = float(trace["tilt"].max())
    max_dy = float(np.abs(base[:, 1] - base[0, 1]).max())
    nonfoot = int(trace["nonfoot"].sum())
    limits = int(trace["joint_limit"].sum())
    max_joint_speed = float(trace["joint_speed"].max())
    passed = (
        airtime >= criteria["minimum_airborne_seconds"]
        and air_dx >= criteria["minimum_airborne_forward_m"]
        and final_dx >= criteria["minimum_final_forward_m"]
        and gain >= criteria["minimum_height_gain_m"]
        and clearance >= criteria["minimum_all_feet_clearance_m"]
        and max_tilt <= criteria["maximum_tilt_rad"]
        and max_dy <= criteria["maximum_lateral_displacement_m"]
        and stable_tail >= criteria["final_stable_seconds"]
        and nonfoot == 0 and limits == 0
        and max_joint_speed <= criteria["maximum_joint_speed_rad_s"]
    )
    return {
        "passed": bool(passed), "airborne_seconds": airtime,
        "airborne_forward_m": air_dx, "final_forward_m": final_dx,
        "height_gain_m": gain, "height_start_m": float(base[0, 2]),
        "height_min_m": float(base[:, 2].min()), "height_max_m": float(base[:, 2].max()),
        "height_final_m": float(base[-1, 2]), "max_all_feet_clearance_m": clearance,
        "takeoff_time_s": float(t[start]) if end > start else None,
        "last_airborne_time_s": float(t[end - 1]) if end > start else None,
        "flight_intervals": flights,
        "max_tilt_rad": max_tilt, "max_lateral_m": max_dy,
        "final_stable_seconds": stable_tail,
        "nonfoot_contact_samples": nonfoot, "joint_limit_samples": limits,
        "max_joint_speed_rad_s": max_joint_speed,
        "torque_saturation_sample_fraction": float(trace["torque_saturation"].mean()),
        "peak_abs_torque_nm": trace["peak_torque"].max(axis=0).tolist(),
    }


def rollout(p, cfg, seed=0, perturb=0.0, duration=8.0, dense=False, capture=False, video=None):
    p.reset(seed=seed, perturb=perturb)
    foot_geoms = np.array([p.model.geom(f"{leg}_foot_collision").id for leg in ("FL", "FR", "RL", "RR")])
    foot_set = set(foot_geoms)
    floor = p.model.geom("floor").id
    limits = p.model.jnt_range[p.jids]
    standing_q = p.data.qpos[p.qadr].copy()
    substeps = round(DT / p.model.opt.timestep)
    records, snapshots, actions = [], [], []

    def record(t):
        touching, nonfoot = set(), False
        for con in p.data.contact:
            if con.dist <= 0 and floor in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == floor else con.geom1
                touching.add(other)
                nonfoot |= other not in foot_set
        lin, ang = p.velocity(p.base_id)
        q = p.data.qpos[p.qadr]
        records.append({
            "time": t, "base_pos": p.data.xpos[p.base_id].copy(),
            "base_vel": lin, "base_ang_vel": ang,
            "feet_contact": len(touching & foot_set), "nonfoot": nonfoot,
            "foot_clearance": np.min(p.data.geom_xpos[foot_geoms, 2] - p.model.geom_size[foot_geoms, 0]),
            "tilt": np.arccos(np.clip(p.data.xmat[p.base_id].reshape(3, 3)[2, 2], -1, 1)),
            "joint_limit": np.any((q < limits[:, 0]) | (q > limits[:, 1])),
            "joint_speed": np.abs(p.data.qvel[p.vadr]).max(),
            "standing_joint_error": np.abs(q - standing_q).max(),
            "torque_saturation": np.mean(np.abs(p.data.actuator_force) >= p.model.actuator_forcerange[:, 1] * 0.99),
            "peak_torque": np.abs(p.data.actuator_force).copy(),
        })

    record(0.0)
    count = round(duration / DT)
    for i in range(count + 1):
        t = i * DT
        if capture:
            snapshots.append(p.snapshot())
        if video is not None and i % 2 == 0:
            video.capture(t)
        if i == count:
            break
        qtarget = jump_target(t, cfg)
        if np.any((qtarget < limits[:, 0]) | (qtarget > limits[:, 1])):
            raise ValueError("scripted joint target exceeds original joint limits")
        if np.max(np.abs((qtarget - p.default_q) / 0.25)) > 4:
            raise ValueError("scripted target exceeds the existing learner action range")
        actions.append((qtarget - p.default_q) / 0.25)
        p.data.ctrl[:] = qtarget[p.act_order]
        if dense:
            for k in range(substeps):
                mujoco.mj_step(p.model, p.data)
                mujoco.mj_forward(p.model, p.data)
                record(t + (k + 1) * p.model.opt.timestep)
        else:
            mujoco.mj_step(p.model, p.data, nstep=substeps)
            mujoco.mj_forward(p.model, p.data)
            record(t + DT)
    trace = {k: np.array([r[k] for r in records]) for k in records[0]}
    arrays = {k: np.array([s[k] for s in snapshots], np.float32) for k in snapshots[0]} if capture else None
    return trace, arrays, np.array(actions)


class JumpVideo(Video):
    """Fixed side camera so forward displacement remains visible."""

    def __init__(self, physics, path):
        super().__init__(physics, path, "AS2 scripted PD | NOT learned | simulation only")
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.65, 90, -12
        self.camera.lookat[:] = [0.12, 0, 0.26]
        self.start_x = float(physics.data.xpos[physics.base_id, 0])

    def capture(self, time):
        from PIL import Image, ImageDraw

        p = self.physics
        self.renderer.update_scene(p.data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 960, 54), fill=(245, 245, 245))
        draw.text((12, 8), self.label, fill=(20, 20, 20))
        dx = p.data.xpos[p.base_id, 0] - self.start_x
        draw.text((12, 30), f"t={time:.2f}s | forward x={dx:+.3f} m | feet on ground={p.diagnostics()['feet_contact']}", fill=(20, 20, 20))
        self.frames.append(np.array(frame))

    def close(self):
        from PIL import Image, ImageDraw

        super().close()
        times = (0.0, 1.6, 2.12, 2.24, 2.4, 4.0)
        sheet = Image.new("RGB", (1440, 584), "white")
        draw = ImageDraw.Draw(sheet)
        for i, t in enumerate(times):
            image = Image.fromarray(self.frames[round(t * 25)]).resize((480, 270))
            x, y = (i % 3) * 480, (i // 3) * 292
            sheet.paste(image, (x, y + 22))
            draw.text((x + 8, y + 5), f"t={t:.2f}s", fill=(20, 20, 20))
        sheet.save(self.path.with_name("jump-contact-sheet.png"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reference(cfg, output, trials, export):
    from as2_motion_imitation.motion import MotionClip, save_clip

    if trials < 1:
        raise ValueError("validation requires at least one perturbation trial")
    raw = ROOT / "data/raw/as2-native-forward-jump"
    if export:
        export = export.resolve()
        if not export.is_relative_to(ROOT / "data/processed"):
            raise ValueError("export must be inside data/processed")
        if export.exists() or export.with_suffix(".json").exists() or raw.exists():
            raise ValueError("refusing to overwrite existing reference or raw capture")
    criteria = json.loads(ACCEPTANCE.read_text())
    p = Physics()
    video = JumpVideo(p, output / "as2-forward-jump-scripted-physics.mp4")
    trace, arrays, actions = rollout(p, cfg, dense=True, capture=True, video=video)
    video.close()
    metrics = summarize(trace, criteria)
    np.savez_compressed(output / "nominal-dense-trace.npz", **trace)
    (output / "nominal-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"nominal": metrics}), flush=True)
    results = []
    perturbations = output / "perturbations"
    perturbations.mkdir()
    for seed in range(100, 100 + trials):
        tr, _, _ = rollout(p, cfg, seed=seed, perturb=0.01, dense=True)
        result = {"seed": seed, **summarize(tr, criteria)}
        results.append(result)
        np.savez_compressed(perturbations / f"seed-{seed}.npz", **tr)
        print(json.dumps({"seed": seed, "passed": result["passed"]}), flush=True)
    summary = {
        "scope": "Scripted controller, initial joint offsets only, NOT learned-policy success",
        "perturbation_rad": 0.01, "reset_settling_s": 3.0,
        "passed": sum(r["passed"] for r in results), "total": trials, "results": results,
    }
    (output / "perturbation-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "kind": "locally-generated-scripted-PD-physics",
        "upstream_motion": False, "learned_policy": False, "robot": "AS2",
        "motion": "stand-crouch-forward-jump-land-stand", "seed": 0,
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": digest(__file__), "parameters": asdict(cfg),
        "physics_dt": 0.002, "control_dt": DT, "settling_seconds": 3.0,
        "duration_seconds": 8.0, "pose_writes_after_reset": False,
        "external_forces": False, "original_gains_mass_and_torque_limits": True,
        "physics_backend": "CPU MuJoCo", "diagnostic_hz": 500,
        "self_collision_enabled": False,
        "self_collision_note": "Unchanged baseline mask; self-intersection is not certified.",
        "selection": "Fixed supplied script parameters; no policy learning in this capture",
        "acceptance_file": str(ACCEPTANCE.relative_to(ROOT)),
        "acceptance_sha256": digest(ACCEPTANCE),
        "nominal_metrics": metrics,
        "actuator_joint_names": [p.model.joint(int(j)).name for j in p.act_joints],
        "actuator_force_limits_nm": p.model.actuator_forcerange.tolist(),
        "max_abs_normalized_action": float(np.abs(actions).max()),
        "license_note": "Locally generated simulation data; model notices remain in baseline.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not metrics["passed"]:
        raise RuntimeError("nominal dense audit failed; reference was not exported")
    if export:
        raw.mkdir(parents=True)
        np.savez_compressed(raw / "physics_capture.npz", **arrays, action=actions,
                            action_time=np.arange(len(actions)) * DT)
        (raw / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        save_clip(export, arrays, p, provenance)
        clip = MotionClip(export)
        print(json.dumps({"exported": str(export), "frames": clip.frames,
                          "duration_s": clip.duration, "perturbations_passed": summary["passed"],
                          "perturbations_total": trials}), flush=True)


def score(metrics):
    m = metrics
    return float(
        8 * min(m["airborne_seconds"], 0.3)
        + 12 * np.clip(m["airborne_forward_m"], -0.3, 0.4)
        + 3 * np.clip(m["final_forward_m"], -0.5, 0.5)
        + 4 * min(m["height_gain_m"], 0.2)
        + min(m["final_stable_seconds"], 2.0)
        - 8 * max(m["max_tilt_rad"] - 0.35, 0)
        - 6 * m["max_lateral_m"]
        - 20 * (m["nonfoot_contact_samples"] > 0)
        - 0.15 * m["nonfoot_contact_samples"]
        - 10 * (m["joint_limit_samples"] > 0)
        - 10 * abs(m["height_final_m"] - m["height_start_m"])
        + 10 * m["passed"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=OUT / "search-01")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-trials", type=int, default=20)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("outputs must stay inside this subproject")
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a fresh output directory to preserve previous runs")
    output.mkdir(parents=True, exist_ok=True)
    criteria = json.loads(ACCEPTANCE.read_text())
    cfg = JumpParams(**json.loads(args.config.read_text())) if args.config else JumpParams()
    identity = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "initial_parameters": asdict(cfg), "bounds": BOUNDS,
                "generator_sha256": digest(__file__), "acceptance_sha256": digest(ACCEPTANCE)}
    (output / "run-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if args.validate:
        validate_reference(cfg, output, args.validation_trials, args.export)
        return
    if args.export:
        raise ValueError("--export requires --validate")
    p, rng = Physics(), np.random.default_rng(args.seed)
    lo, hi = np.array(list(BOUNDS.values())).T
    best, best_cfg = -np.inf, cfg
    start = time.perf_counter()
    with (output / "trials.jsonl").open("w") as log:
        for i in range(args.trials):
            if i:
                center = np.array(list(asdict(best_cfg).values()))
                scale = 0.20 if i < 100 else 0.10
                x = np.clip(center + rng.normal(size=len(lo)) * (hi - lo) * scale, lo, hi)
                cfg = JumpParams(**dict(zip(BOUNDS, x.tolist())))
            trace, _, _ = rollout(p, cfg)
            metrics = summarize(trace, criteria)
            value = score(metrics)
            result = {"trial": i, "score": value, "params": asdict(cfg), "metrics": metrics}
            log.write(json.dumps(result) + "\n")
            log.flush()
            if value > best:
                best, best_cfg = value, cfg
                (output / "best-params.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
                (output / "best-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
                np.savez_compressed(output / "best-trace.npz", **trace)
                print(json.dumps({"trial": i, "score": value, "elapsed_s": time.perf_counter() - start, **metrics}), flush=True)
    print(f"Finished {args.trials} trials in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()

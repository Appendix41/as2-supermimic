"""Record repeated PD jumps in one continuous simulation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from as2_motion_imitation.motion import MotionClip, save_clip, sha256
from as2_motion_imitation.physics import DT, Physics, Video
from as2_motion_imitation.prepare_jump import JumpParams, JumpVideo, jump_target

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/triple-forward-jump-script.json"
ACCEPTANCE = ROOT / "configs/triple-forward-jump-acceptance.json"


def load_sequence(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    params = JumpParams(**json.loads((ROOT / cfg["single_jump_parameters"]).read_text()))
    recovery_end = (1.8 + params.thrust_seconds + params.extended_seconds
                    + 0.10 + params.flight_seconds + 0.8)
    if cfg["count"] < 2 or cfg["period_seconds"] < recovery_end - 0.6:
        raise ValueError("sequence must permit continuous return to standing before the next crouch")
    if cfg["tail_duration_seconds"] < 8.0:
        raise ValueError("keep the original final jump and settling duration")
    duration = cfg["tail_duration_seconds"] + (cfg["count"] - 1) * cfg["period_seconds"]
    if not np.isclose(duration / DT, round(duration / DT)):
        raise ValueError("sequence duration must align with the control clock")
    return cfg, params, duration


def sequence_target(time, params, count, period):
    # Switch templates only at the next crouch onset, when both targets are standing.
    index = int(np.clip(np.floor((time - 0.6) / period), 0, count - 1))
    return jump_target(time - index * period, params)


def summarize_sequence(trace, criteria):
    t, base = trace["time"], trace["base_pos"]
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, atol=1e-8) or not all(
        np.isfinite(value).all() for value in trace.values()
    ):
        raise ValueError("sequence audit requires finite, uniformly sampled physics")
    feet = trace["feet_contact"]
    edges = np.diff(np.r_[False, feet == 0, False].astype(np.int8))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    jumps, all_flights = [], []
    for start, end in spans:
        duration = float((end - start) * dt)
        all_flights.append({"start_s": float(t[start]), "duration_s": duration})
        if duration + 1e-9 < criteria["minimum_counted_airborne_seconds"]:
            continue
        dx = float(base[end - 1, 0] - base[start, 0])
        gain = float(base[start:end, 2].max() - base[0, 2])
        clearance = float(trace["foot_clearance"][start:end].max())
        landed = end < len(t) and feet[end] > 0
        passed = (
            duration >= criteria["minimum_airborne_seconds"]
            and dx >= criteria["minimum_airborne_forward_m"]
            and gain >= criteria["minimum_height_gain_m"]
            and clearance >= criteria["minimum_all_feet_clearance_m"] and landed
        )
        jumps.append({
            "number": len(jumps) + 1, "start_s": float(t[start]),
            "last_airborne_s": float(t[end - 1]),
            "landing_s": float(t[end]) if landed else None,
            "duration_s": duration, "airborne_forward_m": dx,
            "height_gain_m": gain, "max_all_feet_clearance_m": clearance,
            "landed": bool(landed), "passed": bool(passed),
        })
    gaps = [
        b["start_s"] - a["landing_s"]
        for a, b in zip(jumps, jumps[1:]) if a["landing_s"] is not None
    ]
    stable = (
        (feet == 4)
        & (np.linalg.norm(trace["base_vel"], axis=1) <= criteria["final_max_speed_m_s"])
        & (np.linalg.norm(trace["base_ang_vel"], axis=1) <= criteria["final_max_angular_speed_rad_s"])
        & (np.abs(base[:, 2] - base[0, 2]) <= criteria["final_max_height_error_m"])
        & (trace["standing_joint_error"] <= criteria["final_max_joint_error_rad"])
        & ~trace["nonfoot"] & ~trace["joint_limit"]
    )
    unstable = np.flatnonzero(~stable)
    tail = float(t[-1] - t[unstable[-1]]) if len(unstable) else float(t[-1] - t[0])
    final_dx = float(base[-1, 0] - base[0, 0])
    tilt = float(trace["tilt"].max())
    lateral = float(np.abs(base[:, 1] - base[0, 1]).max())
    speed = float(trace["joint_speed"].max())
    nonfoot, limits = int(trace["nonfoot"].sum()), int(trace["joint_limit"].sum())
    saturation = float(trace["torque_saturation"].mean())
    passed = (
        len(jumps) == criteria["expected_jumps"] and all(j["passed"] for j in jumps)
        and len(gaps) == criteria["expected_jumps"] - 1
        and all(criteria["minimum_interjump_ground_seconds"] <= gap
                <= criteria["maximum_interjump_ground_seconds"] for gap in gaps)
        and final_dx >= criteria["minimum_final_forward_m"]
        and tilt <= criteria["maximum_tilt_rad"]
        and lateral <= criteria["maximum_lateral_displacement_m"]
        and speed <= criteria["maximum_joint_speed_rad_s"]
        and tail >= criteria["final_stable_seconds"]
        and saturation <= criteria["maximum_torque_saturation_fraction"]
        and nonfoot == 0 and limits == 0
    )
    return {
        "passed": bool(passed), "jump_count": len(jumps), "jumps": jumps,
        "all_flight_intervals": all_flights, "interjump_ground_seconds": gaps,
        "airborne_seconds": sum(j["duration_s"] for j in jumps),
        "airborne_forward_m": sum(j["airborne_forward_m"] for j in jumps),
        "final_forward_m": final_dx, "final_stable_seconds": tail,
        "height_start_m": float(base[0, 2]), "height_max_m": float(base[:, 2].max()),
        "height_final_m": float(base[-1, 2]), "max_tilt_rad": tilt,
        "max_lateral_m": lateral, "max_joint_speed_rad_s": speed,
        "nonfoot_contact_samples": nonfoot, "joint_limit_samples": limits,
        "torque_saturation_sample_fraction": saturation,
        "peak_abs_torque_nm": trace["peak_torque"].max(axis=0).tolist(),
    }


class SequenceVideo(JumpVideo):
    def __init__(self, physics, path):
        super().__init__(physics, path)
        self.camera.distance = 2.3
        self.camera.lookat[:] = [0.38, 0, 0.26]
        self.label = "AS2 three jumps | scripted PD, NOT learned | simulation"

    def close(self):
        from PIL import Image, ImageDraw

        Video.close(self)
        times = (0.0, 2.14, 2.6, 4.94, 5.4, 7.74, 8.2, 9.2, (len(self.frames) - 1) / 25)
        sheet = Image.new("RGB", (1440, 876), "white")
        draw = ImageDraw.Draw(sheet)
        for i, time in enumerate(times):
            frame = self.frames[min(round(time * 25), len(self.frames) - 1)]
            x, y = (i % 3) * 480, (i // 3) * 292
            sheet.paste(Image.fromarray(frame).resize((480, 270)), (x, y + 22))
            draw.text((x + 8, y + 5), f"t={time:.2f}s", fill=(20, 20, 20))
        sheet.save(self.path.with_name("triple-jump-contact-sheet.png"))


def rollout_sequence(physics, cfg, params, duration, seed=0, perturb=0.0,
                     capture=False, video=None):
    from as2_motion_imitation.evaluate_jump import JumpRecorder

    p = physics
    p.reset(seed=seed, perturb=perturb)
    recorder = JumpRecorder(p)
    recorder.record(0.0)
    snapshots, actions = [], []
    limits = p.model.jnt_range[p.jids]
    steps = round(duration / DT)
    for step in range(steps + 1):
        time = step * DT
        if capture:
            snapshots.append(p.snapshot())
        if video is not None and step % 2 == 0:
            video.capture(time)
        if step == steps:
            break
        target = sequence_target(time, params, cfg["count"], cfg["period_seconds"])
        action = (target - p.default_q) / 0.25
        if np.any((target < limits[:, 0]) | (target > limits[:, 1])) or np.abs(action).max() > 4:
            raise ValueError("script exceeds unchanged joint or action limits")
        actions.append(action)
        p.data.ctrl[:] = target[p.act_order]
        for substep in range(round(DT / p.model.opt.timestep)):
            mujoco.mj_step(p.model, p.data)
            mujoco.mj_forward(p.model, p.data)
            recorder.record(time + (substep + 1) * p.model.opt.timestep)
    arrays = (
        {k: np.array([s[k] for s in snapshots], np.float32) for k in snapshots[0]}
        if capture else None
    )
    return recorder.arrays(), arrays, np.array(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    raw = ROOT / "data/raw/as2-native-triple-forward-jump"
    if args.trials < 1 or (args.probe and args.export):
        raise ValueError("export requires validation trials")
    if args.export:
        args.export = args.export.resolve()
        args.export.relative_to(ROOT / "data/processed")
        if args.export.exists() or args.export.with_suffix(".json").exists() or raw.exists():
            raise ValueError("refusing to overwrite an existing reference")
    output.mkdir(parents=True, exist_ok=False)
    cfg, params, duration = load_sequence(args.config)
    criteria = json.loads(ACCEPTANCE.read_text())
    p = Physics()
    movie = None if args.probe else SequenceVideo(p, output / "scripted-triple-forward-jump.mp4")
    trace, arrays, actions = rollout_sequence(p, cfg, params, duration,
                                              capture=not args.probe, video=movie)
    if movie:
        movie.close()
    nominal = summarize_sequence(trace, criteria)
    np.savez_compressed(output / "nominal-dense-trace.npz", **trace)
    (output / "nominal-metrics.json").write_text(json.dumps(nominal, indent=2) + "\n")
    print(json.dumps({"nominal": nominal}), flush=True)
    if args.probe:
        return
    results = []
    perturbations = output / "perturbations"
    perturbations.mkdir()
    for seed in range(100, 100 + args.trials):
        trial, _, _ = rollout_sequence(p, cfg, params, duration, seed=seed, perturb=0.01)
        result = {"seed": seed, **summarize_sequence(trial, criteria)}
        results.append(result)
        np.savez_compressed(perturbations / f"seed-{seed}.npz", **trial)
        print(json.dumps({"seed": seed, "passed": result["passed"],
                          "jump_count": result["jump_count"]}), flush=True)
    summary = {"scope": "Scripted PD reference, NOT learned", "perturbation_rad": 0.01,
               "passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
    (output / "perturbation-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = [
        Path(__file__), ROOT / "src/as2_motion_imitation/prepare_jump.py",
        ROOT / "src/as2_motion_imitation/evaluate_jump.py",
        ROOT / "src/as2_motion_imitation/physics.py", args.config.resolve(),
        ROOT / cfg["single_jump_parameters"], ACCEPTANCE,
    ]
    provenance = {
        "kind": "locally-generated-scripted-PD-physics", "robot": "AS2",
        "motion": "three-consecutive-forward-jumps-land-stand",
        "upstream_motion": False, "learned_policy": False,
        "sequence": cfg, "single_jump_parameters": asdict(params),
        "generator": str(Path(__file__).relative_to(ROOT)),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "seed": 0, "settling_seconds": 3.0, "duration_seconds": duration,
        "physics_dt": p.model.opt.timestep, "control_dt": DT, "diagnostic_hz": 500,
        "pose_writes_after_reset": False, "resets_between_jumps": False,
        "concatenated_state_clips": False, "external_forces": False,
        "original_gains_mass_and_torque_limits": True, "self_collision_enabled": False,
        "physics_backend": "CPU MuJoCo", "nominal_metrics": nominal,
        "max_abs_normalized_action": float(np.abs(actions).max()),
        "actuator_force_limits_nm": p.model.actuator_forcerange.tolist(),
        "license_note": "Locally generated simulation data; model notices remain in baseline.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not nominal["passed"] or summary["passed"] != args.trials:
        raise RuntimeError("reference validation failed; no clip was exported")
    if args.export:
        raw.mkdir(parents=True)
        np.savez_compressed(raw / "physics_capture.npz", **arrays, action=actions,
                            action_time=np.arange(len(actions)) * DT)
        (raw / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        save_clip(args.export, arrays, p, provenance)
        clip = MotionClip(args.export)
        print(json.dumps({"exported": str(args.export), "frames": clip.frames,
                          "duration_s": clip.duration, "passed": summary["passed"]}), flush=True)


if __name__ == "__main__":
    main()

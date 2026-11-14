"""Generate crouch and standing reference clips in MuJoCo."""

import json
from pathlib import Path

import numpy as np

from as2_motion_imitation.motion import save_clip
from as2_motion_imitation.physics import DT, Physics, Video

ROOT = Path(__file__).resolve().parents[2]


def crouch_action(t, amplitude=0.13):
    def ease(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 - 15 * x + 6 * x**2)

    depth = ease((t - 1) / 1.5) * (1 - ease((t - 3.5) / 1.5))
    a = np.zeros((4, 3))
    a[:, 1], a[:, 2] = amplitude * depth / 0.25, -2 * amplitude * depth / 0.25
    return a.ravel()


def main():
    p = Physics()
    raw = ROOT / "data/raw/as2-native-crouch"
    out = ROOT / "outputs/reference-selection"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    video = Video(p, out / "as2-crouch-scripted-physics.mp4", "AS2 scripted PD physics | NOT learned")
    snapshots, actions, diagnostics = [], [], []
    for i in range(401):
        t = i * DT
        snapshots.append(p.snapshot())
        diagnostics.append(p.diagnostics())
        if i % 2 == 0:
            video.capture(t)
        a = crouch_action(t)
        actions.append(a)
        p.step(a)
    video.close()
    arrays = {k: np.array([s[k] for s in snapshots], np.float32) for k in snapshots[0]}
    np.savez_compressed(raw / "physics_capture.npz", **arrays, action=np.array(actions))
    provenance = {
        "kind": "locally-generated-scripted-PD-physics",
        "upstream_motion": False, "robot": "AS2", "seed": 0,
        "generator": "src/as2_motion_imitation/prepare.py",
        "motion": "stand-crouch-rise-stand", "thigh_offset_rad": 0.13,
        "calf_offset_rad": -0.26, "physics_dt": 0.002, "control_dt": DT,
        "settling_seconds": 3.0, "duration_seconds": 8.0,
        "pose_writes_after_reset": False,
        "license_note": "Locally generated simulation data; model notices remain in baseline.",
    }
    (raw / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    save_clip(ROOT / "data/processed/as2_crouch.npz", arrays, p, provenance)
    # Include a standing clip to test tracking without a crouch motion.
    static = {k: np.repeat(v[:1], 101, axis=0) for k, v in arrays.items()}
    for k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w", "qvel"):
        static[k][:] = 0
    save_clip(ROOT / "data/processed/as2_stand.npz", static, p,
              {**provenance, "kind": "static-FK-control-test", "motion": "stand"})
    result = {
        "scope": "Reference collection, NOT learned policy",
        "height_min": min(d["height"] for d in diagnostics),
        "height_start": diagnostics[0]["height"],
        "height_end": diagnostics[-1]["height"],
        "max_tilt_rad": max(d["tilt"] for d in diagnostics),
        "nonfoot_contact_frames": sum(d["nonfoot_contact"] for d in diagnostics),
        "joint_limit_frames": sum(d["joint_limit"] for d in diagnostics),
        "min_contact_feet": min(d["feet_contact"] for d in diagnostics),
        "max_torque_saturation_fraction": max(d["torque_saturation"] for d in diagnostics),
        "final_two_seconds_max_speed": max(d["speed"] for d in diagnostics[-100:]),
    }
    (out / "reference-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

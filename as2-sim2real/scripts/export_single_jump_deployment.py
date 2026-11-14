"""Export the frozen single-jump actor into a simulator-free NumPy bundle."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2_motion_imitation.deployment.runtime import InferenceBundle, SCHEMA, sha256
from as2_motion_imitation.deployment.offline import verify
from as2_motion_imitation.learner import ROOT, load_actor
from as2_motion_imitation.motion import MotionClip
from as2_motion_imitation.task import OBS_DIM, OBSERVATION_SCHEMA, OFFSETS, TrackingTask, push_history, sensors
from as2_supermimic.robot.morphology import JOINT_NAMES
MOTION_HASH = "75ef2084372ec058067ef30d7f8f7ead8c9a65046a346e4f511de7bffe968f6b"


@torch.no_grad()
def export(output, model, evaluation):
    torch.set_num_threads(2)
    output = output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    motion = ROOT / "data/processed/as2_forward_jump.npz"
    if sha256(motion) != MOTION_HASH:
        raise ValueError("Frozen source artifact hash mismatch")
    clip = MotionClip(motion)
    actor, _ = load_actor(model, clip)
    actor.eval()
    state = actor.state_dict()
    weights = {"mean": state["norm.mean"].numpy(), "var": state["norm.var"].numpy()}
    for i, layer in enumerate((0, 2, 4, 6)):
        weights[f"weight_{i}"] = state[f"net.{layer}.weight"].numpy()
        weights[f"bias_{i}"] = state[f"net.{layer}.bias"].numpy()
    np.savez(output / "actor.npz", **weights)
    # Bake through the original interpolation path, including quaternion gravity.
    times = torch.tensor([i * 0.02 for i in range(400)], dtype=torch.float32)
    refs = clip.sample(times[:, None] + torch.tensor(OFFSETS))
    np.savez(
        output / "reference.npz", states=refs["state"][..., :34].numpy(),
        phases=(times / clip.duration).numpy(),
        default_q=clip.default_q.numpy(),
    )
    task = TrackingTask(clip)
    rows = {key: [] for key in ("step", "sensor", "observation", "action", "target")}
    report = json.loads((evaluation / "metrics.json").read_text())
    if report["checkpoint_sha256"] != sha256(model) or report.get("sensor_noise_std", []):
        raise ValueError("Parity export requires matching, noiseless evaluation trajectories")
    trace_files = sorted(evaluation.glob("seed-*.npz"))
    if len(trace_files) != 20:
        raise ValueError("Expected all 20 preserved final evaluation trajectories")
    traces = []
    for path in trace_files:
        with np.load(path, allow_pickle=False) as data:
            states, poses, actions = (torch.from_numpy(data[k].copy()) for k in ("state", "pose", "action"))
        origin = poses[0:1]
        history = sensors(states[0:1], torch.zeros(1, 12)).repeat(1, 5)
        for i in range(400):
            previous = torch.zeros(1, 12) if i == 0 else actions[i - 1:i]
            sensor = sensors(states[i:i + 1], previous)[0]
            obs = task.observe(history, states[i:i + 1], poses[i:i + 1], times[i:i + 1], origin)
            action = actor(obs).clamp(-4, 4)[0]
            if (action - actions[i]).abs().max() > 1e-5:
                raise ValueError(f"Source actor no longer matches recorded trajectory: {path.name}:{i}")
            values = (i, sensor.numpy(), obs[0].numpy(),
                      action.numpy(), (clip.default_q + 0.25 * action).numpy())
            for key, value in zip(rows, values):
                rows[key].append(value)
            history = push_history(history, states[i + 1:i + 2], actions[i:i + 1])
        traces.append({"name": path.name, "sha256": sha256(path)})
    np.savez_compressed(output / "parity_cases.npz", **{k: np.array(v) for k, v in rows.items()})
    manifest = {
        "schema": SCHEMA, "hardware_execution_enabled": False,
        "state_frame": clip.state_frame, "checkpoint_sha256": sha256(model),
        "observation_schema": OBSERVATION_SCHEMA,
        "motion_sha256": MOTION_HASH, "steps": 400, "dt": 0.02, "duration_s": 8.0,
        "obs_dim": OBS_DIM, "action_dim": 12, "history": 5,
        "joint_names_policy_order": list(JOINT_NAMES), "sdk_motor_map": None,
        "required_pose": False,
        "live_inputs": ["LowState IMU quaternion/gyro", "LowState joint q/dq",
                        "previous applied action", "local phase"],
        "previous_action": "actual applied normalized policy action; not an unexecuted shadow output",
        "normalizer": "(obs-mean)/(sqrt(var)+1e-6), clipped to [-10,10], frozen",
        "action_to_target": "default_q + 0.25 * clip(actor, -4, 4)",
        "reference_offsets_s": list(OFFSETS), "reference_mode": "baked at exact 50 Hz control ticks",
        "files": {n: sha256(output / n) for n in ("actor.npz", "reference.npz", "parity_cases.npz")},
        "source_traces": traces,
        "not_validated": [
            "real motor signs/offsets and IMU frame", "hardware actuation",
            "independent safety controller", "mass/friction/latency robustness",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parity, _ = verify(InferenceBundle(output), output / "parity_cases.npz")
    (output / "export-verification.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(json.dumps(parity, indent=2))
    if not parity["passed"]:
        raise ValueError("Export parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    export(args.output, args.checkpoint, args.evaluation)

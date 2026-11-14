"""Collect completed training and validation evidence without changing any run."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from as2_motion_imitation.learner import ROOT, load_actor
from as2_motion_imitation.motion import MotionClip


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--numpy-evaluation", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    actor, ckpt = load_actor(args.checkpoint, MotionClip(
        ROOT / "data/processed/as2_forward_jump.npz"))
    run = args.checkpoint.parent
    status = read(run / "status.json")
    if status["iteration"] != ckpt["cfg"]["iterations"]:
        raise ValueError("Training has not completed its recorded iteration budget")
    nominal = read(args.evaluation / "metrics.json")
    fields = (
        "airborne_seconds", "airborne_forward_m", "final_forward_m", "height_gain_m",
        "max_tilt_rad", "max_lateral_m", "final_stable_seconds", "mean_tracking_score",
        "max_joint_speed_rad_s", "torque_saturation_sample_fraction",
    )
    rows = nominal["trials_detail"]
    aggregate = {
        key: {"mean": float(np.mean([r[key] for r in rows])),
              "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
        for key in fields
    }
    evaluations = []
    for path in sorted((ROOT / "outputs/evaluation").glob("*/metrics.json")):
        info = read(path)
        if "iteration" in info:
            evaluations.append({
                "path": str(path.relative_to(ROOT)), "iteration": info["iteration"],
                "successes": info["successes"], "trials": info["trials"],
                "sensor_noise_std": info.get("sensor_noise_std", []),
            })
    result = {
        "date_local": "2026-11-12", "timezone": "Asia/Shanghai",
        "training_finished": True, "hardware_execution_enabled": False,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
        "schema": ckpt["schema"], "observation_schema": ckpt["observation_schema"],
        "actor_input_dim": ckpt["obs_dim"],
        "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
        "motion_sha256": ckpt["motion_sha256"], "training_config": ckpt["cfg"],
        "training_status": status, "nominal_successes": nominal["successes"],
        "nominal_trials": nominal["trials"], "nominal_aggregate": aggregate,
        "checkpoint_evaluations": evaluations,
        "independent_nominal_audit": read(args.evaluation / "independent-audit.json"),
        "input_contract": read(ROOT / "outputs/validation/input-contract.json"),
        "export_parity": read(args.bundle / "export-verification.json"),
        "numpy_closed_loop": {
            k: v for k, v in read(args.numpy_evaluation / "metrics.json").items()
            if k not in ("trials_detail", "criteria")
        },
        "robustness": read(args.robustness / "summary.json"),
        "tests": [],
    }
    for path in sorted((ROOT / "outputs/validation").glob("*tests*.xml")):
        suite = ET.parse(path).getroot().find("testsuite")
        result["tests"].append({"path": str(path.relative_to(ROOT)), **suite.attrib})
    original = read(ROOT / "outputs/provenance/training-source.json")
    result["post_snapshot_changed_files"] = [
        name for name, value in original.items() if digest(ROOT / name) != value
    ]
    for name in ("task.py", "learner.py", "gpu_physics.py", "physics.py", "motion.py"):
        key = "src/as2_motion_imitation/" + name
        assert digest(ROOT / key) == original[key], key
    final = {}
    for tree in ("src", "scripts", "configs", "tests", "docs", "data", "baseline"):
        for path in sorted((ROOT / tree).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                final[str(path.relative_to(ROOT))] = digest(path)
    for name in ("README.md", "AGENTS.md", "START_NEXT_CHAT.md", "THIRD_PARTY_NOTICES.md"):
        final[name] = digest(ROOT / name)
    (ROOT / "outputs/provenance/final-files.json").write_text(json.dumps(final, indent=2) + "\n")
    with (ROOT / "outputs/results-summary.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "checkpoint_sha256": result["checkpoint_sha256"],
        "nominal": [result["nominal_successes"], result["nominal_trials"]],
        "aggregate": aggregate,
        "robustness": {name: [row["successes"], row["trials"]]
                       for name, row in result["robustness"]["cases"].items()},
        "post_snapshot_changed_files": result["post_snapshot_changed_files"],
    }, indent=2))


if __name__ == "__main__":
    main()

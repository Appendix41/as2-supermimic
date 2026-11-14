"""Recheck saved backflip evidence and replay actor commands in fresh physics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from as2_motion_imitation.backflip_audit import summarize
from as2_motion_imitation.motion import sha256
from as2_motion_imitation.physics import DT, Physics
from as2_motion_imitation.prepare_backflip import orientation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    output = args.evaluation.resolve()
    output.relative_to(ROOT / "outputs")
    metrics = json.loads((output / "metrics.json").read_text())
    assert sha256(metrics["checkpoint"]) == metrics["checkpoint_sha256"]
    assert sha256(metrics["motion"]) == metrics["motion_sha256"]
    successes = 0
    for row in metrics["trials_detail"]:
        with np.load(output / f"seed-{row['seed']}.npz") as trace:
            np.testing.assert_allclose(np.diff(trace["time"]), .002, atol=1e-12)
            assert len(trace["action"]) * DT == trace["time"][-1]
            physical = summarize({key: trace[key] for key in (
                "time", "base_pos", "base_quat", "base_vel", "base_ang_vel",
                "feet_contact", "nonfoot", "joint_limit", "joint_speed",
                "standing_joint_error", "torque", "torque_saturation",
            )}, metrics["criteria"])
            assert physical["checks"] == row["checks"]
            expected = physical["passed"] and not trace["tracking_failure"].any()
            assert bool(expected) == row["success"]
            successes += bool(expected)
    assert successes == metrics["successes"]
    seed = metrics["trials_detail"][0]["seed"]
    p = Physics(state_frame=metrics["state_frame"])
    p.reset(seed=seed, perturb=metrics["reset_perturbation_rad"])
    errors = {"qpos": 0., "qvel": 0.}
    with np.load(output / f"seed-{seed}.npz") as trace:
        np.testing.assert_allclose(p.data.qpos, trace["qpos"][0], atol=1e-12)
        for step, action in enumerate(trace["action"]):
            p.step(action)
            for key in errors:
                errors[key] = max(errors[key], float(np.abs(
                    getattr(p.data, key) - trace[key][(step+1)*10]).max()))
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        with np.load(metrics["motion"]) as ref:
            reference_time = np.arange(len(ref["qpos"])) * DT
            reference_back, _, _ = orientation(ref["qpos"][:, 3:7])
            back, _, _ = orientation(trace["base_quat"])
            for axis, predicted, target, label in (
                (axes[0], trace["base_pos"][:, 2], ref["qpos"][:, 2], "Root height (m)"),
                (axes[1], np.rad2deg(back-back[0]), np.rad2deg(reference_back-reference_back[0]),
                 "Backward rotation (deg)"),
            ):
                axis.plot(reference_time, target, label="Scripted reference", color="#176b52")
                axis.plot(trace["time"], predicted, label="Actor-only physics", color="#b34c36")
                axis.set_ylabel(label)
                axis.grid(alpha=.2)
            axes[0].legend()
        axes[2].step(trace["time"], trace["feet_contact"], label="Feet on ground", color="#176b52")
        axes[2].fill_between(trace["time"], 0, trace["nonfoot"].astype(int)*4,
                             alpha=.35, color="#b34c36", label="Nonfoot contact")
        axes[2].set(xlabel="Time (s)", ylabel="Contacts", ylim=(-.1, 4.2))
        axes[2].legend()
        figure.suptitle(f"AS2 backflip | iteration {metrics['iteration']} | "
                        f"actor-only evaluation {successes}/{metrics['trials']}")
        figure.savefig(output / "reference-policy-comparison.png", dpi=150)
        plt.close(figure)
    assert errors["qpos"] < 2e-6 and errors["qvel"] < 2e-5
    assert not p.data.xfrc_applied.any() and not p.data.qfrc_applied.any()
    result = {
        "scope": "Saved-trace recheck and actor command replay, not additional policy trials",
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "motion_sha256": metrics["motion_sha256"],
        "trials": metrics["trials"], "successes": successes,
        "replay_seed": seed, "replay_max_abs_errors": errors,
    }
    (output / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

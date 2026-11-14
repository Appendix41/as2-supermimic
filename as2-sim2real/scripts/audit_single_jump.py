"""Independent saved-contact recount and fresh replay of nominal policy actions."""

import argparse
import json
from pathlib import Path

import numpy as np

from as2_motion_imitation.motion import sha256
from as2_motion_imitation.physics import Physics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    out = args.evaluation.resolve()
    out.relative_to(ROOT / "outputs")
    metrics = json.loads((out / "metrics.json").read_text())
    if metrics.get("physics_variation") or metrics.get("action_delay_steps"):
        raise ValueError("This action-replay audit is for nominal physics only")
    assert sha256(metrics["checkpoint"]) == metrics["checkpoint_sha256"]
    flights = []
    physics = Physics()
    limits = physics.model.jnt_range[physics.jids]
    target_violations = 0
    target_margin = float("inf")
    for row in metrics["trials_detail"]:
        with np.load(out / f"seed-{row['seed']}.npz", allow_pickle=False) as trace:
            edges = np.diff(np.r_[False, trace["feet_contact"] == 0, False].astype(int))
            intervals = [(int(a), int(b)) for a, b in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                if (b - a) * 0.002 >= 0.04 - 1e-9]
            assert len(intervals) == 1, (row["seed"], intervals)
            a, b = intervals[0]
            assert b < len(trace["time"])
            assert trace["feet_contact"][b] > 0
            assert not trace["nonfoot"].any() and not trace["joint_limit"].any()
            assert not trace["tracking_failure"].any() and row["success"]
            flights.append({
                "seed": row["seed"], "count": 1,
                "airborne_seconds": (b - a) * 0.002,
                "airborne_forward_m": float(trace["base_pos"][b - 1, 0] - trace["base_pos"][a, 0]),
            })
            targets = physics.default_q + 0.25 * trace["action"]
            margin = np.minimum(targets - limits[:, 0], limits[:, 1] - targets)
            target_violations += int(np.sum(margin < 0))
            target_margin = min(target_margin, float(margin.min()))
    seed = metrics["trials_detail"][0]["seed"]
    physics.reset(seed, metrics["reset_perturbation_rad"])
    errors = {"qpos": 0.0, "qvel": 0.0}
    with np.load(out / f"seed-{seed}.npz", allow_pickle=False) as trace:
        for i, action in enumerate(trace["action"]):
            physics.step(action)
            for key in errors:
                errors[key] = max(errors[key], float(
                    np.abs(getattr(physics.data, key) - trace[key][(i + 1) * 10]).max()))
    assert errors["qpos"] < 2e-6 and errors["qvel"] < 2e-5, errors
    report = {"flights": flights, "minimum_counted_airtime_s": 0.04,
              "replay_max_errors": errors, "passed": True,
              "target_joint_limit_violation_count": target_violations,
              "minimum_target_joint_limit_margin_rad": target_margin,
              "scope": "Independent recount plus action replay, not additional policy trials"}
    (out / "independent-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

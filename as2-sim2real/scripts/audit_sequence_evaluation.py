"""Recount saved flights and replay one actor action sequence in fresh physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from as2_motion_imitation.motion import sha256
from as2_motion_imitation.physics import DT, Physics

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
    criteria = metrics["criteria"]
    counts = []
    for row in metrics["trials_detail"]:
        with np.load(output / f"seed-{row['seed']}.npz") as trace:
            time = trace["time"]
            np.testing.assert_allclose(np.diff(time), 0.002, atol=1e-12)
            assert len(trace["action"]) * DT == time[-1]
            edges = np.diff(np.r_[False, trace["feet_contact"] == 0, False].astype(int))
            flights = [
                (start, end) for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                if (end - start) * 0.002 + 1e-9 >= criteria["minimum_counted_airborne_seconds"]
            ]
            assert len(flights) == row["jump_count"]
            assert len(flights) == criteria["expected_jumps"]
            for (start, end), jump in zip(flights, row["jumps"]):
                seconds = (end - start) * 0.002
                forward = float(trace["base_pos"][end - 1, 0] - trace["base_pos"][start, 0])
                assert seconds >= criteria["minimum_airborne_seconds"]
                assert forward >= criteria["minimum_airborne_forward_m"]
                assert end < len(time) and trace["feet_contact"][end] > 0
                np.testing.assert_allclose(seconds, jump["duration_s"], atol=1e-10)
                np.testing.assert_allclose(forward, jump["airborne_forward_m"], atol=1e-10)
            assert not trace["nonfoot"].any() and not trace["joint_limit"].any()
            assert not trace["tracking_failure"].any()
            assert row["success"]
            counts.append({"seed": row["seed"], "flights": len(flights)})
    seed = metrics["trials_detail"][0]["seed"]
    p = Physics()
    p.reset(seed=seed, perturb=metrics["reset_perturbation_rad"])
    max_qpos, max_qvel = 0.0, 0.0
    with np.load(output / f"seed-{seed}.npz") as trace:
        np.testing.assert_allclose(p.data.qpos, trace["qpos"][0], atol=1e-10)
        substeps = round(DT / p.model.opt.timestep)
        for step, action in enumerate(trace["action"]):
            p.step(action)
            index = (step + 1) * substeps
            max_qpos = max(max_qpos, float(np.abs(p.data.qpos - trace["qpos"][index]).max()))
            max_qvel = max(max_qvel, float(np.abs(p.data.qvel - trace["qvel"][index]).max()))
    assert max_qpos < 2e-6 and max_qvel < 2e-5
    assert not p.data.xfrc_applied.any() and not p.data.qfrc_applied.any()
    rows = metrics["trials_detail"]
    keys = ("final_forward_m", "airborne_forward_m", "airborne_seconds",
            "final_stable_seconds", "max_lateral_m", "max_tilt_rad", "mean_tracking_score")
    aggregate = {
        key: {"mean": float(np.mean([r[key] for r in rows])),
              "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
        for key in keys
    }
    per_jump = [
        {key: float(np.mean([r["jumps"][i][key] for r in rows]))
         for key in ("start_s", "duration_s", "airborne_forward_m", "height_gain_m")}
        for i in range(criteria["expected_jumps"])
    ]
    result = {
        "scope": "Independent saved-trace checks plus recorded actor-action replay; not extra policy trials",
        "checkpoint_sha256": metrics["checkpoint_sha256"], "trials": counts,
        "action_replay_seed": seed, "action_replay_qpos_max_error": max_qpos,
        "action_replay_qvel_max_error": max_qvel,
        "aggregate": aggregate, "mean_per_jump": per_jump,
    }
    (output / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

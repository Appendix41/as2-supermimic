"""Evaluate the policy under the configured simulation perturbations."""

import argparse
import json
from pathlib import Path

from as2_motion_imitation.evaluate_jump import evaluate
from as2_motion_imitation.learner import ROOT


CASES = {
    "sensor-noise": {"sensor_noise_std": (0.01, 0.003, 0.002, 0.03), "perturb": 0.04},
    "mass-minus-10pct": {"variation": {"mass_scale": 0.9}},
    "mass-plus-10pct": {"variation": {"mass_scale": 1.1}},
    "friction-0p4": {"variation": {"friction": 0.4}},
    "friction-0p8": {"variation": {"friction": 0.8}},
    "gains-minus-10pct": {"variation": {"gain_scale": 0.9}},
    "gains-plus-10pct": {"variation": {"gain_scale": 1.1}},
    "command-delay-20ms": {"action_delay_steps": 1},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Separate diagnostic assumptions, not measured hardware parameters",
        "training_domain_randomization": "sensor sample noise only; not mass/friction/delay",
        "cases": {},
    }
    for index, (name, kwargs) in enumerate(CASES.items()):
        count = 20 if name == "sensor-noise" else 5
        start = 200 if name == "sensor-noise" else 300 + index * 10
        summary = evaluate(args.checkpoint, args.output / name,
                           seeds=range(start, start + count), video=False, **kwargs)
        report["cases"][name] = {
            "successes": summary["successes"], "trials": summary["trials"], "settings": kwargs,
            "failures": [{"seed": row["seed"],
                          "airborne_forward_m": row["airborne_forward_m"],
                          "final_stable_seconds": row["final_stable_seconds"],
                          "max_tilt_rad": row["max_tilt_rad"],
                          "joint_limit_samples": row["joint_limit_samples"],
                          "nonfoot_contact_samples": row["nonfoot_contact_samples"],
                          "tracking_failure": row["tracking_failure"]}
                         for row in summary["trials_detail"] if not row["success"]],
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": name, **report["cases"][name]}), flush=True)


if __name__ == "__main__":
    main()

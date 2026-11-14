"""Run the exported NumPy actor closed-loop in the existing isolated AS2 physics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from as2_motion_imitation.deployment.runtime import (
    InferenceBundle, init_history, lowstate_sensor, push_history,
)
from as2_motion_imitation.evaluate_jump import JumpRecorder
from as2_motion_imitation.learner import ROOT
from as2_motion_imitation.motion import MotionClip
from as2_motion_imitation.physics import Physics
from as2_motion_imitation.prepare_jump import summarize
from as2_motion_imitation.task import TrackingTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = InferenceBundle(args.bundle)
    criteria = json.loads((ROOT / "configs/forward-jump-acceptance.json").read_text())
    task = TrackingTask(MotionClip(ROOT / "data/processed/as2_forward_jump.npz"))
    torch.set_num_threads(2)
    results = []
    for seed in range(100, 120):
        physics = Physics(state_frame="forward-axis-v1")
        physics.reset(seed=seed, perturb=0.01)
        state, pose = physics.state()
        previous = np.zeros(12, dtype=np.float32)
        slots = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        calibration = {
            "verified_on_robot": False, "sdk_slots": slots,
            "joint_signs": [1] * 12, "joint_offsets_rad": [0] * 12,
            "policy_from_imu": np.eye(3).tolist(),
        }

        def sensor():
            rotation = physics.data.xmat[physics.base_id].reshape(3, 3)
            _, angular = physics.velocity(physics.base_id)
            motors = [{"slot": i, "q": 0.0, "dq": 0.0} for i in range(35)]
            for j, slot in enumerate(slots):
                motors[slot]["q"] = float(physics.data.qpos[physics.qadr[j]])
                motors[slot]["dq"] = float(physics.data.qvel[physics.vadr[j]])
            row = {
                "topic": "rt/lowstate", "received_monotonic_ns": 0,
                "quaternion_raw": physics.data.xquat[physics.base_id].tolist(),
                "gyroscope_raw": (rotation.T @ angular).tolist(), "motor_slots": motors,
            }
            return lowstate_sensor(row, bundle.default_q, previous, calibration, 0,
                                   allow_unverified=True)

        history = init_history(sensor())
        recorder = JumpRecorder(physics)
        recorder.record(0.0)
        actions, failed = [], False
        for step in range(bundle.steps):
            obs = bundle.observe(history, step)
            previous = bundle.act(obs)
            physics.data.ctrl[:] = bundle.target(previous)[physics.act_order]
            for substep in range(round(bundle.dt / physics.model.opt.timestep)):
                mujoco.mj_step(physics.model, physics.data)
                mujoco.mj_forward(physics.model, physics.data)
                recorder.record(step * bundle.dt + (substep + 1) * physics.model.opt.timestep)
            state, pose = physics.state()
            failed |= bool(task.failure(
                torch.tensor(state)[None], torch.tensor([(step + 1) * bundle.dt])).item())
            history = push_history(history, sensor())
            actions.append(previous.copy())
        trace = recorder.arrays()
        physical = summarize(trace, criteria)
        result = {"seed": seed, "success": physical["passed"] and not failed,
                  "tracking_failure": failed, **physical}
        results.append(result)
        np.savez_compressed(args.output / f"seed-{seed}.npz", **trace, action=np.array(actions))
        print(json.dumps({"seed": seed, "success": result["success"],
                          "forward_m": physical["final_forward_m"]}), flush=True)
    report = {
        "scope": "Exported NumPy actor closed-loop in original CPU MuJoCo model; no DDS or robot access",
        "successes": sum(row["success"] for row in results), "trials": len(results),
        "seeds": list(range(100, 120)), "initial_joint_perturbation_rad": 0.01,
        "same_seeds_as_original_not_blind_test": True,
        "no_reference_pose_writes": True, "no_reference_action_driver": True,
        "mass_friction_delay_randomization": False,
        "input_path": "synthetic LowState quaternion/gyro/q/dq -> shared calibrated decoder",
        "actor_received_actual_pose": False,
        "identity_calibration_is_simulation_only": True,
        "criteria": criteria, "trials_detail": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"successes": report["successes"], "trials": report["trials"]}))
    if report["successes"] != report["trials"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

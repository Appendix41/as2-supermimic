"""Integrated supervisor/ledger in CPU MuJoCo, never a robot connection."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from as2_motion_imitation.deployment.controller import (
    Controller, ControllerConfig, Reply, SLOTS, Snapshot,
)
from as2_motion_imitation.deployment.runtime import InferenceBundle
from as2_motion_imitation.evaluate_jump import JumpRecorder
from as2_motion_imitation.physics import Physics
from as2_motion_imitation.prepare_jump import summarize


ROOT = Path(__file__).resolve().parents[1]


def trial(bundle, document, seed, output):
    physics = Physics(state_frame="forward-axis-v1")
    physics.reset(seed=seed, perturb=0.01)
    limits = physics.model.jnt_range[physics.jids]
    document = json.loads(json.dumps(document))
    document["limits"]["lower"] = limits[:, 0].tolist()
    document["limits"]["upper"] = limits[:, 1].tolist()
    # The loaded model sags by about 0.159 rad at standstill under gravity.
    # This simulation-only start tolerance is not a hardware calibration.
    document["safety"]["initial_error_rad"] = 0.2
    config = ControllerConfig(document)
    controller = Controller(bundle, config)
    start = 1_000_000_000
    now = start

    def feedback():
        rotation = physics.data.xmat[physics.base_id].reshape(3, 3)
        _, angular = physics.velocity(physics.base_id)
        torque = np.zeros(12)
        torque[physics.act_order] = physics.data.actuator_force
        motors = [dict(slot=i, q=0.0, dq=0.0, tau_est=0.0, status=0, warning=0,
                       temperature_raw=[30, 30], voltage_raw=42.0) for i in range(12)]
        for index, slot in enumerate(SLOTS):
            motors[slot].update(q=float(physics.data.qpos[physics.qadr[index]]),
                                dq=float(physics.data.qvel[physics.vadr[index]]),
                                tau_est=float(torque[index]))
        row = dict(topic="rt/lowstate", received_monotonic_ns=now,
                   tick=(now // 1_000_000) % 2 ** 32, mode_machine=1, mode_pr=0,
                   quaternion_raw=physics.data.xquat[physics.base_id].tolist(),
                   gyroscope_raw=(rotation.T @ angular).tolist(), motor_slots=motors,
                   wireless_remote_hex=(b"\x55\x51" + bytes(38)).hex())
        return Snapshot(row, True, now, (0,) * 6)

    controller.begin(now, feedback())
    reply = None
    recorder = JumpRecorder(physics)
    policy_start = None
    events = []
    for substep in range(1, 8001):
        now = start + substep * 2_000_000
        command, request = controller.advance(now, feedback(), reply)
        reply = None
        if command is None:
            break
        if command.kind != "position":
            break
        target = np.asarray(command.raw_q)[SLOTS]
        kp = np.asarray(command.raw_kp)[SLOTS][physics.act_order]
        kd = np.asarray(command.raw_kd)[SLOTS][physics.act_order]
        physics.data.ctrl[:] = target[physics.act_order]
        physics.model.actuator_gainprm[:, 0] = kp
        physics.model.actuator_biasprm[:, 1] = -kp
        physics.model.actuator_biasprm[:, 2] = -kd
        controller.submitted(command, True, now)
        if policy_start is None and controller.step == 1:
            policy_start = now
            recorder.record(0.0)
        if request is not None:
            action = bundle.act(request.observation)
            reply = Reply(request.sequence, request.step, tuple(float(v) for v in action))
        mujoco.mj_step(physics.model, physics.data)
        mujoco.mj_forward(physics.model, physics.data)
        if policy_start is not None and now - policy_start < 8_000_000_000:
            recorder.record((now - policy_start) / 1e9 + 0.002)
        if substep % 10 == 0:
            events.append(dict(time_ns=now, state=controller.state, step=controller.step,
                               command=asdict(command)))
    physical = None
    if policy_start is not None:
        trace = recorder.arrays()
        physical = summarize(trace, json.loads((ROOT / "configs/forward-jump-acceptance.json").read_text()))
        np.savez_compressed(output / f"seed-{seed}.npz", **trace)
    report = {"seed": seed, "policy_steps": controller.step, "reason": controller.reason,
              "physical": physical,
              "passed": bool(controller.reason == "one_shot_complete"
                             and controller.step == 400 and physical and physical["passed"])}
    with (output / f"seed-{seed}-commands.jsonl").open("x") as stream:
        for row in events:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.trials <= 20:
        parser.error("trials must be in [1,20]")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = InferenceBundle(ROOT / "outputs/deployment/proprio-v2/bundle")
    config = json.loads((ROOT / "configs/controller-synthetic.json").read_text())
    results = [trial(bundle, config, seed, args.output) for seed in range(100, 100 + args.trials)]
    report = {"scope": "Full supervisor, history, 2ms command loop, simulated handoff and physical AS2 model",
              "hardware_commands_sent": 0, "dds_entities_created": 0,
              "calibration": "Simulation identity, not hardware",
              "inference_delay": "One 2ms simulation step, no wall-clock deadline qualification",
              "stopping_physics_tested": False,
              "limits": "Original simulator joint limits; other limits from synthetic fixture",
              "simulation_initial_pose_tolerance_rad": 0.2,
              "trials": results, "passed": all(row["passed"] for row in results)}
    with (args.output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

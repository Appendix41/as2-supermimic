"""No-DDS integration fault tests; optional official SDK serialization only."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .controller import ControllerConfig, ControlFault, encode_command, make_command
from .controller_run import SyntheticTransport, run_session
from .controller_worker import ActorWorker
from .runtime import InferenceBundle


class DyingWorker:
    def __init__(self, worker, after=3):
        self.worker, self.after, self.count = worker, after, 0
        self.killed_ns = None

    def submit(self, request):
        self.count += 1
        if self.count == self.after:
            self.killed_ns = time.monotonic_ns()
            self.worker.process.kill()
            self.worker.process.join(1)
        else:
            self.worker.submit(request)

    def poll(self):
        return self.worker.poll()


def process_death(bundle_path, config):
    bundle = InferenceBundle(bundle_path)
    worker = ActorWorker(bundle_path, bundle.manifest["checkpoint_sha256"])
    fault = DyingWorker(worker)
    events = []
    try:
        result = run_session(bundle, config, SyntheticTransport(bundle, config), fault, events)
        passive = [row["time_ns"] for row in events if row.get("event") == "submission"
                   and row["command"]["kind"] == "damping"]
        delay = ((passive[0] - fault.killed_ns) / 1e6
                 if passive and fault.killed_ns is not None else None)
        passed = (fault.killed_ns is not None and worker.process.exitcode == -9
                  and result["reason"] in ("actor_process_dead", "actor_connection_lost")
                  and result["policy_steps_submitted"] == 2 and delay is not None and 0 <= delay <= 40)
        return dict(result=result, passed=passed, producer_exitcode=worker.process.exitcode,
                    passive_decision_after_kill_ms=delay, commands_sent_to_robot=0,
                    scope="Actual actor SIGKILL, memory-only transport and synthetic feedback")
    finally:
        worker.close()


def nominal_trial(bundle_path, config):
    bundle = InferenceBundle(bundle_path)
    worker = ActorWorker(bundle_path, bundle.manifest["checkpoint_sha256"])
    events = []
    try:
        result = run_session(bundle, config, SyntheticTransport(bundle, config), worker, events)
        result["commands_sent_to_robot"] = 0
        result["scope"] = "Synthetic feedback, memory transport; actual exported actor in child process"
        return result
    finally:
        worker.close()


def sdk_frames(config, default_q):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    crc = CRC()
    cases = []
    for kind in ("position", "damping", "stop"):
        for machine in (1, 2, 127):
            command = make_command(default_q, config.kp, config.kd, config, kind, machine)
            frame = encode_command(command, unitree_hg_msg_dds__LowCmd_, crc)
            decoded = LowCmd_.deserialize(frame.serialize())
            checks = [
                decoded.crc == crc.Crc(decoded),
                decoded.mode_machine == machine,
                all(m.mode == m.q == m.kp == m.kd == m.dq == m.tau == m.reserve == 0
                    for m in decoded.motor_cmd[12:]),
                np.allclose([m.q for m in decoded.motor_cmd[:12]], command.raw_q),
                np.allclose([m.kp for m in decoded.motor_cmd[:12]], command.raw_kp),
                np.allclose([m.kd for m in decoded.motor_cmd[:12]], command.raw_kd),
                all(m.mode == int(kind != "stop") for m in decoded.motor_cmd[:12]),
            ]
            expected = decoded.crc
            decoded.motor_cmd[0].q += 0.001
            checks.append(crc.Crc(decoded) != expected)
            cases.append({"kind": kind, "synthetic_mode_machine": machine, "passed": all(checks)})
    return {"passed": all(c["passed"] for c in cases), "cases": cases,
            "scope": "Official SDK serialization only, zero DDS entities"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-frames", action="store_true")
    parser.add_argument("--nominal-trials", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 0 <= args.nominal_trials <= 5:
        parser.error("nominal-trials must be in [0,5]")
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "robot_commands_sent": 0, "dds_entities_created": 0,
              "service_changes": 0, "hardware_execution_enabled": False}
    try:
        config = ControllerConfig(json.loads(args.config.read_text()))
        if config.profile != "simulation-only":
            raise ControlFault("Synthetic profile required for selftest")
        bundle = InferenceBundle(args.bundle)
        report["nominal_trials"] = [
            nominal_trial(args.bundle, config) for _ in range(args.nominal_trials)]
        report["process_death_trials"] = [process_death(args.bundle, config) for _ in range(3)]
        if args.sdk_frames:
            report["sdk_frames"] = sdk_frames(config, bundle.default_q)
        report["passed"] = all(trial["passed"] for trial in report["process_death_trials"])
        report["passed"] &= all(trial["passed"] for trial in report["nominal_trials"])
        if args.sdk_frames:
            report["passed"] &= report["sdk_frames"]["passed"]
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

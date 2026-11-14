"""Validate or run the integrated controller. Default mode never opens DDS."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import fcntl
import json
from pathlib import Path
import signal
import time

import numpy as np

from .controller import Controller, ControllerConfig, ControlFault, Snapshot, SLOTS, release_control
from .controller_worker import ActorWorker, wait_for_snapshot
from .runtime import InferenceBundle, sha256
from .telemetry import network_preflight


class SyntheticTransport:
    """Immediate target following, not physics and not a hardware acceptance test."""

    def __init__(self, bundle, config):
        self.q = np.empty(12)
        self.q[SLOTS] = (bundle.default_q - config.offsets) / config.signs
        self.machine = config.expected_machine
        self.owner = "synthetic-vendor"
        self.commands = []

    def enable_watch(self):
        pass

    def snapshot(self):
        now = time.monotonic_ns()
        motors = [dict(slot=i, q=float(q), dq=0.0, tau_est=0.0, status=0, mode=1,
                       warning=0, temperature_raw=[30, 30], voltage_raw=42.0)
                  for i, q in enumerate(self.q)]
        row = dict(topic="rt/lowstate", received_monotonic_ns=now,
                   tick=(now // 1_000_000) % 2 ** 32, mode_machine=self.machine, mode_pr=0,
                   quaternion_raw=[1, 0, 0, 0], gyroscope_raw=[0, 0, 0],
                   wireless_remote_hex=(b"\x55\x51" + bytes(38)).hex(), motor_slots=motors)
        return Snapshot(row, True, now, (0,) * 6)

    def check_mode(self):
        return 0, {"name": self.owner}

    def release_mode(self):
        self.owner = ""
        return 0, None

    def monitor_owner(self):
        pass

    def publish(self, command):
        self.commands.append(command)
        if command.kind == "position":
            self.q[:] = command.raw_q
        return True

    def close(self):
        pass


class ControllerLock:
    """Exclude another instance in the same deployment, not other DDS hosts."""

    def __init__(self, path):
        self.stream = Path(path).open("a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ControlFault("another_controller_holds_deployment_lock") from error

    def close(self):
        self.stream.close()


def run_session(bundle, config, transport, worker, events, stop_requested=lambda: False,
                clock=time.monotonic_ns, sleep=time.sleep):
    controller = Controller(bundle, config)
    owner_confirmed = False
    gc_was_enabled = gc.isenabled()
    events.append({"event": "session_start", "time_ns": clock()})
    try:
        # The session and log are bounded; defer cyclic GC until publishing ends.
        gc.collect()
        gc.disable()
        initial = wait_for_snapshot(transport, clock)
        # Preflight before service release; this instance is never used to emit.
        Controller(bundle, config).begin(clock(), initial)
        transport.enable_watch()
        if stop_requested():
            raise ControlFault("operator_cancelled_before_handoff")
        expected = (config.document["expected_service_name"] if config.profile == "hardware"
                    else "synthetic-vendor")

        def recheck_before_release():
            if stop_requested():
                raise ControlFault("operator_cancelled_before_handoff")
            snapshot = transport.snapshot()
            Controller(bundle, config).begin(clock(), snapshot)

        release_control(transport, expected, recheck_before_release)
        owner_confirmed = True
        events.append({"event": "release_confirmed", "time_ns": clock()})
        transport.monitor_owner()
        snapshot = transport.snapshot()
        controller.begin(clock(), snapshot)
        start = clock()
        next_wake = start
        while controller.state != "STOPPED":
            now = clock()
            if stop_requested():
                controller.stop("operator_process_signal", now)
            if now - start > 30_000_000_000:
                controller.stop("absolute_session_timeout", now)
            reply = None
            if controller.state not in ("STOPPING", "STOPPED"):
                try:
                    reply = worker.poll()
                except (ControlFault, EOFError, OSError) as error:
                    controller.stop(str(error), now)
            snapshot = transport.snapshot()
            now = clock()
            command, request = controller.advance(now, snapshot, reply)
            if command is not None:
                try:
                    accepted = transport.publish(command)
                except Exception as error:
                    accepted = False
                    events.append({"event": "write_exception", "error": str(error), "time_ns": clock()})
                completed_ns = clock()
                controller.submitted(command, accepted, completed_ns)
                events.append({"event": "submission", "time_ns": completed_ns,
                               "supervisor_and_write_ms": (completed_ns - now) / 1e6,
                               "accepted_by_transport": accepted is True,
                               "motor_acknowledged": False, "state": controller.state,
                               "step": controller.step, "command": asdict(command)})
                if completed_ns - now > config.loop_ms * 1e6:
                    controller.stop("write_deadline", completed_ns)
            if request is not None and controller.state not in ("STOPPING", "STOPPED"):
                try:
                    worker.submit(request)
                    events.append({"event": "actor_request", "time_ns": clock(), "step": request.step,
                                   "feedback": snapshot.low, "feedback_crc_valid": snapshot.crc_valid,
                                   "board_received_ns": snapshot.board_ns, "board": snapshot.board,
                                   "previous_submitted_action": controller.ledger.action.tolist()})
                except (ControlFault, OSError) as error:
                    controller.stop(str(error), clock())
            if len(events) > 18000:
                controller.stop("bounded_log_limit", clock())
            next_wake += 2_000_000
            remaining = (next_wake - clock()) / 1e9
            if remaining > 0:
                sleep(remaining)
            else:
                next_wake = clock()
    except BaseException as error:
        controller.stop(f"{type(error).__name__}: {error}", clock())
        # Only a confirmed owner may emit a passive frame on setup failure.
        if owner_confirmed:
            stop_start = clock()
            while clock() - stop_start < config.stop_s * 1e9:
                try:
                    snapshot = transport.snapshot()
                    command, _ = controller.advance(clock(), snapshot)
                    if command is None:
                        break
                    accepted = transport.publish(command)
                    controller.submitted(command, accepted, clock())
                    events.append({"event": "exception_damping_submission", "time_ns": clock(),
                                   "accepted_by_transport": accepted is True,
                                   "motor_acknowledged": False, "command": asdict(command)})
                except Exception:
                    break
                sleep(0.002)
    finally:
        if gc_was_enabled:
            gc.enable()
    positions = [row for row in events if row.get("event") == "submission"
                 and row["command"]["kind"] == "position"]
    return {
        "controller_state": controller.state, "reason": controller.reason,
        "policy_steps_submitted": controller.step,
        "sdk_accepted_submissions": controller.ledger.submissions,
        "write_failures": controller.write_failures,
        "passive_submissions_accepted": controller.stop_write_successes,
        "release_confirmed": owner_confirmed,
        "vendor_motion_automatically_restored": False,
        "passed": (controller.reason == "one_shot_complete" and controller.step == 400
                   and controller.write_failures == 0 and controller.stop_write_successes > 0
                   and controller.state == "STOPPED"),
        "physical_stop_verified": False,
        "cyclic_gc_deferred_during_bounded_session": True,
        "max_supervisor_and_write_ms": max(
            (row.get("supervisor_and_write_ms", 0) for row in positions), default=0),
        "max_position_submission_gap_ms": max(
            ((b["time_ns"] - a["time_ns"]) / 1e6 for a, b in zip(positions, positions[1:])), default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "simulate", "hardware"), default="validate")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", default="end0")
    parser.add_argument("--confirm-handoff", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "passed": False, "hardware_execution_started": False,
        "scope": "Controller integration, not physical acceptance",
    }
    events = []
    transport = worker = lock = None
    old_handlers = {}
    stopped = [False]
    try:
        bundle = InferenceBundle(args.bundle)
        config = ControllerConfig(json.loads(args.config.read_text()))
        report["config_sha256"] = sha256(args.config)
        report["bundle_manifest_sha256"] = sha256(args.bundle / "manifest.json")
        report["hardware_blockers"] = config.hardware_errors(bundle)
        if args.mode == "validate":
            report.update(passed=True, status="config_structurally_valid",
                          hardware_ready=not report["hardware_blockers"])
        else:
            if args.mode == "hardware":
                config.require_hardware(bundle)
                if not args.confirm_handoff:
                    raise ControlFault("explicit_handoff_confirmation_required")
                report["network"] = network_preflight(args.interface, "192.168.123.0/24")
                if not report["network"]["passed"]:
                    raise ControlFault("robot_network_unavailable")
                lock = ControllerLock(args.bundle.resolve().parent / ".controller.lock")
            elif config.profile != "simulation-only":
                raise ControlFault("simulate_requires_simulation_only_profile")
            worker = ActorWorker(args.bundle, bundle.manifest["checkpoint_sha256"])
            if args.mode == "hardware":
                from .controller_transport import SdkTransport
                transport = SdkTransport(args.interface, config, bundle)
                report["hardware_execution_started"] = True
            else:
                transport = SyntheticTransport(bundle, config)
                report["scope"] = "Synthetic feedback/transport, real exported actor in separate process; no physics or DDS"
            for name in (signal.SIGINT, signal.SIGTERM):
                old_handlers[name] = signal.signal(name, lambda *_: stopped.__setitem__(0, True))
            report.update(run_session(bundle, config, transport, worker, events, lambda: stopped[0]))
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        if worker is not None:
            worker.close()
        if transport is not None:
            transport.close()
        if lock is not None:
            lock.close()
        for name, handler in old_handlers.items():
            signal.signal(name, handler)
        with (args.output / "events.jsonl").open("x") as stream:
            for event in events:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        with (args.output / "summary.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

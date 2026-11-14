"""Offline SDK frames and process-failure tests. No DDS entities or robot access."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time

from .bench_control import (
    Decision, Feedback, Health, Intent, ProbeGuard, build_sdk_frame,
)


def synthetic_feedback(now, tick):
    return Feedback(now, tick % (2 ** 32), 1, 0, (0.0,) * 12,
                    (0.0,) * 12, (0,) * 12, True)


def intent_producer(connection):
    try:
        for sequence in range(300):
            connection.send(Intent(sequence, time.monotonic_ns(), 0.0))
            time.sleep(0.01)
    except (BrokenPipeError, EOFError):
        pass
    finally:
        connection.close()


def simulated_supervisor(intents, results):
    """Separate process with synthetic sensors and a memory-only command sink."""
    try:
        if not intents.poll(3):
            raise TimeoutError("No initial synthetic intent")
        last = intents.recv()
        initial_ns = time.monotonic_ns()
        guard = ProbeGuard(synthetic_feedback(initial_ns, 0), initial_ns, 3, -0.5, 0.5)
        enabled_count = 0
        ready = False
        while time.monotonic_ns() - initial_ns < 4_000_000_000:
            time.sleep(0.005)
            try:
                for _ in range(16):
                    if not intents.poll():
                        break
                    last = intents.recv()
            except EOFError:
                # Keep the last timestamp so the lease deadline is actually tested.
                pass
            now = time.monotonic_ns()
            feedback = synthetic_feedback(now, (now - initial_ns) // 1_000_000)
            decision = guard.evaluate(now, feedback, Health(now, 0, True), last)
            if decision.enabled:
                enabled_count += 1
                if not ready:
                    results.send({"event": "ready"})
                    ready = True
            else:
                results.send({
                    "event": "stopped", "decision": asdict(decision), "stop_ns": now,
                    "intent_age_at_stop_ms": (now - last.generated_ns) / 1e6,
                    "enabled_memory_only_decisions": enabled_count,
                    "sink": "memory-only; no publisher or hardware device",
                })
                return
        raise TimeoutError("Supervisor exceeded its bounded test window")
    except Exception as error:
        results.send({"event": "error", "error": f"{type(error).__name__}: {error}"})
    finally:
        intents.close()
        results.close()


def process_failure_trial():
    context = mp.get_context("spawn")
    intent_read, intent_write = context.Pipe(duplex=False)
    result_read, result_write = context.Pipe(duplex=False)
    producer = context.Process(target=intent_producer, args=(intent_write,))
    supervisor = context.Process(target=simulated_supervisor, args=(intent_read, result_write))
    try:
        producer.start()
        supervisor.start()
        intent_read.close()
        intent_write.close()
        result_write.close()
        if not result_read.poll(5):
            raise TimeoutError("Supervisor did not start")
        startup = result_read.recv()
        if startup.get("event") != "ready":
            raise RuntimeError(f"Supervisor startup failed: {startup}")
        time.sleep(0.15)
        killed_ns = time.monotonic_ns()
        producer.kill()
        producer.join(1)
        if not result_read.poll(2):
            raise TimeoutError("No supervisor stop after producer SIGKILL")
        result = result_read.recv()
        supervisor.join(1)
        result.update(producer_exitcode=producer.exitcode, supervisor_exitcode=supervisor.exitcode,
                      fault="SIGKILL synthetic command producer")
        if result.get("event") == "stopped":
            result["stop_after_kill_ms"] = (result["stop_ns"] - killed_ns) / 1e6
        result["passed"] = (
            result.get("event") == "stopped"
            and result["decision"]["reason"] == "invalid_or_expired_intent"
            and not result["decision"]["enabled"]
            and producer.exitcode == -9 and supervisor.exitcode == 0
            and 0 <= result["stop_after_kill_ms"] <= 100
            and 40 < result["intent_age_at_stop_ms"] <= 60
        )
        return result
    finally:
        for process in (producer, supervisor):
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(2)
        for connection in (intent_read, intent_write, result_read, result_write):
            connection.close()


def sdk_frame_checks():
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    crc = CRC()
    cases = []
    for machine in (1, 2, 127):
        for slot in range(12):
            for enabled in (False, True):
                decision = Decision(enabled, "offline_fixture", slot, 0.01, machine, 0)
                command = build_sdk_frame(decision, unitree_hg_msg_dds__LowCmd_, crc)
                payload = command.serialize()
                restored = LowCmd_.deserialize(payload)
                # Pinned SDK diagnostic only: compare its Python and native implementations.
                words = crc._CRC__PackHGLowCmd(restored)
                checks = [
                    restored.crc == crc.Crc(restored) == crc._crc_py(words),
                    restored.mode_machine == machine,
                    restored.mode_pr == 0,
                    all(m.mode == int(enabled and i == slot)
                        for i, m in enumerate(restored.motor_cmd)),
                    all(m.mode == m.q == m.dq == m.tau == m.kp == m.kd == m.reserve == 0
                        for i, m in enumerate(restored.motor_cmd) if not enabled or i != slot),
                ]
                expected = restored.crc
                restored.motor_cmd[slot].q += 0.001
                checks.append(crc.Crc(restored) != expected)
                cases.append({
                    "enabled_fixture": enabled, "slot": slot, "mode_machine_fixture": machine,
                    "serialized_bytes": len(payload),
                    "serialized_sha256": hashlib.sha256(payload).hexdigest(),
                    "passed": all(checks),
                })
    return {
        "cases": cases, "passed": all(case["passed"] for case in cases),
        "crc_implementation": "official pinned SDK native and Python implementations",
        "crc_native_library": crc.crc_lib._name,
        "hardware_commands_sent": 0,
        "scope": "IDL roundtrip, slot initialization, CRC agreement and tamper detection only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-frames", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
        "architecture": platform.machine(), "python": platform.python_version(),
        "hardware_execution_enabled": False, "robot_commands_sent": 0,
        "dds_entities_created": 0, "motion_services_changed": False,
        "scope": "Offline command protocol and simulated-feedback independent-process tests",
        "limitations": [
            "No real feedback, motor output, service handoff or physical stop is exercised",
            "Process failure test has a memory-only sink, not a DDS/motor watchdog",
            "Bench gain/limit/timing values are provisional software test assumptions",
            "Hardware calibration, watchdog and stop remain unverified",
        ],
    }
    try:
        report["process_failure_trials"] = [process_failure_trial() for _ in range(3)]
        if args.sdk_frames:
            report["sdk_frames"] = sdk_frame_checks()
        report["passed"] = (
            all(trial["passed"] for trial in report["process_failure_trials"])
            and (not args.sdk_frames or report["sdk_frames"]["passed"])
        )
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sdk_frames"},
                     indent=2, allow_nan=False))
    if "sdk_frames" in report:
        print(json.dumps({"sdk_frame_cases": len(report["sdk_frames"]["cases"]),
                          "sdk_frames_passed": report["sdk_frames"]["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

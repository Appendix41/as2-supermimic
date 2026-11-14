"""Bounded hardware-status inspection; no actor, motor writer or RPC client."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .telemetry import network_preflight, record_message


LOW = "rt/lowstate"
SPORT = "rt/lf/sportmodestate"
BOARD = "rt/lf/mainboardstate"
BMS = "rt/lf/bmsstate"
TOPICS = (LOW, SPORT, BOARD, BMS)
BOARD_BITS = (
    "upperlevel_command_timeout", "underlying_feedback_timeout", "imu_timeout",
    "motor_feedback_timeout", "bms_timeout", "physical_remote_timeout",
    "battery_model_error", "softstart_error", "motor_error",
    "motor_overcurrent_lowlimit", "motor_undervoltage_highlimit",
    "motor_overcurrent_highlimit", "soft_emergency_stop_pressed", "serial_number_error",
    "upperlevel_machine_mismatch", "underlying_machine_mismatch", "usb_device_error",
    "joint_limit_error", "secondary_imu_timeout", "battery_undervoltage",
    "motor_undervoltage", "machine_category_mismatch", "special_joint_selftest_error",
    "secondary_bms_timeout",
)


def board_flags(value):
    if type(value) is not int or not 0 <= value < 2 ** 32:
        raise ValueError("Invalid mainboard status word")
    flags = [name for bit, name in enumerate(BOARD_BITS) if value & (1 << bit)]
    unknown = value & ~((1 << len(BOARD_BITS)) - 1)
    if unknown:
        flags.append(f"unknown_bits_0x{unknown:08x}")
    return flags


def status_message(topic, message, now_ns):
    if topic in (LOW, SPORT):
        row = record_message(topic, message, now_ns)
        if topic == LOW:
            row["motor_slots"] = row["motor_slots"][:12]
            for motor, raw in zip(row["motor_slots"], message.motor_state[:12]):
                motor.update(temperature_raw=list(raw.temperature),
                             voltage_raw=float(raw.vol), reserve_raw=list(raw.reserve))
        return row
    row = {
        "topic": topic, "received_monotonic_ns": now_ns,
        "dds_source_timestamp_ns": getattr(message.sample_info, "source_timestamp", None),
    }
    if topic == BOARD:
        row.update(
            fan_state_raw=list(message.fan_state), temperature_raw=list(message.temperature),
            value_raw=list(message.value), state_raw=list(message.state),
            documented_state_flags=board_flags(int(message.state[0])),
        )
    elif topic == BMS:
        row.update(soc=int(message.soc), soh=int(message.soh), fn=int(message.fn),
                   voltage_mv_raw=list(message.bmsvoltage), current_ma=int(message.current),
                   temperature_raw=list(message.temperature), state_raw=list(message.bmsstate))
    else:
        raise ValueError("Unsupported status topic")
    return row


def summarize(rows):
    grouped = {name: [row for row in rows if row["topic"] == name] for name in TOPICS}
    low, sport, board, bms = (grouped[name] for name in TOPICS)
    faults = sorted({flag for row in board for flag in row["documented_state_flags"]})
    result = {
        "hardware_execution_enabled": False,
        "motor_takeover_allowed": False,
        "missing_topics": [name for name, values in grouped.items() if not values],
        "mainboard_flags_observed": faults,
        "soft_estop_asserted_observed": "soft_emergency_stop_pressed" in faults,
        "soft_estop_effectiveness_verified": False,
        "independent_watchdog_verified": False,
        "joint_imu_calibration_verified": False,
        "sport_modes": sorted({row["mode"] for row in sport}),
        "sport_errors": sorted({row["error_code"] for row in sport}),
        "mode_machine_values": sorted({row["mode_machine"] for row in low}),
        "mode_pr_values": sorted({row["mode_pr"] for row in low}),
        "battery_soc_min_max": [min(r["soc"] for r in bms), max(r["soc"] for r in bms)]
        if bms else None,
        "battery_state_words_observed": sorted({tuple(r["state_raw"]) for r in bms}),
        "mainboard_state_words_observed": sorted({tuple(r["state_raw"]) for r in board}),
        "remote_raw_payload_counts": dict(Counter(r["wireless_remote_hex"] for r in low)),
        "limits": [
            "A clear status word is not proof of calibrated hardware or emergency-stop effectiveness",
            "State flags use the archived official AS2 interface, not verified firmware behavior",
            "No low-level controller or watchdog is exercised by this reader",
            "Saved rows are decimated; DDS receipt age is not sensor age",
            "Motor/BMS temperature, status and reserve fields remain raw",
        ],
    }
    if low:
        motors = [motor for row in low for motor in row["motor_slots"]]
        numbers = [float(value) for m in motors for value in
                   (m["q"], m["dq"], m["tau_est"], m["voltage_raw"], *m["temperature_raw"])]
        result["motor_numeric_fields_finite"] = all(math.isfinite(v) for v in numbers)
        result["motor_status_values"] = sorted({m["status"] for m in motors})
        result["motor_mode_values"] = sorted({m["mode"] for m in motors})
        result["motor_warning_values"] = sorted({m["reserve_raw"][3] for m in motors})
        temperatures = [v for m in motors for v in m["temperature_raw"]]
        result["motor_temperature_raw_min_max"] = [min(temperatures), max(temperatures)]
        voltages = [m["voltage_raw"] for m in motors]
        result["motor_voltage_raw_min_max"] = [min(voltages), max(voltages)]
    return result


def collect(interface, duration, output, report):
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, LowState_, MainBoardState_

    xml = (
        '<CycloneDDS><Domain Id="0"><General><Interfaces>'
        f'<NetworkInterface name="{interface}"/></Interfaces>'
        '<AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
    )
    domain = Domain(0, xml)
    participant = DomainParticipant(0)
    qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
    types = (LowState_, SportModeState_, MainBoardState_, BmsState_)
    readers = [(name, DataReader(participant, Topic(participant, name, kind), qos))
               for name, kind in zip(TOPICS, types)]
    stats = {name: {"received": 0, "saved": 0, "first_ns": None, "last_ns": None,
                    "max_receipt_gap_ms": 0.0} for name in TOPICS}
    last_save = {name: 0 for name in TOPICS}
    rows = []
    start = time.monotonic()
    report["topic_statistics"] = stats
    try:
        while time.monotonic() - start < duration:
            for name, reader in readers:
                for message in reader.take(1):
                    if not message.sample_info.valid_data:
                        continue
                    now = time.monotonic_ns()
                    stat = stats[name]
                    if stat["last_ns"] is not None:
                        stat["max_receipt_gap_ms"] = max(
                            stat["max_receipt_gap_ms"], (now - stat["last_ns"]) / 1e6)
                    else:
                        stat["first_ns"] = now
                    stat["last_ns"] = now
                    stat["received"] += 1
                    # Board/BMS messages are low-rate; retain every one for fault review.
                    if name != LOW or now - last_save[name] >= 20_000_000:
                        rows.append(status_message(name, message, now))
                        stat["saved"] += 1
                        last_save[name] = now
            time.sleep(0.001)
    finally:
        report["elapsed_s"] = time.monotonic() - start
        report["inspection"] = summarize(rows)
        report["status"] = "read_only_complete" if not report["inspection"]["missing_topics"] else "missing_feedback"
        with (output / "status.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        del readers, participant, domain
    report["sampling_note"] = "KeepLast(1) latest-sample diagnostic; not lossless or a command-loop test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="end0")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 30:
        parser.error("duration must be finite and in (0,30]")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "status": "preflight",
        "robot_commands_sent": 0, "robot_command_writers_created": 0,
        "rpc_clients_created": 0, "hardware_execution_enabled": False,
        "scope": "Hardware status read only; no motor commands or motion-service changes",
    }
    try:
        report["network"] = network_preflight(args.interface, "192.168.123.0/24")
        if report["network"]["passed"]:
            collect(args.interface, args.duration, args.output, report)
        else:
            report["status"] = "blocked_network"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        with (args.output / "summary.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "read_only_complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

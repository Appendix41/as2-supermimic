"""Read robot telemetry over DDS without publishing commands."""

import argparse
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import time


TOPICS = ("rt/lowstate", "rt/sportmodestate", "rt/lf/lowstate", "rt/lf/sportmodestate")


def network_preflight(interface, network):
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
        raise ValueError("Invalid interface name")
    subnet = ipaddress.ip_network(network)
    carrier = (Path("/sys/class/net") / interface / "carrier").read_text().strip()
    completed = subprocess.run(
        ["ip", "-j", "address", "show", "dev", interface],
        capture_output=True, text=True, check=True, timeout=5,
    )
    records = json.loads(completed.stdout)
    addresses = [
        entry["local"] for record in records for entry in record.get("addr_info", [])
        if entry["family"] == "inet"
    ]
    usable = carrier == "1" and any(ipaddress.ip_address(a) in subnet for a in addresses)
    return {"interface": interface, "carrier": carrier, "ipv4": addresses,
            "required_network": network, "passed": usable}


def record_message(topic, message, received_ns):
    row = {
        "topic": topic, "received_monotonic_ns": received_ns,
        "dds_source_timestamp_ns": getattr(message.sample_info, "source_timestamp", None),
        "clock_note": "Receipt and DDS source clocks are not assumed synchronized",
    }
    if topic.endswith("/lowstate"):
        row.update({
            "tick": int(message.tick), "mode_pr": int(message.mode_pr),
            "mode_machine": int(message.mode_machine), "version": list(message.version),
            "quaternion_raw": list(message.imu_state.quaternion),
            "gyroscope_raw": list(message.imu_state.gyroscope),
            "accelerometer_raw": list(message.imu_state.accelerometer),
            "rpy_raw": list(message.imu_state.rpy),
            "motor_slots": [
                {"slot": i, "mode": int(m.mode), "q": float(m.q), "dq": float(m.dq),
                 "tau_est": float(m.tau_est), "status": int(m.motorstate)}
                for i, m in enumerate(message.motor_state)
            ],
            "mapping_note": "Raw protocol slot numbers, NOT verified AS2 policy joint order",
            "wireless_remote_hex": bytes(message.wireless_remote).hex(),
        })
    else:
        row.update({
            "mode": int(message.mode), "error_code": int(message.error_code),
            "position_raw": list(message.position), "velocity_raw": list(message.velocity),
            "rpy_raw": list(message.imu_state.rpy),
            "stamp": {"sec": int(message.stamp.sec), "nanosec": int(message.stamp.nanosec)},
            "foot_force_raw": list(message.foot_force),
            "availability_note": "Observed in present mode only, not a low-level-mode guarantee",
        })
    return row


def collect(interface, domain_id, duration, output, report):
    # Check the network before loading the DDS readers.
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    xml = (
        f'<CycloneDDS><Domain Id="{domain_id}"><General><Interfaces>'
        f'<NetworkInterface name="{interface}"/></Interfaces>'
        '<AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
    )
    domain = Domain(domain_id, xml)
    participant = DomainParticipant(domain_id)
    qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(8))
    readers = [
        (name, DataReader(participant, Topic(
            participant, name, LowState_ if name.endswith("/lowstate") else SportModeState_), qos))
        for name in TOPICS
    ]
    counts = {name: {"received": 0, "saved": 0, "first_ns": None, "last_ns": None,
                     "last_save_ns": 0, "max_receipt_gap_ms": 0.0} for name in TOPICS}
    report.update(status="collecting", topics=counts)
    start = time.monotonic()
    with (output / "telemetry.jsonl").open("x") as stream:
        while time.monotonic() - start < duration:
            for name, reader in readers:
                for message in reader.take(16):
                    if not message.sample_info.valid_data:
                        continue
                    now = time.monotonic_ns()
                    stat = counts[name]
                    if stat["last_ns"] is not None:
                        stat["max_receipt_gap_ms"] = max(
                            stat["max_receipt_gap_ms"], (now - stat["last_ns"]) / 1e6)
                    stat["first_ns"] = now if stat["first_ns"] is None else stat["first_ns"]
                    stat["last_ns"] = now
                    stat["received"] += 1
                    if now - stat["last_save_ns"] >= 20_000_000:
                        row = record_message(name, message, now)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stat["saved"] += 1
                        stat["last_save_ns"] = now
            time.sleep(0.002)
    for stat in counts.values():
        span = (stat["last_ns"] - stat["first_ns"]) if stat["received"] > 1 else 0
        stat["observed_receive_hz"] = (stat["received"] - 1) * 1e9 / span if span else 0
    report["elapsed_s"] = time.monotonic() - start
    report["status"] = "received" if counts["rt/lowstate"]["received"] else "no_high_rate_lowstate"
    report["rate_note"] = "Best-effort receipt counts, not certified publisher rates or lossless capture"
    # Keep subscriptions alive for the capture session.
    del readers, participant, domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="end0")
    parser.add_argument("--robot-network", default="192.168.123.0/24")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.duration <= 30 or not 0 <= args.domain <= 230:
        parser.error("duration must be in (0,30] and domain in [0,230]")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight", "robot_commands_sent": 0, "robot_command_writers_created": 0,
        "scope": "Read-only telemetry; no actuation, no mode switching, no firmware queries",
    }
    try:
        report["network"] = network_preflight(args.interface, args.robot_network)
        if not report["network"]["passed"]:
            report["status"] = "blocked_network"
        else:
            collect(args.interface, args.domain, args.duration, args.output, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "received":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

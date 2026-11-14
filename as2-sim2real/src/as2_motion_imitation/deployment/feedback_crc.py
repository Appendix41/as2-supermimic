"""Bounded real LowState CRC inspection. Explicit reader only, no commands."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .telemetry import network_preflight


def check_crc(message, crc):
    received = message.crc
    if type(received) is not int or not 0 <= received < 2 ** 32:
        raise ValueError("Malformed LowState CRC field")
    computed = crc.Crc(message)
    if type(computed) is not int or not 0 <= computed < 2 ** 32:
        raise ValueError("Malformed computed CRC")
    return {"received_crc": received, "computed_crc": computed, "valid": received == computed}


def collect(interface, duration, report):
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC

    xml = (
        '<CycloneDDS><Domain Id="0"><General><Interfaces>'
        f'<NetworkInterface name="{interface}"/></Interfaces>'
        '<AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
    )
    domain = Domain(0, xml)
    participant = DomainParticipant(0)
    reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_),
                        Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1)))
    crc = CRC()
    samples = []
    start = time.monotonic()
    try:
        while time.monotonic() - start < duration:
            for message in reader.take(1):
                if not message.sample_info.valid_data:
                    continue
                received_ns = time.monotonic_ns()
                result = check_crc(message, crc)
                result.update(tick=int(message.tick), mode_machine=int(message.mode_machine),
                              received_monotonic_ns=received_ns)
                samples.append(result)
            time.sleep(0.005)
    finally:
        report["elapsed_s"] = time.monotonic() - start
        report["samples"] = samples
        report["checked"] = len(samples)
        report["mismatches"] = sum(not row["valid"] for row in samples)
        report["passed"] = bool(samples) and not report["mismatches"]
        del reader, participant, domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="end0")
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 10:
        parser.error("duration must be finite and in (0,10]")
    if args.output.exists():
        parser.error("output already exists")
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual LowState integrity check, not a control or calibration test",
        "robot_commands_sent": 0, "robot_command_writers_created": 0,
        "hardware_execution_enabled": False,
        "passed": False,
    }
    try:
        report["network"] = network_preflight(args.interface, "192.168.123.0/24")
        if report["network"]["passed"]:
            collect(args.interface, args.duration, report)
        else:
            report["error"] = "Robot network unavailable"
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"},
                     indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Bounded remote-stop signal observation; no command writer or service RPC."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .feedback_crc import check_crc
from .readiness import BOARD, LOW, SPORT, board_flags
from .telemetry import network_preflight


BUTTONS = (
    "R1", "L1", "START", "SELECT", "R2", "L2", "reserved6", "reserved7",
    "A", "B", "X", "Y", "UP", "RIGHT", "DOWN", "LEFT",
)


def decode_remote(payload):
    raw = bytes(payload)
    if len(raw) != 40:
        raise ValueError("Expected 40-byte LowState wireless_remote")
    mask = int.from_bytes(raw[2:4], "little")
    return {
        "wireless_remote_hex": raw.hex(), "button_mask": mask,
        "buttons": [name for bit, name in enumerate(BUTTONS) if mask & (1 << bit)],
        "header_matches_recorded_fixture": raw[:2] == b"\x55\x51",
        "l2_b": bool(mask & (1 << 5) and mask & (1 << 9)),
    }


class SignalLog:
    """Record observed transitions, never infer physical stopping from a flag."""

    def __init__(self, stream):
        self.stream = stream
        self.counts = Counter()
        self.previous = {}
        self.last_saved = {}
        self.last_received = {}
        self.max_gap_ms = {}
        self.last_tick = None
        self.saw_combo = False
        self.release_after_combo = False
        self.saw_board_stop = False
        self.board_assertions = 0
        self.board_clears = 0
        self.board_stopped = None
        self.modes = set()
        self.ready_announced = False

    def event(self, topic, signature, row, now_ns, *, always=False):
        changed = topic not in self.previous or self.previous[topic] != signature
        if always or changed or now_ns - self.last_saved.get(topic, 0) >= 1_000_000_000:
            row.update(topic=topic, received_monotonic_ns=now_ns,
                       time_utc=datetime.now(timezone.utc).isoformat(),
                       changed=changed)
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.stream.flush()
            self.last_saved[topic] = now_ns
            if changed:
                print(json.dumps(row, allow_nan=False), flush=True)
        self.previous[topic] = signature

    def received(self, topic, now_ns):
        self.counts[topic] += 1
        if topic in self.last_received:
            gap = (now_ns - self.last_received[topic]) / 1e6
            self.max_gap_ms[topic] = max(self.max_gap_ms.get(topic, 0), gap)
        self.last_received[topic] = now_ns

    def low(self, message, now_ns, integrity):
        self.received(LOW, now_ns)
        tick = int(message.tick)
        if not integrity["valid"]:
            self.counts["crc_mismatches"] += 1
            self.event(LOW, ("crc_invalid",), {"crc": integrity, "tick": tick},
                       now_ns, always=True)
            return
        self.counts["crc_valid"] += 1
        remote = decode_remote(message.wireless_remote)
        # A repeated or out-of-order sample cannot establish a button release.
        forward = (self.last_tick is None
                   or 0 < (tick - self.last_tick) % 2 ** 32 < 2 ** 31)
        if not forward:
            self.counts["nonforward_ticks"] += 1
        else:
            self.last_tick = tick
        if forward and remote["header_matches_recorded_fixture"]:
            self.saw_combo |= remote["l2_b"]
            if self.saw_combo and not remote["button_mask"] & ((1 << 5) | (1 << 9)):
                self.release_after_combo = True
        elif not remote["header_matches_recorded_fixture"]:
            self.counts["unrecognized_remote_headers"] += 1
        row = dict(remote, tick=tick, crc=integrity, forward_tick=forward,
                   mode_machine=int(message.mode_machine), mode_pr=int(message.mode_pr),
                   dds_source_timestamp_ns=getattr(message.sample_info, "source_timestamp", None))
        signature = (remote["button_mask"], remote["header_matches_recorded_fixture"],
                     row["mode_machine"], row["mode_pr"], forward)
        self.event(LOW, signature, row, now_ns)

    def board(self, message, now_ns):
        self.received(BOARD, now_ns)
        words = [int(value) for value in message.state]
        flags = board_flags(words[0])
        stopped = bool(words[0] & (1 << 12))
        self.saw_board_stop |= stopped
        if self.board_stopped is False and stopped:
            self.board_assertions += 1
        if self.board_stopped is True and not stopped:
            self.board_clears += 1
        self.board_stopped = stopped
        self.event(BOARD, tuple(words), {
            "state_raw": words, "documented_state_flags": flags,
            "dds_source_timestamp_ns": getattr(message.sample_info, "source_timestamp", None),
        }, now_ns)

    def sport(self, message, now_ns):
        self.received(SPORT, now_ns)
        mode, error = int(message.mode), int(message.error_code)
        self.modes.add(mode)
        self.event(SPORT, (mode, error), {"mode": mode, "error_code": error}, now_ns)

    def summary(self):
        return {
            "counts": dict(self.counts),
            "max_receipt_gap_ms": self.max_gap_ms,
            "required_feedback_received": bool(self.counts["crc_valid"] and self.counts[BOARD]),
            "l2_b_observed_in_crc_valid_feedback": self.saw_combo,
            "both_buttons_released_after_combo_observed": self.release_after_combo,
            "mainboard_soft_estop_asserted_observed": self.saw_board_stop,
            "mainboard_soft_estop_rising_edges": self.board_assertions,
            "mainboard_soft_estop_falling_edges": self.board_clears,
            "sport_modes_observed": sorted(self.modes),
            "physical_stop_effectiveness_verified": False,
            "stop_after_service_release_verified": False,
            "hardware_execution_enabled": False,
        }


def collect(interface, duration, output, report):
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, MainBoardState_
    from unitree_sdk2py.utils.crc import CRC

    xml = (
        '<CycloneDDS><Domain Id="0"><General><Interfaces>'
        f'<NetworkInterface name="{interface}"/></Interfaces>'
        '<AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
    )
    domain = Domain(0, xml)
    participant = DomainParticipant(0)
    readers = [
        (name, DataReader(participant, Topic(participant, name, kind),
                          Qos(Policy.Reliability.BestEffort,
                              Policy.History.KeepLast(1 if name == LOW else 32))))
        for name, kind in ((LOW, LowState_), (BOARD, MainBoardState_), (SPORT, SportModeState_))
    ]
    crc = CRC()
    start = time.monotonic()
    with (output / "events.jsonl").open("x") as stream:
        log = SignalLog(stream)
        try:
            while time.monotonic() - start < duration:
                for topic, reader in readers:
                    for message in reader.take(1 if topic == LOW else 32):
                        if not message.sample_info.valid_data:
                            continue
                        now = time.monotonic_ns()
                        if topic == LOW:
                            log.low(message, now, check_crc(message, crc))
                        elif topic == BOARD:
                            log.board(message, now)
                        else:
                            log.sport(message, now)
                if not log.ready_announced and log.counts["crc_valid"] and log.counts[BOARD]:
                    log.ready_announced = True
                    print("LISTENER_READY: read-only LowState and mainboard feedback received",
                          flush=True)
                time.sleep(0.001)
        finally:
            report["elapsed_s"] = time.monotonic() - start
            report["observation"] = log.summary()
            del readers, participant, domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="end0")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 120:
        parser.error("duration must be finite and in (0,120]")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "status": "preflight",
        "robot_commands_sent": 0, "robot_command_writers_created": 0, "rpc_clients_created": 0,
        "hardware_execution_enabled": False,
        "scope": "Remote signal observation only; vendor services unchanged",
        "limitations": [
            "Button decoding follows the pinned SDK; header matches only a recorded fixture",
            "Best-effort latest-sample LowState capture can miss brief edges",
            "Mainboard flags and button events are independent observations, not causality proof",
            "Receipt and DDS source clocks are not assumed synchronized",
            "No motor output, low-level handoff or physical stop is tested",
        ],
    }
    try:
        report["network"] = network_preflight(args.interface, "192.168.123.0/24")
        if report["network"]["passed"]:
            collect(args.interface, args.duration, args.output, report)
            observation = report["observation"]
            report["status"] = (
                "missing_feedback" if not observation["required_feedback_received"]
                else "feedback_crc_error" if observation["counts"].get("crc_mismatches")
                else "observation_complete"
            )
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
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if report["status"] != "observation_complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

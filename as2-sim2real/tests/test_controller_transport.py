"""SDK adapter tested with injected SDK objects, no network initialization."""

import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest

from as2_motion_imitation.deployment import controller_transport as transport
from as2_motion_imitation.deployment.controller import ControllerConfig, ControlFault, make_command


ROOT = Path(__file__).resolve().parents[1]


def adapter():
    result = transport.SdkTransport.__new__(transport.SdkTransport)
    result._lock = Lock()
    result._fault = result._low = result._last_tick = None
    result._crc_valid = False
    result._board_ns, result._board = 0, ()
    result._watch = False
    result._expected_machine = 1
    result._owner_ns = result._owner_thread = None
    result._closing = Event()
    result.crc = SimpleNamespace(Crc=lambda _: 0)
    result.readers, result.publisher = [], None
    return result


def message(mask=0, tick=1):
    imu = SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0],
                          accelerometer=[0, 0, 0], rpy=[0, 0, 0])
    motors = [SimpleNamespace(mode=1, q=0.0, dq=0.0, tau_est=0.0, motorstate=0,
                              temperature=[30, 30], vol=42.0, reserve=[0] * 4)
              for _ in range(35)]
    return SimpleNamespace(crc=0, tick=tick, mode_machine=1, mode_pr=0, version=[0, 0],
                           imu_state=imu, motor_state=motors,
                           sample_info=SimpleNamespace(source_timestamp=123),
                           wireless_remote=b"\x55\x51" + mask.to_bytes(2, "little") + bytes(36))


def test_stop_between_supervisor_polls_remains_latched():
    io = adapter()
    io.enable_watch()
    io._on_low(message(544, 10))
    io._on_low(message(0, 11))
    assert io.snapshot().transport_fault == "remote_stop"
    io._on_low(message(4, 12))
    assert io.snapshot().transport_fault == "remote_stop"


@pytest.mark.parametrize("fault", ["crc", "mode", "tick", "motor", "malformed", "header"])
def test_callback_fault_cannot_be_overwritten_by_new_good_sample(fault):
    io = adapter()
    io.enable_watch()
    io._on_low(message(0, 100))
    bad = message(0, 101)
    if fault == "crc":
        bad.crc = 123
    elif fault == "mode":
        bad.mode_machine = 2
    elif fault == "tick":
        bad.tick = 99
    elif fault == "motor":
        bad.motor_state[0].motorstate = 3
    elif fault == "header":
        bad.wireless_remote = bytes(40)
    else:
        bad.wireless_remote = bytes(1)
    io._on_low(bad)
    io._on_low(message(0, 102))
    assert io.snapshot().transport_fault is not None


def test_board_flag_is_latched_independently_of_remote():
    io = adapter()
    io._on_board(SimpleNamespace(state=[4096, 0, 0, 0, 0, 0]))
    io._on_board(SimpleNamespace(state=[0] * 6))
    io._on_low(message())
    assert io.snapshot().transport_fault == "transport_mainboard_fault"


def test_latched_stop_prevents_position_write_but_allows_passive(monkeypatch):
    config = ControllerConfig(json.loads((ROOT / "configs/controller-synthetic.json").read_text()))
    io = adapter()
    io._fault = "remote_stop"
    io.factory = lambda: None
    frames = []
    io.publisher = SimpleNamespace(Write=lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(transport, "encode_command", lambda command, *_: command)
    command = make_command(np.zeros(12), config.kp, config.kd, config)
    with pytest.raises(ControlFault, match="remote_stop"):
        io.publish(command)
    passive = make_command(np.zeros(12), config.kp, config.kd, config, "damping")
    assert io.publish(passive)
    assert frames == [passive]


def test_sdk_constructor_checks_hardware_gate_before_importing_sdk():
    config = ControllerConfig(json.loads((ROOT / "configs/controller-synthetic.json").read_text()))
    with pytest.raises(ControlFault, match="hardware profile"):
        transport.SdkTransport("end0", config, SimpleNamespace(manifest={}))


def test_owner_monitor_timeout_is_latched(monkeypatch):
    io = adapter()
    io._on_low(message())
    io._owner_ns = 1
    monkeypatch.setattr(transport.time, "monotonic_ns", lambda: 400_000_001)
    assert io.snapshot().transport_fault == "motion_owner_monitor_timeout"


def test_owner_monitor_detects_vendor_reactivation():
    io = adapter()
    io.check_mode = lambda: (0, {"name": "ai"})
    io.monitor_owner()
    io._owner_thread.join(0.25)
    io.close()
    assert io._fault == "motion_owner_lost"

"""Pure status inspection does not grant permission for physical control."""

from types import SimpleNamespace

import pytest

from as2_motion_imitation.deployment import readiness


def test_mainboard_flags_and_unknown_bits_are_not_ignored():
    assert readiness.board_flags(0) == []
    assert readiness.board_flags((1 << 12) | (1 << 5)) == [
        "physical_remote_timeout", "soft_emergency_stop_pressed"]
    assert readiness.board_flags(1 << 31) == ["unknown_bits_0x80000000"]
    with pytest.raises(ValueError):
        readiness.board_flags(-1)


def test_no_packets_cannot_be_readiness_pass():
    result = readiness.summarize([])
    assert result["missing_topics"] == list(readiness.TOPICS)
    assert not result["hardware_execution_enabled"]
    assert not result["motor_takeover_allowed"]


def test_asserted_estop_not_confused_with_verified_stop():
    row = {"topic": readiness.BOARD, "state_raw": [4096, 0, 0, 0, 0, 0],
           "documented_state_flags": ["soft_emergency_stop_pressed"]}
    result = readiness.summarize([row])
    assert result["soft_estop_asserted_observed"]
    assert not result["soft_estop_effectiveness_verified"]
    assert not result["motor_takeover_allowed"]


def test_mainboard_and_battery_are_recorded_without_rpc():
    info = SimpleNamespace(source_timestamp=1)
    board = SimpleNamespace(sample_info=info, fan_state=[0] * 6, temperature=[25] * 6,
                            value=[58, 58, 0, 0, 0, 0], state=[32, 0, 0, 0, 0, 0])
    row = readiness.status_message(readiness.BOARD, board, 123)
    assert row["documented_state_flags"] == ["physical_remote_timeout"]
    bms = SimpleNamespace(sample_info=info, soc=75, soh=99, fn=0, bmsvoltage=[58000] * 3,
                          current=-500, temperature=[25] * 12, bmsstate=[0] * 5)
    battery = readiness.status_message(readiness.BMS, bms, 124)
    assert battery["soc"] == 75
    assert battery["current_ma"] == -500


def test_network_failure_never_initializes_sdk(monkeypatch, tmp_path):
    import json
    import sys
    output = tmp_path / "capture"
    monkeypatch.setattr(sys, "argv", ["readiness", "--output", str(output)])
    monkeypatch.setattr(readiness, "network_preflight", lambda *a: {"passed": False})
    monkeypatch.setattr(readiness, "collect", lambda *a: pytest.fail("must not start DDS"))
    with pytest.raises(SystemExit) as error:
        readiness.main()
    assert error.value.code == 2
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "blocked_network"
    assert report["robot_commands_sent"] == report["rpc_clients_created"] == 0

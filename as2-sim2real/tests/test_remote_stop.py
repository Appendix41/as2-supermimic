"""Remote observations cannot grant hardware execution or stop verification."""

import io
import json
import sys
from types import SimpleNamespace

import pytest

from as2_motion_imitation.deployment import remote_stop as rs


def payload(mask=0, header=b"\x55\x51"):
    return header + mask.to_bytes(2, "little") + bytes(36)


def low(mask=0, tick=1, header=b"\x55\x51"):
    return SimpleNamespace(wireless_remote=payload(mask, header), tick=tick,
                           mode_machine=1, mode_pr=0,
                           sample_info=SimpleNamespace(source_timestamp=123))


GOOD = {"valid": True, "received_crc": 4, "computed_crc": 4}
BAD = {"valid": False, "received_crc": 4, "computed_crc": 5}
COMBO = (1 << 5) | (1 << 9)


@pytest.mark.parametrize("bit,name", list(enumerate(rs.BUTTONS)))
def test_each_button_matches_pinned_sdk_bit_order(bit, name):
    result = rs.decode_remote(payload(1 << bit))
    assert result["buttons"] == [name]
    assert not result["l2_b"]


def test_combo_requires_both_bits_and_correct_length():
    assert rs.decode_remote(payload(COMBO))["l2_b"]
    with pytest.raises(ValueError):
        rs.decode_remote(bytes(39))


def test_valid_press_release_is_only_signal_evidence():
    stream = io.StringIO()
    log = rs.SignalLog(stream)
    for tick, mask in enumerate((0, 1 << 5, COMBO, 1 << 5, 0), 1):
        log.low(low(mask, tick), tick * 1_000_000, GOOD)
    result = log.summary()
    assert result["l2_b_observed_in_crc_valid_feedback"]
    assert result["both_buttons_released_after_combo_observed"]
    assert not result["required_feedback_received"]
    assert not result["physical_stop_effectiveness_verified"]
    assert not result["stop_after_service_release_verified"]
    assert not result["hardware_execution_enabled"]
    assert len(stream.getvalue().splitlines()) == 5


def test_bad_crc_or_unknown_header_cannot_establish_press():
    log = rs.SignalLog(io.StringIO())
    log.low(low(COMBO), 1, BAD)
    log.low(low(COMBO, 2, b"\x00\x00"), 2, GOOD)
    assert not log.summary()["l2_b_observed_in_crc_valid_feedback"]
    assert log.counts["crc_mismatches"] == 1
    assert log.counts["unrecognized_remote_headers"] == 1


def test_nonforward_tick_cannot_establish_press_or_release():
    log = rs.SignalLog(io.StringIO())
    log.low(low(0, 10), 1, GOOD)
    log.low(low(COMBO, 9), 2, GOOD)
    assert not log.saw_combo
    log.low(low(COMBO, 11), 3, GOOD)
    log.low(low(0, 11), 4, GOOD)
    assert not log.release_after_combo
    log.low(low(0, 12), 5, GOOD)
    assert log.release_after_combo


def test_tick_wrap_accepts_forward_sample():
    log = rs.SignalLog(io.StringIO())
    log.low(low(COMBO, 2 ** 32 - 1), 1, GOOD)
    log.low(low(0, 0), 2, GOOD)
    assert log.release_after_combo
    assert not log.counts["nonforward_ticks"]


def test_board_edges_and_initial_assertion_are_distinct():
    log = rs.SignalLog(io.StringIO())
    message = SimpleNamespace(state=[4096, 0], sample_info=SimpleNamespace())
    log.board(message, 1)
    assert log.saw_board_stop and log.board_assertions == 0
    message.state[0] = 0
    log.board(message, 2)
    message.state[0] = 4096
    log.board(message, 3)
    result = log.summary()
    assert result["mainboard_soft_estop_rising_edges"] == 1
    assert result["mainboard_soft_estop_falling_edges"] == 1
    assert not result["physical_stop_effectiveness_verified"]


def test_unchanged_buttons_are_periodic_but_brief_changes_are_not_decimated():
    stream = io.StringIO()
    log = rs.SignalLog(stream)
    for tick, mask in enumerate((0, 0, COMBO, 0), 1):
        log.low(low(mask, tick), tick * 1_000_000, GOOD)
    assert len(stream.getvalue().splitlines()) == 3
    log.low(low(0, 5), 1_100_000_000, GOOD)
    assert len(stream.getvalue().splitlines()) == 4


def test_network_failure_never_initializes_sdk(monkeypatch, tmp_path):
    output = tmp_path / "capture"
    monkeypatch.setattr(sys, "argv", ["remote_stop", "--output", str(output)])
    monkeypatch.setattr(rs, "network_preflight", lambda *a: {"passed": False})
    monkeypatch.setattr(rs, "collect", lambda *a: pytest.fail("DDS must not start"))
    with pytest.raises(SystemExit) as error:
        rs.main()
    assert error.value.code == 2
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "blocked_network"
    assert report["robot_commands_sent"] == report["rpc_clients_created"] == 0


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "-1", "121"])
def test_unbounded_duration_rejected(monkeypatch, tmp_path, duration):
    output = tmp_path / "capture"
    monkeypatch.setattr(sys, "argv", ["remote_stop", "--output", str(output),
                                     "--duration", duration])
    with pytest.raises(SystemExit):
        rs.main()
    assert not output.exists()

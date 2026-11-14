"""Integrity helpers reject mismatches; zero CRC does not mean unchecked/pass."""

from types import SimpleNamespace

import pytest

from as2_motion_imitation.deployment.feedback_crc import check_crc


@pytest.mark.parametrize("received,computed,expected", [
    (123, 123, True), (123, 124, False), (0, 124, False), (0, 0, True),
])
def test_crc_comparison(received, computed, expected):
    crc = SimpleNamespace(Crc=lambda message: computed)
    assert check_crc(SimpleNamespace(crc=received), crc)["valid"] is expected


@pytest.mark.parametrize("bad", [-1, 2 ** 32, True, float("nan"), None])
def test_invalid_crc_field(bad):
    with pytest.raises(ValueError):
        check_crc(SimpleNamespace(crc=bad), SimpleNamespace(Crc=lambda message: 0))

from pathlib import Path

import pytest
from mypy import api as mypy_api

from mdx_assessment.usecases.mdx.domain import (
    Bps,
    Percent,
    bps_to_percent,
    change_bps,
    percent_to_bps,
    relative_change_pct,
)


def test_adb_worked_example_is_exactly_three_bps() -> None:
    # JPY 3M 0.04% -> 0.07%: "+75%" but +3 bps (ADB's near-zero example, scenario S2).
    assert change_bps(Percent(0.04), Percent(0.07)) == 3.0
    assert relative_change_pct(0.04, 0.07) == 75.0


@pytest.mark.parametrize(("pct", "bps"), [(0.04, 4.0), (1.0, 100.0), (-0.25, -25.0), (0.0, 0.0)])
def test_percent_bps_conversion_both_ways(pct: float, bps: float) -> None:
    assert percent_to_bps(Percent(pct)) == bps
    assert bps_to_percent(Bps(bps)) == pct


def test_genuine_move_example() -> None:
    # GBP 2Y +18 bps (the planted genuine move).
    assert change_bps(Percent(4.12), Percent(4.30)) == 18.0


def test_relative_change_undefined_from_zero() -> None:
    assert relative_change_pct(0.0, 0.05) is None


def test_relative_change_uses_magnitude_of_previous() -> None:
    assert relative_change_pct(-0.10, -0.05) == 50.0


def test_mixing_units_is_a_type_error(tmp_path: Path) -> None:
    """The point of Percent/Bps: mypy rejects passing one unit where the other belongs."""
    snippet = tmp_path / "misuse.py"
    snippet.write_text(
        "from mdx_assessment.usecases.mdx.domain.units import Percent, change_bps\n"
        "move = change_bps(Percent(0.04), Percent(0.07))\n"
        "change_bps(move, Percent(0.07))\n"  # Bps given where Percent is expected
        "change_bps(0.04, 0.07)\n"  # bare floats: unit not stated
    )
    stdout, _, status = mypy_api.run(
        ["--strict", "--no-incremental", "--cache-dir=/dev/null", str(snippet)]
    )
    assert status == 1
    assert stdout.count('incompatible type "Bps"; expected "Percent"') == 1
    assert stdout.count('incompatible type "float"; expected "Percent"') == 2

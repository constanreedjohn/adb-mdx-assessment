"""Type-safe rate units: percent vs basis points.

`Percent` and `Bps` are distinct types for mypy, so a value in one unit can't be
passed where the other is expected, and a bare float has to be labelled first:

    change = change_bps(Percent(0.04), Percent(0.07))   # Bps(3.0)
    percent_to_bps(change)                              # mypy error: Bps is not Percent

Results are rounded to `PRECISION_DP` decimal places to remove binary floating-point
noise (0.07 - 0.04 = 0.030000000000000006), so 3 bps is exactly 3.0.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NewType

Percent = NewType("Percent", float)
Bps = NewType("Bps", float)

BPS_PER_PERCENT = 100
PRECISION_DP = 10


class Unit(StrEnum):
    """Units a market-data value can be quoted or measured in (plan: native units)."""

    PERCENT = "percent"
    BPS = "bps"
    PIPS = "pips"
    VOL_POINTS = "vol_points"
    PRICE = "price"


def percent_to_bps(value: Percent) -> Bps:
    return Bps(round(value * BPS_PER_PERCENT, PRECISION_DP))


def bps_to_percent(value: Bps) -> Percent:
    return Percent(round(value / BPS_PER_PERCENT, PRECISION_DP))


def change_bps(previous: Percent, current: Percent) -> Bps:
    """Absolute change of a rate quoted in percent, in basis points."""
    return percent_to_bps(Percent(current - previous))


def relative_change_pct(previous: float, current: float) -> float | None:
    """Relative change in percent, as MDX's percentage rule computes it.

    Returns None when `previous` is zero (undefined). Near-zero handling, i.e.
    when this number should not be trusted, is a check (E8-01), not a unit rule.
    """
    if previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, PRECISION_DP)

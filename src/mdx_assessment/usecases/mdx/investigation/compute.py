"""What each investigation tool computes: pure functions, no I/O and no LLM (E8-03).

The toolkit (`toolkit.py`) reads the data through the MDX tool and hands it here. Every
number the student and the teacher see comes from these functions; the LLM never
calculates. Values stay in the quote unit (rates: percent), changes are in the measure
unit (rates: bps).

The results state facts, not conclusions: "both neighbours moved +9 bps" rather than
"a real move". Drawing the conclusion is the student's job, checked by the teacher.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import pairwise
from typing import Literal

from pydantic import Field

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.usecases.mdx.checks.normalize import NormalizedFlag
from mdx_assessment.usecases.mdx.config import MdxRuleSettings
from mdx_assessment.usecases.mdx.domain import Observation, Tenor
from mdx_assessment.usecases.mdx.domain.units import PRECISION_DP, relative_change_pct

CurveValues = Mapping[tuple[str, Tenor], float]
"""Every point's value on one date, keyed by (curve ID, tenor)."""

FlagFrequencySource = Literal["mdx_flags", "configured_rule"]


def _r(value: float) -> float:
    return round(value, PRECISION_DP)


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


class PointMove(DomainModel):
    curve_id: str
    tenor: str
    previous: float
    current: float
    change: float
    """In the measure unit."""


class NeighbourMoves(DomainModel):
    """Did the adjacent tenors on the same curve move like the flagged one?"""

    flag_id: str
    curve_id: str
    tenor: str
    change: float
    unit: str
    previous_date: date
    neighbours: tuple[PointMove, ...]
    """Nearest tenors first on each side, in tenor order."""
    neighbour_mean_change: float | None
    gap_to_neighbour_mean: float | None
    """The flagged change minus the neighbours' mean change."""
    same_direction: int = Field(ge=0)
    """Neighbours that moved in the flagged point's direction."""


class SameTenorOtherCurves(DomainModel):
    """Did this tenor move in other curves today?"""

    flag_id: str
    curve_id: str
    tenor: str
    change: float
    unit: str
    previous_date: date
    others: tuple[PointMove, ...]
    """Every other curve with this tenor, by curve ID."""
    others_mean_change: float | None
    same_direction: int = Field(ge=0)


class HistoryPoint(DomainModel):
    business_date: date
    value: float
    change: float | None
    """From the business day before, in the measure unit; None on the first day."""


class PointHistory(DomainModel):
    """The point over a chosen window, today included."""

    flag_id: str
    curve_id: str
    tenor: str
    days_requested: int
    days: int
    """Business days returned: fewer than requested when MDX holds less history."""
    quote_unit: str
    unit: str
    observations: tuple[HistoryPoint, ...]
    min_value: float
    max_value: float
    largest_change: float | None
    """The day-to-day change largest in size, today included."""
    largest_change_date: date | None
    stdev_change: float | None
    """Sample standard deviation of the day-to-day changes before today."""


class FlagFrequency(DomainModel):
    """How often MDX's rule flagged (or would have flagged) this point in its history."""

    flag_id: str
    curve_id: str
    tenor: str
    source: FlagFrequencySource
    rule: dict[str, str | float]
    """The rule counted: MDX's flags by rule name, or the configured rule re-applied."""
    source_note: str
    window_days: int
    days_checked: int = Field(ge=0)
    days_flagged: int = Field(ge=0)
    flagged_dates: tuple[date, ...]
    first_date: date | None
    last_date: date | None


def _move(
    curve_id: str, tenor: Tenor, prev: CurveValues, curr: CurveValues, factor: float
) -> PointMove | None:
    key = (curve_id, tenor)
    if key not in prev or key not in curr:
        return None
    return PointMove(
        curve_id=curve_id,
        tenor=str(tenor),
        previous=prev[key],
        current=curr[key],
        change=_r((curr[key] - prev[key]) * factor),
    )


def _mean(values: Sequence[float]) -> float | None:
    return _r(statistics.fmean(values)) if values else None


def neighbour_moves(
    flag: NormalizedFlag,
    today: CurveValues,
    previous: CurveValues,
    previous_date: date,
    factor: float,
    each_side: int,
) -> NeighbourMoves:
    tenors = sorted(t for c, t in today if c == flag.curve_id)
    i = tenors.index(flag.tenor) if flag.tenor in tenors else None
    near = (
        []
        if i is None
        else [*tenors[max(0, i - each_side) : i], *tenors[i + 1 : i + 1 + each_side]]
    )
    moves = [m for t in near if (m := _move(flag.curve_id, t, previous, today, factor))]
    mean = _mean([m.change for m in moves])
    return NeighbourMoves(
        flag_id=flag.flag_id,
        curve_id=flag.curve_id,
        tenor=str(flag.tenor),
        change=flag.change,
        unit=flag.measure_unit.value,
        previous_date=previous_date,
        neighbours=tuple(moves),
        neighbour_mean_change=mean,
        gap_to_neighbour_mean=_r(flag.change - mean) if mean is not None else None,
        same_direction=sum(1 for m in moves if _sign(m.change) == _sign(flag.change) != 0),
    )


def same_tenor_other_curves(
    flag: NormalizedFlag,
    today: CurveValues,
    previous: CurveValues,
    previous_date: date,
    factor: float,
) -> SameTenorOtherCurves:
    curves = sorted({c for c, t in today if t == flag.tenor and c != flag.curve_id})
    moves = [m for c in curves if (m := _move(c, flag.tenor, previous, today, factor))]
    return SameTenorOtherCurves(
        flag_id=flag.flag_id,
        curve_id=flag.curve_id,
        tenor=str(flag.tenor),
        change=flag.change,
        unit=flag.measure_unit.value,
        previous_date=previous_date,
        others=tuple(moves),
        others_mean_change=_mean([m.change for m in moves]),
        same_direction=sum(1 for m in moves if _sign(m.change) == _sign(flag.change) != 0),
    )


def point_history(
    flag: NormalizedFlag,
    observations: Sequence[Observation],
    today: date,
    days_requested: int,
    factor: float,
) -> PointHistory:
    obs = list(observations)[-days_requested:]
    changes: list[float | None] = [None] + [
        _r((b.value - a.value) * factor) for a, b in pairwise(obs)
    ]
    points = tuple(
        HistoryPoint(business_date=o.business_date, value=o.value, change=c)
        for o, c in zip(obs, changes, strict=True)
    )
    moved = [p for p in points if p.change is not None]
    largest = max(moved, key=lambda p: abs(p.change or 0.0), default=None)
    before = [p.change for p in points if p.change is not None and p.business_date < today]
    return PointHistory(
        flag_id=flag.flag_id,
        curve_id=flag.curve_id,
        tenor=str(flag.tenor),
        days_requested=days_requested,
        days=len(points),
        quote_unit=flag.quote_unit.value,
        unit=flag.measure_unit.value,
        observations=points,
        min_value=min(o.value for o in obs),
        max_value=max(o.value for o in obs),
        largest_change=largest.change if largest else None,
        largest_change_date=largest.business_date if largest else None,
        stdev_change=_r(statistics.stdev(before)) if len(before) >= 2 else None,
    )


def is_flagged_by_rule(previous: float, current: float, rule: MdxRuleSettings) -> bool:
    """MDX's relative-change rule: |relative change| above the threshold. A move from
    exactly 0 has no relative change and counts as flagged when the value changed."""
    rel = relative_change_pct(previous, current)
    if rel is None:
        return current != previous
    return abs(rel) > rule.relative_change_pct


def _window(
    observations: Sequence[Observation], today: date, window_days: int
) -> list[Observation]:
    return [o for o in observations if o.business_date < today][-window_days:]


def flag_frequency_from_rule(
    flag: NormalizedFlag,
    observations: Sequence[Observation],
    today: date,
    window_days: int,
    rule: MdxRuleSettings,
    why: str,
) -> FlagFrequency:
    """Re-apply the configured rule to each day before today that has a day before it."""
    window = _window(observations, today, window_days)
    flagged = tuple(
        b.business_date for a, b in pairwise(window) if is_flagged_by_rule(a.value, b.value, rule)
    )
    return FlagFrequency(
        flag_id=flag.flag_id,
        curve_id=flag.curve_id,
        tenor=str(flag.tenor),
        source="configured_rule",
        rule={"name": rule.name, "relative_change_pct": rule.relative_change_pct},
        source_note=why,
        window_days=window_days,
        days_checked=max(0, len(window) - 1),
        days_flagged=len(flagged),
        flagged_dates=flagged,
        first_date=window[1].business_date if len(window) > 1 else None,
        last_date=window[-1].business_date if len(window) > 1 else None,
    )


def flag_frequency_from_mdx(
    flag: NormalizedFlag,
    flagged_on: Mapping[date, bool],
    window_days: int,
) -> FlagFrequency:
    """Count the days MDX itself flagged this point, from its past flags."""
    days = sorted(flagged_on)
    return FlagFrequency(
        flag_id=flag.flag_id,
        curve_id=flag.curve_id,
        tenor=str(flag.tenor),
        source="mdx_flags",
        rule={"name": flag.rule},
        source_note=f"MDX's own flags for the {len(days)} business days checked",
        window_days=window_days,
        days_checked=len(days),
        days_flagged=sum(1 for d in days if flagged_on[d]),
        flagged_dates=tuple(d for d in days if flagged_on[d]),
        first_date=days[0] if days else None,
        last_date=days[-1] if days else None,
    )

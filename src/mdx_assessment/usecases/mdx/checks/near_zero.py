"""The near-zero check: is an MDX flag a real move or an artifact of the percentage rule?

Scenario S2 (E8-01). One pure function per flag, no I/O and no LLM: the same inputs
always give the same label. The rules are written out for the team in
docs/business-rules.md §3; settings are in `config/near_zero.yaml` and the floor in the
data-type profile.

Measures, per normalized flag (values in the quote unit, changes in the measure unit):

- `change`: absolute change (rates: bps), from the intake step.
- `relative_change_pct`: what MDX's rule saw. **Unreliable** when the previous value is
  below the near-zero floor, because a tiny base makes any change look huge.
- `z = change / stdev(daily changes)`: today's move in normal days' moves, over the
  `history_window_days` business days before today. Today is left out, so a large move
  can't inflate its own yardstick.

Labels, first that holds wins:

1. `genuine_large_move`: |z| ≥ `large_move_z`. A large move is real however small the rate.
2. `threshold_artifact`: near zero, |change| ≤ `artifact_max_change`, and |z| below the
   large-move cut-off: a big percentage, a small move, a normal day.
3. `ordinary`: anything else.

`insufficient_history` replaces all three when z can't be computed: fewer than
`min_history_changes` daily changes, or a perfectly flat history (stale data, E8-05).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from itertools import pairwise

from pydantic import Field

from mdx_assessment.core.domain import DomainModel, Severity
from mdx_assessment.usecases.mdx.checks.normalize import NormalizedFlag
from mdx_assessment.usecases.mdx.config import DataTypeProfile, NearZeroSettings
from mdx_assessment.usecases.mdx.domain import Observation
from mdx_assessment.usecases.mdx.domain.units import PRECISION_DP, conversion_factor

Z_DP = 2


class Classification(StrEnum):
    GENUINE_LARGE_MOVE = "genuine_large_move"
    THRESHOLD_ARTIFACT = "threshold_artifact"
    ORDINARY = "ordinary"
    INSUFFICIENT_HISTORY = "insufficient_history"


SEVERITY = {
    Classification.GENUINE_LARGE_MOVE: Severity.HIGH,
    Classification.INSUFFICIENT_HISTORY: Severity.MEDIUM,
    Classification.ORDINARY: Severity.MEDIUM,
    Classification.THRESHOLD_ARTIFACT: Severity.LOW,
}
_REVIEW_ORDER = (Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO)


class HistoryStats(DomainModel):
    """The normal daily move of one point, before today."""

    changes: int = Field(ge=0)
    stdev: float | None
    """Sample standard deviation of the daily changes, in the measure unit."""
    first_date: date | None
    last_date: date | None


class NearZeroAssessment(DomainModel):
    flag_id: str
    item_key: str
    classification: Classification
    severity: Severity
    previous: float
    current: float
    change: float
    relative_change_pct: float | None
    near_zero: bool
    relative_change_reliable: bool
    z: float | None
    history: HistoryStats
    reasons: tuple[str, ...]
    """Which rule fired and why, in words, for the reviewer and the explanation."""


def history_stats(
    observations: Sequence[Observation],
    today: date,
    profile: DataTypeProfile,
    settings: NearZeroSettings,
) -> HistoryStats:
    before = [o for o in observations if o.business_date < today][-settings.history_window_days :]
    factor = conversion_factor(profile.quote_unit, profile.measure_unit) or 1.0
    changes = [(b.value - a.value) * factor for a, b in pairwise(before)]
    stdev = statistics.stdev(changes) if len(changes) >= 2 else None
    return HistoryStats(
        changes=len(changes),
        stdev=round(stdev, PRECISION_DP) if stdev is not None else None,
        first_date=before[0].business_date if before else None,
        last_date=before[-1].business_date if before else None,
    )


def _fmt(value: float, unit: str) -> str:
    return f"{value:+.2f} {unit}"


def assess_near_zero(
    flag: NormalizedFlag,
    history: HistoryStats,
    profile: DataTypeProfile,
    settings: NearZeroSettings,
) -> NearZeroAssessment:
    unit = profile.measure_unit.value
    floor = profile.near_zero_floor
    near_zero = abs(flag.previous) < floor
    reliable = not near_zero and flag.relative_change_pct is not None
    reasons = [
        f"previous {flag.previous}% is below the {floor}% floor: relative change unreliable"
        if near_zero
        else f"previous {flag.previous}% is at or above the {floor}% floor"
    ]

    usable = (
        history.stdev is not None
        and history.stdev > 0
        and history.changes >= settings.min_history_changes
    )
    z = round(flag.change / history.stdev, Z_DP) if usable and history.stdev else None
    small = abs(flag.change) <= settings.artifact_max_change

    if z is None:
        label = Classification.INSUFFICIENT_HISTORY
        reasons.append(
            f"only {history.changes} daily changes, or no variation, in the history: "
            f"z can't be computed (needs {settings.min_history_changes})"
        )
    elif abs(z) >= settings.large_move_z:
        label = Classification.GENUINE_LARGE_MOVE
        reasons.append(
            f"z {z:+.2f}: at or beyond the {settings.large_move_z:g} large-move cut-off "
            f"(change {_fmt(flag.change, unit)}, normal daily move {history.stdev:.2f} {unit})"
        )
    elif near_zero and small:
        label = Classification.THRESHOLD_ARTIFACT
        reasons.append(
            f"change {_fmt(flag.change, unit)} is within {settings.artifact_max_change:g} {unit} "
            f"and z {z:+.2f} is normal: the percentage overstates a small move"
        )
    else:
        label = Classification.ORDINARY
        why = (
            f"change {_fmt(flag.change, unit)} is above {settings.artifact_max_change:g} {unit}"
            if near_zero
            else "the rate isn't near zero"
        )
        reasons.append(f"z {z:+.2f} is below the large-move cut-off; {why}")

    return NearZeroAssessment(
        flag_id=flag.flag_id,
        item_key=flag.item_key,
        classification=label,
        severity=SEVERITY[label],
        previous=flag.previous,
        current=flag.current,
        change=flag.change,
        relative_change_pct=flag.relative_change_pct,
        near_zero=near_zero,
        relative_change_reliable=reliable,
        z=z,
        history=history,
        reasons=tuple(reasons),
    )


def review_order(items: Sequence[NearZeroAssessment]) -> list[NearZeroAssessment]:
    """Severity first, then the largest |z|, then the item key for a stable order."""
    return sorted(
        items,
        key=lambda a: (_REVIEW_ORDER.index(a.severity), -abs(a.z or 0.0), a.item_key),
    )

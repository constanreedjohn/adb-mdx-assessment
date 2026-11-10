"""Flag normalization: MDX flags in their quote unit, made comparable in native units (E7-02).

One pure function; the Input Validation Agent calls it with what the MDX tool returned.
For each flag:

1. Look up the data-type profile for its kind (`config/profiles/data_types.yaml`): the
   quote unit values arrive in and the measure unit changes are judged in (rates:
   percent → bps).
2. Take today's value from MDX's curve values, the stored point, and check the flag
   reports the same value.
3. Compute the absolute change in the measure unit, and the relative change in percent
   as MDX's percentage rule sees it.

A flag that can't be normalized isn't dropped silently: it's listed with the reason, so
the review point can reconcile every flag MDX raised (C-1, E10-01).

| Reason                | When                                                       |
| --------------------- | ---------------------------------------------------------- |
| `no_profile`          | No data-type profile for the flag's kind                   |
| `unit_mismatch`       | The flag's unit isn't the profile's quote unit             |
| `no_conversion`       | No plain rule from the quote unit to the measure unit      |
| `no_previous_value`   | The flag has no previous value to measure the change from  |
| `no_curve_value`      | MDX has no stored value for the flagged point today        |
| `value_mismatch`      | The flag's current value differs from MDX's stored value   |
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field

from mdx_assessment.core.domain import DomainModel, EvidenceId
from mdx_assessment.usecases.mdx.config import DataTypeProfiles
from mdx_assessment.usecases.mdx.domain import (
    ColorCode,
    CurveId,
    MarketDataKind,
    MdxFlag,
    Point,
    Tenor,
    Unit,
)
from mdx_assessment.usecases.mdx.domain.units import change_in, relative_change_pct

VALUE_TOLERANCE = 1e-9
"""Flag and stored value are both MDX's own numbers, so they must agree exactly."""


class IssueReason(StrEnum):
    NO_PROFILE = "no_profile"
    UNIT_MISMATCH = "unit_mismatch"
    NO_CONVERSION = "no_conversion"
    NO_PREVIOUS_VALUE = "no_previous_value"
    NO_CURVE_VALUE = "no_curve_value"
    VALUE_MISMATCH = "value_mismatch"


class NormalizedFlag(DomainModel):
    """One MDX flag with its change in the native unit. Values stay in the quote unit."""

    flag_id: str = Field(min_length=1)
    curve_id: CurveId
    tenor: Tenor
    kind: MarketDataKind
    color_code: ColorCode
    rule: str
    previous: float
    current: float
    quote_unit: Unit
    change: float
    """Absolute change in `measure_unit`, e.g. +3.0 bps."""
    measure_unit: Unit
    relative_change_pct: float | None
    """As MDX's percentage rule computes it; None when `previous` is 0."""
    comment: str | None = None
    """MDX's free text, untouched: untrusted data, never instructions."""
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)

    @property
    def item_key(self) -> str:
        return f"{self.curve_id} {self.tenor}"


class FlagIssue(DomainModel):
    flag_id: str = Field(min_length=1)
    reason: IssueReason
    detail: str


class Normalization(DomainModel):
    flags: tuple[NormalizedFlag, ...] = ()
    issues: tuple[FlagIssue, ...] = ()


def _point_key(curve_id: str, tenor: Tenor) -> tuple[str, str]:
    return (curve_id, str(tenor))


def normalize_flags(
    flags: Sequence[MdxFlag],
    points: Sequence[Point],
    profiles: DataTypeProfiles,
    evidence_ids: dict[str, tuple[EvidenceId, ...]],
) -> Normalization:
    """`evidence_ids[flag_id]`: the evidence recorded for that flag and its stored point."""
    stored = {_point_key(p.curve_id, p.tenor): p for p in points}
    normalized: list[NormalizedFlag] = []
    issues: list[FlagIssue] = []

    def issue(flag: MdxFlag, reason: IssueReason, detail: str) -> None:
        issues.append(FlagIssue(flag_id=flag.flag_id, reason=reason, detail=detail))

    for flag in flags:
        profile = profiles.profiles.get(flag.kind)
        if profile is None:
            issue(flag, IssueReason.NO_PROFILE, f"no data-type profile for {flag.kind}")
            continue
        if flag.unit is not profile.quote_unit:
            issue(
                flag,
                IssueReason.UNIT_MISMATCH,
                f"quoted in {flag.unit}, but {flag.kind} is quoted in {profile.quote_unit}",
            )
            continue
        if flag.previous_value is None:
            issue(flag, IssueReason.NO_PREVIOUS_VALUE, "the flag has no previous value")
            continue
        point = stored.get(_point_key(flag.curve_id, flag.tenor))
        if point is None:
            issue(flag, IssueReason.NO_CURVE_VALUE, "MDX has no stored value for this point")
            continue
        if flag.current_value is not None and abs(flag.current_value - point.value) > (
            VALUE_TOLERANCE
        ):
            issue(
                flag,
                IssueReason.VALUE_MISMATCH,
                f"the flag reports {flag.current_value}, MDX stores {point.value}",
            )
            continue
        change = change_in(
            flag.previous_value, point.value, profile.quote_unit, profile.measure_unit
        )
        if change is None:
            issue(
                flag,
                IssueReason.NO_CONVERSION,
                f"no rule converts {profile.quote_unit} to {profile.measure_unit}",
            )
            continue
        normalized.append(
            NormalizedFlag(
                flag_id=flag.flag_id,
                curve_id=flag.curve_id,
                tenor=flag.tenor,
                kind=flag.kind,
                color_code=flag.color_code,
                rule=flag.rule,
                previous=flag.previous_value,
                current=point.value,
                quote_unit=profile.quote_unit,
                change=change,
                measure_unit=profile.measure_unit,
                relative_change_pct=relative_change_pct(flag.previous_value, point.value),
                comment=flag.comment,
                evidence_ids=evidence_ids[flag.flag_id],
            )
        )
    return Normalization(flags=tuple(normalized), issues=tuple(issues))

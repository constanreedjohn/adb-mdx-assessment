"""Schema for `config/sim/linear_day.yaml`: the simulated linear MDX day (plan §2.3)."""

from __future__ import annotations

from datetime import time
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.usecases.mdx.domain import ColorCode, CurveId, Tenor

SIM_LINEAR_DAY_PATH = "sim/linear_day.yaml"

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class SimCurve(DomainModel):
    """One rate curve. Values in percent; the base shape is long + (short - long)·e^(-t/decay)."""

    id: CurveId
    short_rate: float
    long_rate: float
    decay_years: float = Field(gt=0)
    daily_vol_bps: float = Field(gt=0)
    """Typical daily move (1 standard deviation) in bps."""
    color_code: ColorCode
    color_overrides: dict[str, ColorCode] = Field(default_factory=dict)
    """Per-tenor colour codes that differ from the curve's, e.g. {"30Y": greyed}."""


class PlantedKind(StrEnum):
    THRESHOLD_ARTIFACT = (
        "threshold_artifact"  # big % on a near-zero rate, small bps, normal history
    )
    GENUINE_LARGE_MOVE = "genuine_large_move"  # real market move, far outside history


class PlantedCase(DomainModel):
    id: Identifier
    kind: PlantedKind
    curve: CurveId
    tenor: Tenor
    previous: float | None = None
    """Exact previous-day value (percent). The whole history is shifted to end here."""
    current: float | None = None
    """Exact value today (percent). Use either `current` or `change_bps`."""
    change_bps: float | None = None
    neighbour_change_bps: float = 0.0
    """Move applied today to the adjacent tenors, so the move is coherent across the curve."""

    @model_validator(mode="after")
    def _one_way_to_set_today(self) -> Self:
        if (self.current is None) == (self.change_bps is None):
            raise ValueError("set exactly one of `current` or `change_bps`")
        return self


class MdxRule(DomainModel):
    """MDX's current generic rule: flag when |relative change| exceeds a percentage."""

    relative_change_pct: float = Field(gt=0)


class SimLinearDay(DomainModel):
    seed: int
    history_days: int = Field(ge=20)
    """Business days of history before the business date."""
    file_id: Identifier
    """Entry in expected_files.yaml that this simulated file delivers."""
    arrival_time: time
    """Local arrival time of the vendor file (timezone from expected_files.yaml)."""
    mean_reversion: float = Field(gt=0, lt=1)
    idiosyncratic_share: float = Field(ge=0, le=1)
    """Share of each tenor's daily variance that is its own (the rest is common to the curve)."""
    tenors: tuple[Tenor, ...] = Field(min_length=2)
    curves: tuple[SimCurve, ...] = Field(min_length=1)
    mdx_rule: MdxRule
    planted: tuple[PlantedCase, ...] = ()

    @field_validator("tenors")
    @classmethod
    def _sorted_unique(cls, tenors: tuple[Tenor, ...]) -> tuple[Tenor, ...]:
        if list(tenors) != sorted(set(tenors)):
            raise ValueError("tenors must be unique and in increasing order")
        return tenors

    @model_validator(mode="after")
    def _references_exist(self) -> Self:
        curve_ids = [c.id for c in self.curves]
        if len(set(curve_ids)) != len(curve_ids):
            raise ValueError("duplicate curve id")
        for curve in self.curves:
            unknown = set(curve.color_overrides) - {str(t) for t in self.tenors}
            if unknown:
                raise ValueError(
                    f"{curve.id}: colour override for unknown tenor(s) {sorted(unknown)}"
                )
        for case in self.planted:
            if case.curve not in curve_ids:
                raise ValueError(f"planted {case.id}: unknown curve {case.curve}")
            if case.tenor not in self.tenors:
                raise ValueError(f"planted {case.id}: unknown tenor {case.tenor}")
        if len({(c.curve, c.tenor) for c in self.planted}) != len(self.planted):
            raise ValueError("two planted cases on the same point")
        return self

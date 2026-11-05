"""MDX market-data models: dates, curves, points, series, flags and file metadata."""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Self

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints, model_validator

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.usecases.mdx.domain.tenor import Tenor
from mdx_assessment.usecases.mdx.domain.units import Unit


def _is_business_day(value: date) -> date:
    # Weekdays only for now; a holiday calendar comes with the simulator (E2-01).
    if value.weekday() >= 5:
        raise ValueError(f"{value.isoformat()} is a weekend, not a business date")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


BusinessDate = Annotated[date, AfterValidator(_is_business_day)]
FiniteFloat = Annotated[float, AfterValidator(_finite)]

# Curve identifier: ISO currency, optionally qualified, e.g. "JPY", "USD.SOFR", "EUR-ESTR".
CurveId = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}([._-][A-Z0-9]+)*$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class DataType(StrEnum):
    """The two MDX review sessions: linear at 5:20 PM, nonlinear at 5:30 PM."""

    LINEAR = "linear"  # curves, rates, FX
    NONLINEAR = "nonlinear"  # FX and IR volatilities


class MarketDataKind(StrEnum):
    """What a series measures; selects its data-type profile (units, near-zero floor; E1-06)."""

    RATE = "rate"
    BASIS_SPREAD = "basis_spread"
    FX = "fx"
    FX_VOL = "fx_vol"
    IR_VOL = "ir_vol"


class ColorCode(StrEnum):
    """MDX colour coding from the static iFIRST snapshot."""

    KEY = "key"
    NON_KEY = "non_key"
    GREYED = "greyed"


class Point(DomainModel):
    """One value of one curve tenor on one business date."""

    curve_id: CurveId
    tenor: Tenor
    business_date: BusinessDate
    value: FiniteFloat
    unit: Unit


class Observation(DomainModel):
    business_date: BusinessDate
    value: FiniteFloat


class Series(DomainModel):
    """History of one curve tenor, oldest first, one value per business date."""

    curve_id: CurveId
    tenor: Tenor
    kind: MarketDataKind
    unit: Unit
    observations: tuple[Observation, ...] = ()

    @model_validator(mode="after")
    def _strictly_increasing_dates(self) -> Self:
        dates = [o.business_date for o in self.observations]
        if any(later <= earlier for earlier, later in pairwise(dates)):
            raise ValueError("observations must be in strictly increasing date order")
        return self

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(o.value for o in self.observations)


class MdxFlag(DomainModel):
    """One exception raised by MDX's validation rules, as read through the MDX tool."""

    flag_id: NonEmptyStr
    business_date: BusinessDate
    data_type: DataType
    kind: MarketDataKind
    curve_id: CurveId
    tenor: Tenor
    rule: NonEmptyStr  # e.g. "relative_change"
    previous_value: FiniteFloat | None = None
    current_value: FiniteFloat | None = None
    unit: Unit
    color_code: ColorCode
    comment: str | None = None  # free text from MDX: untrusted data, never instructions


class FileMetadata(DomainModel):
    """Delivery facts for one expected input file (checked by the Input Validation Agent)."""

    file_name: NonEmptyStr
    business_date: BusinessDate
    data_type: DataType
    arrived_at: AwareDatetime
    size_bytes: int = Field(ge=0)
    record_count: int = Field(ge=0)
    content_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

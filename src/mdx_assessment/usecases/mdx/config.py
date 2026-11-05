"""Config schemas owned by the MDX pack.

- `profiles/data_types.yaml`: per market-data kind, the quoted unit, the native unit
  for measures, and the near-zero floor below which relative change isn't trusted.
- `expected_files.yaml`: the input files each session expects (Input Validation Agent).
"""

from __future__ import annotations

import re
from datetime import date, time
from typing import Annotated, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, StringConstraints, field_validator, model_validator

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.usecases.mdx.domain import DataType, MarketDataKind, Unit

DATA_TYPES_PATH = "profiles/data_types.yaml"
EXPECTED_FILES_PATH = "expected_files.yaml"

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


# --- profiles/data_types.yaml --------------------------------------------------


class DataTypeProfile(DomainModel):
    quote_unit: Unit  # how values arrive from MDX, e.g. percent for rates
    measure_unit: Unit  # native unit for absolute change, e.g. bps for rates
    near_zero_floor: float = Field(gt=0)
    """In quote units. |previous| below this → relative change is marked unreliable (E8-01)."""
    provisional: bool = False
    """True while a value is a placeholder awaiting ADB confirmation."""
    note: str = ""


# Unit pairs that make sense for each kind; guards against e.g. a rate measured in pips.
_ALLOWED_UNITS: dict[MarketDataKind, tuple[set[Unit], set[Unit]]] = {
    MarketDataKind.RATE: ({Unit.PERCENT}, {Unit.BPS}),
    MarketDataKind.BASIS_SPREAD: ({Unit.BPS, Unit.PERCENT}, {Unit.BPS}),
    MarketDataKind.FX: ({Unit.PRICE}, {Unit.PIPS}),
    MarketDataKind.FX_VOL: ({Unit.PERCENT, Unit.VOL_POINTS}, {Unit.VOL_POINTS}),
    MarketDataKind.IR_VOL: ({Unit.BPS, Unit.PERCENT, Unit.VOL_POINTS}, {Unit.BPS, Unit.VOL_POINTS}),
}


class DataTypeProfiles(DomainModel):
    profiles: dict[MarketDataKind, DataTypeProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def _units_fit_kind(self) -> Self:
        problems = []
        for kind, profile in self.profiles.items():
            quotes, measures = _ALLOWED_UNITS[kind]
            if profile.quote_unit not in quotes:
                problems.append(
                    f"{kind}: quote_unit {profile.quote_unit} not one of {sorted(quotes)}"
                )
            if profile.measure_unit not in measures:
                problems.append(
                    f"{kind}: measure_unit {profile.measure_unit} not one of {sorted(measures)}"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def for_kind(self, kind: MarketDataKind) -> DataTypeProfile:
        try:
            return self.profiles[kind]
        except KeyError:
            raise KeyError(f"no data-type profile for {kind}") from None


# --- expected_files.yaml ---------------------------------------------------------

_DATE_PLACEHOLDER = "{yyyymmdd}"


class ExpectedFile(DomainModel):
    id: Identifier
    data_type: DataType
    name_pattern: str
    """File name with a `{yyyymmdd}` business-date token, e.g. `mdx_linear_{yyyymmdd}.csv`."""
    expected_record_count: int = Field(gt=0)
    expected_by: time | None = None
    """Latest expected arrival (local time in `timezone`); used from E7-03."""
    description: str = ""

    @field_validator("name_pattern")
    @classmethod
    def _has_single_date_token(cls, value: str) -> str:
        if value.count(_DATE_PLACEHOLDER) != 1:
            raise ValueError(f"must contain {_DATE_PLACEHOLDER} exactly once")
        literal = value.replace(_DATE_PLACEHOLDER, "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", literal):
            raise ValueError("may only contain letters, digits, '_', '.', '-' besides the token")
        return value

    def file_name_for(self, business_date: date) -> str:
        return self.name_pattern.replace(_DATE_PLACEHOLDER, business_date.strftime("%Y%m%d"))


class ExpectedFiles(DomainModel):
    timezone: Annotated[str, StringConstraints(min_length=1)]
    """IANA zone for `expected_by` times, e.g. Asia/Manila (ADB HQ)."""
    files: tuple[ExpectedFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids_and_patterns(self) -> Self:
        for attr in ("id", "name_pattern"):
            values = [getattr(f, attr) for f in self.files]
            dupes = sorted({v for v in values if values.count(v) > 1})
            if dupes:
                raise ValueError(f"duplicate {attr}: {dupes}")
        return self

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown IANA timezone {value!r}") from None
        return value

    def for_data_type(self, data_type: DataType) -> tuple[ExpectedFile, ...]:
        return tuple(f for f in self.files if f.data_type is data_type)

"""Tenor: a maturity label such as "1W", "3M" or "2Y".

- Parses case-insensitively ("3m" → "3M"); units D (days), W (weeks), M (months), Y (years).
- Orders by length of time: 1W < 1M < 3M < 2Y. Days are converted at 365 a year.
- Equality is by label: "12M" and "1Y" are different points on an MDX curve even
  though they are the same length; ordering puts the month form first, so the
  order stays total and deterministic.
- In pydantic models a Tenor reads from, and serializes to, its label string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from typing import Any, ClassVar

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

_PATTERN = re.compile(r"^\s*(\d+)\s*([DWMY])\s*$", re.IGNORECASE)
_YEARS_PER_UNIT: dict[str, Fraction] = {
    "D": Fraction(1, 365),
    "W": Fraction(7, 365),
    "M": Fraction(1, 12),
    "Y": Fraction(1),
}
_UNIT_RANK = {"D": 0, "W": 1, "M": 2, "Y": 3}


@total_ordering
@dataclass(frozen=True, slots=True)
class Tenor:
    count: int
    unit: str

    LABEL_PATTERN: ClassVar[str] = r"^\d+[DWMYdwmy]$"

    def __post_init__(self) -> None:
        if self.unit not in _YEARS_PER_UNIT:
            raise ValueError(f"unknown tenor unit {self.unit!r}; expected one of D, W, M, Y")
        if self.count <= 0:
            raise ValueError(f"tenor count must be positive, got {self.count}")

    @classmethod
    def parse(cls, label: str) -> Tenor:
        match = _PATTERN.match(label)
        if not match:
            raise ValueError(f"invalid tenor {label!r}; expected e.g. '1W', '3M', '2Y'")
        return cls(int(match.group(1)), match.group(2).upper())

    @property
    def years(self) -> Fraction:
        """Length in years (exact). 1D = 1/365 years."""
        return self.count * _YEARS_PER_UNIT[self.unit]

    def _sort_key(self) -> tuple[Fraction, int]:
        return (self.years, _UNIT_RANK[self.unit])

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tenor):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __str__(self) -> str:
        return f"{self.count}{self.unit}"

    def __repr__(self) -> str:
        return f"Tenor({str(self)!r})"

    # --- pydantic integration: a Tenor field accepts "3M" (or a Tenor) and dumps "3M".

    @classmethod
    def _validate(cls, value: Any) -> Tenor:
        if isinstance(value, Tenor):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        # ValueError (not TypeError) so pydantic reports it as a validation error.
        raise ValueError(f"tenor must be a string like '3M', got {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "pattern": cls.LABEL_PATTERN, "examples": ["1W", "3M", "2Y"]}

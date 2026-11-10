"""Numbers in model-written text, and whether they match the evidence (E5-03).

    stated = extract_numbers("JPY 3M rose +3 bps (+75%), a 0.03% move.")
    # [+3 bps, +75 %, 0.03 %]   "3M" is a tenor, not a number
    matches(stated[2], Quantity(3.0, Unit.BPS), tolerance)   # True: 0.03% is 3 bps

**What counts as a stated number.** Decimals, integers, thousands separators
(`1,250.5`), signs (`+3`, `-3`, and `3` after a U+2212 minus sign) and the units `%`/`percent`/`pct` and
`bps`/`bp`/`basis points`. Skipped, because they are labels rather than claims: ISO dates
and times, numbers glued to letters (tenors like `3M` or `10Y`, versions like `v1`) and
numbers inside IDs (`P-1`, `fact-2`). Skipping errs towards letting a label through,
never towards rejecting a correct text.

**When a stated number matches a quantity.**
- Units must be compatible: percent and bps convert (1% = 100 bps); a plain number in
  the text matches any quantity's raw value; a number with a unit never matches a
  plain count (28 flags is not 28%).
- Without an explicit sign, the size is compared ("fell 3 bps" matches -3 bps). With
  one, the sign must agree too.
- The difference may be up to half a unit of the last digit written (18.04 → "18 bps",
  5.14 → "5.1"), or the configured relative tolerance, whichever is larger.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from mdx_assessment.core.domain import DomainModel

BPS_PER_PERCENT = 100.0
UNICODE_MINUS = "\u2212"


class Unit(StrEnum):
    PERCENT = "percent"
    BPS = "bps"


_TO_BPS = {Unit.PERCENT: BPS_PER_PERCENT, Unit.BPS: 1.0}


@dataclass(frozen=True)
class Quantity:
    """A number from the evidence. `unit=None` is a plain number (a count, a z-score)."""

    value: float
    unit: Unit | None = None


@dataclass(frozen=True)
class StatedNumber:
    """A number as written in the text."""

    text: str
    value: float
    unit: Unit | None
    decimals: int
    signed: bool

    def __str__(self) -> str:
        return self.text


class NumberTolerance(DomainModel):
    """`config/guardrails.yaml` → `number_matching`."""

    relative_tolerance: float = Field(default=0.001, ge=0, le=0.1)
    """Allowed difference as a share of the evidence value (0.001 = 0.1%)."""
    accept_rounding: bool = True
    """Accept a number rounded to the precision written (18.04 → "18")."""
    ignore_plain_integers_up_to: int = Field(default=10, ge=0)
    """Plain whole numbers up to this ("3 steps", "1 file") are not checked. A number with
    a unit is always checked."""


_DATE_OR_TIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?\b|\b\d{1,2}:\d{2}\b"
)
_NUMBER = re.compile(
    r"""
    (?<![\w.,])                       # not part of a word, ID or another number
    (?<![A-Za-z]-)                    # not the tail of an ID like P-1
    (?P<sign>[+\-\u2212])?
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?P<unit>\s*(?:%|percent\b|per\s+cent\b|pct\b|bps\b|bp\b|basis\s+points?\b))?
    (?![\w])                          # not glued to letters: 3M, 10Y, 2nd, v1
    (?!\.\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _unit(raw: str | None) -> Unit | None:
    if not raw:
        return None
    word = raw.strip().lower()
    if word == "%" or word.startswith(("percent", "per", "pct")):
        return Unit.PERCENT
    return Unit.BPS


def extract_numbers(text: str) -> list[StatedNumber]:
    """Every number the text states, in order."""
    cleaned = _DATE_OR_TIME.sub(lambda m: " " * len(m.group()), text)
    found = []
    for match in _NUMBER.finditer(cleaned):
        digits = match["digits"]
        sign = match["sign"]
        value = float(digits.replace(",", ""))
        if sign in ("-", UNICODE_MINUS):
            value = -value
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(
            StatedNumber(
                text=text[match.start() : match.end()].strip(),
                value=value,
                unit=_unit(match["unit"]),
                decimals=decimals,
                signed=sign is not None,
            )
        )
    return found


def needs_checking(stated: StatedNumber, tolerance: NumberTolerance) -> bool:
    plain_small_integer = (
        stated.unit is None
        and stated.decimals == 0
        and abs(stated.value) <= tolerance.ignore_plain_integers_up_to
    )
    return not plain_small_integer


def _in_unit(quantity: Quantity, unit: Unit | None) -> float | None:
    """The quantity's value expressed in `unit`, or None if the units don't mix."""
    if unit is None:
        return quantity.value
    if quantity.unit is None:
        return None
    return quantity.value * _TO_BPS[quantity.unit] / _TO_BPS[unit]


def matches(stated: StatedNumber, quantity: Quantity, tolerance: NumberTolerance) -> bool:
    expected = _in_unit(quantity, stated.unit)
    if expected is None:
        return False
    actual = stated.value
    if not stated.signed:
        expected, actual = abs(expected), abs(actual)
    allowed = tolerance.relative_tolerance * abs(expected)
    if tolerance.accept_rounding:
        allowed = max(allowed, 0.5 * 10.0**-stated.decimals)
    return math.isclose(actual, expected, abs_tol=allowed + 1e-12)

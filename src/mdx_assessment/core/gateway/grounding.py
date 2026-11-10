"""Evidence citation and number matching: model text must stand on the evidence (E5-03).

The caller builds an `EvidenceBundle` from what it gives the model:

- **items** the model may cite, each with an ID (`fact-flag-count`, `ev-…`) and the
  numbers it holds;
- **context**: further numbers the model was shown and may quote, but not cite
  (a playbook's limits, say).

Output models that make claims implement `Grounded`: the IDs they cite and the texts
they write. `check_grounding` then requires that

1. at least one ID is cited, if the caller asks for that;
2. every cited ID, and every ID named in the text, is in the bundle;
3. every number the text states matches a number of a cited item or of the context,
   unit-aware and within tolerance (`numbers.matches`).

The gateway runs this on any call given a bundle, and rejects the output if it fails;
plan validation runs it as P-4.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.core.gateway.numbers import (
    NumberTolerance,
    Quantity,
    Unit,
    extract_numbers,
    matches,
    needs_checking,
)

GUARDRAILS_PATH = "guardrails.yaml"


class GuardrailsConfig(DomainModel):
    """`config/guardrails.yaml`."""

    version: str = Field(min_length=1)
    number_matching: NumberTolerance
    provisional: bool = False


_NAMED_ID = re.compile(r"\b(?:fact|obs)-[a-z0-9]+(?:-[a-z0-9]+)*|\bev-[0-9a-f]{16}\b")


@runtime_checkable
class Grounded(Protocol):
    """An output whose texts must be backed by the evidence it cites."""

    def cited_ids(self) -> Sequence[str]: ...

    def grounded_texts(self) -> Sequence[str]: ...


def quantities_in(data: JsonValue, units: Mapping[str, Unit] | None = None) -> tuple[Quantity, ...]:
    """Every number in a JSON value. `units` gives the unit of a key, e.g. `change_bps`."""
    found: list[Quantity] = []

    def walk(value: JsonValue, unit: Unit | None) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            found.append(Quantity(float(value), unit))
        elif isinstance(value, dict):
            for key, inner in value.items():
                walk(inner, (units or {}).get(key, unit))
        elif isinstance(value, list):
            for inner in value:
                walk(inner, unit)

    walk(data, None)
    return tuple(found)


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    quantities: tuple[Quantity, ...] = ()

    @classmethod
    def from_data(
        cls, id: str, data: JsonValue, units: Mapping[str, Unit] | None = None
    ) -> EvidenceItem:
        return cls(id, quantities_in(data, units))


@dataclass(frozen=True)
class EvidenceBundle:
    items: Mapping[str, EvidenceItem]
    context: tuple[Quantity, ...] = ()
    tolerance: NumberTolerance = field(default_factory=NumberTolerance)

    @classmethod
    def of(
        cls,
        items: Iterable[EvidenceItem],
        *,
        context: Iterable[Quantity] = (),
        tolerance: NumberTolerance | None = None,
    ) -> EvidenceBundle:
        by_id: dict[str, EvidenceItem] = {}
        for item in items:
            if item.id in by_id:
                raise ValueError(f"evidence {item.id!r} is in the bundle twice")
            by_id[item.id] = item
        return cls(by_id, tuple(context), tolerance or NumberTolerance())

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self.items)


@dataclass(frozen=True)
class GroundingReport:
    citation_errors: tuple[str, ...] = ()
    number_errors: tuple[str, ...] = ()
    numbers_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.citation_errors and not self.number_errors

    @property
    def errors(self) -> tuple[str, ...]:
        return self.citation_errors + self.number_errors


def check_grounding(
    output: Grounded, bundle: EvidenceBundle, *, require_citation: bool = True
) -> GroundingReport:
    cited = list(dict.fromkeys(output.cited_ids()))
    texts = [t for t in output.grounded_texts() if t]
    citation_errors = [] if cited or not require_citation else ["no evidence is cited"]
    citation_errors += [f"cited ID {i!r} does not exist" for i in cited if i not in bundle.items]
    named = {m for text in texts for m in _NAMED_ID.findall(text)}
    citation_errors += [f"the text names unknown ID {i!r}" for i in sorted(named - bundle.ids)]

    known = [q for i in cited if i in bundle.items for q in bundle.items[i].quantities]
    known += bundle.context
    number_errors: list[str] = []
    checked = 0
    for text in texts:
        for stated in extract_numbers(text):
            if not needs_checking(stated, bundle.tolerance):
                continue
            checked += 1
            if not any(matches(stated, q, bundle.tolerance) for q in known):
                number_errors.append(f"{str(stated)!r} does not match the cited evidence")
    return GroundingReport(
        citation_errors=tuple(citation_errors),
        number_errors=tuple(dict.fromkeys(number_errors)),
        numbers_checked=checked,
    )

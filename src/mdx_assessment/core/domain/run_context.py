"""Run context: what code establishes at session start, before any reasoning (plan §2.5).

**Facts** drive playbook selection and are what the Master Agent's plan must cite.
**Observations** are borderline situations the Master should know about but that do
not select a playbook (for example a file late but still inside its grace period).

Each fact has a stable ID (`fact-flag-count`), which plans cite, and a content-derived
evidence ID. A value that could not be established is `None`: never guessed.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import EvidenceId, compute_evidence_id

FactId = Annotated[str, StringConstraints(pattern=r"^fact-[a-z0-9-]+$")]
FactName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
EVIDENCE_KIND = "run_fact"


class Fact(DomainModel):
    id: FactId
    name: FactName
    value: JsonValue
    expected: JsonValue = None
    threshold: JsonValue = None
    source: str = Field(min_length=1)
    """Where it came from, e.g. "mdx_data.get_flags"."""
    evidence_id: EvidenceId

    @classmethod
    def create(
        cls,
        name: str,
        value: JsonValue,
        *,
        source: str,
        scope: dict[str, JsonValue],
        expected: JsonValue = None,
        threshold: JsonValue = None,
    ) -> Self:
        """`scope` identifies the run (e.g. business date), so each day's fact is distinct."""
        evidence_id = compute_evidence_id(
            EVIDENCE_KIND,
            source,
            {
                "name": name,
                "value": value,
                "expected": expected,
                "threshold": threshold,
                "scope": scope,
            },
        )
        return cls(
            id="fact-" + name.replace("_", "-"),
            name=name,
            value=value,
            expected=expected,
            threshold=threshold,
            source=source,
            evidence_id=evidence_id,
        )


class RunObservation(DomainModel):
    id: Annotated[str, StringConstraints(pattern=r"^obs-[a-z0-9-]+$")]
    text: str = Field(min_length=1)
    fact_ids: tuple[FactId, ...] = ()


class RunContext(DomainModel):
    facts: tuple[Fact, ...] = ()
    observations: tuple[RunObservation, ...] = ()

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [f.id for f in self.facts] + [o.id for o in self.observations]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate fact or observation id(s): {duplicates}")
        return self

    def fact(self, name: str) -> Fact | None:
        return next((f for f in self.facts if f.name == name), None)

    @property
    def citable_ids(self) -> frozenset[str]:
        """IDs a plan may cite (P-4)."""
        return frozenset({f.id for f in self.facts} | {o.id for o in self.observations})

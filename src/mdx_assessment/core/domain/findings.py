"""Measures and findings: what deterministic checks and agents conclude, with evidence."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field, field_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import EvidenceId, NonEmptyStr


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Measure(DomainModel):
    """One computed number with its unit, e.g. change_bps = 3.0 bps."""

    name: NonEmptyStr
    value: float
    unit: NonEmptyStr
    evidence_ids: tuple[EvidenceId, ...] = ()

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("measure value must be finite")
        return v


class Finding(DomainModel):
    """A conclusion about one subject (an item, dataset or run), citing its evidence.

    `kind` is use-case defined (e.g. "threshold_artifact", "dataset_incomplete");
    core treats it as an opaque label.
    """

    finding_id: NonEmptyStr
    kind: NonEmptyStr
    subject: NonEmptyStr
    summary: NonEmptyStr
    severity: Severity = Severity.INFO
    measures: tuple[Measure, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)

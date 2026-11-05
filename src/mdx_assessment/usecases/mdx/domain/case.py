"""Exception case: one assessed item recorded for the analyst (PDF §2.1 exception cases store)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field

from mdx_assessment.core.domain import DomainModel, EvidenceId, Finding, Measure
from mdx_assessment.usecases.mdx.domain.market import CurveId, NonEmptyStr
from mdx_assessment.usecases.mdx.domain.tenor import Tenor


class CaseStatus(StrEnum):
    OPEN = "open"
    DECIDED = "decided"  # an analyst recorded a decision (SP9)
    CLOSED = "closed"


class ExplanationSource(StrEnum):
    LLM = "llm"
    TEMPLATE = "template"  # deterministic fallback text (gateway retry failed or LLM down)


class Case(DomainModel):
    """One prioritized item (a curve tenor in the skeleton; a curve or surface group later).

    Holds what the agents found and why. It has no field for a final decision by
    the solution: analyst decisions are recorded separately (SP9, PDF §3.1).
    """

    case_id: NonEmptyStr
    session_id: NonEmptyStr
    curve_id: CurveId
    tenor: Tenor | None = None  # None when the case covers a whole curve
    status: CaseStatus = CaseStatus.OPEN
    measures: tuple[Measure, ...] = ()
    findings: tuple[Finding, ...] = ()
    explanation: str | None = None
    explanation_source: ExplanationSource | None = None
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    created_at: AwareDatetime

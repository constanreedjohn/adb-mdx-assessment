"""Review points: deterministic check results and the Master Agent's decisions (plan §2.6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import EvidenceId, NonEmptyStr
from mdx_assessment.core.domain.planning import PlanStep


class ReviewCheckResult(DomainModel):
    """Outcome of one deterministic check at a review point, e.g. C-1 flag reconciliation."""

    check_id: NonEmptyStr
    name: NonEmptyStr
    passed: bool
    details: str = ""
    evidence_ids: tuple[EvidenceId, ...] = ()


class DecisionKind(StrEnum):
    ACCEPT = "accept"
    FOLLOW_UP = "follow_up"
    FLAG_FOR_HUMAN = "flag_for_human"


class Decision(DomainModel):
    """What the Master Agent decides after a review point. Advisory: never a final business decision."""

    step_id: NonEmptyStr
    kind: DecisionKind
    reason: NonEmptyStr
    follow_up: PlanStep | None = None
    evidence_ids: tuple[EvidenceId, ...] = ()

    @model_validator(mode="after")
    def _follow_up_iff_kind(self) -> Self:
        if self.kind is DecisionKind.FOLLOW_UP and self.follow_up is None:
            raise ValueError("a follow_up decision must name the follow-up step")
        if self.kind is not DecisionKind.FOLLOW_UP and self.follow_up is not None:
            raise ValueError(f"a {self.kind} decision must not carry a follow-up step")
        return self

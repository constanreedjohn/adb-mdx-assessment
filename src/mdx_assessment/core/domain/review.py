"""Review points: deterministic check results and the Master Agent's decisions (plan §2.6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import EvidenceId, NonEmptyStr
from mdx_assessment.core.domain.planning import PlanStep


class ReviewCheckResult(DomainModel):
    """Outcome of one deterministic check at a review point, e.g. C-1 flag reconciliation."""

    step_id: NonEmptyStr | None = None
    """The review point's step; set by the review point."""
    check_id: NonEmptyStr
    name: NonEmptyStr
    passed: bool
    details: str = ""
    evidence_ids: tuple[EvidenceId, ...] = ()


class DecisionKind(StrEnum):
    ACCEPT = "accept"
    FOLLOW_UP = "follow_up"
    FLAG_FOR_HUMAN = "flag_for_human"


class DecisionSource(StrEnum):
    RULE = "rule"  # code decided: every check passed, or the step failed or was skipped
    LLM_REVIEW = "llm_review"  # a check failed and the Master's LLM review decided
    RULE_FALLBACK = "rule_fallback"  # a check failed, the LLM review wasn't usable: flagged


class Decision(DomainModel):
    """What the Master Agent decides after a review point. Advisory: never a final business decision."""

    step_id: NonEmptyStr
    kind: DecisionKind
    reason: NonEmptyStr
    source: DecisionSource = DecisionSource.RULE
    follow_up: PlanStep | None = None
    evidence_ids: tuple[EvidenceId, ...] = ()

    @model_validator(mode="after")
    def _follow_up_iff_kind(self) -> Self:
        if self.kind is DecisionKind.FOLLOW_UP and self.follow_up is None:
            raise ValueError("a follow_up decision must name the follow-up step")
        if self.kind is not DecisionKind.FOLLOW_UP and self.follow_up is not None:
            raise ValueError(f"a {self.kind} decision must not carry a follow-up step")
        return self


class ItemDecisionKind(StrEnum):
    """The Master's decision on one item of a step's output (E10-05), e.g. one flag's draft."""

    ACCEPT = "accept"
    SEND_BACK = "send_back"  # sent back to its agent once, with a reason
    UNRESOLVED = "unresolved"
    """The item ends without an accepted draft: code rejected it, or the Master rejected
    its revision, or its send-back couldn't run. The pack decides what that means."""
    NOT_VALIDATED = "not_validated"  # the LLM was off or its review wasn't usable


class ItemDecision(DomainModel):
    item_id: NonEmptyStr
    kind: ItemDecisionKind
    reason: str = ""
    source: DecisionSource
    revision: int = 0
    """Which draft was reviewed: 0 the first, 1 after a send-back."""


class StepValidation(DomainModel):
    """The Master's per-item review of one step's output, after the step's checks."""

    step_id: NonEmptyStr
    validated: bool
    """False when no item was reviewed by the Master (LLM off or review unusable)."""
    summary: str
    decisions: tuple[ItemDecision, ...] = ()
    llm_attempts: int = 0


class RunOpinion(DomainModel):
    """The Master Agent's summary of the run for the analyst (E10-02). Advisory."""

    summary: NonEmptyStr
    counts: str = ""
    """The day's counts, written by code: the summary doesn't state them."""
    review_first: tuple[str, ...] = ()
    """Item IDs to look at first, in order."""
    uncertainties: tuple[str, ...] = ()
    """What the analyst shouldn't take for granted."""
    evidence_ids: tuple[str, ...] = ()
    source: Literal["llm", "template_fallback"]
    fallback_reason: str | None = None
    """Why code wrote it: the LLM was off, unreachable, or its output unusable twice."""
    attempts: int = 0


class Reliability(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RunReview(DomainModel):
    """The run as a whole, after the last review point (simple rule, E10-01; E10-06 refines).

    - `low`: a step failed or was skipped, or a review point flagged a step for a human.
    - `medium`: otherwise, if the default plan ran, a step used a template fallback, or a
      check failed and the Master's review accepted the step anyway.
    - `high`: every step done, every check passed, every decision accept.
    """

    reliability: Reliability
    reasons: tuple[str, ...] = ()
    human_review_required: tuple[str, ...] = ()
    """Step IDs a review point flagged for a human."""
    llm_reviews: int = 0
    """Review points that needed the Master's LLM review (option C)."""

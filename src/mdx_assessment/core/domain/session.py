"""Session result: the run-level record returned by `GET /sessions/{id}` (plan §2.4).

Core keeps this use-case-agnostic: `data_type` is an opaque label and agent
outputs are filled in by the packs. It grows story by story; fields added later
(run context, playbook, agents, review, run opinion) are optional or defaulted
so earlier records stay valid.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import NonEmptyStr
from mdx_assessment.core.domain.findings import Finding
from mdx_assessment.core.domain.planning import Plan, Task
from mdx_assessment.core.domain.review import Decision, ReviewCheckResult


class SessionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_DEGRADED = "completed_degraded"  # completed with a fallback (plan §3.3)
    DEFAULT_COMPLETE = "default_complete"  # Master planning failed; the default plan ran
    FAILED = "failed"


class Versions(DomainModel):
    """Everything a session ran with, for reproducibility (PDF §3.4)."""

    config: str | None = None
    prompts: dict[str, str] = Field(default_factory=dict)
    llm_deployment: str | None = None
    ml_model: str | None = None


class SessionResult(DomainModel):
    session_id: NonEmptyStr
    business_date: date
    data_type: NonEmptyStr
    status: SessionStatus = SessionStatus.RUNNING
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    versions: Versions = Field(default_factory=Versions)
    plan: Plan | None = None
    tasks: tuple[Task, ...] = ()
    review_checks: tuple[ReviewCheckResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    findings: tuple[Finding, ...] = ()
    fallbacks: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        if self.status is SessionStatus.RUNNING and self.finished_at is not None:
            raise ValueError("a running session has no finished_at")
        fallback_statuses = (SessionStatus.COMPLETED_DEGRADED, SessionStatus.DEFAULT_COMPLETE)
        if self.status in fallback_statuses and not self.fallbacks:
            raise ValueError(f"a {self.status} session must list the fallbacks it used")
        return self

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

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import NonEmptyStr
from mdx_assessment.core.domain.findings import Finding
from mdx_assessment.core.domain.planning import Plan, PlanSource, Task, TaskStatus
from mdx_assessment.core.domain.review import (
    Decision,
    ReviewCheckResult,
    RunOpinion,
    RunReview,
    StepValidation,
)
from mdx_assessment.core.domain.run_context import RunContext


class SessionStatus(StrEnum):
    """The run's lifecycle, as the team's `run_requests` status (adb-mdx-agent ADR-0001).

    It says only whether the run finished. How well it went is elsewhere: the plan's
    `source`, each step's `TaskStatus`, `review.reliability` and `fallbacks`.
    """

    PENDING = "pending"  # queued, not claimed yet (unused while the api runs sessions inline)
    RUNNING = "running"
    SUCCEEDED = "succeeded"  # the run reached its end, whatever its steps' statuses
    FAILED = "failed"  # the run stopped before its end


class Versions(DomainModel):
    """Everything a session ran with, for reproducibility (PDF §3.4)."""

    config: str | None = None
    prompts: dict[str, str] = Field(default_factory=dict)
    llm_deployment: str | None = None
    ml_model: str | None = None


class TraceRef(DomainModel):
    mlflow_trace_id: str
    url: str | None = None


class SessionResult(DomainModel):
    session_id: NonEmptyStr
    business_date: date
    data_type: NonEmptyStr
    llm_enabled: bool = True
    status: SessionStatus = SessionStatus.RUNNING
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    versions: Versions = Field(default_factory=Versions)
    run_context: RunContext = Field(default_factory=RunContext)
    playbook: dict[str, JsonValue] | None = None
    plan: Plan | None = None
    execution: tuple[Task, ...] = ()
    agent_results: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    review_checks: tuple[ReviewCheckResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    validations: tuple[StepValidation, ...] = ()
    """The Master's per-item review of each step that has one (E10-05)."""
    findings: tuple[Finding, ...] = ()
    run_opinion: RunOpinion | None = None
    review: RunReview | None = None
    fallbacks: tuple[str, ...] = ()
    analyst_notice: str | None = None
    trace: TraceRef | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        unfinished = (SessionStatus.PENDING, SessionStatus.RUNNING)
        if self.status in unfinished and self.finished_at is not None:
            raise ValueError(f"a {self.status} session has no finished_at")
        degraded = any(
            t.status in (TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.DONE_WITH_FALLBACK)
            for t in self.execution
        ) or (self.plan is not None and self.plan.source is PlanSource.DEFAULT)
        if self.status is SessionStatus.SUCCEEDED and degraded and not self.fallbacks:
            raise ValueError("a session that used the default plan or a fallback must list it")
        return self

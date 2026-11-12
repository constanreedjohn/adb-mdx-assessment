"""Plans and tasks: what the Master Agent schedules and the dispatcher runs (plan §2.5, §2.6)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.evidence import NonEmptyStr


class PlanSource(StrEnum):
    MASTER = "master"  # produced by the Master Agent and validated
    DEFAULT = "default"  # the selected playbook's default plan (safe fallback)


class PlanStep(DomainModel):
    """One step of a plan, naming a registered task (by ID, never by code reference)."""

    step_id: NonEmptyStr
    task: NonEmptyStr
    rationale: str = ""
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    """What a follow-up step works on, e.g. the flags a send-back names (E6-05). The task
    validates them; planned steps have none."""


class PlanValidation(DomainModel):
    """How a plan fared against P-1 to P-4 (plan §2.4). `valid` is False on a default plan."""

    valid: bool
    checks: dict[str, bool]
    attempts: int = Field(ge=0)
    errors: tuple[str, ...] = ()
    fallback_reason: str | None = None


class Plan(DomainModel):
    source: PlanSource
    steps: tuple[PlanStep, ...] = Field(min_length=1)
    follow_ups: tuple[PlanStep, ...] = ()
    rationale: str = ""
    cited_fact_ids: tuple[str, ...] = ()
    validation: PlanValidation | None = None

    @model_validator(mode="after")
    def _unique_step_ids(self) -> Self:
        ids = [s.step_id for s in (*self.steps, *self.follow_ups)]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate step_id(s): {duplicates}")
        return self


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    DONE_WITH_FALLBACK = "done_with_fallback"
    """Done, but part of the output is a code template because the LLM's wasn't usable."""
    FAILED = "failed"
    SKIPPED = "skipped"
    """Not run: a step it depends on failed or was skipped."""


class Task(DomainModel):
    """A dispatched instance of a registered task within one session."""

    task_id: NonEmptyStr
    session_id: NonEmptyStr
    step_id: NonEmptyStr
    task: NonEmptyStr  # registered task ID
    status: TaskStatus = TaskStatus.PENDING
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _consistent_status(self) -> Self:
        if self.status is TaskStatus.FAILED and not self.error:
            raise ValueError("a failed task needs an error message")
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        return self

    @property
    def duration_seconds(self) -> float | None:
        start: datetime | None = self.started_at
        end: datetime | None = self.finished_at
        return (end - start).total_seconds() if start and end else None

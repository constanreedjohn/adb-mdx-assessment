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


class Plan(DomainModel):
    source: PlanSource
    steps: tuple[PlanStep, ...] = Field(min_length=1)
    follow_ups: tuple[PlanStep, ...] = ()
    rationale: str = ""
    cited_fact_ids: tuple[str, ...] = ()

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
    FAILED = "failed"


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

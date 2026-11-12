"""Shared state for one session (PDF §2.1: one LangGraph state per session, checkpointed).

Agents hand work to each other through this state; the dispatcher (E6-02) moves the
`next_step` cursor; the Master Agent reads it to plan, review and decide. It is
checkpointed to PostgreSQL after every step, so an interrupted session resumes where
it stopped without repeating a step.

Use-case results live
in `agent_results` as JSON, so core never needs a pack's types.

List fields use an "append" reducer: a node returns only its new items, and LangGraph
adds them to what is already there. `agent_results` merges by key (step ID).
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from mdx_assessment.core.domain import (
    Decision,
    DomainModel,
    Plan,
    PlanStep,
    ReviewCheckResult,
    RunContext,
    RunOpinion,
    RunReview,
    SessionStatus,
    StepValidation,
    TaskStatus,
    Versions,
)
from mdx_assessment.core.orchestration.playbooks import ResolvedPlaybook


def merge_results(
    current: dict[str, dict[str, JsonValue]], update: dict[str, dict[str, JsonValue]]
) -> dict[str, dict[str, JsonValue]]:
    """Reducer for `agent_results`: later results for the same step replace earlier ones."""
    return {**current, **update}


class ExecutionEntry(DomainModel):
    """One plan step as the dispatcher ran it (the execution log, plan §2.4)."""

    step_id: str
    task: str
    status: TaskStatus
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _failed_has_error(self) -> Self:
        if self.status is TaskStatus.FAILED and not self.error:
            raise ValueError("a failed step needs an error message")
        return self


class RunState(BaseModel):
    """The LangGraph state of one session. Thread ID = session ID."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64)
    business_date: date
    data_type: str = Field(min_length=1)
    versions: Versions = Field(default_factory=Versions)
    llm_enabled: bool = True
    """False runs the default plan with no LLM call: the analyst turned the LLM off."""

    run_context: RunContext = Field(default_factory=RunContext)
    """Facts and observations gathered by code at session start."""
    playbook: ResolvedPlaybook | None = None
    """The selected playbook and its selection record."""

    plan: Plan | None = None
    next_step: int = Field(default=0, ge=0)
    """Index of the next plan step the dispatcher will run (E6-02)."""
    queued_step: PlanStep | None = None
    """A follow-up a review point decided (a send-back, E6-05). It runs before the next
    plan step, then gets its own review point."""
    execution: Annotated[list[ExecutionEntry], operator.add] = Field(default_factory=list)
    agent_results: Annotated[dict[str, dict[str, JsonValue]], merge_results] = Field(
        default_factory=dict
    )
    """Each step's result, keyed by step ID (JSON; the pack defines the shape)."""

    review_checks: Annotated[list[ReviewCheckResult], operator.add] = Field(default_factory=list)
    decisions: Annotated[list[Decision], operator.add] = Field(default_factory=list)
    validations: Annotated[list[StepValidation], operator.add] = Field(default_factory=list)
    """The Master's per-item review of each step whose task has a validator (E10-05)."""
    run_opinion: RunOpinion | None = None
    """The Master Agent's run opinion (E10-02)."""
    review: RunReview | None = None
    """The run's reliability and what needs a human, set at the end (E10-01)."""

    status: SessionStatus = SessionStatus.RUNNING
    analyst_notice: str | None = None
    """Set when the run stopped and an analyst must be told, e.g. the model is unreachable."""
    fallbacks: Annotated[list[str], operator.add] = Field(default_factory=list)

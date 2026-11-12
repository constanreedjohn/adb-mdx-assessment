"""Task, agent and review-check registries (plan §2.6, §2.7).

A use-case pack registers what its agents can do; core runs it without naming it.
The dispatcher looks a plan step's task up here, the Master Agent's catalogue is
built from here (E6-04), and review points run the checks a task declares.

    registries.agents.register(AgentDefinition("input_validation", "Checks inputs ..."))
    registries.tasks.register(TaskDefinition(
        id="input_validation.check_files_and_intake", agent="input_validation",
        description="...", output_model=IntakeResult, handler=check_files_and_intake,
    ))
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, JsonValue

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import PlanStep, ReviewCheckResult, StepValidation
from mdx_assessment.core.gateway.gateway import LLMGateway
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState
from mdx_assessment.core.tools.client import AgentToolClient, TaskBudget


class TaskInput(BaseModel):
    """What every task receives about its run (more comes from `TaskContext.state`)."""

    session_id: str
    business_date: date
    data_type: str
    step_id: str


@dataclass(frozen=True)
class TaskContext:
    """Everything a task handler may use. Tools are pre-bound to the task's own agent."""

    inputs: TaskInput
    state: RunState
    step: PlanStep
    tools_client: AgentToolClient | None
    gateway: LLMGateway | None
    loaded_config: LoadedConfig | None = None

    @property
    def config(self) -> LoadedConfig:
        """The session's validated `config/` (the same version the session records)."""
        if self.loaded_config is None:
            raise RuntimeError("no configuration is available to this run")
        return self.loaded_config

    @property
    def tools(self) -> AgentToolClient:
        if self.tools_client is None:
            raise RuntimeError("no tools are available to this run")
        return self.tools_client

    def result_of(self, task_id: str) -> dict[str, JsonValue] | None:
        """The latest successful result of an earlier step that ran `task_id`."""
        for entry in reversed(self.state.execution):
            if entry.task == task_id and entry.status.value == "done":
                return self.state.agent_results.get(entry.step_id)
        return None


TaskHandler = Callable[[TaskContext], Awaitable[BaseModel]]


@dataclass(frozen=True)
class TaskDefinition:
    """A registered task (satisfies `core.registry.TaskSpec`)."""

    id: str
    agent: str
    description: str
    output_model: type[BaseModel]
    handler: TaskHandler
    input_model: type[BaseModel] = TaskInput
    review_checks: tuple[str, ...] = ()
    budget: TaskBudget = field(default_factory=TaskBudget)
    prompts: tuple[tuple[str, str], ...] = ()
    """(prompt ID, version) pairs the task may call, recorded with every session."""
    depends_on: tuple[str, ...] = ()
    """Task IDs whose output this task needs. If the latest run of one failed or was
    skipped, the dispatcher skips this task instead of running it on missing input."""
    validate: ItemValidator | None = None
    """The Master's per-item review of this task's output (E10-05), run at its review
    point after the checks. It may propose a send-back."""


@dataclass(frozen=True)
class ValidationContext:
    """What a validator may use at a review point."""

    state: RunState
    entry: ExecutionEntry
    gateway: LLMGateway | None
    loaded_config: LoadedConfig | None
    send_back_refusal: Callable[[PlanStep], str | None]
    """Why a send-back with this step would be refused (P-2, P-3), or None if it can run."""

    @property
    def config(self) -> LoadedConfig:
        if self.loaded_config is None:
            raise RuntimeError("no configuration is available to this run")
        return self.loaded_config


@dataclass(frozen=True)
class ValidationOutcome:
    validation: StepValidation
    send_back: PlanStep | None = None
    """The follow-up to run for the items sent back, if any."""


ItemValidator = Callable[[ValidationContext], Awaitable[ValidationOutcome]]


USED_FALLBACK = "used_fallback"
"""A task's output may carry `used_fallback: bool`. True marks the step
`done_with_fallback`: done, but part of it is a code template instead of the LLM's output."""


@dataclass(frozen=True)
class AgentDefinition:
    """A registered agent (satisfies `core.registry.AgentSpec`)."""

    id: str
    description: str


ReviewCheckFn = Callable[[RunState, ExecutionEntry], ReviewCheckResult]


@dataclass(frozen=True)
class ReviewCheckDefinition:
    id: str  # e.g. "C-1"
    name: str
    check: ReviewCheckFn


@dataclass
class AgentRegistry:
    _agents: dict[str, AgentDefinition] = field(default_factory=dict)

    def register(self, agent: AgentDefinition) -> None:
        if agent.id in self._agents:
            raise ValueError(f"agent {agent.id!r} is already registered")
        self._agents[agent.id] = agent

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

    def __iter__(self) -> Iterator[AgentDefinition]:
        return iter(self._agents.values())


@dataclass
class ReviewCheckRegistry:
    _checks: dict[str, ReviewCheckDefinition] = field(default_factory=dict)

    def register(self, check: ReviewCheckDefinition) -> None:
        if check.id in self._checks:
            raise ValueError(f"review check {check.id!r} is already registered")
        self._checks[check.id] = check

    def get(self, check_id: str) -> ReviewCheckDefinition:
        try:
            return self._checks[check_id]
        except KeyError:
            raise KeyError(f"unknown review check {check_id!r}") from None

    def __contains__(self, check_id: object) -> bool:
        return check_id in self._checks


@dataclass
class TaskRegistry:
    _tasks: dict[str, TaskDefinition] = field(default_factory=dict)

    def register(self, task: TaskDefinition) -> None:
        if task.id in self._tasks:
            raise ValueError(f"task {task.id!r} is already registered")
        self._tasks[task.id] = task

    def get(self, task_id: str) -> TaskDefinition | None:
        return self._tasks.get(task_id)

    def __iter__(self) -> Iterator[TaskDefinition]:
        return iter(self._tasks.values())

    def catalogue(self) -> list[dict[str, Any]]:
        """The tasks as data, for the Master Agent's planning prompt (E6-04)."""
        return [
            {"task": t.id, "agent": t.agent, "description": t.description}
            for t in self._tasks.values()
        ]


def check_consistency(
    tasks: TaskRegistry, agents: AgentRegistry, checks: ReviewCheckRegistry
) -> list[str]:
    """Problems between the registries (empty when consistent)."""
    problems = []
    for task in tasks:
        if task.agent not in agents:
            problems.append(f"task {task.id!r}: unknown agent {task.agent!r}")
        problems += [
            f"task {task.id!r}: unknown review check {c!r}"
            for c in task.review_checks
            if c not in checks
        ]
    return problems

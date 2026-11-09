"""The session graph: plan-driven dispatcher with a review point after every step (plan §2.6).

    START → run_context → select_playbook → master_plan ─┬─→ dispatch → review_point ─┐
                                                        │        ↑                    │ more steps
                                                        │        └────────────────────┘
                                                        └──────────→ master_synthesize → END

- **The Master decides, code executes.** The decision points (gather context, select
  the playbook, plan, decide after each step, synthesize) are `GraphHooks`, filled in
  by later stories (E6-03, E6-04, E10-01, E10-02). The graph itself never changes:
  new steps, agents and playbooks are registered, not wired in (plan §2.7).
- **The dispatcher** runs the step at `next_step` by looking its task up in the task
  registry, gives it a tool client bound to the task's own agent (so the permission
  matrix applies), validates its output and records an execution-log entry.
- **Safe failure.** An unknown task, a crash or invalid output fails that step only:
  it is recorded with its error, flagged at its review point, and the run continues
  and completes, marked degraded (PDF §3.3).
- Every plan, task and decision is an audit event.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import JsonValue, ValidationError

from mdx_assessment.core.domain import (
    Decision,
    DecisionKind,
    NewAuditEvent,
    Plan,
    ReviewCheckResult,
    RunContext,
    SessionStatus,
    TaskStatus,
)
from mdx_assessment.core.gateway.gateway import LLMGateway
from mdx_assessment.core.observability.tracing import Tracer
from mdx_assessment.core.orchestration.playbooks import ResolvedPlaybook
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState
from mdx_assessment.core.orchestration.tasks import (
    ReviewCheckRegistry,
    TaskContext,
    TaskInput,
    TaskRegistry,
)
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.tools.client import ToolHub

MASTER_AGENT = "master_agent"
ORCHESTRATOR = "orchestrator"
Clock = Callable[[], datetime]

ContextHook = Callable[[RunState], Awaitable[RunContext]]
PlaybookHook = Callable[[RunState], Awaitable[ResolvedPlaybook | None]]
PlanHook = Callable[[RunState], Awaitable[Plan]]
DecideHook = Callable[[RunState, ExecutionEntry, list[ReviewCheckResult]], Awaitable[Decision]]
SynthesizeHook = Callable[[RunState], Awaitable[dict[str, JsonValue] | None]]


async def _no_context(state: RunState) -> RunContext:
    return RunContext()


async def _no_playbook(state: RunState) -> ResolvedPlaybook | None:
    return None


async def accept_unless_failed(
    state: RunState, entry: ExecutionEntry, checks: list[ReviewCheckResult]
) -> Decision:
    """Stand-in decide step until the Master's review (E10-01): accept, or flag a failure."""
    failed_checks = [c.check_id for c in checks if not c.passed]
    if entry.status is TaskStatus.FAILED or failed_checks:
        reason = entry.error or f"checks failed: {', '.join(failed_checks)}"
        return Decision(step_id=entry.step_id, kind=DecisionKind.FLAG_FOR_HUMAN, reason=reason)
    return Decision(
        step_id=entry.step_id, kind=DecisionKind.ACCEPT, reason="step done, checks passed"
    )


async def _no_opinion(state: RunState) -> dict[str, JsonValue] | None:
    return None


@dataclass(frozen=True)
class GraphHooks:
    """The decision points. `plan` is required; the rest default to stand-ins."""

    plan: PlanHook
    gather_context: ContextHook = _no_context
    select_playbook: PlaybookHook = _no_playbook
    decide: DecideHook = accept_unless_failed
    synthesize: SynthesizeHook = _no_opinion


@dataclass
class SessionRuntime:
    """What the graph needs at run time. Not part of the checkpointed state: rebuilt on resume."""

    tasks: TaskRegistry
    checks: ReviewCheckRegistry
    audit: AuditTrail
    hub: ToolHub | None = None
    gateway: LLMGateway | None = None
    tracer: Tracer = field(default_factory=lambda: Tracer(None))
    clock: Clock = field(default=lambda: datetime.now(UTC))

    async def record(
        self, state: RunState, actor: str, event_type: str, payload: dict[str, JsonValue]
    ) -> None:
        event = NewAuditEvent(
            session_id=state.session_id, actor=actor, event_type=event_type, payload=payload
        )
        await asyncio.to_thread(self.audit.append, event)


def build_session_graph(runtime: SessionRuntime, hooks: GraphHooks) -> StateGraph[RunState]:
    """The graph for one session. Compile it with a checkpointer to run or resume."""

    async def run_context(state: RunState) -> dict[str, Any]:
        update: dict[str, Any] = {}
        try:
            context = await hooks.gather_context(state)
        except Exception as exc:  # safe failure: no facts, so the default playbook is selected
            context = RunContext()
            update["fallbacks"] = [f"run_context_failed:{type(exc).__name__}"]
        update["run_context"] = context
        await runtime.record(
            state,
            ORCHESTRATOR,
            "run_context",
            {
                "facts": [f.model_dump(mode="json") for f in context.facts],
                "observations": [o.model_dump(mode="json") for o in context.observations],
            },
        )
        return update

    async def select_playbook(state: RunState) -> dict[str, Any]:
        playbook = await hooks.select_playbook(state)
        if playbook is not None:
            await runtime.record(
                state,
                ORCHESTRATOR,
                "playbook_selected",
                {
                    "playbook": playbook.id,
                    "version": playbook.version,
                    "selected_by": playbook.selected_by,
                    "selection": playbook.selection.model_dump(mode="json"),
                },
            )
        return {"playbook": playbook}

    async def master_plan(state: RunState) -> dict[str, Any]:
        plan = await hooks.plan(state)
        await runtime.record(
            state,
            MASTER_AGENT,
            "plan",
            {
                "source": plan.source.value,
                "steps": [{"step_id": s.step_id, "task": s.task} for s in plan.steps],
                "follow_ups": [s.task for s in plan.follow_ups],
                "cited_fact_ids": list(plan.cited_fact_ids),
            },
        )
        return {"plan": plan, "next_step": 0}

    async def dispatch(state: RunState) -> dict[str, Any]:
        assert state.plan is not None  # guaranteed by routing
        step = state.plan.steps[state.next_step]
        started_at, started = runtime.clock(), time.perf_counter()
        status, error, result = TaskStatus.DONE, None, None
        definition = runtime.tasks.get(step.task)
        if definition is None:
            status, error = TaskStatus.FAILED, f"unknown task {step.task!r}"
        else:
            context = TaskContext(
                inputs=TaskInput(
                    session_id=state.session_id,
                    business_date=state.business_date,
                    data_type=state.data_type,
                    step_id=step.step_id,
                ),
                state=state,
                step=step,
                tools_client=(
                    runtime.hub.client(
                        definition.agent,
                        session_id=state.session_id,
                        task_id=f"{state.session_id}/{step.step_id}",
                        budget=definition.budget,
                    )
                    if runtime.hub is not None
                    else None
                ),
                gateway=runtime.gateway,
            )
            try:
                with runtime.tracer.span(
                    f"task:{step.task}", kind="chain", metadata={"step_id": step.step_id}
                ):
                    output = await definition.handler(context)
                result = definition.output_model.model_validate(output.model_dump()).model_dump(
                    mode="json"
                )
            except ValidationError as exc:
                status, error = TaskStatus.FAILED, f"invalid output: {exc.error_count()} error(s)"
            except Exception as exc:
                status, error = TaskStatus.FAILED, f"{type(exc).__name__}: {exc}"[:500]

        entry = ExecutionEntry(
            step_id=step.step_id,
            task=step.task,
            status=status,
            started_at=started_at,
            finished_at=runtime.clock(),
            error=error,
        )
        await runtime.record(
            state,
            definition.agent if definition else ORCHESTRATOR,
            "task_completed",
            {
                "step_id": step.step_id,
                "task": step.task,
                "status": status.value,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                **({"error": error} if error else {}),
            },
        )
        update: dict[str, Any] = {"execution": [entry], "next_step": state.next_step + 1}
        if result is not None:
            update["agent_results"] = {step.step_id: result}
        return update

    async def review_point(state: RunState) -> dict[str, Any]:
        entry = state.execution[-1]
        definition = runtime.tasks.get(entry.task)
        check_ids = definition.review_checks if definition else ()
        results = [runtime.checks.get(c).check(state, entry) for c in check_ids]
        decision = await hooks.decide(state, entry, results)
        await runtime.record(
            state,
            MASTER_AGENT,
            "decision",
            {
                "step_id": entry.step_id,
                "kind": decision.kind.value,
                "reason": decision.reason,
                "checks": {r.check_id: r.passed for r in results},
            },
        )
        return {"review_checks": results, "decisions": [decision]}

    async def master_synthesize(state: RunState) -> dict[str, Any]:
        failed = [e.step_id for e in state.execution if e.status is TaskStatus.FAILED]
        update: dict[str, Any] = {"run_opinion": await hooks.synthesize(state)}
        if failed:
            update["status"] = SessionStatus.COMPLETED_DEGRADED
            update["fallbacks"] = [f"step_failed:{s}" for s in failed]
        else:
            update["status"] = SessionStatus.COMPLETED
        payload: dict[str, JsonValue] = {
            "status": update["status"].value,
            "steps": len(state.execution),
            "failed": [step_id for step_id in failed],
        }
        await runtime.record(state, MASTER_AGENT, "session_completed", payload)
        return update

    def after_plan(state: RunState) -> Literal["dispatch", "master_synthesize"]:
        return "dispatch" if state.plan and state.plan.steps else "master_synthesize"

    def after_review(state: RunState) -> Literal["dispatch", "master_synthesize"]:
        assert state.plan is not None
        return "dispatch" if state.next_step < len(state.plan.steps) else "master_synthesize"

    graph: StateGraph[RunState] = StateGraph(RunState)
    graph.add_node("run_context", run_context)
    graph.add_node("select_playbook", select_playbook)
    graph.add_node("master_plan", master_plan)
    graph.add_node("dispatch", dispatch)
    graph.add_node("review_point", review_point)
    graph.add_node("master_synthesize", master_synthesize)
    graph.add_edge(START, "run_context")
    graph.add_edge("run_context", "select_playbook")
    graph.add_edge("select_playbook", "master_plan")
    graph.add_conditional_edges("master_plan", after_plan)
    graph.add_edge("dispatch", "review_point")
    graph.add_conditional_edges("review_point", after_review)
    graph.add_edge("master_synthesize", END)
    return graph

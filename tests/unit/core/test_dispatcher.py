"""The plan-driven dispatcher and review points, with toy tasks (no use-case pack, no tools)."""

import logging
from datetime import UTC, date, datetime
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, JsonValue

from mdx_assessment.core.domain import (
    DecisionKind,
    Fact,
    Plan,
    PlanSource,
    PlanStep,
    ReviewCheckResult,
    RunContext,
    SessionStatus,
    TaskStatus,
)
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.playbooks import (
    PlaybookContent,
    PlaybookLimits,
    ResolvedPlaybook,
    SelectionRecord,
)
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState
from mdx_assessment.core.orchestration.tasks import (
    AgentDefinition,
    AgentRegistry,
    ReviewCheckDefinition,
    ReviewCheckRegistry,
    TaskContext,
    TaskDefinition,
    TaskRegistry,
    check_consistency,
)
from mdx_assessment.core.persistence import InMemoryAuditTrail

pytestmark = pytest.mark.anyio
T0 = datetime(2026, 10, 15, 9, 20, tzinfo=UTC)
PLAYBOOK = ResolvedPlaybook(
    id="normal",
    version="1",
    selection=SelectionRecord(rules=()),
    content=PlaybookContent(
        title="Normal",
        description="",
        mandatory_steps=("toy.produce",),
        allowed_follow_ups=(),
        limits=PlaybookLimits(max_follow_up_rounds=0, max_run_seconds=60, max_ai_explanations=1),
        default_plan=(PlanStep(step_id="s1", task="toy.produce"),),
    ),
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Count(BaseModel):
    n: int


class Other(BaseModel):
    text: str


async def produce(ctx: TaskContext) -> BaseModel:
    previous = ctx.result_of("toy.produce")
    return Count(n=(previous or {}).get("n", 0) + 1)  # type: ignore[operator]


async def explode(ctx: TaskContext) -> BaseModel:
    raise RuntimeError("tool unreachable")


async def wrong_type(ctx: TaskContext) -> BaseModel:
    return Other(text="not a Count")


def always_fails(state: RunState, entry: ExecutionEntry) -> ReviewCheckResult:
    return ReviewCheckResult(check_id="C-X", name="always fails", passed=False, details="test")


def registries() -> tuple[TaskRegistry, ReviewCheckRegistry]:
    tasks, checks = TaskRegistry(), ReviewCheckRegistry()
    checks.register(ReviewCheckDefinition("C-X", "always fails", always_fails))
    for task_id, handler, check_ids in (
        ("toy.produce", produce, ()),
        ("toy.explode", explode, ()),
        ("toy.wrong_type", wrong_type, ()),
        ("toy.checked", produce, ("C-X",)),
    ):
        tasks.register(
            TaskDefinition(
                id=task_id,
                agent="toy_agent",
                description=task_id,
                output_model=Count,
                handler=handler,
                review_checks=check_ids,
            )
        )
    return tasks, checks


def plan_of(*task_ids: str) -> Plan:
    return Plan(
        source=PlanSource.DEFAULT,
        steps=tuple(PlanStep(step_id=f"s{i}", task=t) for i, t in enumerate(task_ids, 1)),
    )


async def run(
    plan: Plan, hooks: dict[str, Any] | None = None
) -> tuple[RunState, InMemoryAuditTrail]:
    tasks, checks = registries()
    audit = InMemoryAuditTrail()
    runtime = SessionRuntime(tasks=tasks, checks=checks, audit=audit, clock=lambda: T0)

    async def fixed_plan(state: RunState) -> Plan:
        return plan

    graph = build_session_graph(runtime, GraphHooks(plan=fixed_plan, **(hooks or {})))
    app = graph.compile(checkpointer=InMemorySaver(serde=checkpoint_serializer()))
    initial = RunState(session_id="sess-1", business_date=date(2026, 10, 15), data_type="linear")
    final = RunState.model_validate(await app.ainvoke(initial, thread_config("sess-1")))
    return final, audit


# --- acceptance: a plan of three stub steps runs end to end ----------------------------------


async def test_three_steps_run_in_order_and_are_logged() -> None:
    final, audit = await run(plan_of("toy.produce", "toy.produce", "toy.produce"))
    assert [(e.step_id, e.task, e.status) for e in final.execution] == [
        ("s1", "toy.produce", TaskStatus.DONE),
        ("s2", "toy.produce", TaskStatus.DONE),
        ("s3", "toy.produce", TaskStatus.DONE),
    ]
    assert final.agent_results == {
        "s1": {"n": 1},
        "s2": {"n": 2},
        "s3": {"n": 3},
    }  # each saw the last
    assert [d.kind for d in final.decisions] == [DecisionKind.ACCEPT] * 3
    assert final.status is SessionStatus.SUCCEEDED
    assert final.next_step == 3
    assert [e.event_type for e in audit.read()] == [
        "run_context",
        "plan",
        "task_completed",
        "decision",
        "task_completed",
        "decision",
        "task_completed",
        "decision",
        "session_completed",
    ]
    assert audit.verify().valid


# --- safe failure: one bad step doesn't stop the run ----------------------------------------


@pytest.mark.parametrize(
    ("bad_task", "error"),
    [
        ("toy.missing", "unknown task 'toy.missing'"),
        ("toy.explode", "RuntimeError: tool unreachable"),
        ("toy.wrong_type", "invalid output"),
    ],
)
async def test_a_failed_step_is_flagged_and_the_run_completes(bad_task: str, error: str) -> None:
    final, _ = await run(plan_of("toy.produce", bad_task, "toy.produce"))
    statuses = [e.status for e in final.execution]
    assert statuses == [TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.DONE]
    assert error in (final.execution[1].error or "")
    assert final.decisions[1].kind is DecisionKind.FLAG_FOR_HUMAN
    assert "s2" not in final.agent_results
    assert final.status is SessionStatus.SUCCEEDED
    assert final.fallbacks == ["step_failed:s2"]


async def test_a_crash_is_logged_with_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="mdx_assessment.core.orchestration.graph"):
        await run(plan_of("toy.explode"))
    assert "step s1 (toy.explode) crashed" in caplog.text
    assert "RuntimeError: tool unreachable" in caplog.text
    assert "Traceback" in caplog.text


async def test_invalid_output_names_the_fields(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="mdx_assessment.core.orchestration.graph"):
        final, _ = await run(plan_of("toy.wrong_type"))
    assert "n: Field required" in (final.execution[0].error or "")
    assert "returned invalid output: n: Field required" in caplog.text


# --- review points ---------------------------------------------------------------------------


async def test_registered_checks_run_at_the_review_point() -> None:
    final, audit = await run(plan_of("toy.produce", "toy.checked"))
    assert [c.check_id for c in final.review_checks] == ["C-X"]
    assert final.decisions[1].kind is DecisionKind.FLAG_FOR_HUMAN
    assert "C-X" in final.decisions[1].reason
    decision_events = [e for e in audit.read() if e.event_type == "decision"]
    assert decision_events[1].payload["checks"] == {"C-X": False}


# --- hooks -------------------------------------------------------------------------------------


async def test_hooks_fill_the_state() -> None:
    fact = Fact.create("flag_count", 28, source="test", scope={})

    async def context(state: RunState) -> RunContext:
        return RunContext(facts=(fact,))

    async def playbook(state: RunState) -> ResolvedPlaybook | None:
        return PLAYBOOK

    async def opinion(state: RunState) -> dict[str, JsonValue] | None:
        return {"summary": f"{len(state.execution)} steps"}

    final, audit = await run(
        plan_of("toy.produce"),
        {"gather_context": context, "select_playbook": playbook, "synthesize": opinion},
    )
    assert final.run_context.fact("flag_count") == fact
    assert final.playbook == PLAYBOOK
    assert final.run_opinion == {"summary": "1 steps"}
    events = [e.event_type for e in audit.read()]
    assert events[:3] == ["run_context", "playbook_selected", "plan"]


async def test_failing_to_gather_facts_stops_the_run(caplog: pytest.LogCaptureFixture) -> None:
    async def broken(state: RunState) -> RunContext:
        raise RuntimeError("MDX is down")

    with caplog.at_level(logging.ERROR, logger="mdx_assessment.core.orchestration.graph"):
        final, audit = await run(plan_of("toy.produce"), {"gather_context": broken})
    assert final.status is SessionStatus.FAILED
    assert final.analyst_notice is not None and "source system" in final.analyst_notice
    assert "MDX is down" not in final.analyst_notice
    assert final.execution == []
    assert final.plan is None
    assert "MDX is down" in caplog.text
    assert "Traceback" in caplog.text
    assert audit.read() == []


async def test_an_empty_plan_goes_straight_to_synthesis() -> None:
    final, audit = await run(
        Plan.model_construct(source=PlanSource.DEFAULT, steps=(), follow_ups=())
    )
    assert final.execution == []
    assert final.status is SessionStatus.SUCCEEDED
    assert [e.event_type for e in audit.read()] == ["run_context", "plan", "session_completed"]


# --- registry consistency -----------------------------------------------------------------------


def test_consistency_reports_unknown_agents_and_checks() -> None:
    tasks, checks = registries()
    tasks.register(
        TaskDefinition(
            id="toy.orphan",
            agent="ghost",
            description="",
            output_model=Count,
            handler=produce,
            review_checks=("C-404",),
        )
    )
    agents = AgentRegistry()
    agents.register(AgentDefinition("toy_agent", "toy"))
    assert check_consistency(tasks, agents, checks) == [
        "task 'toy.orphan': unknown agent 'ghost'",
        "task 'toy.orphan': unknown review check 'C-404'",
    ]


def test_catalogue_is_data() -> None:
    tasks, _ = registries()
    assert tasks.catalogue()[0] == {
        "task": "toy.produce",
        "agent": "toy_agent",
        "description": "toy.produce",
    }

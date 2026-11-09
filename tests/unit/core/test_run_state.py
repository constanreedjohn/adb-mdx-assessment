"""RunState and the strict checkpoint serializer, without a database (in-memory checkpointer)."""

from collections import Counter
from datetime import UTC, date, datetime
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mdx_assessment.core.domain import (
    Decision,
    DecisionKind,
    Plan,
    PlanSource,
    PlanStep,
    ReviewCheckResult,
    SessionStatus,
    TaskStatus,
    Versions,
)
from mdx_assessment.core.orchestration.checkpoint import (
    checkpoint_serializer,
    model_types,
    thread_config,
)
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings("error:Deserializing unregistered type"),
]
STEPS = ("step_1", "step_2", "step_3")
T0 = datetime(2026, 10, 15, 9, 20, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def initial(session_id: str = "sess-1") -> RunState:
    return RunState(
        session_id=session_id,
        business_date=date(2026, 10, 15),
        data_type="linear",
        versions=Versions(config="sha256:abc", prompts={"master_plan": "v1"}),
        plan=Plan(
            source=PlanSource.DEFAULT,
            steps=tuple(PlanStep(step_id=s, task=f"stub.{s}") for s in STEPS),
        ),
    )


def build_graph(runs: Counter[str]) -> StateGraph[RunState]:
    graph: StateGraph[RunState] = StateGraph(RunState)

    def make(step: str) -> Any:
        async def node(state: RunState) -> dict[str, Any]:
            runs[step] += 1
            return {
                "execution": [
                    ExecutionEntry(
                        step_id=step,
                        task=f"stub.{step}",
                        status=TaskStatus.DONE,
                        started_at=T0,
                        finished_at=T0,
                    )
                ],
                "agent_results": {step: {"ok": True, "items": runs[step]}},
                "review_checks": [ReviewCheckResult(check_id="C-1", name="reconcile", passed=True)],
                "decisions": [
                    Decision(step_id=step, kind=DecisionKind.ACCEPT, reason="checks passed")
                ],
                "next_step": state.next_step + 1,
            }

        return node

    previous = START
    for step in STEPS:
        graph.add_node(step, make(step))
        graph.add_edge(previous, step)
        previous = step
    graph.add_edge(previous, END)
    return graph


# --- the allow-list ------------------------------------------------------------------------


def test_allow_list_is_derived_from_the_state() -> None:
    names = {t.__name__ for t in model_types(RunState)}
    assert names == {
        "RunState",
        "Versions",
        "Plan",
        "PlanStep",
        "PlanSource",
        "PlanValidation",
        "ExecutionEntry",
        "TaskStatus",
        "ReviewCheckResult",
        "Decision",
        "DecisionKind",
        "SessionStatus",
        # E6-03: facts and the selected playbook
        "RunContext",
        "Fact",
        "RunObservation",
        "ResolvedPlaybook",
        "SelectionRecord",
        "RuleOutcome",
        "PlaybookContent",
        "PlaybookLimits",
    }


def test_allowed_classes_come_back_as_classes() -> None:
    serde = checkpoint_serializer()
    plan = initial().plan
    assert serde.loads_typed(serde.dumps_typed(plan)) == plan
    assert isinstance(serde.loads_typed(serde.dumps_typed(SessionStatus.COMPLETED)), SessionStatus)


class Intruder(BaseModel):
    payload: str


@pytest.mark.filterwarnings("ignore")
def test_classes_outside_the_allow_list_are_not_rebuilt() -> None:
    serde = checkpoint_serializer()
    restored = serde.loads_typed(serde.dumps_typed(Intruder(payload="x")))
    assert not isinstance(restored, Intruder)


# --- reducers, interrupt and resume (in memory) -----------------------------------------------


async def test_steps_append_and_merge_into_the_state() -> None:
    runs: Counter[str] = Counter()
    app = build_graph(runs).compile(checkpointer=InMemorySaver(serde=checkpoint_serializer()))
    final = RunState.model_validate(await app.ainvoke(initial(), thread_config("sess-1")))
    assert [e.step_id for e in final.execution] == list(STEPS)
    assert set(final.agent_results) == set(STEPS)
    assert len(final.decisions) == len(final.review_checks) == 3
    assert final.next_step == 3
    assert isinstance(final.plan, Plan)


async def test_interrupt_then_resume_runs_each_step_once() -> None:
    runs: Counter[str] = Counter()
    app = build_graph(runs).compile(
        checkpointer=InMemorySaver(serde=checkpoint_serializer()), interrupt_after=["step_2"]
    )
    config = thread_config("sess-1")
    await app.ainvoke(initial(), config)
    snapshot = await app.aget_state(config)
    assert snapshot.next == ("step_3",)
    assert [
        e["step_id"] if isinstance(e, dict) else e.step_id for e in snapshot.values["execution"]
    ] == ["step_1", "step_2"]

    final = RunState.model_validate(await app.ainvoke(None, config))
    assert [e.step_id for e in final.execution] == list(STEPS)
    assert runs == Counter(dict.fromkeys(STEPS, 1))

"""Acceptance (E6-01): interrupt after step N, resume, no step repeated — on PostgreSQL,
across a fresh checkpointer connection (as after a process restart)."""

import uuid
from collections import Counter
from datetime import UTC, date, datetime
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph
from sqlalchemy import URL, Engine, text

from mdx_assessment.core.domain import Plan, PlanSource, PlanStep, TaskStatus
from mdx_assessment.core.orchestration.checkpoint import (
    CHECKPOINT_SCHEMA,
    open_checkpointer,
    thread_config,
)
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState

pytestmark = [
    pytest.mark.db,
    pytest.mark.anyio,
    pytest.mark.filterwarnings("error:Deserializing unregistered type"),
]
STEPS = ("step_1", "step_2", "step_3")
T0 = datetime(2026, 10, 15, 9, 20, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def new_session() -> RunState:
    return RunState(
        session_id=f"sess-{uuid.uuid4().hex[:12]}",
        business_date=date(2026, 10, 15),
        data_type="linear",
        plan=Plan(
            source=PlanSource.DEFAULT,
            steps=tuple(PlanStep(step_id=s, task=f"stub.{s}") for s in STEPS),
        ),
    )


def build_graph(runs: Counter[str], crash_once: set[str] | None = None) -> StateGraph[RunState]:
    crash = crash_once if crash_once is not None else set()
    graph: StateGraph[RunState] = StateGraph(RunState)

    def make(step: str) -> Any:
        async def node(state: RunState) -> dict[str, Any]:
            runs[step] += 1
            if step in crash:
                crash.discard(step)
                raise RuntimeError(f"{step} crashed")
            entry = ExecutionEntry(
                step_id=step,
                task=f"stub.{step}",
                status=TaskStatus.DONE,
                started_at=T0,
                finished_at=T0,
            )
            return {"execution": [entry], "next_step": state.next_step + 1}

        return node

    previous = START
    for step in STEPS:
        graph.add_node(step, make(step))
        graph.add_edge(previous, step)
        previous = step
    graph.add_edge(previous, END)
    return graph


async def test_resume_after_interrupt_on_a_fresh_connection(
    test_db_url: URL, migrated_engine: Engine
) -> None:
    runs: Counter[str] = Counter()
    session = new_session()
    config = thread_config(session.session_id)

    async with open_checkpointer(test_db_url) as saver:  # first process
        app = build_graph(runs).compile(checkpointer=saver, interrupt_after=["step_2"])
        await app.ainvoke(session, config)
        assert (await app.aget_state(config)).next == ("step_3",)

    async with open_checkpointer(test_db_url) as saver:  # "after a restart"
        app = build_graph(runs).compile(checkpointer=saver)
        final = RunState.model_validate(await app.ainvoke(None, config))

    assert [e.step_id for e in final.execution] == list(STEPS)
    assert final.next_step == 3
    assert isinstance(final.plan, Plan)
    assert runs == Counter(dict.fromkeys(STEPS, 1))  # no step repeated


async def test_a_crashed_step_is_retried_alone(test_db_url: URL, migrated_engine: Engine) -> None:
    runs: Counter[str] = Counter()
    crash = {"step_3"}
    session = new_session()
    config = thread_config(session.session_id)
    async with open_checkpointer(test_db_url) as saver:
        app = build_graph(runs, crash).compile(checkpointer=saver)
        with pytest.raises(RuntimeError, match="step_3 crashed"):
            await app.ainvoke(session, config)
        final = RunState.model_validate(await app.ainvoke(None, config))
    assert [e.step_id for e in final.execution] == list(STEPS)
    assert runs == Counter({"step_1": 1, "step_2": 1, "step_3": 2})  # only the failed step re-ran


async def test_sessions_do_not_share_state(test_db_url: URL, migrated_engine: Engine) -> None:
    runs: Counter[str] = Counter()
    a, b = new_session(), new_session()
    async with open_checkpointer(test_db_url) as saver:
        app = build_graph(runs).compile(checkpointer=saver, interrupt_after=["step_1"])
        await app.ainvoke(a, thread_config(a.session_id))
        await app.ainvoke(b, thread_config(b.session_id))
        await app.ainvoke(None, thread_config(a.session_id))  # finish a only
        state_a = await app.aget_state(thread_config(a.session_id))
        state_b = await app.aget_state(thread_config(b.session_id))
    assert len(state_a.values["execution"]) == 3
    assert len(state_b.values["execution"]) == 1
    assert state_b.next == ("step_2",)


async def test_checkpoint_tables_live_in_their_own_schema(
    test_db_url: URL, migrated_engine: Engine
) -> None:
    async with open_checkpointer(test_db_url):
        pass
    with migrated_engine.connect() as conn:
        tables = dict(
            conn.execute(
                text(
                    "SELECT table_name, table_schema FROM information_schema.tables "
                    "WHERE table_name LIKE 'checkpoint%'"
                )
            ).all()
        )
    assert set(tables) >= {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
    assert set(tables.values()) == {CHECKPOINT_SCHEMA}

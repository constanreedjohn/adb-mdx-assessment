"""E6-05 in core: a review point's `follow_up` decision runs as a send-back, with toy tasks.

Acceptance (with the MDX pack in tests/unit/mdx/test_send_back_round.py): a scripted
send-back re-runs the step once; a second send-back is refused.
"""

from datetime import date
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from tests.unit.core.test_dispatcher import T0, Count, plan_of, registries

from mdx_assessment.core.domain import (
    Decision,
    DecisionKind,
    DecisionSource,
    PlanStep,
    ReviewCheckResult,
    SessionStatus,
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
from mdx_assessment.core.orchestration.tasks import TaskContext, TaskDefinition
from mdx_assessment.core.persistence import InMemoryAuditTrail

pytestmark = pytest.mark.anyio
REVISE = "toy.revise"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def playbook(allowed: tuple[str, ...] = (REVISE,), send_backs: int = 1) -> ResolvedPlaybook:
    return ResolvedPlaybook(
        id="normal",
        version="1",
        selection=SelectionRecord(rules=()),
        content=PlaybookContent(
            title="Normal",
            description="",
            mandatory_steps=("toy.produce",),
            allowed_follow_ups=allowed,
            limits=PlaybookLimits(
                max_follow_up_rounds=0,
                max_run_seconds=600,
                max_ai_explanations=1,
                max_send_backs=send_backs,
            ),
            default_plan=(PlanStep(step_id="s1", task="toy.produce"),),
        ),
    )


async def revise(ctx: TaskContext) -> Count:
    return Count(n=len(ctx.step.arguments.get("items", [])))  # type: ignore[arg-type]


def send_back(after: set[str], step_id: str = "r1", task: str = REVISE) -> Any:
    """A decide hook that sends back each step in `after`, naming two items."""

    async def decide(
        state: RunState, entry: ExecutionEntry, checks: list[ReviewCheckResult]
    ) -> Decision:
        if entry.step_id in after:
            n = sum(d.kind is DecisionKind.FOLLOW_UP for d in state.decisions) + 1
            return Decision(
                step_id=entry.step_id,
                kind=DecisionKind.FOLLOW_UP,
                reason="two items need another look",
                source=DecisionSource.LLM_REVIEW,
                follow_up=PlanStep(
                    step_id=step_id if n == 1 else f"{step_id}_{n}",
                    task=task,
                    arguments={"of_step": entry.step_id, "items": ["a", "b"]},
                ),
            )
        return Decision(step_id=entry.step_id, kind=DecisionKind.ACCEPT, reason="fine")

    return decide


async def run(
    decide: Any, book: ResolvedPlaybook | None = None
) -> tuple[RunState, InMemoryAuditTrail]:
    tasks, checks = registries()
    tasks.register(
        TaskDefinition(
            id=REVISE, agent="toy_agent", description="revise", output_model=Count, handler=revise
        )
    )
    audit = InMemoryAuditTrail()
    runtime = SessionRuntime(tasks=tasks, checks=checks, audit=audit, clock=lambda: T0)
    plan = plan_of("toy.produce", "toy.produce", "toy.produce")

    async def fixed_plan(state: RunState) -> Any:
        return plan

    async def select(state: RunState) -> ResolvedPlaybook:
        return book or playbook()

    app = build_session_graph(
        runtime, GraphHooks(plan=fixed_plan, select_playbook=select, decide=decide)
    ).compile(checkpointer=InMemorySaver(serde=checkpoint_serializer()))
    initial = RunState(
        session_id="sess-1",
        business_date=date(2026, 10, 15),
        data_type="linear",
        playbook=book or playbook(),
    )
    return RunState.model_validate(await app.ainvoke(initial, thread_config("sess-1"))), audit


async def test_a_send_back_runs_next_then_the_plan_resumes() -> None:
    final, audit = await run(send_back({"s2"}))
    assert [(e.step_id, e.task) for e in final.execution] == [
        ("s1", "toy.produce"),
        ("s2", "toy.produce"),
        ("r1", REVISE),
        ("s3", "toy.produce"),
    ]
    assert final.agent_results["r1"] == {"n": 2}  # the step got its arguments
    assert [(d.step_id, d.kind) for d in final.decisions] == [
        ("s1", DecisionKind.ACCEPT),
        ("s2", DecisionKind.FOLLOW_UP),
        ("r1", DecisionKind.ACCEPT),  # the send-back gets its own review point
        ("s3", DecisionKind.ACCEPT),
    ]
    assert (final.queued_step, final.next_step, final.status) == (None, 3, SessionStatus.SUCCEEDED)
    decision = next(
        e for e in audit.read() if e.event_type == "decision" and e.payload["kind"] == "follow_up"
    )
    assert decision.payload["follow_up"]["task"] == REVISE  # type: ignore[index]
    assert audit.verify().valid


async def test_a_second_send_back_is_refused_and_flagged() -> None:
    final, audit = await run(send_back({"s2", "r1"}))
    assert [e.step_id for e in final.execution] == ["s1", "s2", "r1", "s3"]  # r1 ran once
    refused = final.decisions[2]
    assert (refused.step_id, refused.kind, refused.source) == (
        "r1",
        DecisionKind.FLAG_FOR_HUMAN,
        DecisionSource.RULE,
    )
    assert "P-3: the playbook allows 1 send-back(s) per run, and 1 already ran" in refused.reason
    [event] = [e for e in audit.read() if e.event_type == "send_back_refused"]
    assert event.payload["step_id"] == "r1"
    assert final.review is not None and final.review.human_review_required == ("r1",)


@pytest.mark.parametrize(
    ("decide", "book", "message"),
    [
        (
            send_back({"s2"}, task="toy.produce"),
            None,
            "P-2: follow-up 'toy.produce' is not allowed",
        ),
        (send_back({"s2"}), playbook(allowed=()), "P-2: follow-up 'toy.revise' is not allowed"),
        (send_back({"s2"}), playbook(send_backs=0), "P-3: the playbook allows 0 send-back(s)"),
        (send_back({"s2"}, step_id="s3"), None, "step_id 's3' is already used"),
    ],
    ids=["task-not-allowed", "none-allowed", "limit-zero", "step-id-taken"],
)
async def test_send_backs_outside_the_playbook_are_refused(
    decide: Any, book: ResolvedPlaybook | None, message: str
) -> None:
    final, _ = await run(decide, book)
    assert [e.step_id for e in final.execution] == ["s1", "s2", "s3"]
    assert final.decisions[1].kind is DecisionKind.FLAG_FOR_HUMAN
    assert message in final.decisions[1].reason

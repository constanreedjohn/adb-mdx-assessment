"""Master planning: validation P-1 to P-4, retry and fallback, with toy tasks and a fake model."""

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from mdx_assessment.core.domain import (
    Fact,
    Plan,
    PlanSource,
    PlanStep,
    RunContext,
    SessionStatus,
)
from mdx_assessment.core.gateway import (
    FakeProvider,
    LLMGateway,
    LLMUnavailableError,
    PromptStore,
)
from mdx_assessment.core.gateway.provider import LLMRequest
from mdx_assessment.core.master.planning import (
    MasterPlanOutput,
    master_plan_hook,
    validate_plan,
)
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.playbooks import (
    PlaybookContent,
    PlaybookLimits,
    ResolvedPlaybook,
    SelectionRecord,
)
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.orchestration.tasks import (
    ReviewCheckRegistry,
    TaskContext,
    TaskDefinition,
    TaskRegistry,
)
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.core.tools.client import TaskBudget

pytestmark = pytest.mark.anyio
FIRST, SECOND, EXTRA = "toy.first", "toy.second", "toy.extra"
FACT = Fact.create("flag_count", 28, source="test", scope={})
CONTEXT = RunContext(facts=(FACT,))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Count(BaseModel):
    n: int


async def produce(ctx: TaskContext) -> BaseModel:
    return Count(n=1)


def playbook(
    follow_ups: tuple[str, ...] = (), rounds: int = 0, seconds: int = 600
) -> ResolvedPlaybook:
    return ResolvedPlaybook(
        id="toy_playbook",
        version="1",
        selection=SelectionRecord(rules=()),
        content=PlaybookContent(
            title="Toy",
            description="A toy playbook.",
            mandatory_steps=(FIRST, SECOND),
            allowed_follow_ups=follow_ups,
            limits=PlaybookLimits(
                max_follow_up_rounds=rounds, max_run_seconds=seconds, max_ai_explanations=1
            ),
            default_plan=(
                PlanStep(step_id="default_1", task=FIRST),
                PlanStep(step_id="default_2", task=SECOND),
            ),
        ),
    )


def registry() -> TaskRegistry:
    tasks = TaskRegistry()
    for task_id in (FIRST, SECOND, EXTRA):
        tasks.register(
            TaskDefinition(
                id=task_id,
                agent="toy_agent",
                description=f"Does {task_id}",
                output_model=Count,
                handler=produce,
                budget=TaskBudget(max_calls=1, max_seconds=60),
            )
        )
    return tasks


def step(task: str, step_id: str | None = None, rationale: str = "why") -> dict[str, str]:
    return {"step_id": step_id or task.replace(".", "_"), "task": task, "rationale": rationale}


def reply(**overrides: Any) -> dict[str, Any]:
    return {
        "steps": [step(FIRST), step(SECOND)],
        "follow_ups": [],
        "rationale": "28 flags, see fact-flag-count.",
        "cited_fact_ids": ["fact-flag-count"],
        **overrides,
    }


def check(
    data: dict[str, Any], book: ResolvedPlaybook | None = None
) -> tuple[dict[str, bool], str]:
    report = validate_plan(
        MasterPlanOutput.model_validate(data),
        book or playbook(),
        CONTEXT.citable_ids,
        registry(),
    )
    return report.checks, " | ".join(report.errors)


class Harness:
    def __init__(self, tmp_path: Path, script: Any, book: ResolvedPlaybook | None = None) -> None:
        folder = tmp_path / "prompts" / "master_plan"
        folder.mkdir(parents=True)
        (folder / "v1.md").write_text("Plan the run.\n")
        self.fake = FakeProvider({"master_plan": script})
        self.audit = InMemoryAuditTrail()
        self.book = book or playbook()
        self.runtime = SessionRuntime(
            tasks=registry(),
            checks=ReviewCheckRegistry(),
            audit=self.audit,
            gateway=LLMGateway(self.fake, PromptStore(tmp_path), self.audit),
        )

    def state(self, context: RunContext = CONTEXT) -> RunState:
        return RunState(
            session_id="s1",
            business_date=date(2026, 10, 15),
            data_type="linear",
            run_context=context,
            playbook=self.book,
        )

    async def plan(self, context: RunContext = CONTEXT) -> Plan:
        return await master_plan_hook(self.runtime)(self.state(context))

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [e.payload for e in self.audit.read() if e.event_type == event_type]


# --- the validator -------------------------------------------------------------------------


def test_a_valid_plan_passes_every_check() -> None:
    checks, errors = check(reply())
    assert checks == {"P-1": True, "P-2": True, "P-3": True, "P-4": True}
    assert errors == ""


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([step(FIRST)], "'toy.second' is missing"),
        ([step(SECOND), step(FIRST)], "must run in this order"),
        ([step(FIRST), step(FIRST, "again"), step(SECOND)], "appears 2 times"),
        ([], "no steps"),
    ],
)
def test_p1_mandatory_steps(steps: list[dict[str, str]], message: str) -> None:
    checks, errors = check(reply(steps=steps))
    assert checks["P-1"] is False
    assert message in errors


def test_p2_rejects_an_extra_task_in_steps() -> None:
    checks, errors = check(reply(steps=[step(FIRST), step(SECOND), step(EXTRA)]))
    assert (checks["P-1"], checks["P-2"]) == (True, False)
    assert "'toy.extra' is not a mandatory step" in errors


def test_p2_rejects_a_follow_up_the_playbook_does_not_allow() -> None:
    checks, errors = check(reply(follow_ups=[step(EXTRA)]))
    assert checks["P-2"] is False
    assert "follow-up 'toy.extra' is not allowed" in errors


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([step(FIRST, "same"), step(SECOND, "same")], "'same' is used more than once"),
        ([step(FIRST, " "), step(SECOND)], "needs a step_id"),
    ],
)
def test_p2_step_ids(steps: list[dict[str, str]], message: str) -> None:
    checks, errors = check(reply(steps=steps))
    assert checks["P-2"] is False
    assert message in errors


def test_p3_follow_up_round_limit() -> None:
    book = playbook(follow_ups=(EXTRA,), rounds=0)
    checks, errors = check(reply(follow_ups=[step(EXTRA)]), book)
    assert (checks["P-2"], checks["P-3"]) == (True, False)
    assert "1 follow-up(s) exceed the limit of 0" in errors


def test_p3_allows_a_follow_up_within_the_limit() -> None:
    checks, _ = check(reply(follow_ups=[step(EXTRA)]), playbook(follow_ups=(EXTRA,), rounds=1))
    assert all(checks.values())


def test_p3_time_budgets_must_fit_the_run_limit() -> None:
    checks, errors = check(reply(), playbook(seconds=100))
    assert checks["P-3"] is False
    assert "total 120s, over the run limit of 100s" in errors


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cited_fact_ids": []}, "cites no facts"),
        ({"cited_fact_ids": ["fact-invented"]}, "'fact-invented' does not exist"),
        ({"rationale": "See fact-invented."}, "unknown ID 'fact-invented'"),
        (
            {"steps": [step(FIRST, rationale="obs-made-up says so."), step(SECOND)]},
            "unknown ID 'obs-made-up'",
        ),
    ],
)
def test_p4_evidence(overrides: dict[str, Any], message: str) -> None:
    checks, errors = check(reply(**overrides))
    assert checks["P-4"] is False
    assert message in errors


# --- planning: retry and fallback ----------------------------------------------------------


async def test_a_valid_plan_is_used_after_one_call(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply())
    plan = await harness.plan()
    assert plan.source is PlanSource.MASTER
    assert [s.task for s in plan.steps] == [FIRST, SECOND]
    assert plan.cited_fact_ids == ("fact-flag-count",)
    assert plan.validation is not None
    assert (plan.validation.valid, plan.validation.attempts) == (True, 1)
    assert len(harness.fake.requests) == 1


async def test_the_catalogue_and_playbook_reach_the_model_as_data(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply())
    await harness.plan()
    request = harness.fake.requests[0]
    assert (request.prompt_id, request.prompt_version) == ("master_plan", "v1")
    assert request.output_model is MasterPlanOutput
    assert "toy.first" not in request.system_prompt
    for expected in ("toy.first", "Does toy.extra", "fact-flag-count", "toy_playbook"):
        assert expected in request.user_message
    assert "previous_attempt" not in request.user_message


async def test_an_invalid_plan_is_retried_with_the_errors_attached(tmp_path: Path) -> None:
    harness = Harness(tmp_path, [reply(steps=[step(FIRST)]), reply()])
    plan = await harness.plan()
    assert plan.source is PlanSource.MASTER
    assert plan.validation is not None and plan.validation.attempts == 2
    retry = harness.fake.requests[1].user_message
    assert "previous_attempt" in retry
    assert "P-1: mandatory step 'toy.second' is missing" in retry


@pytest.mark.parametrize(
    ("bad", "check_id", "message"),
    [
        (reply(steps=[step(FIRST)]), "P-1", "'toy.second' is missing"),
        (reply(steps=[step(SECOND), step(FIRST)]), "P-1", "must run in this order"),
        (reply(cited_fact_ids=["fact-invented"]), "P-4", "'fact-invented' does not exist"),
    ],
)
async def test_invalid_twice_falls_back_to_the_default_plan(
    tmp_path: Path, bad: dict[str, Any], check_id: str, message: str
) -> None:
    harness = Harness(tmp_path, bad)
    plan = await harness.plan()
    assert plan.source is PlanSource.DEFAULT
    assert [s.step_id for s in plan.steps] == ["default_1", "default_2"]
    assert plan.follow_ups == ()
    validation = plan.validation
    assert validation is not None
    assert (validation.valid, validation.attempts) == (False, 2)
    assert validation.fallback_reason == "plan_invalid"
    assert validation.checks[check_id] is False
    assert any(message in e for e in validation.errors)
    assert len(harness.fake.requests) == 2


async def test_unusable_output_is_retried_then_falls_back(tmp_path: Path) -> None:
    harness = Harness(tmp_path, '{"not": "valid"')
    plan = await harness.plan()
    assert plan.source is PlanSource.DEFAULT
    assert plan.validation is not None
    assert plan.validation.fallback_reason == "plan_invalid"
    assert plan.validation.errors[0].startswith("output:")
    assert len(harness.fake.requests) == 2


def down(request: LLMRequest) -> dict[str, Any]:
    raise LLMUnavailableError("connection refused")


async def test_an_unavailable_model_is_not_retried_and_uses_the_default_plan(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, down)
    plan = await harness.plan()
    assert plan.source is PlanSource.DEFAULT
    assert [s.step_id for s in plan.steps] == ["default_1", "default_2"]
    assert plan.validation is not None
    assert (plan.validation.fallback_reason, plan.validation.attempts) == ("model_unavailable", 1)
    assert len(harness.fake.requests) == 1


async def test_with_the_llm_off_the_model_is_not_called(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply())
    state = harness.state().model_copy(update={"llm_enabled": False})
    plan = await master_plan_hook(harness.runtime)(state)
    assert plan.source is PlanSource.DEFAULT
    assert plan.validation is not None
    assert (plan.validation.fallback_reason, plan.validation.attempts) == ("llm_off", 0)
    assert harness.fake.requests == []


async def test_without_facts_the_model_is_not_called(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply())
    plan = await harness.plan(RunContext())
    assert plan.source is PlanSource.DEFAULT
    assert plan.validation is not None
    assert (plan.validation.fallback_reason, plan.validation.attempts) == ("no_facts", 0)
    assert harness.fake.requests == []


async def test_a_gateway_is_required_unless_the_llm_is_off(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply())
    harness.runtime.gateway = None
    with pytest.raises(ValueError, match="LLM gateway"):
        await harness.plan()
    off = harness.state().model_copy(update={"llm_enabled": False})
    assert (await master_plan_hook(harness.runtime)(off)).source is PlanSource.DEFAULT


# --- audit and the graph -------------------------------------------------------------------


async def test_every_attempt_is_logged_not_audited(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = Harness(tmp_path, [reply(steps=[step(FIRST)]), reply()])
    with caplog.at_level(logging.INFO, logger="mdx_assessment.core.master.planning"):
        await harness.plan()
    first, second = [r.getMessage() for r in caplog.records]
    assert "plan attempt 1: valid=False" in first
    assert "'toy.second' is missing" in first
    assert "plan attempt 2: valid=True" in second
    assert harness.events("plan_validation") == []


async def run_graph(harness: Harness) -> RunState:
    async def context(state: RunState) -> RunContext:
        return CONTEXT

    async def select(state: RunState) -> ResolvedPlaybook:
        return harness.book

    hooks = GraphHooks(
        plan=master_plan_hook(harness.runtime), gather_context=context, select_playbook=select
    )
    app = build_session_graph(harness.runtime, hooks).compile(
        checkpointer=InMemorySaver(serde=checkpoint_serializer())
    )
    initial = RunState(session_id="s1", business_date=date(2026, 10, 15), data_type="linear")
    return RunState.model_validate(await app.ainvoke(initial, thread_config("s1")))


async def test_the_validated_plan_drives_the_run(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply())
    final = await run_graph(harness)
    assert final.plan is not None and final.plan.source is PlanSource.MASTER
    assert [e.task for e in final.execution] == [FIRST, SECOND]
    assert final.fallbacks == []
    assert final.status is SessionStatus.COMPLETED


async def test_the_default_plan_ends_as_default_complete(tmp_path: Path) -> None:
    harness = Harness(tmp_path, reply(steps=[step(FIRST)]))
    final = await run_graph(harness)
    assert final.plan is not None and final.plan.source is PlanSource.DEFAULT
    assert [e.step_id for e in final.execution] == ["default_1", "default_2"]
    assert final.fallbacks == ["plan_default:plan_invalid"]
    assert final.status is SessionStatus.DEFAULT_COMPLETE
    assert final.analyst_notice is not None and "usable plan" in final.analyst_notice
    assert harness.events("plan")[0]["source"] == "default"
    assert harness.events("plan")[0]["fallback_reason"] == "plan_invalid"
    assert harness.audit.verify().valid


async def test_a_failed_step_on_the_default_plan_is_still_listed(tmp_path: Path) -> None:
    book = playbook()
    broken = (PlanStep(step_id="default_1", task="toy.missing"),)
    book = book.model_copy(
        update={"content": book.content.model_copy(update={"default_plan": broken})}
    )
    harness = Harness(tmp_path, reply(steps=[step(FIRST)]), book)
    final = await run_graph(harness)
    assert final.status is SessionStatus.DEFAULT_COMPLETE
    assert final.fallbacks == ["plan_default:plan_invalid", "step_failed:default_1"]


async def test_an_unavailable_model_runs_the_default_plan_and_tells_the_analyst(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = Harness(tmp_path, down)
    with caplog.at_level(logging.ERROR, logger="mdx_assessment.core.master.planning"):
        final = await run_graph(harness)
    assert final.status is SessionStatus.DEFAULT_COMPLETE
    assert final.fallbacks == ["plan_default:model_unavailable"]
    assert final.analyst_notice is not None and "developer" in final.analyst_notice
    assert "connection refused" not in final.analyst_notice
    assert "connection refused" in caplog.text
    assert final.plan is not None and final.plan.source is PlanSource.DEFAULT
    assert [e.step_id for e in final.execution] == ["default_1", "default_2"]
    assert harness.events("plan")[0]["fallback_reason"] == "model_unavailable"
    assert len(harness.events("session_completed")) == 1

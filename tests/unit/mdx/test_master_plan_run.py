"""E6-04 with the real config, registries and tools. The live test calls Azure OpenAI."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from tests.support import explain_from_inputs

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import LoadedConfig, load_prompt
from mdx_assessment.core.domain import PlanSource, SessionStatus
from mdx_assessment.core.gateway import FakeProvider, LLMProvider, create_llm_gateway
from mdx_assessment.core.gateway.factory import create_provider
from mdx_assessment.core.master.planning import master_plan_hook
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.run_context import context_hook, playbook_hook
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
STEP_NAMES = ("validate", "assess", "record")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def config() -> LoadedConfig:
    return load_config(PROJECT_ROOT / "config")


@pytest.fixture(scope="module")
def data_dir(config: LoadedConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(config, DAY), path, config_version=config.version)
    return path


async def run_session(
    config: LoadedConfig, data_dir: Path, audit: InMemoryAuditTrail, provider: LLMProvider
) -> RunState:
    registries = build_registries()
    assert registries.facts is not None
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    context = ToolContext(config=config, settings=settings, audit=audit, cases=InMemoryCaseStore())
    permissions = config.get(PERMISSIONS_PATH, PermissionMatrix)
    async with ToolHub(registries.tools, permissions, context) as hub:
        runtime = SessionRuntime(
            tasks=registries.tasks,
            checks=registries.checks,
            audit=audit,
            config=config,
            hub=hub,
            gateway=create_llm_gateway(audit, settings=settings, provider=provider),
        )
        hooks = GraphHooks(
            plan=master_plan_hook(runtime),
            gather_context=context_hook(registries.facts, hub, config),
            select_playbook=playbook_hook(PlaybookStore.from_config(config)),
        )
        app = build_session_graph(runtime, hooks).compile(
            checkpointer=InMemorySaver(serde=checkpoint_serializer())
        )
        initial = RunState(session_id="sess-plan", business_date=DAY, data_type="linear")
        return RunState.model_validate(await app.ainvoke(initial, thread_config("sess-plan")))


def mandatory(config: LoadedConfig) -> tuple[str, ...]:
    return resolve_playbook([], PlaybookStore.from_config(config)).content.mandatory_steps


def valid_reply(tasks: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    def reply(request: Any) -> dict[str, Any]:
        return {
            "steps": [
                {"step_id": name, "task": task, "rationale": "Needed today."}
                for name, task in zip(STEP_NAMES, tasks)
            ],
            "follow_ups": [],
            "rationale": "A normal day: fact-flag-count is under its threshold.",
            "cited_fact_ids": ["fact-flag-count", "fact-files-received"],
        }

    return reply


async def test_the_master_plans_the_simulated_day(config: LoadedConfig, data_dir: Path) -> None:
    tasks = mandatory(config)
    audit = InMemoryAuditTrail()
    fake = FakeProvider({"master_plan": valid_reply(tasks), "mda_explain": explain_from_inputs})
    final = await run_session(config, data_dir, audit, fake)

    assert final.plan is not None and final.plan.source is PlanSource.MASTER
    assert [e.step_id for e in final.execution] == list(STEP_NAMES)
    assert final.status is SessionStatus.SUCCEEDED
    message = fake.requests[0].user_message
    assert all(task in message for task in tasks)  # the catalogue, as data
    assert '"fact-flag-count"' in message
    master_events = [
        e.event_type
        for e in audit.read()
        if e.actor == "master_agent" and e.event_type != "tool_call"
    ]
    assert master_events[:2] == [
        "llm_call",
        "plan",
    ]
    assert audit.verify().valid


async def test_a_rejected_plan_still_runs_the_day_on_the_default(
    config: LoadedConfig, data_dir: Path
) -> None:
    fake = FakeProvider(
        {"master_plan": valid_reply(mandatory(config)[:2]), "mda_explain": explain_from_inputs}
    )
    final = await run_session(config, data_dir, InMemoryAuditTrail(), fake)
    assert final.plan is not None and final.plan.source is PlanSource.DEFAULT
    assert [e.step_id for e in final.execution] == [
        "validate_inputs",
        "assess_movements",
        "record_cases",
    ]
    assert final.fallbacks == ["plan_default:plan_invalid"]
    assert final.status is SessionStatus.SUCCEEDED


def test_the_prompt_names_no_task_or_playbook(config: LoadedConfig) -> None:
    text = load_prompt(PROJECT_ROOT / "config", "master_plan", "v1")
    assert "master_plan" in config.prompts
    names = {t.id for t in build_registries().tasks} | {
        p.id for p in PlaybookStore.from_config(config).playbooks
    }
    assert [n for n in names if n in text] == []


@pytest.mark.live
async def test_live_azure_openai_produces_a_valid_plan(
    config: LoadedConfig, data_dir: Path
) -> None:
    audit = InMemoryAuditTrail()
    provider = create_provider()
    try:
        final = await run_session(config, data_dir, audit, provider)
    finally:
        await provider.aclose()
    plan = final.plan
    assert plan is not None
    assert plan.source is PlanSource.MASTER, plan.validation
    assert plan.validation is not None and plan.validation.valid
    assert [s.task for s in plan.steps] == list(mandatory(config))
    assert set(plan.cited_fact_ids) <= final.run_context.citable_ids
    assert final.status is SessionStatus.SUCCEEDED
    print(f"\nattempts: {plan.validation.attempts}\ncited: {plan.cited_fact_ids}\n{plan.rationale}")

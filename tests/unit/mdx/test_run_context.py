"""Acceptance (E6-03): the selected playbook, its selection record and the facts appear in
the state, on the simulated MDX day, with the real config, registries and tools."""

import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import ConfigError, LoadedConfig
from mdx_assessment.core.domain import Plan, PlanSource, RunContext
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.run_context import (
    FactsRequest,
    context_hook,
    playbook_hook,
)
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)


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


@asynccontextmanager
async def open_hub(
    config: LoadedConfig, data_dir: Path, audit: InMemoryAuditTrail
) -> AsyncIterator[ToolHub]:
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    context = ToolContext(config=config, settings=settings, audit=audit)
    async with ToolHub(
        build_registries().tools, config.get(PERMISSIONS_PATH, PermissionMatrix), context
    ) as hub:
        yield hub


async def facts_for(config: LoadedConfig, data_dir: Path, business_date: date) -> RunContext:
    builder = build_registries().facts
    assert builder is not None
    async with open_hub(config, data_dir, InMemoryAuditTrail()) as hub:
        tools = hub.client("master_agent", session_id="s1", task_id="s1/run_context")
        state = RunState(session_id="s1", business_date=business_date, data_type="linear")
        return await builder(FactsRequest(state=state, tools=tools, config=config))


# --- facts ---------------------------------------------------------------------------------


async def test_facts_on_the_simulated_day(config: LoadedConfig, data_dir: Path) -> None:
    context = await facts_for(config, data_dir, DAY)
    summary = {f.id: (f.value, f.expected, f.threshold) for f in context.facts}
    assert summary == {
        "fact-data-type": ("linear", None, None),
        "fact-business-date": ("2026-10-15", None, None),
        "fact-files-received": (1, 1, None),
        "fact-records-received": (80, 80, None),
        "fact-flag-count": (28, None, 300),
    }
    assert all(f.evidence_id.startswith("ev-") for f in context.facts)
    assert context.observations == ()
    assert "fact-flag-count" in context.citable_ids


async def test_same_day_gives_the_same_evidence_ids(config: LoadedConfig, data_dir: Path) -> None:
    first, second = [await facts_for(config, data_dir, DAY) for _ in range(2)]
    assert [f.evidence_id for f in first.facts] == [f.evidence_id for f in second.facts]


async def test_a_day_mdx_has_no_data_for(config: LoadedConfig, data_dir: Path) -> None:
    context = await facts_for(config, data_dir, date(2026, 10, 16))
    files, records, flags = (
        context.fact(n) for n in ("files_received", "records_received", "flag_count")
    )
    assert files is not None and (files.value, files.expected) == (0, 1)
    assert records is not None and records.value == 0
    assert flags is not None and flags.value is None  # not guessed


# --- acceptance: playbook, selection record and facts in the state -----------------------------


async def run_session(config: LoadedConfig, data_dir: Path, audit: InMemoryAuditTrail) -> RunState:
    registries = build_registries()
    assert registries.facts is not None
    store = PlaybookStore.from_config(config)

    async with open_hub(config, data_dir, audit) as hub:

        async def default_plan(state: RunState) -> Plan:
            assert state.playbook is not None
            return Plan(source=PlanSource.DEFAULT, steps=state.playbook.content.default_plan)

        hooks = GraphHooks(
            plan=default_plan,
            gather_context=context_hook(registries.facts, hub, config),
            select_playbook=playbook_hook(store),
        )
        runtime = SessionRuntime(
            tasks=registries.tasks, checks=registries.checks, audit=audit, hub=hub
        )
        app = build_session_graph(runtime, hooks).compile(
            checkpointer=InMemorySaver(serde=checkpoint_serializer())
        )
        initial = RunState(session_id="sess-ctx", business_date=DAY, data_type="linear")
        return RunState.model_validate(await app.ainvoke(initial, thread_config("sess-ctx")))


async def test_playbook_selection_and_facts_appear_in_the_state(
    config: LoadedConfig, data_dir: Path
) -> None:
    audit = InMemoryAuditTrail()
    final = await run_session(config, data_dir, audit)

    assert final.run_context.fact("flag_count") is not None
    assert final.playbook is not None
    assert (final.playbook.id, final.playbook.version, final.playbook.selected_by) == (
        "normal_eod",
        "1",
        "selector",
    )
    [rule] = final.playbook.selection.rules
    assert (rule.playbook, rule.met, rule.rule) == ("normal_eod", True, "default")
    assert not final.playbook.selection.multi_condition_day
    # The default plan from the playbook ran through the dispatcher.
    assert [e.step_id for e in final.execution] == [
        "validate_inputs",
        "assess_movements",
        "record_cases",
    ]

    events = [
        e for e in audit.read() if e.event_type in ("run_context", "playbook_selected", "plan")
    ]
    assert [e.event_type for e in events] == ["run_context", "playbook_selected", "plan"]
    assert events[1].payload["playbook"] == "normal_eod"


# --- config ------------------------------------------------------------------------------------


def test_real_normal_eod_is_selected_by_default(config: LoadedConfig) -> None:
    resolved = resolve_playbook([], PlaybookStore.from_config(config))
    assert resolved.id == "normal_eod"
    assert resolved.content.allowed_follow_ups == ()  # none in Phase 1
    assert [s.step_id for s in resolved.content.default_plan] == [
        "validate_inputs",
        "assess_movements",
        "record_cases",
    ]


def test_a_playbook_naming_an_unknown_task_fails_startup(tmp_path: Path) -> None:
    broken = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", broken)
    path = broken / "playbooks" / "normal_eod.yaml"
    path.write_text(
        path.read_text().replace("escalation_reporting.record_cases", "nobody.did_this")
    )
    with pytest.raises(ConfigError, match=r"unknown task 'nobody\.did_this'"):
        load_config(broken)

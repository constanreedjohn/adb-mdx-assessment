"""Acceptance (E6-02) with the MDX pack: the three stub steps run end to end through the
real registries, the permission-checked tools and the simulated MDX day."""

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import DecisionKind, Plan, PlanSource, PlanStep, SessionStatus
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.skeleton import (
    CHECK_FILES_AND_INTAKE,
    NEAR_ZERO_CHECK,
    RECORD_CASES,
)
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio

SKELETON_PLAN = Plan(
    source=PlanSource.DEFAULT,
    steps=(
        PlanStep(step_id="validate_inputs", task=CHECK_FILES_AND_INTAKE),
        PlanStep(step_id="assess_movements", task=NEAR_ZERO_CHECK),
        PlanStep(step_id="record_cases", task=RECORD_CASES),
    ),
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def config() -> LoadedConfig:
    return load_config(PROJECT_ROOT / "config")


@pytest.fixture(scope="module")
def data_dir(config: LoadedConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(config, date(2026, 10, 15)), path, config_version=config.version)
    return path


@pytest.fixture
async def outcome(
    config: LoadedConfig, data_dir: Path
) -> AsyncIterator[tuple[RunState, InMemoryAuditTrail]]:
    registries = build_registries()
    audit = InMemoryAuditTrail()
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    hub = ToolHub(
        registries.tools,
        config.get(PERMISSIONS_PATH, PermissionMatrix),
        ToolContext(config=config, settings=settings, audit=audit),
    )

    async def default_plan(state: RunState) -> Plan:
        return SKELETON_PLAN

    async with hub:
        runtime = SessionRuntime(
            tasks=registries.tasks, checks=registries.checks, audit=audit, hub=hub
        )
        app = build_session_graph(runtime, GraphHooks(plan=default_plan)).compile(
            checkpointer=InMemorySaver(serde=checkpoint_serializer())
        )
        initial = RunState(
            session_id="sess-sk", business_date=date(2026, 10, 15), data_type="linear"
        )
        final = RunState.model_validate(await app.ainvoke(initial, thread_config("sess-sk")))
    yield final, audit


async def test_three_stub_steps_run_end_to_end(
    outcome: tuple[RunState, InMemoryAuditTrail],
) -> None:
    final, _ = outcome
    assert [(e.step_id, e.status.value) for e in final.execution] == [
        ("validate_inputs", "done"),
        ("assess_movements", "done"),
        ("record_cases", "done"),
    ]
    assert [d.kind for d in final.decisions] == [DecisionKind.ACCEPT] * 3
    assert final.status is SessionStatus.COMPLETED


async def test_the_input_validation_stub_reads_real_mdx_data(
    outcome: tuple[RunState, InMemoryAuditTrail],
) -> None:
    final, audit = outcome
    assert final.agent_results["validate_inputs"] == {
        "stub": True,
        "files_expected": 1,
        "files_received": 1,
        "record_counts": {"mdx_linear_rates_20261015.csv": 80},
        "flags_read": 28,
    }
    assert final.agent_results["assess_movements"] == {"stub": True, "items_received": 28}
    tool_calls = [e for e in audit.read() if e.event_type == "tool_call"]
    assert [(e.actor, e.payload["operation"], e.payload["status"]) for e in tool_calls] == [
        ("input_validation", "list_expected_files", "ok"),
        ("input_validation", "get_file_metadata", "ok"),
        ("input_validation", "get_flags", "ok"),
    ]
    assert all(e.session_id == "sess-sk" for e in audit.read())
    assert audit.verify().valid


def test_registries_are_consistent_with_the_permission_matrix(config: LoadedConfig) -> None:
    registries = build_registries()
    assert registries.problems(config.get(PERMISSIONS_PATH, PermissionMatrix)) == []
    assert {a.id for a in registries.agents} == {
        "master_agent",
        "input_validation",
        "market_data_assessment",
        "escalation_reporting",
    }

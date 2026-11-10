"""Acceptance (E3-03) with the real permission matrix and real tools: Escalation and
Reporting → MDX data is denied; budgets; one audit event per call. Plus the PDF §2.1 cells."""

import shutil
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import pytest

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import ConfigError, LoadedConfig
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import (
    BudgetExceededError,
    TaskBudget,
    ToolHub,
    ToolPermissionError,
)
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = "2026-10-15"
FLAGS = ("mdx_data", "get_flags", {"business_date": DAY})


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
def audit() -> InMemoryAuditTrail:
    return InMemoryAuditTrail()


@pytest.fixture
async def hub(
    config: LoadedConfig, data_dir: Path, audit: InMemoryAuditTrail
) -> AsyncIterator[ToolHub]:
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    context = ToolContext(config=config, settings=settings, audit=audit, cases=InMemoryCaseStore())
    async with ToolHub(
        build_registries().tools, config.get(PERMISSIONS_PATH, PermissionMatrix), context
    ) as h:
        yield h


async def test_escalation_reporting_cannot_read_mdx(
    hub: ToolHub, audit: InMemoryAuditTrail
) -> None:
    with pytest.raises(ToolPermissionError, match="escalation_reporting may not read on mdx_data"):
        await hub.client("escalation_reporting", session_id="s1", task_id="t1").call(*FLAGS)
    [event] = audit.read()
    assert (event.actor, event.payload["status"]) == ("escalation_reporting", "denied")


@pytest.mark.parametrize("agent", ["master_agent", "input_validation", "market_data_assessment"])
async def test_mdx_readers(hub: ToolHub, agent: str) -> None:
    result = await hub.client(agent, session_id="s1", task_id="t1").call(*FLAGS)
    assert result["count"] == 28


@pytest.mark.parametrize(
    ("agent", "operation", "allowed"),
    [
        ("master_agent", "read_events", True),
        ("escalation_reporting", "read_events", True),
        ("input_validation", "read_events", False),
        ("market_data_assessment", "verify_chain", False),
        ("input_validation", "append_event", True),
        ("market_data_assessment", "append_event", True),
    ],
)
async def test_audit_tool_cells_match_the_pdf(
    hub: ToolHub, agent: str, operation: str, allowed: bool
) -> None:
    args = {"actor": agent, "event_type": "note"} if operation == "append_event" else {}
    client = hub.client(agent, session_id="s1", task_id="t1")
    if allowed:
        await client.call("activity_and_audit", operation, args)
    else:
        with pytest.raises(ToolPermissionError):
            await client.call("activity_and_audit", operation, args)


CASE = {"session_id": "s1", "item_key": "JPY 3M", "payload": {"finding": "threshold_artifact"}}


async def test_escalation_reporting_records_and_reads_cases(
    hub: ToolHub, audit: InMemoryAuditTrail
) -> None:
    client = hub.client("escalation_reporting", session_id="s1", task_id="t1")
    created = await client.call("exception_cases", "create_case", CASE)
    listed = await client.call("exception_cases", "list_cases", {"session_id": "s1"})
    assert listed["count"] == 1
    assert listed["cases"][0]["case_id"] == created["case"]["case_id"]
    statuses = [(e.payload["operation"], e.payload["status"]) for e in audit.read()]
    assert statuses == [("create_case", "ok"), ("list_cases", "ok")]


@pytest.mark.parametrize("agent", ["master_agent", "input_validation", "market_data_assessment"])
async def test_only_escalation_reporting_may_write_cases(
    hub: ToolHub, audit: InMemoryAuditTrail, agent: str
) -> None:
    with pytest.raises(ToolPermissionError, match=f"{agent} may not write on exception_cases"):
        await hub.client(agent, session_id="s1", task_id="t1").call(
            "exception_cases", "create_case", CASE
        )
    [event] = audit.read()
    assert (event.actor, event.payload["status"]) == (agent, "denied")


@pytest.mark.parametrize(
    ("agent", "allowed"),
    [
        ("master_agent", True),
        ("market_data_assessment", True),
        ("escalation_reporting", True),
        ("input_validation", False),
    ],
)
async def test_case_readers(hub: ToolHub, agent: str, allowed: bool) -> None:
    writer = hub.client("escalation_reporting", session_id="s1", task_id="t0")
    case_id = (await writer.call("exception_cases", "create_case", CASE))["case"]["case_id"]
    client = hub.client(agent, session_id="s1", task_id="t1")
    if allowed:
        case = await client.call("exception_cases", "get_case", {"case_id": case_id})
        assert case["item_key"] == "JPY 3M"
    else:
        with pytest.raises(ToolPermissionError):
            await client.call("exception_cases", "get_case", {"case_id": case_id})


async def test_budget_and_one_audit_event_per_call(hub: ToolHub, audit: InMemoryAuditTrail) -> None:
    client = hub.client(
        "input_validation", session_id="s1", task_id="t1", budget=TaskBudget(max_calls=2)
    )
    await client.call(*FLAGS)
    await client.call("mdx_data", "get_file_metadata", {"business_date": DAY})
    with pytest.raises(BudgetExceededError):
        await client.call(*FLAGS)
    statuses = [e.payload["status"] for e in audit.read() if e.event_type == "tool_call"]
    assert statuses == ["ok", "ok", "budget_exceeded"]


def test_granting_write_on_an_adb_system_fails_startup(tmp_path: Path) -> None:
    broken = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", broken)
    text = (
        (broken / PERMISSIONS_PATH)
        .read_text()
        .replace(
            "    input_validation: [read]\n    market_data_assessment: [read]\n    escalation_reporting: []",
            "    input_validation: [read, write]\n    market_data_assessment: [read]\n    escalation_reporting: []",
            1,
        )
    )
    (broken / PERMISSIONS_PATH).write_text(text)
    with pytest.raises(ConfigError, match=r"mdx_data.input_validation grants \['write'\]"):
        load_config(broken)


def test_every_registered_tool_has_a_row(config: LoadedConfig) -> None:
    matrix = config.get(PERMISSIONS_PATH, PermissionMatrix)
    assert matrix.check_against_tools(build_registries().tools.specs) == []
    assert set(matrix.tools) == {"mdx_data", "activity_and_audit", "exception_cases"}

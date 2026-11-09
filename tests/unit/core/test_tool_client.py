"""Tool client wrapper, with a toy tool: permissions, budgets, one audit event per call, tracing."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ValidationError
from tests.support import CapturedTraces

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain.canonical import content_hash
from mdx_assessment.core.observability import Tracer
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.core.registry import ToolAccess, ToolGroup
from mdx_assessment.core.tools.client import (
    BudgetExceededError,
    TaskBudget,
    ToolCallError,
    ToolHub,
    ToolPermissionError,
)
from mdx_assessment.core.tools.permissions import PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext, ToolRegistry
from mdx_assessment.settings import get_settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ToySpec(BaseModel):
    id: str = "toy"
    group: ToolGroup = ToolGroup.SOLUTION_DATA
    description: str = "toy"
    operations: dict[str, ToolAccess] = {
        "peek": ToolAccess.READ,
        "poke": ToolAccess.WRITE,
        "slow": ToolAccess.READ,
        "fail": ToolAccess.READ,
    }


class Counter(BaseModel):
    n: int


def toy_server(calls: list[str]) -> MCPServer:
    server: MCPServer = MCPServer(name="toy")

    @server.tool(name="peek")
    def peek(x: int = 0) -> Counter:
        calls.append("peek")
        return Counter(n=x + 1)

    @server.tool(name="poke")
    def poke() -> Counter:
        calls.append("poke")
        return Counter(n=-1)

    @server.tool(name="slow")
    async def slow() -> Counter:
        calls.append("slow")
        await asyncio.sleep(2)
        return Counter(n=0)

    @server.tool(name="fail")
    def fail() -> Counter:
        calls.append("fail")
        raise ToolError("no data for that date")

    return server


MATRIX = PermissionMatrix.model_validate(
    {
        "version": "1",
        "agents": ["reader", "writer", "nobody"],
        "tools": {"toy": {"reader": ["read"], "writer": ["read", "write"], "nobody": []}},
    }
)


@pytest.fixture
def audit() -> InMemoryAuditTrail:
    return InMemoryAuditTrail()


@pytest.fixture
def calls() -> list[str]:
    return []


def make_hub(calls: list[str], audit: InMemoryAuditTrail, tracer: Tracer | None = None) -> ToolHub:
    registry = ToolRegistry()
    registry.register(ToySpec(), lambda ctx: toy_server(calls))
    context = ToolContext(
        config=LoadedConfig(config_dir=Path(), version="v", documents={}, prompts={}),
        settings=get_settings(),
        audit=audit,
    )
    return ToolHub(registry, MATRIX, context, tracer)


@pytest.fixture
async def hub(calls: list[str], audit: InMemoryAuditTrail) -> AsyncIterator[ToolHub]:
    async with make_hub(calls, audit) as h:
        yield h


def last_event(audit: InMemoryAuditTrail) -> dict[str, Any]:
    event = audit.read()[-1]
    return {
        "actor": event.actor,
        "type": event.event_type,
        "session": event.session_id,
        **event.payload,
    }


# --- permissions ---------------------------------------------------------------------------


async def test_permitted_call_returns_the_structured_result(
    hub: ToolHub, audit: InMemoryAuditTrail
) -> None:
    result = await hub.client("reader", session_id="s1", task_id="t1").call(
        "toy", "peek", {"x": 41}
    )
    assert result == {"n": 42}
    event = last_event(audit)
    assert (event["actor"], event["type"], event["session"], event["status"]) == (
        "reader",
        "tool_call",
        "s1",
        "ok",
    )


async def test_denied_call_never_reaches_the_server(
    hub: ToolHub, audit: InMemoryAuditTrail, calls: list[str]
) -> None:
    with pytest.raises(ToolPermissionError, match="reader may not write on toy"):
        await hub.client("reader", session_id="s1", task_id="t1").call("toy", "poke")
    assert calls == []
    assert last_event(audit)["status"] == "denied"


async def test_write_access_is_granted_explicitly(hub: ToolHub) -> None:
    assert await hub.client("writer", session_id="s1", task_id="t1").call("toy", "poke") == {
        "n": -1
    }


@pytest.mark.parametrize(
    ("agent", "tool", "operation", "message"),
    [
        ("nobody", "toy", "peek", "nobody may not read on toy"),
        ("stranger", "toy", "peek", "stranger may not read on toy"),
        ("reader", "toy", "drop_table", "unknown operation toy.drop_table"),
        ("reader", "other", "peek", "unknown operation other.peek"),
    ],
)
async def test_anything_not_granted_is_denied(
    hub: ToolHub, calls: list[str], agent: str, tool: str, operation: str, message: str
) -> None:
    with pytest.raises(ToolPermissionError, match=message):
        await hub.client(agent, session_id="s1", task_id="t1").call(tool, operation)
    assert calls == []


# --- budgets -------------------------------------------------------------------------------


async def test_call_budget(hub: ToolHub, audit: InMemoryAuditTrail, calls: list[str]) -> None:
    client = hub.client("reader", session_id="s1", task_id="t1", budget=TaskBudget(max_calls=2))
    await client.call("toy", "peek")
    await client.call("toy", "peek")
    with pytest.raises(BudgetExceededError, match="call budget of 2 used up"):
        await client.call("toy", "peek")
    assert calls == ["peek", "peek"]
    assert last_event(audit)["status"] == "budget_exceeded"


async def test_budget_is_per_task(hub: ToolHub) -> None:
    first = hub.client("reader", session_id="s1", task_id="t1", budget=TaskBudget(max_calls=1))
    await first.call("toy", "peek")
    second = hub.client("reader", session_id="s1", task_id="t2", budget=TaskBudget(max_calls=1))
    assert await second.call("toy", "peek") == {"n": 1}


async def test_time_budget_used_up_before_a_call(hub: ToolHub, calls: list[str]) -> None:
    now = [100.0]
    client = hub.client(
        "reader",
        session_id="s1",
        task_id="t1",
        budget=TaskBudget(max_seconds=5),
        clock=lambda: now[0],
    )
    await client.call("toy", "peek")
    now[0] += 6
    with pytest.raises(BudgetExceededError, match="time budget of 5s used up"):
        await client.call("toy", "peek")
    assert calls == ["peek"]


async def test_a_call_that_outlasts_the_time_left_is_cut_off(
    hub: ToolHub, audit: InMemoryAuditTrail
) -> None:
    client = hub.client("reader", session_id="s1", task_id="t1", budget=TaskBudget(max_seconds=0.3))
    with pytest.raises(BudgetExceededError, match="outlasted the time budget"):
        await client.call("toy", "slow")
    assert last_event(audit)["status"] == "budget_exceeded"


def test_budget_must_allow_something() -> None:
    with pytest.raises(ValueError, match="at least one call"):
        TaskBudget(max_calls=0)


# --- audit: exactly one event per call --------------------------------------------------------


async def test_one_audit_event_per_call(hub: ToolHub, audit: InMemoryAuditTrail) -> None:
    client = hub.client("reader", session_id="s1", task_id="t1")
    await client.call("toy", "peek", {"x": 1})
    with pytest.raises(ToolPermissionError):
        await client.call("toy", "poke")
    with pytest.raises(ToolCallError, match="no data for that date"):
        await client.call("toy", "fail")
    events = audit.read()
    assert [e.payload["status"] for e in events] == ["ok", "denied", "tool_error"]
    assert all(e.event_type == "tool_call" and e.actor == "reader" for e in events)
    assert audit.verify().valid


async def test_audit_event_records_a_hash_not_the_arguments(
    hub: ToolHub, audit: InMemoryAuditTrail
) -> None:
    await hub.client("reader", session_id="s1", task_id="t9").call("toy", "peek", {"x": 424242})
    event = last_event(audit)
    assert event["tool"] == "toy" and event["operation"] == "peek" and event["task_id"] == "t9"
    assert event["arguments_sha256"] == content_hash({"x": 424242})
    assert "x" not in event and 424242 not in event.values()
    assert event["result_bytes"] > 0
    assert isinstance(event["latency_ms"], float)


# --- matrix and hub consistency ---------------------------------------------------------------


def test_hub_rejects_a_matrix_without_a_row_for_a_registered_tool(
    calls: list[str], audit: InMemoryAuditTrail
) -> None:
    registry = ToolRegistry()
    registry.register(ToySpec(), lambda ctx: toy_server(calls))
    registry.register(ToySpec(id="toy2"), lambda ctx: toy_server(calls))
    context = ToolContext(
        config=LoadedConfig(Path(), "v", {}, {}), settings=get_settings(), audit=audit
    )
    with pytest.raises(ValueError, match="registered tool 'toy2' has no row"):
        ToolHub(registry, MATRIX, context)


def test_matrix_rejects_unknown_agents() -> None:
    with pytest.raises(ValidationError, match="unknown agent"):
        PermissionMatrix.model_validate(
            {"version": "1", "agents": ["a"], "tools": {"toy": {"b": ["read"]}}}
        )


def test_matrix_reports_grants_a_tool_cannot_use() -> None:
    matrix = PermissionMatrix.model_validate(
        {"version": "1", "agents": ["a"], "tools": {"toy": {"a": ["read", "call"]}}}
    )
    [problem] = matrix.check_against_tools([ToySpec()])
    assert "toy.a grants ['call'], but toy only has ['read', 'write'] operations" in problem


def test_registry_rejects_duplicates_and_empty_tools(calls: list[str]) -> None:
    registry = ToolRegistry()
    registry.register(ToySpec(), lambda ctx: toy_server(calls))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ToySpec(), lambda ctx: toy_server(calls))
    with pytest.raises(ValueError, match="no operations"):
        registry.register(ToySpec(id="empty", operations={}), lambda ctx: toy_server(calls))


# --- tracing: one tool span per call -----------------------------------------------------------


async def test_each_call_is_a_tool_span(
    calls: list[str],
    audit: InMemoryAuditTrail,
    captured_traces: CapturedTraces,
) -> None:
    tracer, traces = captured_traces
    with tracer.session_trace("s1"):
        async with make_hub(calls, audit, tracer) as hub:
            await hub.client("reader", session_id="s1", task_id="t1").call("toy", "peek", {"x": 1})
    [span] = [s for t in traces for s in t.data.spans if s.name == "toy.peek"]
    assert span.span_type == "TOOL"
    assert span.inputs == {"x": 1}

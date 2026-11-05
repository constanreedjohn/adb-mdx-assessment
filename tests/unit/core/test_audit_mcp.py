"""Activity and Audit MCP server, called through a real MCP client in-process."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from mdx_assessment.core.domain import ChainVerification
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.core.registry import ToolSpec
from mdx_assessment.core.tools.audit import (
    APPEND_EVENT,
    AUDIT_TOOL_SPEC,
    READ_EVENTS,
    VERIFY_CHAIN,
    AppendEventResult,
    ReadEventsResult,
    create_audit_server,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def trail() -> InMemoryAuditTrail:
    return InMemoryAuditTrail()


@pytest.fixture
async def client(trail: InMemoryAuditTrail) -> AsyncIterator[Client]:
    async with Client(create_audit_server(trail)) as c:
        yield c


async def call(client: Client, tool: str, **args: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error, error_text(result)
    assert result.structured_content is not None
    return result.structured_content


def error_text(result: CallToolResult) -> str:
    return " ".join(c.text for c in result.content if isinstance(c, TextContent))


async def test_tools_are_listed_with_schemas_and_hints(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {APPEND_EVENT, READ_EVENTS, VERIFY_CHAIN}
    append = tools[APPEND_EVENT]
    assert append.input_schema["required"] == ["actor", "event_type"]
    assert append.output_schema is not None
    assert append.annotations is not None and append.annotations.read_only_hint is False
    for name in (READ_EVENTS, VERIFY_CHAIN):
        annotations = tools[name].annotations
        assert annotations is not None and annotations.read_only_hint is True


async def test_append_read_and_verify(client: Client, trail: InMemoryAuditTrail) -> None:
    first = AppendEventResult.model_validate(
        await call(
            client,
            APPEND_EVENT,
            actor="master_agent",
            event_type="plan",
            session_id="s1",
            payload={"steps": ["validate_inputs", "assess_movements"], "source": "master"},
        )
    )
    await call(
        client, APPEND_EVENT, actor="input_validation", event_type="tool_call", session_id="s1"
    )
    await call(client, APPEND_EVENT, actor="system", event_type="startup")  # no session

    assert first.seq == 1
    assert first.hash == trail.read()[0].hash

    s1 = ReadEventsResult.model_validate(await call(client, READ_EVENTS, session_id="s1"))
    assert s1.count == 2
    assert [e.event_type for e in s1.events] == ["plan", "tool_call"]
    assert s1.events[0].payload == {
        "steps": ["validate_inputs", "assess_movements"],
        "source": "master",
    }
    assert ReadEventsResult.model_validate(await call(client, READ_EVENTS)).count == 3

    for scope in ({}, {"session_id": "s1"}):
        verification = ChainVerification.model_validate(await call(client, VERIFY_CHAIN, **scope))
        assert verification.valid


async def test_verify_reports_tampering(client: Client, trail: InMemoryAuditTrail) -> None:
    for i in range(3):
        await call(client, APPEND_EVENT, actor="a", event_type="e", payload={"i": i})
    trail._events[1] = trail._events[1].model_copy(update={"payload": {"i": 99}})  # tamper
    verification = ChainVerification.model_validate(await call(client, VERIFY_CHAIN))
    assert (verification.valid, verification.first_broken_seq) == (False, 2)


@pytest.mark.parametrize(
    ("args", "field"),
    [
        ({"actor": "", "event_type": "plan"}, "actor"),
        ({"actor": "a" * 101, "event_type": "plan"}, "actor"),
        ({"actor": "master_agent"}, "event_type"),
        ({"actor": "master_agent", "event_type": "plan", "session_id": ""}, "session_id"),
    ],
)
async def test_bad_arguments_are_rejected_with_a_readable_error(
    client: Client, trail: InMemoryAuditTrail, args: dict[str, Any], field: str
) -> None:
    result = await client.call_tool(APPEND_EVENT, args)
    assert result.is_error
    assert field in error_text(result)
    assert trail.read() == []  # nothing recorded


async def test_tool_spec_matches_the_registry_protocol() -> None:
    assert isinstance(AUDIT_TOOL_SPEC, ToolSpec)
    assert AUDIT_TOOL_SPEC.id == "activity_and_audit"

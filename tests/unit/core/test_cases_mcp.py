"""Exception cases MCP server (E3-04), called through a real MCP client in-process."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from mdx_assessment.core.persistence import InMemoryCaseStore
from mdx_assessment.core.registry import ToolSpec
from mdx_assessment.core.registry.specs import ToolAccess
from mdx_assessment.core.tools.cases import (
    CASES_TOOL_SPEC,
    CREATE_CASE,
    GET_CASE,
    LIST_CASES,
    CreateCaseResult,
    ListCasesResult,
    create_cases_server,
)

pytestmark = pytest.mark.anyio
PAYLOAD = {"finding": "threshold_artifact", "change_bps": 3.0, "evidence_ids": ["ev-1"]}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def store() -> InMemoryCaseStore:
    return InMemoryCaseStore()


@pytest.fixture
async def client(store: InMemoryCaseStore) -> AsyncIterator[Client]:
    async with Client(create_cases_server(store)) as c:
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
    assert set(tools) == {CREATE_CASE, GET_CASE, LIST_CASES}
    create = tools[CREATE_CASE]
    assert create.input_schema["required"] == ["session_id", "item_key"]
    assert create.output_schema is not None
    assert create.annotations is not None
    assert (create.annotations.read_only_hint, create.annotations.idempotent_hint) == (False, True)
    for name in (GET_CASE, LIST_CASES):
        annotations = tools[name].annotations
        assert annotations is not None and annotations.read_only_hint is True


async def test_create_then_get_and_list(client: Client) -> None:
    created = CreateCaseResult.model_validate(
        await call(client, CREATE_CASE, session_id="s1", item_key="JPY 3M", payload=PAYLOAD)
    )
    assert created.created is True
    case = created.case
    assert (case.session_id, case.item_key, case.status) == ("s1", "JPY 3M", "open")
    assert case.payload == PAYLOAD
    assert case.case_id.startswith("case-")

    assert (
        await call(client, GET_CASE, case_id=case.case_id)
        == created.model_dump(mode="json")["case"]
    )

    await call(client, CREATE_CASE, session_id="s1", item_key="GBP 2Y")
    await call(client, CREATE_CASE, session_id="s2", item_key="JPY 3M")
    listed = ListCasesResult.model_validate(await call(client, LIST_CASES, session_id="s1"))
    assert listed.count == 2
    assert [c.item_key for c in listed.cases] == ["JPY 3M", "GBP 2Y"]
    assert listed.cases[1].payload == {}


async def test_creating_the_same_item_again_returns_the_first_case(
    client: Client, store: InMemoryCaseStore
) -> None:
    first = CreateCaseResult.model_validate(
        await call(client, CREATE_CASE, session_id="s1", item_key="JPY 3M", payload=PAYLOAD)
    )
    again = CreateCaseResult.model_validate(
        await call(client, CREATE_CASE, session_id="s1", item_key="JPY 3M", payload={"x": 1})
    )
    assert again.created is False
    assert again.case == first.case  # unchanged: the second payload is ignored
    assert len(store.list_for_session("s1")) == 1


async def test_unknown_case_is_a_readable_error(client: Client) -> None:
    result = await client.call_tool(GET_CASE, {"case_id": "case-nope"})
    assert result.is_error
    assert "unknown case 'case-nope'" in error_text(result)


@pytest.mark.parametrize(
    ("tool", "args", "field"),
    [
        (CREATE_CASE, {"session_id": "s1", "item_key": ""}, "item_key"),
        (CREATE_CASE, {"session_id": "s1", "item_key": "x" * 201}, "item_key"),
        (CREATE_CASE, {"item_key": "JPY 3M"}, "session_id"),
        (CREATE_CASE, {"session_id": "s" * 65, "item_key": "JPY 3M"}, "session_id"),
        (LIST_CASES, {}, "session_id"),
        (GET_CASE, {"case_id": ""}, "case_id"),
    ],
)
async def test_bad_arguments_are_rejected_with_a_readable_error(
    client: Client, store: InMemoryCaseStore, tool: str, args: dict[str, Any], field: str
) -> None:
    result = await client.call_tool(tool, args)
    assert result.is_error
    assert field in error_text(result)
    assert store.list_for_session("s1") == []  # nothing recorded


def test_tool_spec_matches_the_registry_protocol() -> None:
    assert isinstance(CASES_TOOL_SPEC, ToolSpec)
    assert CASES_TOOL_SPEC.id == "exception_cases"
    assert CASES_TOOL_SPEC.operations == {
        CREATE_CASE: ToolAccess.WRITE,
        GET_CASE: ToolAccess.READ,
        LIST_CASES: ToolAccess.READ,
    }

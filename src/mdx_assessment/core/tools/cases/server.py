"""Exception cases MCP server, minimal (PDF §2.1 tool layer; §3.4 exception cases store).

Three tools over a `CaseStore` (Postgres in the app, in-memory in tests):

| Tool        | Does                                                        | Kind      |
| ----------- | ----------------------------------------------------------- | --------- |
| create_case | Open a case for one item of a session; idempotent per item  | write     |
| get_case    | One case by ID                                              | read-only |
| list_cases  | A session's cases, oldest first                             | read-only |

Core doesn't know what a case describes: the use-case pack chooses the item key
(e.g. "JPY 3M") and the payload (measures, findings, explanation, evidence IDs).

Who may call what (Escalation and Reporting writes; Master and Market Data Assessment
read) is enforced by the MCP client wrapper with the permission matrix, as for every
tool. Updating a case (an analyst decision, closing it) comes with SP9.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, JsonValue, ValidationError

from mdx_assessment.core.persistence.case_store import CaseStore, UnknownSessionError
from mdx_assessment.core.persistence.repositories import CaseRecord
from mdx_assessment.core.registry.specs import ToolAccess, ToolGroup

SERVER_NAME = "exception-cases"
TOOL_ID = "exception_cases"
CREATE_CASE = "create_case"
GET_CASE = "get_case"
LIST_CASES = "list_cases"

# Limits match CaseRecord and the cases columns, so bad arguments are rejected by the
# MCP input schema before the tool runs.
SessionIdArg = Annotated[
    str, Field(min_length=1, max_length=64, description="Session the case belongs to.")
]
ItemKeyArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        description="The item the case is about, unique within the session, e.g. 'JPY 3M'.",
    ),
]
CaseIdArg = Annotated[str, Field(min_length=1, max_length=64, description="The case's ID.")]


class CasesToolSpec(BaseModel):
    """This tool as the registries see it (satisfies `core.registry.ToolSpec`)."""

    id: str = TOOL_ID
    group: ToolGroup = ToolGroup.SOLUTION_DATA
    description: str = (
        "The solution's exception cases: one open case per assessed item, with what the "
        "agents found and the evidence behind it."
    )
    operations: dict[str, ToolAccess] = {
        CREATE_CASE: ToolAccess.WRITE,
        GET_CASE: ToolAccess.READ,
        LIST_CASES: ToolAccess.READ,
    }


CASES_TOOL_SPEC = CasesToolSpec()


class CreateCaseResult(BaseModel):
    case: CaseRecord
    created: bool = Field(
        description="False when the session already had a case for this item; that case is returned."
    )


class ListCasesResult(BaseModel):
    cases: list[CaseRecord]
    count: int


def create_cases_server(store: CaseStore) -> MCPServer:
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        title="Exception cases",
        instructions=(
            "Exception cases for analysts. Open one case per assessed item with create_case; "
            "calling it again for the same item returns the existing case."
        ),
    )

    @server.tool(
        name=CREATE_CASE,
        description=(
            "Open a case for one item of a session, with status 'open'. If the session already "
            "has a case for this item, that case is returned unchanged (created=false)."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    def create_case(
        session_id: SessionIdArg,
        item_key: ItemKeyArg,
        payload: Annotated[
            dict[str, JsonValue] | None,
            Field(description="What the case records (findings, evidence IDs, ...) as JSON."),
        ] = None,
    ) -> CreateCaseResult:
        try:
            result = store.create(session_id, item_key, payload or {})
        except UnknownSessionError as exc:
            raise ToolError(str(exc)) from exc
        except ValidationError as exc:  # report the fields, not the values
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            raise ToolError(f"invalid case: {', '.join(fields)}") from exc
        return CreateCaseResult(case=result.case, created=result.created)

    @server.tool(
        name=GET_CASE,
        description="Read one case by its ID.",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def get_case(case_id: CaseIdArg) -> CaseRecord:
        case = store.get(case_id)
        if case is None:
            raise ToolError(f"unknown case {case_id!r}")
        return case

    @server.tool(
        name=LIST_CASES,
        description="List a session's cases, oldest first.",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def list_cases(session_id: SessionIdArg) -> ListCasesResult:
        cases = store.list_for_session(session_id)
        return ListCasesResult(cases=cases, count=len(cases))

    return server

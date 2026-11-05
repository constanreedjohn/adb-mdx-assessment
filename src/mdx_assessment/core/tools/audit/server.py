"""Activity and Audit MCP server (PDF §2.1 tool layer; §3.4 activity and audit store).

Three tools over an `AuditTrail` (Postgres in the app, in-memory in tests):

| Tool           | Does                                             | Kind       |
| -------------- | ------------------------------------------------ | ---------- |
| append_event   | Record one event; returns its seq and hash       | write      |
| read_events    | Events in seq order, for one session or all      | read-only  |
| verify_chain   | Check the hash chain (one session or all)        | read-only  |

Who may call what (PDF §2.1: Master and Escalation & Reporting append + read;
Input Validation and Market Data Assessment append only) is enforced by the MCP
client wrapper with the permission matrix (E3-03), not here: the server serves
every caller the same way, and the wrapper is the single place that knows the caller.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, JsonValue, ValidationError

from mdx_assessment.core.domain.audit import AuditEvent, ChainVerification, NewAuditEvent
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.registry.specs import ToolGroup

SERVER_NAME = "activity-and-audit"
TOOL_ID = "activity_and_audit"
APPEND_EVENT = "append_event"
READ_EVENTS = "read_events"
VERIFY_CHAIN = "verify_chain"

# Limits match NewAuditEvent and the audit_events columns, so bad arguments are
# rejected by the MCP input schema before the tool runs.
SessionIdArg = Annotated[
    str | None,
    Field(
        min_length=1,
        max_length=64,
        description="Session the event belongs to; omit for events outside a session.",
    ),
]
ActorArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=100,
        description="Who acted, e.g. 'master_agent', 'input_validation'.",
    ),
]
EventTypeArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=100,
        description="What happened, e.g. 'plan', 'tool_call', 'decision'.",
    ),
]


class AuditToolSpec(BaseModel):
    """This tool as the registries see it (satisfies `core.registry.ToolSpec`)."""

    id: str = TOOL_ID
    group: ToolGroup = ToolGroup.SOLUTION_DATA
    description: str = "Append-only, hash-chained record of plans, tasks, tool and model calls, reviews and decisions."


AUDIT_TOOL_SPEC = AuditToolSpec()


class AppendEventResult(BaseModel):
    seq: int
    hash: str
    ts: str = Field(description="When the event was recorded (ISO 8601, UTC).")


class ReadEventsResult(BaseModel):
    events: list[AuditEvent]
    count: int


def create_audit_server(trail: AuditTrail) -> MCPServer:
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        title="Activity and Audit",
        instructions=(
            "Append-only audit trail. Record each plan, task, tool call, model call, review "
            "and decision with append_event. Events can't be changed or removed."
        ),
    )

    @server.tool(
        name=APPEND_EVENT,
        description="Record one event in the append-only audit trail. Returns its sequence number and hash.",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    def append_event(
        actor: ActorArg,
        event_type: EventTypeArg,
        session_id: SessionIdArg = None,
        payload: Annotated[
            dict[str, JsonValue] | None, Field(description="Event details as JSON.")
        ] = None,
    ) -> AppendEventResult:
        try:
            new = NewAuditEvent(
                session_id=session_id, actor=actor, event_type=event_type, payload=payload or {}
            )
            event = trail.append(new)
        except ValidationError as exc:  # anticipated: report the fields, not the values
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            raise ToolError(f"invalid event: {', '.join(fields)}") from exc
        except ValueError as exc:  # e.g. a payload that can't be hashed canonically
            raise ToolError(f"invalid event: {exc}") from exc
        return AppendEventResult(seq=event.seq, hash=event.hash, ts=event.ts.isoformat())

    @server.tool(
        name=READ_EVENTS,
        description="Read audit events in sequence order, for one session or (session_id omitted) all.",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def read_events(session_id: SessionIdArg = None) -> ReadEventsResult:
        events = trail.read(session_id)
        return ReadEventsResult(events=events, count=len(events))

    @server.tool(
        name=VERIFY_CHAIN,
        description=(
            "Verify the audit hash chain for one session or (session_id omitted) the whole trail. "
            "Reports the first broken sequence number if anything was changed or removed."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def verify_chain(session_id: SessionIdArg = None) -> ChainVerification:
        return trail.verify(session_id)

    return server

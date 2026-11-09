"""The only way an agent calls a tool (PDF §2.1 tool layer, §3.2 bounded tool use, §3.5).

    async with ToolHub(registry, matrix, context, tracer) as hub:      # one per session
        tools = hub.client("input_validation", session_id=..., task_id=..., budget=...)
        flags = await tools.call("mdx_data", "get_flags", {"business_date": "2026-10-15"})

Every call goes through four steps, in order:
1. **Permission**: the operation's access level must be granted to this agent on
   this tool in the permission matrix; otherwise `ToolPermissionError`. The server is
   never contacted.
2. **Budget**: at most `max_calls` calls and `max_seconds` of wall time per task;
   otherwise `BudgetExceededError`. A call that would outlast the remaining time is
   cut off and counts as over budget.
3. **Call** through MCP, inside a "tool" trace span.
4. **Audit**: exactly one audit event per attempt (allowed, denied or failed) with
   the tool, operation, a SHA-256 of the arguments (not the arguments themselves),
   the outcome, result size and latency.

The audit event is written straight to the audit trail (not through the Activity and
Audit MCP tool), so recording a call can never itself need a recorded call.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any, Self

from mcp import Client
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue

from mdx_assessment.core.domain import NewAuditEvent
from mdx_assessment.core.domain.canonical import content_hash
from mdx_assessment.core.observability.tracing import Tracer
from mdx_assessment.core.tools.permissions import PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext, ToolRegistry

TOOL_CALL_EVENT = "tool_call"
MonotonicClock = Callable[[], float]


class ToolPermissionError(PermissionError):
    """The agent may not perform this operation on this tool."""


class BudgetExceededError(RuntimeError):
    """The task used up its tool-call budget (calls or time)."""


class ToolCallError(RuntimeError):
    """The tool ran but reported an error (e.g. unknown date). Message is the tool's."""


class CallStatus(StrEnum):
    OK = "ok"
    DENIED = "denied"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_ERROR = "tool_error"
    FAILED = "failed"  # transport or unexpected error


@dataclass(frozen=True)
class TaskBudget:
    """Bounded tool use per task (PDF §3.2). Defaults suit a skeleton step; tasks override."""

    max_calls: int = 20
    max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_calls < 1 or self.max_seconds <= 0:
            raise ValueError("budget needs at least one call and a positive time limit")


class ToolHub:
    """Connects to every registered tool server once per session and hands out clients."""

    def __init__(
        self,
        registry: ToolRegistry,
        matrix: PermissionMatrix,
        context: ToolContext,
        tracer: Tracer | None = None,
    ) -> None:
        problems = matrix.check_against_tools(registry.specs)
        if problems:
            raise ValueError("permission matrix does not match the tools: " + "; ".join(problems))
        self._registry = registry
        self.matrix = matrix
        self.context = context
        self.tracer = tracer or Tracer(None)
        self._clients: dict[str, Client] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        for registration in self._registry:
            server = registration.factory(self.context)
            self._clients[registration.spec.id] = await self._stack.enter_async_context(
                Client(server)
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._clients.clear()
        await self._stack.aclose()

    def client(
        self,
        agent_id: str,
        *,
        session_id: str | None,
        task_id: str,
        budget: TaskBudget | None = None,
        clock: MonotonicClock = time.monotonic,
    ) -> AgentToolClient:
        """A client for one agent doing one task (its own budget)."""
        return AgentToolClient(
            self, agent_id, session_id=session_id, task_id=task_id, budget=budget, clock=clock
        )

    def _connection(self, tool_id: str) -> Client:
        try:
            return self._clients[tool_id]
        except KeyError:
            raise ToolPermissionError(f"unknown tool {tool_id!r}") from None

    def _registry_operations(self, tool_id: str) -> dict[str, Any]:
        try:
            return dict(self._registry.get(tool_id).spec.operations)
        except KeyError:
            return {}


class AgentToolClient:
    def __init__(
        self,
        hub: ToolHub,
        agent_id: str,
        *,
        session_id: str | None,
        task_id: str,
        budget: TaskBudget | None,
        clock: MonotonicClock,
    ) -> None:
        self._hub = hub
        self.agent_id = agent_id
        self.session_id = session_id
        self.task_id = task_id
        self.budget = budget or TaskBudget()
        self._clock = clock
        self._started = clock()
        self.calls_made = 0

    @property
    def seconds_left(self) -> float:
        return self.budget.max_seconds - (self._clock() - self._started)

    async def call(
        self, tool_id: str, operation: str, arguments: Mapping[str, JsonValue] | None = None
    ) -> dict[str, Any]:
        """Call `operation` on `tool_id`. Returns the tool's structured result."""
        args: dict[str, JsonValue] = dict(arguments or {})
        started = time.perf_counter()
        status, result_bytes, error = CallStatus.FAILED, 0, ""
        try:
            self._check_permission(tool_id, operation)
            self._check_budget()
            self.calls_made += 1
            result = await self._invoke(tool_id, operation, args)
            text = _text_of(result)
            result_bytes = len(text.encode("utf-8"))
            if result.is_error:
                status, error = CallStatus.TOOL_ERROR, text
                raise ToolCallError(text)
            status = CallStatus.OK
            return result.structured_content or {}
        except ToolPermissionError as exc:
            status, error = CallStatus.DENIED, str(exc)
            raise
        except BudgetExceededError as exc:
            status, error = CallStatus.BUDGET_EXCEEDED, str(exc)
            raise
        except ToolCallError:
            raise
        except Exception as exc:
            status, error = CallStatus.FAILED, type(exc).__name__
            raise
        finally:
            await self._audit(tool_id, operation, args, status, result_bytes, error, started)

    # --- steps ---------------------------------------------------------------------------

    def _check_permission(self, tool_id: str, operation: str) -> None:
        access = self._hub._registry_operations(tool_id).get(operation)
        if access is None:
            raise ToolPermissionError(f"unknown operation {tool_id}.{operation}")
        if not self._hub.matrix.allows(self.agent_id, tool_id, access):
            raise ToolPermissionError(
                f"{self.agent_id} may not {access.value} on {tool_id} ({operation})"
            )

    def _check_budget(self) -> None:
        if self.calls_made >= self.budget.max_calls:
            raise BudgetExceededError(
                f"task {self.task_id}: call budget of {self.budget.max_calls} used up"
            )
        if self.seconds_left <= 0:
            raise BudgetExceededError(
                f"task {self.task_id}: time budget of {self.budget.max_seconds:g}s used up"
            )

    async def _invoke(
        self, tool_id: str, operation: str, args: dict[str, JsonValue]
    ) -> CallToolResult:
        connection = self._hub._connection(tool_id)
        with self._hub.tracer.span(
            f"{tool_id}.{operation}",
            kind="tool",
            input=args,
            metadata={"agent": self.agent_id, "task": self.task_id},
        ) as span:
            try:
                result = await asyncio.wait_for(
                    connection.call_tool(operation, args), timeout=max(self.seconds_left, 0.001)
                )
            except TimeoutError:
                raise BudgetExceededError(
                    f"task {self.task_id}: {tool_id}.{operation} outlasted the time budget"
                ) from None
            if span is not None:
                span.update(
                    output=result.structured_content,
                    error="the tool reported an error" if result.is_error else None,
                )
            return result

    async def _audit(
        self,
        tool_id: str,
        operation: str,
        args: dict[str, JsonValue],
        status: CallStatus,
        result_bytes: int,
        error: str,
        started: float,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "tool": tool_id,
            "operation": operation,
            "task_id": self.task_id,
            "arguments_sha256": _arguments_hash(args),
            "status": status.value,
            "result_bytes": result_bytes,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if error:
            payload["error"] = error[:500]
        event = NewAuditEvent(
            session_id=self.session_id,
            actor=self.agent_id,
            event_type=TOOL_CALL_EVENT,
            payload=payload,
        )
        await asyncio.to_thread(self._hub.context.audit.append, event)


def _arguments_hash(args: dict[str, JsonValue]) -> str:
    try:
        return content_hash(args)
    except ValueError:  # e.g. NaN: still record that a call happened
        return content_hash(json.dumps(args, default=str, sort_keys=True))


def _text_of(result: CallToolResult) -> str:
    if result.structured_content is not None and not result.is_error:
        return json.dumps(result.structured_content, separators=(",", ":"), default=str)
    return " ".join(c.text for c in result.content if isinstance(c, TextContent))

"""Interfaces for what a use-case pack registers with core.

These are structural protocols only. The registries that store them, and the
dispatcher that runs tasks, arrive in SP2 (E6-02). Core code depends on these
shapes and never on a concrete MDX task, agent or tool, so new capability is
added by registering, not by editing core (plan §2.7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ToolGroup(StrEnum):
    """Tool groups from the architecture's tool layer (PDF §2.1)."""

    ADB_SYSTEM = "adb_system"  # read-only ADB systems: MDX, iFIRST, OneView, Reconciliation
    SOLUTION_DATA = "solution_data"  # the solution's own stores: cases, materiality, audit
    MODEL_ACCESS = "model_access"  # LLM gateway, Exception scoring model


class ToolAccess(StrEnum):
    """Access levels, as in the PDF §2.1 tool table (Read / Append / Write / Call)."""

    READ = "read"
    APPEND = "append"  # add records, never change or remove them (audit trail)
    WRITE = "write"  # create or update the solution's own records (cases, materiality)
    CALL = "call"  # invoke a model (LLM gateway, Exception scoring model)


@runtime_checkable
class ToolSpec(Protocol):
    """An MCP tool (server) an agent may be permitted to call."""

    @property
    def id(self) -> str:
        """Stable identifier, used as the row key in the permission matrix."""
        ...

    @property
    def group(self) -> ToolGroup: ...

    @property
    def description(self) -> str: ...

    @property
    def operations(self) -> Mapping[str, ToolAccess]:
        """Each MCP operation (tool name) on this server and the access level it needs."""
        ...


@runtime_checkable
class AgentSpec(Protocol):
    """A specialist agent (or the Master Agent) that owns one part of the run."""

    @property
    def id(self) -> str: ...

    @property
    def description(self) -> str:
        """Shown to the Master Agent as part of its catalogue (catalogue as data)."""
        ...


@runtime_checkable
class TaskSpec(Protocol):
    """A unit of work the dispatcher can run as a plan step."""

    @property
    def id(self) -> str:
        """Stable identifier, e.g. "<agent>.<task>". Plans refer to tasks by this ID."""
        ...

    @property
    def agent(self) -> str:
        """ID of the agent that owns this task."""
        ...

    @property
    def description(self) -> str: ...

    @property
    def input_model(self) -> type[BaseModel]: ...

    @property
    def output_model(self) -> type[BaseModel]: ...

    @property
    def review_checks(self) -> Sequence[str]:
        """IDs of the deterministic checks run at the review point after this task."""
        ...

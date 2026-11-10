"""Tool registry: which MCP tool servers exist, and how to build each one at runtime.

Packs register a `ToolSpec` (identity and operations) plus a factory, because a
server often needs runtime things to exist first: the MDX server needs the data
directory and the expected-file manifest, the audit server needs the audit trail and
the cases server the case store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from mcp.server.mcpserver import MCPServer

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.persistence.case_store import CaseStore
from mdx_assessment.core.registry.specs import ToolSpec
from mdx_assessment.settings import Settings


@dataclass(frozen=True)
class ToolContext:
    """What a tool factory may use to build its server."""

    config: LoadedConfig
    settings: Settings
    audit: AuditTrail
    cases: CaseStore


ToolFactory = Callable[[ToolContext], MCPServer]


@dataclass(frozen=True)
class ToolRegistration:
    spec: ToolSpec
    factory: ToolFactory


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolRegistration] = field(default_factory=dict)

    def register(self, spec: ToolSpec, factory: ToolFactory) -> None:
        if spec.id in self._tools:
            raise ValueError(f"tool {spec.id!r} is already registered")
        if not spec.operations:
            raise ValueError(f"tool {spec.id!r} declares no operations")
        self._tools[spec.id] = ToolRegistration(spec, factory)

    def get(self, tool_id: str) -> ToolRegistration:
        try:
            return self._tools[tool_id]
        except KeyError:
            raise KeyError(f"unknown tool {tool_id!r}") from None

    def __iter__(self) -> Iterator[ToolRegistration]:
        return iter(self._tools.values())

    @property
    def specs(self) -> list[ToolSpec]:
        return [r.spec for r in self._tools.values()]

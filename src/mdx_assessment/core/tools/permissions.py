"""Per-agent tool permission matrix (`config/permissions.yaml`, PDF §2.1 tool table).

Rows are tools, columns are agents, cells are access levels (read, append, write,
call). Anything not granted is denied: an agent missing from a row, a tool missing
from the matrix, or an operation needing a level the cell doesn't list.

Changing who may do what is a reviewed, versioned config change (PDF §3.1), and the
matrix is checked at startup against the registered tools (`check_against_tools`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.core.registry.specs import ToolAccess, ToolSpec

PERMISSIONS_PATH = "permissions.yaml"

AgentId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
ToolId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class PermissionMatrix(DomainModel):
    version: Annotated[str, StringConstraints(min_length=1)]
    agents: tuple[AgentId, ...] = Field(min_length=1)
    """Every agent that may hold a tool permission."""
    tools: dict[ToolId, dict[AgentId, tuple[ToolAccess, ...]]]

    @model_validator(mode="after")
    def _known_agents_no_duplicates(self) -> Self:
        problems = []
        if len(set(self.agents)) != len(self.agents):
            problems.append("duplicate agent in `agents`")
        for tool, row in self.tools.items():
            unknown = sorted(set(row) - set(self.agents))
            if unknown:
                problems.append(f"{tool}: unknown agent(s) {unknown}")
            for agent, levels in row.items():
                if len(set(levels)) != len(levels):
                    problems.append(f"{tool}.{agent}: duplicate access level")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def allows(self, agent_id: str, tool_id: str, access: ToolAccess) -> bool:
        return access in self.tools.get(tool_id, {}).get(agent_id, ())

    def check_against_tools(self, tools: Iterable[ToolSpec]) -> list[str]:
        """Problems between the matrix and the registered tools (empty when consistent)."""
        specs = {t.id: t for t in tools}
        problems = [
            f"{PERMISSIONS_PATH}: tool {tool!r} has a row but is not registered"
            for tool in sorted(set(self.tools) - set(specs))
        ]
        problems += [
            f"{PERMISSIONS_PATH}: registered tool {tool!r} has no row (add one, even if empty)"
            for tool in sorted(set(specs) - set(self.tools))
        ]
        for tool_id, row in self.tools.items():
            spec = specs.get(tool_id)
            if spec is None:
                continue
            available = {access.value for access in spec.operations.values()}
            for agent, levels in row.items():
                unusable = sorted({level.value for level in levels} - available)
                if unusable:
                    problems.append(
                        f"{PERMISSIONS_PATH}: {tool_id}.{agent} grants {unusable}, but "
                        f"{tool_id} only has {sorted(available)} operations"
                    )
        return problems

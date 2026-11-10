"""The registries a use-case pack registers into, bundled so `register()` keeps one signature.

Lives at `core.platform` (not in `core.registry`) because it pulls in many subsystems;
`core.registry` stays a low-level package of interfaces that anything can import.

- `config`: config schemas (E1-06)
- `tools`: MCP tool servers and their operations (E3-03)
- `agents`, `tasks`, `checks`: agents, the tasks they run, and review checks (E6-02)
- `facts`: the pack's run-context facts builder (E6-03)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mdx_assessment.core.config import ConfigSchemaRegistry
from mdx_assessment.core.orchestration.graph import MASTER_AGENT
from mdx_assessment.core.orchestration.playbooks import PLAYBOOKS_PATTERN, PlaybookDocument
from mdx_assessment.core.orchestration.run_context import FactsBuilder
from mdx_assessment.core.orchestration.tasks import (
    AgentDefinition,
    AgentRegistry,
    ReviewCheckRegistry,
    TaskRegistry,
    check_consistency,
)
from mdx_assessment.core.tools.audit import AUDIT_TOOL_SPEC, create_audit_server
from mdx_assessment.core.tools.cases import CASES_TOOL_SPEC, create_cases_server
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolRegistry


@dataclass
class PlatformRegistries:
    config: ConfigSchemaRegistry = field(default_factory=ConfigSchemaRegistry)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    agents: AgentRegistry = field(default_factory=AgentRegistry)
    tasks: TaskRegistry = field(default_factory=TaskRegistry)
    checks: ReviewCheckRegistry = field(default_factory=ReviewCheckRegistry)
    facts: FactsBuilder | None = None
    """The pack's run-context facts (one use case per deployment for now, plan §2.7)."""

    def problems(self, permissions: PermissionMatrix) -> list[str]:
        """Inconsistencies between the registries and the permission matrix."""
        found = check_consistency(self.tasks, self.agents, self.checks)
        found += permissions.check_against_tools(self.tools.specs)
        found += [
            f"{PERMISSIONS_PATH}: registered agent {a.id!r} is not in `agents`"
            for a in self.agents
            if a.id not in permissions.agents
        ]
        return found


def create_platform_registries() -> PlatformRegistries:
    """Registries pre-loaded with what core itself owns."""
    registries = PlatformRegistries()
    registries.config.register(PLAYBOOKS_PATTERN, PlaybookDocument, owner="core")
    registries.config.register(PERMISSIONS_PATH, PermissionMatrix, owner="core")
    registries.tools.register(AUDIT_TOOL_SPEC, lambda ctx: create_audit_server(ctx.audit))
    registries.tools.register(CASES_TOOL_SPEC, lambda ctx: create_cases_server(ctx.cases))
    registries.agents.register(
        AgentDefinition(
            MASTER_AGENT,
            "Supervisor: plans the run within the selected playbook, reviews every step's "
            "output, decides what happens next and writes the run opinion.",
        )
    )
    return registries

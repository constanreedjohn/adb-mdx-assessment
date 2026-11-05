"""The registries a use-case pack registers into, bundled so `register()` keeps one signature.

Lives at `core.platform` (not in `core.registry`) because it pulls in many subsystems;
`core.registry` stays a low-level package of interfaces that anything can import.

- `config`: config schemas (E1-06)
- `tools`: MCP tool servers and their operations (E3-03)

SP2 adds the task and agent registries (E6-02) as further fields; packs'
`register(registries)` signatures stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mdx_assessment.core.config import ConfigSchemaRegistry
from mdx_assessment.core.orchestration.playbooks import PLAYBOOKS_PATTERN, PlaybookDocument
from mdx_assessment.core.tools.audit import AUDIT_TOOL_SPEC, create_audit_server
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolRegistry


@dataclass
class PlatformRegistries:
    config: ConfigSchemaRegistry = field(default_factory=ConfigSchemaRegistry)
    tools: ToolRegistry = field(default_factory=ToolRegistry)


def create_platform_registries() -> PlatformRegistries:
    """Registries pre-loaded with what core itself owns."""
    registries = PlatformRegistries()
    registries.config.register(PLAYBOOKS_PATTERN, PlaybookDocument, owner="core")
    registries.config.register(PERMISSIONS_PATH, PermissionMatrix, owner="core")
    registries.tools.register(AUDIT_TOOL_SPEC, lambda ctx: create_audit_server(ctx.audit))
    return registries

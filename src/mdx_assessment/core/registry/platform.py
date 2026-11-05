"""The registries a use-case pack registers into, bundled so `register()` keeps one signature.

SP1 has the config schema registry. SP2 adds the task, agent and tool registries
(E6-02, E3-03) as further fields; packs' `register(registries)` signatures stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mdx_assessment.core.config import ConfigSchemaRegistry
from mdx_assessment.core.orchestration.playbooks import PLAYBOOKS_PATTERN, PlaybookDocument


@dataclass
class PlatformRegistries:
    config: ConfigSchemaRegistry = field(default_factory=ConfigSchemaRegistry)


def create_platform_registries() -> PlatformRegistries:
    """Registries pre-loaded with what core itself owns (e.g. the playbook schema)."""
    registries = PlatformRegistries()
    registries.config.register(PLAYBOOKS_PATTERN, PlaybookDocument, owner="core")
    return registries

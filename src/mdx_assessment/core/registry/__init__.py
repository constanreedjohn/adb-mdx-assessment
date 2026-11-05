"""Task, agent and tool registries. Use-case packs register into these; core never names them."""

from mdx_assessment.core.registry.platform import PlatformRegistries, create_platform_registries
from mdx_assessment.core.registry.specs import AgentSpec, TaskSpec, ToolGroup, ToolSpec

__all__ = [
    "AgentSpec",
    "PlatformRegistries",
    "TaskSpec",
    "ToolGroup",
    "ToolSpec",
    "create_platform_registries",
]

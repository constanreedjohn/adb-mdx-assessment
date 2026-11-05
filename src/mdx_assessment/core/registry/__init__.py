"""Interfaces for what packs register: tasks, agents, tools. Low-level: imports nothing else
from core, so any module can depend on it. The registries bundle lives in `core.platform`."""

from mdx_assessment.core.registry.specs import AgentSpec, TaskSpec, ToolAccess, ToolGroup, ToolSpec

__all__ = ["AgentSpec", "TaskSpec", "ToolAccess", "ToolGroup", "ToolSpec"]

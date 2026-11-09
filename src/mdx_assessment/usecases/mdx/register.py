"""Registers the MDX pack's config schemas, tools, agents and tasks with core.

Filled in as each piece lands:
- E1-06: config schemas (data-type profiles, expected-file manifest)
- E2-01: simulator config (`sim/linear_day.yaml`, optional)
- E3-01/E3-03: MDX data tool (simulated backend); its permission row is in config/permissions.yaml
- E6-02: agents and stub tasks `input_validation.check_files_and_intake`,
  `market_data.near_zero_check`, `escalation_reporting.record_cases` (agents/skeleton.py)
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mdx_assessment.core.platform import PlatformRegistries
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.usecases.mdx.agents.skeleton import register_skeleton
from mdx_assessment.usecases.mdx.config import (
    DATA_TYPES_PATH,
    EXPECTED_FILES_PATH,
    DataTypeProfiles,
    ExpectedFiles,
)
from mdx_assessment.usecases.mdx.sim.config import SIM_LINEAR_DAY_PATH, SimLinearDay
from mdx_assessment.usecases.mdx.tools.mdx import (
    MDX_TOOL_SPEC,
    SimulatedMdxBackend,
    create_mdx_server,
)

USE_CASE_ID = "mdx"


def _mdx_server(ctx: ToolContext) -> MCPServer:
    """MDX data over the simulated backend. Production swaps in a backend over ADB's MDX."""
    backend = SimulatedMdxBackend(
        ctx.settings.app.data_dir, ctx.config.get(EXPECTED_FILES_PATH, ExpectedFiles)
    )
    return create_mdx_server(backend)


def register(registries: PlatformRegistries) -> None:
    """Register this pack with core. Called once by the composition root."""
    registries.config.register(DATA_TYPES_PATH, DataTypeProfiles, owner=USE_CASE_ID)
    registries.config.register(EXPECTED_FILES_PATH, ExpectedFiles, owner=USE_CASE_ID)
    # Simulator settings: optional, so a deployment on real data needs no sim config.
    registries.config.register(SIM_LINEAR_DAY_PATH, SimLinearDay, owner=USE_CASE_ID, required=False)
    registries.tools.register(MDX_TOOL_SPEC, _mdx_server)
    register_skeleton(registries)

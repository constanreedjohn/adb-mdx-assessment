"""Registers the MDX pack's config schemas (and later tasks, agents and tools) with core.

Filled in as each piece lands:
- E1-06: config schemas (data-type profiles, expected-file manifest)
- E2-01: simulator config (`sim/linear_day.yaml`, optional)
- SP2 (E6-02): stub tasks `input_validation.check_files_and_intake`,
  `market_data.near_zero_check`, `escalation_reporting.record_cases`
- SP2 (E3-01, E3-03): MDX data tool and its permission-matrix rows
"""

from __future__ import annotations

from mdx_assessment.core.registry import PlatformRegistries
from mdx_assessment.usecases.mdx.config import (
    DATA_TYPES_PATH,
    EXPECTED_FILES_PATH,
    DataTypeProfiles,
    ExpectedFiles,
)
from mdx_assessment.usecases.mdx.sim.config import SIM_LINEAR_DAY_PATH, SimLinearDay

USE_CASE_ID = "mdx"


def register(registries: PlatformRegistries) -> None:
    """Register this pack with core. Called once by the composition root."""
    registries.config.register(DATA_TYPES_PATH, DataTypeProfiles, owner=USE_CASE_ID)
    registries.config.register(EXPECTED_FILES_PATH, ExpectedFiles, owner=USE_CASE_ID)
    # Simulator settings: optional, so a deployment on real data needs no sim config.
    registries.config.register(SIM_LINEAR_DAY_PATH, SimLinearDay, owner=USE_CASE_ID, required=False)

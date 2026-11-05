"""Build a simulated day from a validated config tree (shared by the CLI and tests)."""

from __future__ import annotations

from datetime import date

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.usecases.mdx.config import (
    DATA_TYPES_PATH,
    EXPECTED_FILES_PATH,
    DataTypeProfiles,
    ExpectedFiles,
)
from mdx_assessment.usecases.mdx.domain import MarketDataKind
from mdx_assessment.usecases.mdx.sim.config import SIM_LINEAR_DAY_PATH, SimLinearDay
from mdx_assessment.usecases.mdx.sim.generator import SimulatedDay, generate_linear_day


def build_linear_day(
    config: LoadedConfig, business_date: date, *, seed: int | None = None
) -> SimulatedDay:
    sim = config.get(SIM_LINEAR_DAY_PATH, SimLinearDay)
    manifest = config.get(EXPECTED_FILES_PATH, ExpectedFiles)
    matches = [f for f in manifest.files if f.id == sim.file_id]
    if not matches:
        raise ValueError(f"sim file_id {sim.file_id!r} is not in {EXPECTED_FILES_PATH}")
    rate = config.get(DATA_TYPES_PATH, DataTypeProfiles).for_kind(MarketDataKind.RATE)
    return generate_linear_day(
        sim,
        business_date,
        expected_file=matches[0],
        timezone=manifest.timezone,
        near_zero_floor=rate.near_zero_floor,
        seed=seed,
    )

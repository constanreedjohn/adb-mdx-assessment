"""Registers the MDX pack's tasks, agents, tools and config schemas with core.

Stub for now. Filled in as each piece lands:
- E1-06: config schemas (data-type profiles, expected-file manifest)
- SP2 (E6-02): stub tasks `input_validation.check_files_and_intake`,
  `market_data.near_zero_check`, `escalation_reporting.record_cases`
- SP2 (E3-01, E3-03): MDX data tool and its permission-matrix rows
"""

from __future__ import annotations

USE_CASE_ID = "mdx"


def register() -> None:
    """Register this pack with core. Called once by the composition root."""

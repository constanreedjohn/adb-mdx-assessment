"""Run-context facts for an MDX session (plan §2.5). Code only, no reasoning.

In the skeleton every fact comes from the MDX data tool: expected against received
files, record counts and the raw flag count. Later stories add processing and export
status, the portfolio snapshot and borderline observations (E6-06).

If MDX can't answer for the date, the facts say so: no files received, and a
`flag_count` of None rather than a guess.
"""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue

from mdx_assessment.core.domain import Fact, RunContext
from mdx_assessment.core.orchestration.run_context import FactsRequest
from mdx_assessment.core.tools.client import ToolCallError
from mdx_assessment.usecases.mdx.checks.completeness import assess_completeness
from mdx_assessment.usecases.mdx.config import RUN_CONTEXT_PATH, RunContextSettings
from mdx_assessment.usecases.mdx.tools.mdx import (
    GET_FILE_METADATA,
    GET_FLAGS,
    LIST_EXPECTED_FILES,
    ExpectedFilesResult,
    FileMetadataResult,
)
from mdx_assessment.usecases.mdx.tools.mdx import TOOL_ID as MDX_DATA


def _source(operation: str) -> str:
    return f"{MDX_DATA}.{operation}"


async def _call_or_none(
    request: FactsRequest, operation: str, arguments: dict[str, JsonValue]
) -> dict[str, Any] | None:
    try:
        return await request.tools.call(MDX_DATA, operation, arguments)
    except ToolCallError:  # e.g. MDX has no data for this date
        return None


async def build_mdx_facts(request: FactsRequest) -> RunContext:
    state = request.state
    settings = request.config.get(RUN_CONTEXT_PATH, RunContextSettings)
    business_date = state.business_date.isoformat()
    day: dict[str, JsonValue] = {"business_date": business_date}
    typed: dict[str, JsonValue] = {**day, "data_type": state.data_type}
    scope: dict[str, JsonValue] = dict(typed)

    expected = ExpectedFilesResult.model_validate(
        await request.tools.call(MDX_DATA, LIST_EXPECTED_FILES, typed)
    ).files
    metadata = await _call_or_none(request, GET_FILE_METADATA, day)
    flags = await _call_or_none(request, GET_FLAGS, typed)
    received = FileMetadataResult.model_validate(metadata).files if metadata else []
    completeness = assess_completeness(expected, received)  # the same check as E7-01

    return RunContext(
        facts=(
            Fact.create("data_type", state.data_type, source="session", scope=scope),
            Fact.create("business_date", business_date, source="session", scope=scope),
            Fact.create(
                "files_received",
                completeness.files_received,
                expected=completeness.files_expected,
                source=_source(GET_FILE_METADATA),
                scope=scope,
            ),
            Fact.create(
                "records_received",
                completeness.records_received,
                expected=completeness.records_expected,
                source=_source(GET_FILE_METADATA),
                scope=scope,
            ),
            Fact.create(
                "flag_count",
                flags["count"] if flags is not None else None,
                threshold=settings.high_volume_flag_threshold,
                source=_source(GET_FLAGS),
                scope=scope,
            ),
        ),
    )

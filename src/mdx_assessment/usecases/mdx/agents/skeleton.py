"""The MDX pack's agents and their walking-skeleton tasks (SP2 stubs).

Registered with core so the dispatcher can run a three-step Normal EOD plan end to
end. SP3 replaces each stub's handler with the real step (E7-01/02, E8-01/02,
E11-01/02); the task IDs, agents and graph stay the same.

- The Input Validation stub really reads MDX through its tool client (expected files,
  file metadata, flags), so the permission matrix and audit already apply.
- The Market Data Assessment stub picks the intake result up from the shared state.
- The Escalation and Reporting stub creates no cases yet.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mdx_assessment.core.orchestration.tasks import (
    AgentDefinition,
    TaskContext,
    TaskDefinition,
)
from mdx_assessment.core.platform import PlatformRegistries
from mdx_assessment.core.tools.client import TaskBudget
from mdx_assessment.usecases.mdx.tools.mdx import GET_FILE_METADATA, GET_FLAGS, LIST_EXPECTED_FILES
from mdx_assessment.usecases.mdx.tools.mdx import TOOL_ID as MDX_DATA

INPUT_VALIDATION = "input_validation"
MARKET_DATA_ASSESSMENT = "market_data_assessment"
ESCALATION_REPORTING = "escalation_reporting"

CHECK_FILES_AND_INTAKE = "input_validation.check_files_and_intake"
NEAR_ZERO_CHECK = "market_data.near_zero_check"
RECORD_CASES = "escalation_reporting.record_cases"

AGENTS = (
    AgentDefinition(
        INPUT_VALIDATION,
        "Makes sure inputs arrived, were processed and are usable: file presence and "
        "record counts, reading and normalizing MDX flags.",
    ),
    AgentDefinition(
        MARKET_DATA_ASSESSMENT,
        "Turns exceptions into a ranked, explained list: movement checks such as "
        "near-zero threshold artifacts, materiality, ranking and classification.",
    ),
    AgentDefinition(
        ESCALATION_REPORTING,
        "Records outcomes and prepares what the analyst needs to act: case records "
        "and the run report.",
    ),
)


class _Stub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stub: Literal[True] = True


class IntakeStubResult(_Stub):
    files_expected: int = Field(ge=0)
    files_received: int = Field(ge=0)
    record_counts: dict[str, int]
    flags_read: int = Field(ge=0)


class AssessmentStubResult(_Stub):
    items_received: int = Field(ge=0)


class CasesStubResult(_Stub):
    cases_created: int = Field(ge=0)


async def check_files_and_intake(ctx: TaskContext) -> IntakeStubResult:
    day = {"business_date": ctx.inputs.business_date.isoformat()}
    typed = {**day, "data_type": ctx.inputs.data_type}
    expected = await ctx.tools.call(MDX_DATA, LIST_EXPECTED_FILES, typed)
    metadata = await ctx.tools.call(MDX_DATA, GET_FILE_METADATA, day)
    flags = await ctx.tools.call(MDX_DATA, GET_FLAGS, typed)
    return IntakeStubResult(
        files_expected=len(expected["files"]),
        files_received=len(metadata["files"]),
        record_counts={f["file_name"]: f["record_count"] for f in metadata["files"]},
        flags_read=flags["count"],
    )


async def near_zero_check(ctx: TaskContext) -> AssessmentStubResult:
    intake = ctx.result_of(CHECK_FILES_AND_INTAKE) or {}
    flags = intake.get("flags_read", 0)
    return AssessmentStubResult(items_received=flags if isinstance(flags, int) else 0)


async def record_cases(ctx: TaskContext) -> CasesStubResult:
    return CasesStubResult(cases_created=0)


TASKS = (
    TaskDefinition(
        id=CHECK_FILES_AND_INTAKE,
        agent=INPUT_VALIDATION,
        description="Check the expected MDX files arrived with the expected record counts, "
        "then read the exception flags. (Skeleton stub: reads, doesn't normalize yet.)",
        output_model=IntakeStubResult,
        handler=check_files_and_intake,
        budget=TaskBudget(max_calls=10, max_seconds=30),
    ),
    TaskDefinition(
        id=NEAR_ZERO_CHECK,
        agent=MARKET_DATA_ASSESSMENT,
        description="Assess each flag for a near-zero threshold artifact against 60-day "
        "history. (Skeleton stub: counts the items it received.)",
        output_model=AssessmentStubResult,
        handler=near_zero_check,
    ),
    TaskDefinition(
        id=RECORD_CASES,
        agent=ESCALATION_REPORTING,
        description="Record one case per assessed item and produce the run report. "
        "(Skeleton stub: creates no cases.)",
        output_model=CasesStubResult,
        handler=record_cases,
    ),
)


def register_skeleton(registries: PlatformRegistries) -> None:
    for agent in AGENTS:
        registries.agents.register(agent)
    for task in TASKS:
        registries.tasks.register(task)

"""The MDX pack's agents and their walking-skeleton tasks (SP2 stubs).

Registered with core so the dispatcher can run a three-step Normal EOD plan end to
end. SP3 replaces each stub's handler with the real step (E7-01/02, E8-01/02,
E11-01/02); the task IDs, agents and graph stay the same.

- Input Validation is real (E7-01, E7-02, `input_validation.py`): file presence and
  record counts with a completeness finding, then the flags normalized to bps.
- Market Data Assessment is real (E8-01, `market_data.py`): the near-zero check.
- The Escalation and Reporting stub creates no cases yet.
"""

from __future__ import annotations

from mdx_assessment.core.orchestration.tasks import (
    AgentDefinition,
    TaskDefinition,
)
from mdx_assessment.core.platform import PlatformRegistries
from mdx_assessment.core.tools.client import TaskBudget
from mdx_assessment.usecases.mdx.agents.escalation import RecordCasesResult, record_cases
from mdx_assessment.usecases.mdx.agents.explanations import PROMPT_ID as EXPLAIN_PROMPT_ID
from mdx_assessment.usecases.mdx.agents.explanations import (
    PROMPT_VERSION as EXPLAIN_PROMPT_VERSION,
)
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    ESCALATION_REPORTING,
    INPUT_VALIDATION,
    MARKET_DATA_ASSESSMENT,
    NEAR_ZERO_CHECK,
    RECORD_CASES,
    REVISE_DRAFTS,
)
from mdx_assessment.usecases.mdx.agents.input_validation import (
    IntakeResult,
    check_files_and_intake,
)
from mdx_assessment.usecases.mdx.agents.investigate import PROMPT_ID as INVESTIGATE_PROMPT_ID
from mdx_assessment.usecases.mdx.agents.investigate import (
    PROMPT_VERSION as INVESTIGATE_PROMPT_VERSION,
)
from mdx_assessment.usecases.mdx.agents.market_data import (
    MarketAssessmentResult,
    RevisionResult,
    near_zero_check,
    revise_drafts,
)
from mdx_assessment.usecases.mdx.agents.teacher import PROMPT_ID as VALIDATE_PROMPT_ID
from mdx_assessment.usecases.mdx.agents.teacher import PROMPT_VERSION as VALIDATE_PROMPT_VERSION
from mdx_assessment.usecases.mdx.agents.teacher import validate_drafts
from mdx_assessment.usecases.mdx.checks.review import (
    CHECKS,
    DATASET_COMPLETENESS,
    FLAG_NORMALIZATION,
    FLAG_RECONCILIATION,
)

__all__ = [
    "AGENTS",
    "CHECK_FILES_AND_INTAKE",
    "ESCALATION_REPORTING",
    "INPUT_VALIDATION",
    "MARKET_DATA_ASSESSMENT",
    "NEAR_ZERO_CHECK",
    "RECORD_CASES",
    "TASKS",
    "register_skeleton",
]

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


TASKS = (
    TaskDefinition(
        id=CHECK_FILES_AND_INTAKE,
        agent=INPUT_VALIDATION,
        description="Check the expected MDX files arrived with the expected record counts, "
        "and record a dataset completeness finding; then read the exception flags.",
        output_model=IntakeResult,
        handler=check_files_and_intake,
        budget=TaskBudget(max_calls=10, max_seconds=30),
        review_checks=(DATASET_COMPLETENESS, FLAG_NORMALIZATION),
    ),
    TaskDefinition(
        id=NEAR_ZERO_CHECK,
        agent=MARKET_DATA_ASSESSMENT,
        description="Assess each flag for a near-zero threshold artifact: absolute change, "
        "relative change only above the near-zero floor, z-score against 60-day history; "
        "label each item, list them in review order and explain each in plain words; "
        "then investigate where unsure and draft a verdict per flag.",
        output_model=MarketAssessmentResult,
        handler=near_zero_check,
        # One history call per flag, a few for the investigation tools; time for the
        # explanation call and up to six student rounds.
        budget=TaskBudget(max_calls=60, max_seconds=300),
        prompts=(
            (EXPLAIN_PROMPT_ID, EXPLAIN_PROMPT_VERSION),
            (INVESTIGATE_PROMPT_ID, INVESTIGATE_PROMPT_VERSION),
            (VALIDATE_PROMPT_ID, VALIDATE_PROMPT_VERSION),
        ),
        validate=validate_drafts,
        review_checks=(FLAG_RECONCILIATION,),
        depends_on=(CHECK_FILES_AND_INTAKE,),
    ),
    TaskDefinition(
        id=REVISE_DRAFTS,
        agent=MARKET_DATA_ASSESSMENT,
        description="Send-back only, never planned: revise the drafts of the flags a review "
        "point named, with the teacher's reasons and each flag's remaining tool budget.",
        output_model=RevisionResult,
        handler=revise_drafts,
        # The budget's history reads again (one per flag sent back), plus the tools.
        budget=TaskBudget(max_calls=30, max_seconds=180),
        prompts=(
            (INVESTIGATE_PROMPT_ID, INVESTIGATE_PROMPT_VERSION),
            (VALIDATE_PROMPT_ID, VALIDATE_PROMPT_VERSION),
        ),
        depends_on=(NEAR_ZERO_CHECK,),
        validate=validate_drafts,
    ),
    TaskDefinition(
        id=RECORD_CASES,
        agent=ESCALATION_REPORTING,
        description="Record one open case per flag, in list order (verdict, then severity), "
        "each with its verdict, report, investigation trail and evidence.",
        output_model=RecordCasesResult,
        handler=record_cases,
        budget=TaskBudget(max_calls=60, max_seconds=60),  # one create_case per flag
        depends_on=(NEAR_ZERO_CHECK,),
    ),
)


def register_skeleton(registries: PlatformRegistries) -> None:
    for agent in AGENTS:
        registries.agents.register(agent)
    for task in TASKS:
        registries.tasks.register(task)
    for check in CHECKS:
        registries.checks.register(check)

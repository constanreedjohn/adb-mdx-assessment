"""Escalation and Reporting Agent, task `escalation_reporting.record_cases` (E11-01).

Code only, no LLM. One case per flag, written through the exception cases tool (the
only agent allowed to), status `open`, in the order the analyst should read them:

1. **Verdict**, as the teacher left it (`teacher.validated_drafts`): likely real first,
   then can't tell, then likely harmless.
2. **Then the E8-01 review order**: severity, then the largest |z|.

Each case is the flag's ROI report: what MDX flagged, the crucial results, the verdict
and the report, a suggested action, and the investigation trail (the rule trace, the
tools the student called and why, every teacher decision), with every evidence ID it
rests on. Cases are created in list order, so the store lists them in that order; each
also carries its `rank`.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mdx_assessment.core.domain import DomainModel, ItemDecision, ItemDecisionKind, Measure
from mdx_assessment.core.orchestration.tasks import TaskContext
from mdx_assessment.core.tools.cases import CREATE_CASE
from mdx_assessment.core.tools.cases import TOOL_ID as CASES
from mdx_assessment.usecases.mdx.agents.ids import NEAR_ZERO_CHECK
from mdx_assessment.usecases.mdx.agents.investigate import ToolCallRecord
from mdx_assessment.usecases.mdx.agents.market_data import MarketAssessmentResult
from mdx_assessment.usecases.mdx.agents.teacher import validated_drafts
from mdx_assessment.usecases.mdx.checks.verdict import Verdict
from mdx_assessment.usecases.mdx.domain import ExplanationSource

VERDICT_ORDER = (Verdict.LIKELY_REAL, Verdict.CANT_TELL, Verdict.LIKELY_HARMLESS)


class SuggestedAction(StrEnum):
    """Set by code from the verdict. The analyst decides; nothing is closed or hidden."""

    REVIEW_FIRST = "review_first"
    REVIEW = "review"
    QUICK_CONFIRM = "quick_confirm"


ACTIONS = {
    Verdict.LIKELY_REAL: SuggestedAction.REVIEW_FIRST,
    Verdict.CANT_TELL: SuggestedAction.REVIEW,
    Verdict.LIKELY_HARMLESS: SuggestedAction.QUICK_CONFIRM,
}


class Report(DomainModel):
    text: str
    source: ExplanationSource
    fallback_reason: str | None = None


class Trail(DomainModel):
    """How the verdict was reached."""

    label: str
    """Code's classification (E8-01)."""
    rule_trace: tuple[str, ...]
    explanation: str
    starting_verdict: Verdict
    tool_calls: tuple[ToolCallRecord, ...]
    revision: int
    teacher_feedback: str | None
    teacher_decisions: tuple[ItemDecision, ...]


class CaseReport(DomainModel):
    """A case's payload: one flag's ROI report."""

    rank: int = Field(ge=1)
    flag_id: str
    item: str
    verdict: Verdict
    suggested_action: SuggestedAction
    validated: bool
    """True when the teacher accepted or decided on the draft; false when not reviewed."""
    teacher_decision: ItemDecisionKind | None
    severity: str
    mdx_shows: float | None
    """MDX's relative change, in percent, as its rule saw it."""
    change: float
    unit: str
    measures: tuple[Measure, ...]
    report: Report
    trail: Trail
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class CaseLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    case_id: str
    item: str
    verdict: Verdict
    created: bool


class RecordCasesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: tuple[CaseLine, ...]
    """In list order."""
    cases_created: int = Field(ge=0)
    cases_existing: int = Field(ge=0)
    """Already recorded for this session, e.g. when a step is resumed."""
    verdicts: dict[Verdict, int]
    not_validated: int = Field(ge=0)


def _latest_assessment(ctx: TaskContext) -> MarketAssessmentResult:
    raw = ctx.result_of(NEAR_ZERO_CHECK)
    if raw is None:
        raise ValueError("recording cases needs a completed near-zero step")
    return MarketAssessmentResult.model_validate(raw)


def case_reports(ctx: TaskContext) -> list[CaseReport]:
    """Every flag's report, in list order."""
    state = ctx.state
    assessed = _latest_assessment(ctx)
    outcomes = validated_drafts(state.execution, state.agent_results, state.validations)
    decisions: dict[str, list[ItemDecision]] = {}
    for v in state.validations:
        for decision in v.decisions:
            decisions.setdefault(decision.item_id, []).append(decision)

    rows = []
    for position, item in enumerate(assessed.items):  # E8-01 review order
        a = item.assessment
        outcome = outcomes[a.flag_id]
        d = outcome.draft
        evidence = dict.fromkeys(
            (
                *item.finding.evidence_ids,
                *d.evidence_ids,
                *(c.evidence_id for c in d.tool_calls if c.evidence_id),
            )
        )
        rows.append(
            (
                (VERDICT_ORDER.index(outcome.verdict), position),
                CaseReport(
                    rank=1,  # set below, once the list is ordered
                    flag_id=a.flag_id,
                    item=a.item_key,
                    verdict=outcome.verdict,
                    suggested_action=ACTIONS[outcome.verdict],
                    validated=outcome.validated,
                    teacher_decision=outcome.decision,
                    severity=a.severity.value,
                    mdx_shows=a.relative_change_pct,
                    change=a.change,
                    unit=item.finding.measures[0].unit if item.finding.measures else "",
                    measures=item.finding.measures,
                    report=Report(
                        text=d.report, source=d.source, fallback_reason=d.fallback_reason
                    ),
                    trail=Trail(
                        label=a.classification.value,
                        rule_trace=a.reasons,
                        explanation=item.explanation.text,
                        starting_verdict=d.starting_verdict,
                        tool_calls=d.tool_calls,
                        revision=d.revision,
                        teacher_feedback=d.teacher_feedback,
                        teacher_decisions=tuple(decisions.get(a.flag_id, ())),
                    ),
                    evidence_ids=tuple(evidence),
                ),
            )
        )
    rows.sort(key=lambda r: r[0])
    return [report.model_copy(update={"rank": n}) for n, (_, report) in enumerate(rows, start=1)]


async def record_cases(ctx: TaskContext) -> RecordCasesResult:
    reports = case_reports(ctx)
    lines = []
    for report in reports:
        payload: dict[str, JsonValue] = report.model_dump(mode="json")
        raw = await ctx.tools.call(
            CASES,
            CREATE_CASE,
            {"session_id": ctx.inputs.session_id, "item_key": report.item, "payload": payload},
        )
        case = raw["case"]
        lines.append(
            CaseLine(
                rank=report.rank,
                case_id=case["case_id"],
                item=report.item,
                verdict=report.verdict,
                created=bool(raw.get("created")),
            )
        )
    verdicts = Counter(line.verdict for line in lines)
    return RecordCasesResult(
        cases=tuple(lines),
        cases_created=sum(1 for line in lines if line.created),
        cases_existing=sum(1 for line in lines if not line.created),
        verdicts={v: verdicts.get(v, 0) for v in Verdict},
        not_validated=sum(1 for r in reports if not r.validated),
    )

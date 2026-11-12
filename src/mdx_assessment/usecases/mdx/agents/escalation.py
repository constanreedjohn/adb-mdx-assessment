"""Escalation and Reporting Agent, task `escalation_reporting.record_cases` (E11-01, E11-02).

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

Then the **raw run report** (E11-02): the day's figures, counted by code from the run's
own results, so they are right even when the LLM is down. Files checked, flags read and
normalized, flags assessed and investigated, labels, verdicts, tools called, send-backs,
the teacher's decisions and cases created.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mdx_assessment.core.domain import DomainModel, ItemDecision, ItemDecisionKind, Measure
from mdx_assessment.core.orchestration.tasks import TaskContext
from mdx_assessment.core.tools.cases import CREATE_CASE
from mdx_assessment.core.tools.cases import TOOL_ID as CASES
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    NEAR_ZERO_CHECK,
    REVISE_DRAFTS,
)
from mdx_assessment.usecases.mdx.agents.input_validation import IntakeResult
from mdx_assessment.usecases.mdx.agents.investigate import ToolCallRecord
from mdx_assessment.usecases.mdx.agents.market_data import MarketAssessmentResult
from mdx_assessment.usecases.mdx.agents.teacher import validated_drafts
from mdx_assessment.usecases.mdx.checks.near_zero import Classification
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


class FileLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    status: str
    expected_records: int
    received_records: int | None


class RunReport(BaseModel):
    """The day's figures, by code (E11-02)."""

    model_config = ConfigDict(extra="forbid")

    business_date: str
    data_type: str
    files_expected: int = Field(ge=0)
    files_received: int = Field(ge=0)
    files: tuple[FileLine, ...]
    dataset_complete: bool
    flags_read: int = Field(ge=0)
    """MDX's flags for the date."""
    flags_normalized: int = Field(ge=0)
    flags_not_normalized: int = Field(ge=0)
    flags_assessed: int = Field(ge=0)
    """Flags that went through the crucial steps."""
    flags_investigated: int = Field(ge=0)
    """Flags for which the student called at least one tool."""
    labels: dict[str, int]
    """Code's classification (E8-01): threshold artifacts, large moves, ordinary, ..."""
    threshold_artifacts: int = Field(ge=0)
    large_moves: int = Field(ge=0)
    verdicts: dict[Verdict, int]
    """Final verdicts, after the teacher."""
    review_first: tuple[str, ...]
    """Items with a likely real verdict, in list order."""
    tools_called: int = Field(ge=0)
    tools_by_name: dict[str, int]
    tool_errors: int = Field(ge=0)
    send_backs: int = Field(ge=0)
    """Flags the teacher sent back."""
    send_back_rounds: int = Field(ge=0)
    teacher_decisions: dict[ItemDecisionKind, int]
    """The latest decision on each flag."""
    templates: dict[str, int]
    """Explanations and reports written by code instead of the LLM."""
    cases_created: int = Field(ge=0)
    cases_existing: int = Field(ge=0)


class RecordCasesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: tuple[CaseLine, ...]
    """In list order."""
    cases_created: int = Field(ge=0)
    cases_existing: int = Field(ge=0)
    """Already recorded for this session, e.g. when a step is resumed."""
    verdicts: dict[Verdict, int]
    not_validated: int = Field(ge=0)
    report: RunReport


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
    created = sum(1 for line in lines if line.created)
    return RecordCasesResult(
        cases=tuple(lines),
        cases_created=created,
        cases_existing=len(lines) - created,
        verdicts={v: verdicts.get(v, 0) for v in Verdict},
        not_validated=sum(1 for r in reports if not r.validated),
        report=run_report(ctx, reports, created, len(lines) - created),
    )


def run_report(
    ctx: TaskContext, reports: list[CaseReport], created: int, existing: int
) -> RunReport:
    state = ctx.state
    raw = ctx.result_of(CHECK_FILES_AND_INTAKE)
    if raw is None:
        raise ValueError("the run report needs a completed intake step")
    intake = IntakeResult.model_validate(raw)
    assessed = _latest_assessment(ctx)
    calls = [c for r in reports for c in r.trail.tool_calls]
    latest = {r.flag_id: r.teacher_decision for r in reports if r.teacher_decision is not None}
    sent_back = {
        d.item_id
        for v in state.validations
        for d in v.decisions
        if d.kind is ItemDecisionKind.SEND_BACK
    }
    labels = Counter(i.assessment.classification.value for i in assessed.items)
    verdicts = Counter(r.verdict for r in reports)
    decisions = Counter(latest.values())
    files = intake.completeness
    return RunReport(
        business_date=ctx.inputs.business_date.isoformat(),
        data_type=ctx.inputs.data_type,
        files_expected=files.files_expected,
        files_received=files.files_received,
        files=tuple(
            FileLine(
                file_name=f.file_name,
                status=f.status.value,
                expected_records=f.expected_records,
                received_records=f.received_records,
            )
            for f in files.files
        ),
        dataset_complete=intake.dataset_complete,
        flags_read=intake.flags_read or 0,
        flags_normalized=len(intake.normalized_flags),
        flags_not_normalized=len(intake.flag_issues),
        flags_assessed=len(assessed.items),
        flags_investigated=sum(1 for r in reports if r.trail.tool_calls),
        labels={c.value: labels.get(c.value, 0) for c in Classification},
        threshold_artifacts=labels.get(Classification.THRESHOLD_ARTIFACT.value, 0),
        large_moves=labels.get(Classification.GENUINE_LARGE_MOVE.value, 0),
        verdicts={v: verdicts.get(v, 0) for v in Verdict},
        review_first=tuple(r.item for r in reports if r.verdict is Verdict.LIKELY_REAL),
        tools_called=len(calls),
        tools_by_name=dict(sorted(Counter(c.tool for c in calls).items())),
        tool_errors=sum(1 for c in calls if c.error),
        send_backs=len(sent_back),
        send_back_rounds=sum(1 for e in state.execution if e.task == REVISE_DRAFTS),
        teacher_decisions={k: decisions.get(k, 0) for k in ItemDecisionKind},
        templates={
            "explanations": sum(
                1 for i in assessed.items if i.explanation.source is ExplanationSource.TEMPLATE
            ),
            "reports": sum(1 for r in reports if r.report.source is ExplanationSource.TEMPLATE),
        },
        cases_created=created,
        cases_existing=existing,
    )

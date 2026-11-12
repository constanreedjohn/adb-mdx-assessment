"""Market Data Assessment Agent, task `market_data.near_zero_check` (E8-01 to E8-04).

Deterministic core, then explanation, then investigation:

1. Take the normalized flags and data-type profiles from the intake step's result.
2. For each flag, read the point's history through the MDX tool and measure its normal
   daily move (`history_stats`), recorded as evidence.
3. Label it with `assess_near_zero` (docs/business-rules.md §3), record the measures as
   evidence, and write one finding per flag citing the raw flag, the stored point, the
   history and the measures.
4. List the items in review order: severity first, then the largest |z|.
5. Explain each item in plain words (`explanations.explain`, E8-02): one batched LLM
   call, every number checked against the item's own evidence, templates as fallback.
6. Investigate and draft a verdict per flag (`investigate.investigate`, E8-04): the
   student asks for tools where the crucial results leave it unsure, within a budget per
   flag, and drafts a verdict that may only move from code's starting verdict towards
   caution. Tools read through the same `MdxReads` as step 2, so they reuse its reads.

The labels and the review order come only from steps 1 to 4. Nothing after can change them.

Task `market_data.revise_drafts` (E6-05) is the send-back: it runs only when a review
point decides a follow-up naming flags of an earlier near-zero step, and re-runs step 6
for those flags with the teacher's reasons. `current_drafts` gives each flag's latest
draft across both.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mdx_assessment.core.domain import Evidence, Finding, Measure
from mdx_assessment.core.gateway.grounding import GUARDRAILS_PATH, GuardrailsConfig
from mdx_assessment.core.orchestration.state import ExecutionEntry
from mdx_assessment.core.orchestration.tasks import TaskContext
from mdx_assessment.usecases.mdx.agents.explanations import (
    FALLBACK_REJECTED,
    FALLBACK_UNAVAILABLE,
    Explanation,
    ItemToExplain,
    explain,
)
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    NEAR_ZERO_CHECK,
    REVISE_DRAFTS,
)
from mdx_assessment.usecases.mdx.agents.input_validation import IntakeResult
from mdx_assessment.usecases.mdx.agents.investigate import (
    UNUSABLE,
    Draft,
    ToRevise,
    investigate,
    revise,
)
from mdx_assessment.usecases.mdx.checks.near_zero import (
    Classification,
    HistoryStats,
    NearZeroAssessment,
    assess_near_zero,
    history_stats,
    review_order,
)
from mdx_assessment.usecases.mdx.checks.normalize import NormalizedFlag
from mdx_assessment.usecases.mdx.checks.verdict import Verdict
from mdx_assessment.usecases.mdx.config import (
    INVESTIGATION_PATH,
    NEAR_ZERO_PATH,
    DataTypeProfile,
    InvestigationSettings,
    NearZeroSettings,
)
from mdx_assessment.usecases.mdx.domain import DataType, ExplanationSource, Unit
from mdx_assessment.usecases.mdx.investigation.reads import MdxReads
from mdx_assessment.usecases.mdx.investigation.toolkit import Investigation
from mdx_assessment.usecases.mdx.tools.mdx import GET_HISTORY
from mdx_assessment.usecases.mdx.tools.mdx import TOOL_ID as MDX_DATA

DEFAULT_MAX_AI_EXPLANATIONS = 100


class AssessedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: NearZeroAssessment
    finding: Finding
    explanation: Explanation
    draft: Draft


class MarketAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[AssessedItem, ...]
    """In review order."""
    counts: dict[Classification, int]
    explanation_sources: dict[ExplanationSource, int]
    explanation_attempts: int = Field(ge=0)
    """LLM calls made for the explanations (0 when templates were used without one)."""
    explanation_rejections: tuple[str, ...] = ()
    """Why rejected explanation attempts were rejected."""
    verdicts: dict[Verdict, int]
    draft_sources: dict[ExplanationSource, int]
    investigation_rounds: int = Field(ge=0)
    """Student rounds run (each one batched LLM call, plus any retry)."""
    investigation_attempts: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    """Investigation tool calls over every flag."""
    investigation_rejections: tuple[str, ...] = ()
    """Why the gateway rejected student rounds (structure)."""
    evidence: tuple[Evidence, ...] = Field(default=())
    """The records this step added (history, measures, tool results); flags and points
    are the intake step's."""
    not_assessed: tuple[str, ...] = ()
    """Flag IDs the intake step couldn't normalize, carried for reconciliation (C-1)."""
    used_fallback: bool = False
    """True when the LLM's explanations or drafts weren't usable and templates replaced
    them: the step is then `done_with_fallback`. Templates by design (LLM off, over the
    limit) aren't, and neither is a spent tool budget: that is a verdict, not a failure."""


def _summary(a: NearZeroAssessment, unit: str) -> str:
    rel = (
        f"{a.relative_change_pct:+.1f}%"
        if a.relative_change_pct is not None
        else "no relative change (previous value 0)"
    )
    if a.relative_change_pct is not None and not a.relative_change_reliable:
        rel += " (unreliable near zero)"
    z = f", z {a.z:+.2f}" if a.z is not None else ""
    label = a.classification.value.replace("_", " ")
    return f"{a.item_key}: {rel}, {a.change:+.2f} {unit}{z}: {label}."


def measure_units(profile: DataTypeProfile) -> dict[str, Unit]:
    """The unit of each number field in the evidence records, for number matching."""
    quote, measure = profile.quote_unit, profile.measure_unit
    return {
        "previous_value": quote,
        "current_value": quote,
        "value": quote,
        "previous": quote,
        "current": quote,
        "near_zero_floor": quote,
        "relative_change_pct": Unit.PERCENT,
        "change": measure,
        "normal_daily_move": measure,
        "stdev": measure,
        "artifact_max_change": measure,
        # investigation tools (E8-03)
        "min_value": quote,
        "max_value": quote,
        "neighbour_mean_change": measure,
        "gap_to_neighbour_mean": measure,
        "others_mean_change": measure,
        "largest_change": measure,
        "stdev_change": measure,
    }


def _measures_evidence(
    a: NearZeroAssessment, profile: DataTypeProfile, settings: NearZeroSettings
) -> Evidence:
    """What the check computed and the settings it used: citable, so explanations can
    quote "+3 bps" or "z 0.85" and have them checked."""
    data: dict[str, JsonValue] = {
        "flag_id": a.flag_id,
        "previous": a.previous,
        "current": a.current,
        "change": a.change,
        "relative_change_pct": a.relative_change_pct,
        "relative_change_reliable": a.relative_change_reliable,
        "z": a.z,
        "normal_daily_move": a.history.stdev,
        "near_zero_floor": profile.near_zero_floor,
        "large_move_z": settings.large_move_z,
        "artifact_max_change": settings.artifact_max_change,
        "quote_unit": profile.quote_unit.value,
        "measure_unit": profile.measure_unit.value,
    }
    return Evidence.create("near_zero_measures", NEAR_ZERO_CHECK, data)


def _finding(
    a: NearZeroAssessment, flag: NormalizedFlag, history: Evidence, measures: Evidence
) -> Finding:
    ids = (*flag.evidence_ids, history.evidence_id, measures.evidence_id)
    unit = flag.measure_unit.value
    computed = (measures.evidence_id,)
    rows = [Measure(name="change", value=a.change, unit=unit, evidence_ids=computed)]
    if a.relative_change_pct is not None:
        rows.append(
            Measure(
                name="relative_change",
                value=a.relative_change_pct,
                unit="percent",
                evidence_ids=computed,
            )
        )
    if a.z is not None and a.history.stdev is not None:
        rows += [
            Measure(name="z", value=a.z, unit="z", evidence_ids=computed),
            Measure(
                name="normal_daily_move",
                value=a.history.stdev,
                unit=unit,
                evidence_ids=(history.evidence_id,),
            ),
        ]
    return Finding(
        finding_id=f"{a.flag_id}/near-zero",
        kind=a.classification.value,
        subject=a.item_key,
        summary=_summary(a, unit),
        severity=a.severity,
        measures=tuple(rows),
        evidence_ids=ids,
    )


def _history_evidence(flag: NormalizedFlag, stats: HistoryStats, window: int) -> Evidence:
    data: dict[str, JsonValue] = {
        "curve_id": flag.curve_id,
        "tenor": str(flag.tenor),
        "window_days": window,
        **stats.model_dump(mode="json"),
        "unit": flag.measure_unit.value,
    }
    return Evidence.create("history_stats", f"{MDX_DATA}.{GET_HISTORY}", data)


async def near_zero_check(ctx: TaskContext) -> MarketAssessmentResult:
    raw = ctx.result_of(CHECK_FILES_AND_INTAKE)
    if raw is None:
        raise ValueError("the near-zero check needs a completed intake step")
    intake = IntakeResult.model_validate(raw)
    settings = ctx.config.get(NEAR_ZERO_PATH, NearZeroSettings)
    today = ctx.inputs.business_date
    intake_evidence = {e.evidence_id: e for e in intake.evidence}
    reads = MdxReads(ctx.tools)

    by_flag: dict[str, ItemToExplain] = {}
    added: list[Evidence] = []
    for flag in intake.normalized_flags:
        profile = intake.profiles[flag.kind]
        # The window, plus today. Read through `reads` so the investigation tools (E8-04)
        # reuse it instead of reading the point again.
        series = await reads.history(
            flag.curve_id, flag.tenor, today, settings.history_window_days + 1
        )
        stats = history_stats(series.observations, today, profile, settings)
        assessment = assess_near_zero(flag, stats, profile, settings)
        history = _history_evidence(flag, stats, settings.history_window_days)
        measures = _measures_evidence(assessment, profile, settings)
        added += [history, measures]
        records = [intake_evidence[i] for i in flag.evidence_ids] + [history, measures]
        by_flag[flag.flag_id] = ItemToExplain(
            assessment=assessment,
            finding=_finding(assessment, flag, history, measures),
            evidence=tuple(records),
            units=measure_units(profile),
            measure_unit=profile.measure_unit,
            comment=flag.comment,
        )

    ordered = [by_flag[a.flag_id] for a in review_order([i.assessment for i in by_flag.values()])]
    playbook = ctx.state.playbook
    max_ai = (
        playbook.content.limits.max_ai_explanations
        if playbook is not None
        else DEFAULT_MAX_AI_EXPLANATIONS
    )
    tolerance = ctx.config.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching
    outcome = await explain(
        ordered,
        gateway=ctx.gateway,
        llm_enabled=ctx.state.llm_enabled,
        session_id=ctx.inputs.session_id,
        max_ai_explanations=max_ai,
        tolerance=tolerance,
    )

    tools = ctx.config.get(INVESTIGATION_PATH, InvestigationSettings)
    studied = await investigate(
        ordered,
        outcome.explanations,
        gateway=ctx.gateway,
        llm_enabled=ctx.state.llm_enabled,
        session_id=ctx.inputs.session_id,
        investigation=Investigation(
            reads,
            intake.normalized_flags,
            intake.profiles,
            today,
            DataType(ctx.inputs.data_type),
            tools,
            settings.history_window_days + 1,
            ctx.tracer,
        ),
        settings=tools,
        max_ai_explanations=max_ai,
        tolerance=tolerance,
    )
    added += studied.tool_evidence

    items = tuple(
        AssessedItem(
            assessment=i.assessment,
            finding=i.finding,
            explanation=outcome.explanations[i.assessment.flag_id],
            draft=studied.drafts[i.assessment.flag_id],
        )
        for i in ordered
    )
    counts = Counter(i.assessment.classification for i in items)
    sources = Counter(i.explanation.source for i in items)
    verdicts = Counter(i.draft.verdict for i in items)
    draft_sources = Counter(i.draft.source for i in items)
    return MarketAssessmentResult(
        items=items,
        counts={c: counts.get(c, 0) for c in Classification},
        explanation_sources={s: sources.get(s, 0) for s in ExplanationSource},
        explanation_attempts=outcome.llm_attempts,
        explanation_rejections=outcome.rejections,
        verdicts={v: verdicts.get(v, 0) for v in Verdict},
        draft_sources={s: draft_sources.get(s, 0) for s in ExplanationSource},
        investigation_rounds=studied.rounds,
        investigation_attempts=studied.llm_attempts,
        tool_calls=sum(len(i.draft.tool_calls) for i in items),
        investigation_rejections=studied.rejections,
        evidence=tuple({e.evidence_id: e for e in added}.values()),
        not_assessed=tuple(i.flag_id for i in intake.flag_issues),
        used_fallback=any(
            i.explanation.fallback_reason in (FALLBACK_REJECTED, FALLBACK_UNAVAILABLE)
            or i.draft.fallback_reason in UNUSABLE
            for i in items
        ),
    )


# --- the send-back round (E6-05) -----------------------------------------------------------


class ReviseArguments(BaseModel):
    """The follow-up step's `arguments`, as the teacher's send-back decision gives them."""

    model_config = ConfigDict(extra="forbid")

    of_step: str = Field(min_length=1)
    """The near-zero step whose drafts are sent back."""
    flags: dict[str, str] = Field(min_length=1)
    """Flag ID → why the teacher sent its draft back."""


class RevisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    of_step: str
    drafts: tuple[Draft, ...]
    """The revised drafts, in review order."""
    verdicts: dict[Verdict, int]
    """Of the revised drafts."""
    investigation_rounds: int = Field(ge=0)
    investigation_attempts: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    """Tool calls made in this revision."""
    investigation_rejections: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    """The records the revision's tools returned."""
    used_fallback: bool = False


def item_to_explain(
    item: AssessedItem, intake: IntakeResult, records: Mapping[str, Evidence]
) -> ItemToExplain:
    """An assessed item with its crucial evidence and units, as the loop and checks need."""
    flag = next(f for f in intake.normalized_flags if f.flag_id == item.assessment.flag_id)
    profile = intake.profiles[flag.kind]
    return ItemToExplain(
        assessment=item.assessment,
        finding=item.finding,
        evidence=tuple(records[i] for i in item.finding.evidence_ids),
        units=measure_units(profile),
        measure_unit=profile.measure_unit,
        comment=flag.comment,
    )


def _sent_back(ctx: TaskContext, args: ReviseArguments) -> MarketAssessmentResult:
    ran = next((e for e in ctx.state.execution if e.step_id == args.of_step), None)
    if ran is None or ran.task != NEAR_ZERO_CHECK:
        raise ValueError(f"step {args.of_step!r} is not a near-zero step of this run")
    raw = ctx.state.agent_results.get(args.of_step)
    if raw is None:
        raise ValueError(f"step {args.of_step!r} has no result to revise")
    return MarketAssessmentResult.model_validate(raw)


async def revise_drafts(ctx: TaskContext) -> RevisionResult:
    args = ReviseArguments.model_validate(ctx.step.arguments)
    original = _sent_back(ctx, args)
    raw = ctx.result_of(CHECK_FILES_AND_INTAKE)
    if raw is None:
        raise ValueError("revising drafts needs a completed intake step")
    intake = IntakeResult.model_validate(raw)
    unknown = sorted(set(args.flags) - {i.assessment.flag_id for i in original.items})
    if unknown:
        raise ValueError(f"step {args.of_step!r} has no draft for flag(s) {unknown}")

    records = {e.evidence_id: e for e in (*intake.evidence, *original.evidence)}
    normalized = {f.flag_id: f for f in intake.normalized_flags}
    to_revise = [
        ToRevise(
            item=item_to_explain(item, intake, records),
            explanation=item.explanation,
            draft=item.draft,
            feedback=args.flags[item.assessment.flag_id],
            tool_evidence=tuple(
                records[c.evidence_id] for c in item.draft.tool_calls if c.evidence_id
            ),
        )
        for item in original.items  # review order
        if item.assessment.flag_id in args.flags
    ]

    near_zero = ctx.config.get(NEAR_ZERO_PATH, NearZeroSettings)
    tools = ctx.config.get(INVESTIGATION_PATH, InvestigationSettings)
    outcome = await revise(
        to_revise,
        gateway=ctx.gateway,
        llm_enabled=ctx.state.llm_enabled,
        session_id=ctx.inputs.session_id,
        investigation=Investigation(
            MdxReads(ctx.tools),
            [normalized[f.item.assessment.flag_id] for f in to_revise],
            intake.profiles,
            ctx.inputs.business_date,
            DataType(ctx.inputs.data_type),
            tools,
            near_zero.history_window_days + 1,
            ctx.tracer,
        ),
        settings=tools,
        tolerance=ctx.config.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching,
    )
    drafts = tuple(outcome.drafts[f.item.assessment.flag_id] for f in to_revise)
    verdicts = Counter(d.verdict for d in drafts)
    return RevisionResult(
        of_step=args.of_step,
        drafts=drafts,
        verdicts={v: verdicts.get(v, 0) for v in Verdict},
        investigation_rounds=outcome.rounds,
        investigation_attempts=outcome.llm_attempts,
        tool_calls=sum(1 for d in drafts for c in d.tool_calls if c.revision == d.revision),
        investigation_rejections=outcome.rejections,
        evidence=outcome.tool_evidence,
        used_fallback=any(d.fallback_reason in UNUSABLE for d in drafts),
    )


def current_drafts(
    execution: Sequence[ExecutionEntry], results: Mapping[str, Mapping[str, JsonValue]]
) -> dict[str, Draft]:
    """Each flag's latest draft: the near-zero step's, replaced by any revision of it that
    ran. In review order (the near-zero step's)."""
    drafts: dict[str, Draft] = {}
    for entry in execution:
        raw = results.get(entry.step_id)
        if raw is None or entry.status.value not in ("done", "done_with_fallback"):
            continue
        if entry.task == NEAR_ZERO_CHECK:
            drafts = {
                i.draft.flag_id: i.draft for i in MarketAssessmentResult.model_validate(raw).items
            }
        elif entry.task == REVISE_DRAFTS:
            for d in RevisionResult.model_validate(raw).drafts:
                if d.flag_id in drafts:
                    drafts[d.flag_id] = d
    return drafts

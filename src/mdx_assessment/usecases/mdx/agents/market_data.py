"""Market Data Assessment Agent, task `market_data.near_zero_check` (E8-01; E8-02 next).

Deterministic core, no LLM:

1. Take the normalized flags and data-type profiles from the intake step's result.
2. For each flag, read the point's history through the MDX tool and measure its normal
   daily move (`history_stats`), recorded as evidence.
3. Label it with `assess_near_zero` (docs/business-rules.md §3) and write one finding
   per flag citing the raw flag, the stored point and the history.
4. List the items in review order: severity first, then the largest |z|.

The labels come only from here. E8-02 adds an LLM explanation per item, which can't
change a label.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mdx_assessment.core.domain import Evidence, Finding, Measure
from mdx_assessment.core.orchestration.tasks import TaskContext
from mdx_assessment.usecases.mdx.agents.ids import CHECK_FILES_AND_INTAKE
from mdx_assessment.usecases.mdx.agents.input_validation import IntakeResult
from mdx_assessment.usecases.mdx.checks.near_zero import (
    Classification,
    HistoryStats,
    NearZeroAssessment,
    assess_near_zero,
    history_stats,
    review_order,
)
from mdx_assessment.usecases.mdx.checks.normalize import NormalizedFlag
from mdx_assessment.usecases.mdx.config import NEAR_ZERO_PATH, NearZeroSettings
from mdx_assessment.usecases.mdx.domain import Series
from mdx_assessment.usecases.mdx.tools.mdx import GET_HISTORY
from mdx_assessment.usecases.mdx.tools.mdx import TOOL_ID as MDX_DATA


class AssessedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: NearZeroAssessment
    finding: Finding


class MarketAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[AssessedItem, ...]
    """In review order."""
    counts: dict[Classification, int]
    evidence: tuple[Evidence, ...] = Field(default=())
    """The history records this step added; flags and points are the intake step's."""
    not_assessed: tuple[str, ...] = ()
    """Flag IDs the intake step couldn't normalize, carried for reconciliation (C-1)."""


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


def _finding(a: NearZeroAssessment, flag: NormalizedFlag, history_evidence: Evidence) -> Finding:
    ids = (*flag.evidence_ids, history_evidence.evidence_id)
    unit = flag.measure_unit.value
    measures = [Measure(name="change", value=a.change, unit=unit, evidence_ids=flag.evidence_ids)]
    if a.relative_change_pct is not None:
        measures.append(
            Measure(
                name="relative_change",
                value=a.relative_change_pct,
                unit="percent",
                evidence_ids=flag.evidence_ids,
            )
        )
    if a.z is not None and a.history.stdev is not None:
        measures += [
            Measure(name="z", value=a.z, unit="z", evidence_ids=ids),
            Measure(
                name="normal_daily_move",
                value=a.history.stdev,
                unit=unit,
                evidence_ids=(history_evidence.evidence_id,),
            ),
        ]
    return Finding(
        finding_id=f"{a.flag_id}/near-zero",
        kind=a.classification.value,
        subject=a.item_key,
        summary=_summary(a, unit),
        severity=a.severity,
        measures=tuple(measures),
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

    items: list[AssessedItem] = []
    evidence: list[Evidence] = []
    for flag in intake.normalized_flags:
        profile = intake.profiles[flag.kind]
        series = Series.model_validate(
            await ctx.tools.call(
                MDX_DATA,
                GET_HISTORY,
                {
                    "curve_id": flag.curve_id,
                    "tenor": str(flag.tenor),
                    "end_date": today.isoformat(),
                    "days": settings.history_window_days + 1,  # the window, plus today
                },
            )
        )
        stats = history_stats(series.observations, today, profile, settings)
        assessment = assess_near_zero(flag, stats, profile, settings)
        record = _history_evidence(flag, stats, settings.history_window_days)
        evidence.append(record)
        items.append(
            AssessedItem(assessment=assessment, finding=_finding(assessment, flag, record))
        )

    ordered = review_order([i.assessment for i in items])
    by_flag = {i.assessment.flag_id: i for i in items}
    counts = Counter(i.assessment.classification for i in items)
    return MarketAssessmentResult(
        items=tuple(by_flag[a.flag_id] for a in ordered),
        counts={c: counts.get(c, 0) for c in Classification},
        evidence=tuple({e.evidence_id: e for e in evidence}.values()),
        not_assessed=tuple(i.flag_id for i in intake.flag_issues),
    )

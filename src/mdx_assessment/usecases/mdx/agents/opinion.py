"""What the MDX pack gives the Master to write the run opinion from (E10-02).

Registered as `PlatformRegistries.opinion`; core's `master.opinion` makes the call.

- **Inputs**: the day's figures (the run report, E11-02) and every flag's validated
  verdict, in list order. Flags that need attention (likely real, can't tell) and the
  threshold artifacts carry their measures as evidence; the ordinary flags are listed
  with their verdict and change only, to keep the call small.
- **Citable**: those measures records, with their units. **Quotable**: the figures.
- **Must review**: every flag whose final verdict is likely real.
- **Template**, by code from the same material: what to look at first and why, the
  near-zero artifacts and what MDX's percentage hid, the count of ordinary moves, and
  what remains uncertain.
"""

from __future__ import annotations

from pydantic import JsonValue

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import Evidence, ItemDecisionKind
from mdx_assessment.core.gateway import EvidenceBundle, EvidenceItem
from mdx_assessment.core.gateway import Unit as GroundingUnit
from mdx_assessment.core.gateway.grounding import GUARDRAILS_PATH, GuardrailsConfig
from mdx_assessment.core.master.opinion import OpinionMaterial, OpinionOutput
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.usecases.mdx.agents.escalation import (
    CaseReport,
    RecordCasesResult,
    RunReport,
    case_reports,
    latest_result,
    run_report,
)
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    NEAR_ZERO_CHECK,
    RECORD_CASES,
)
from mdx_assessment.usecases.mdx.agents.input_validation import IntakeResult
from mdx_assessment.usecases.mdx.agents.market_data import measure_units
from mdx_assessment.usecases.mdx.checks.near_zero import Classification
from mdx_assessment.usecases.mdx.checks.verdict import Verdict
from mdx_assessment.usecases.mdx.domain import Unit

MEASURES = "near_zero_measures"
_GROUNDING = {Unit.PERCENT: GroundingUnit.PERCENT, Unit.BPS: GroundingUnit.BPS}
_WORDS = {
    Verdict.LIKELY_REAL: "likely real",
    Verdict.CANT_TELL: "can't tell",
    Verdict.LIKELY_HARMLESS: "likely harmless",
}


def _records(state: RunState) -> dict[str, Evidence]:
    found: dict[str, Evidence] = {}
    for result in state.agent_results.values():
        for raw in result.get("evidence") or []:  # type: ignore[union-attr]
            e = Evidence.model_validate(raw)
            found[e.evidence_id] = e
    return found


def _measures_of(report: CaseReport, records: dict[str, Evidence]) -> Evidence | None:
    return next(
        (records[i] for i in report.evidence_ids if i in records and records[i].kind == MEASURES),
        None,
    )


def _notable(report: CaseReport) -> bool:
    return (
        report.verdict is not Verdict.LIKELY_HARMLESS
        or report.trail.label == Classification.THRESHOLD_ARTIFACT.value
    )


def _figures(state: RunState, reports: list[CaseReport]) -> RunReport:
    raw = latest_result(state, RECORD_CASES)
    if raw is not None:
        return RecordCasesResult.model_validate(raw).report
    return run_report(state, reports, created=0, existing=0)


def _line(r: CaseReport, measures: Evidence | None) -> str:
    data = measures.data if measures else {}
    z = data.get("z")
    detail = f"{r.change:+.2f} {r.unit}" + (f", z {z:+.2f}" if isinstance(z, int | float) else "")
    return f"{r.item} ({detail}, {_WORDS[r.verdict]})"


def _counts(reports: list[CaseReport], figures: RunReport) -> str:
    """The day's counts, by code: the opinion's text doesn't state them."""
    verdicts = figures.verdicts
    harmless = verdicts[Verdict.LIKELY_HARMLESS]
    artifacts = figures.threshold_artifacts
    line = (
        f"{figures.flags_assessed} flags assessed: {verdicts[Verdict.LIKELY_REAL]} likely real, "
        f"{verdicts[Verdict.CANT_TELL]} can't tell, {harmless} likely harmless "
        f"({artifacts} near-zero threshold artifacts, {harmless - artifacts} ordinary moves)."
    )
    not_validated = sum(1 for r in reports if not r.validated)
    unresolved = sum(1 for r in reports if r.teacher_decision is ItemDecisionKind.UNRESOLVED)
    extra = [f"{not_validated} not validated by the Master"] if not_validated else []
    extra += [f"{unresolved} unresolved after the Master's review"] if unresolved else []
    if not extra:
        return line
    rest = "; ".join(extra)
    return f"{line} {rest[0].upper()}{rest[1:]}."


def _template(
    reports: list[CaseReport], figures: RunReport, records: dict[str, Evidence]
) -> OpinionOutput:
    """Code's opinion. Like the Master's, it states no counts (`_counts` does) and names
    only flags whose measures it cites."""
    measures = {r.flag_id: _measures_of(r, records) for r in reports}
    first = [r for r in reports if r.verdict is Verdict.LIKELY_REAL]
    unsure = [r for r in reports if r.verdict is Verdict.CANT_TELL]
    artifacts = [
        r
        for r in reports
        if r.verdict is Verdict.LIKELY_HARMLESS
        and r.trail.label == Classification.THRESHOLD_ARTIFACT.value
    ]
    ordinary = figures.flags_assessed - len(first) - len(unsure) - len(artifacts)

    parts = []
    if first:
        parts.append(
            "Review first: " + "; ".join(_line(r, measures[r.flag_id]) for r in first) + "."
        )
    else:
        parts.append("No flag stands out to be reviewed first.")
    if unsure:
        parts.append(
            "Can't tell, to review: "
            + "; ".join(_line(r, measures[r.flag_id]) for r in unsure)
            + "."
        )
    if artifacts:
        hidden = "; ".join(
            f"{r.item} (MDX {r.mdx_shows:+.1f}%, actually {r.change:+.2f} {r.unit})"
            if r.mdx_shows is not None
            else f"{r.item} ({r.change:+.2f} {r.unit})"
            for r in artifacts
        )
        parts.append(
            "Near-zero threshold artifacts, likely harmless: MDX's percentage overstates a "
            f"small move: {hidden}."
        )
    if ordinary > 0:
        parts.append("The other flags are ordinary moves for their curves.")

    uncertainties = [f"{r.item}: can't tell. {r.report.text}" for r in unsure]
    if any(not r.validated for r in reports):
        uncertainties.append("Some verdicts were not validated by the Master.")
    if any(r.teacher_decision is ItemDecisionKind.UNRESOLVED for r in reports):
        uncertainties.append("Some drafts ended unresolved after the Master's review.")
    if figures.templates["reports"]:
        uncertainties.append("Some reports were written by code, not the AI model.")

    cited = [m.evidence_id for r in (*first, *unsure, *artifacts) if (m := measures[r.flag_id])]
    return OpinionOutput(
        summary=" ".join(parts),
        review_first=tuple(r.flag_id for r in (*first, *unsure)),
        uncertainties=tuple(uncertainties),
        evidence_ids=tuple(dict.fromkeys(cited)),
    )


def opinion_material(state: RunState, config: LoadedConfig) -> OpinionMaterial | None:
    """None when no flag was assessed: there is nothing for an opinion to be about."""
    if latest_result(state, NEAR_ZERO_CHECK) is None:
        return None
    intake_raw = latest_result(state, CHECK_FILES_AND_INTAKE)
    if intake_raw is None:
        return None
    intake = IntakeResult.model_validate(intake_raw)
    reports = case_reports(state)
    figures = _figures(state, reports)
    records = _records(state)
    kinds = {f.flag_id: f.kind for f in intake.normalized_flags}

    flags: list[JsonValue] = []
    citable: list[EvidenceItem] = []
    named: dict[str, tuple[str, ...]] = {}
    for r in reports:
        row: dict[str, JsonValue] = {
            "flag_id": r.flag_id,
            "item": r.item,
            "rank": r.rank,
            "verdict": r.verdict.value,
            "label": r.trail.label,
            "suggested_action": r.suggested_action.value,
            "validated": r.validated,
            "teacher_decision": r.teacher_decision.value if r.teacher_decision else None,
            "change": r.change,
            "unit": r.unit,
        }
        measures = _measures_of(r, records)
        named[r.item] = ()
        if _notable(r) and measures is not None:
            named[r.item] = (measures.evidence_id,)
            units = measure_units(intake.profiles[kinds[r.flag_id]])
            grounding = {k: _GROUNDING[u] for k, u in units.items() if u in _GROUNDING}
            row["mdx_shows_pct"] = r.mdx_shows
            row["report"] = r.report.text
            row["evidence"] = [
                {"id": measures.evidence_id, "kind": measures.kind, "data": measures.data}
            ]
            citable.append(EvidenceItem.from_data(measures.evidence_id, measures.data, grounding))
        flags.append(row)

    figures_json: dict[str, JsonValue] = figures.model_dump(mode="json")
    tolerance = config.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching
    counts = _counts(reports, figures)
    return OpinionMaterial(
        inputs={"figures": figures_json, "counts": counts, "flags": flags},
        # Only the cited measures are quotable: the counts are code's (`counts`).
        bundle=EvidenceBundle.of(citable, tolerance=tolerance),
        known_items=frozenset(r.flag_id for r in reports),
        named_items=named,
        counts=counts,
        must_review=tuple(r.flag_id for r in reports if r.verdict is Verdict.LIKELY_REAL),
        template=_template(reports, figures, records),
    )

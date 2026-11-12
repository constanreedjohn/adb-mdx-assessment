"""The MDX pack's view of a finished run, for `GET /sessions/{id}` (E12-02, plan §2.4).

Registered as `PlatformRegistries.session_view`. Built from the run's own results when
the run ends, and stored with the session:

- `agents`: what each agent produced, in the §2.4 shape. An agent whose step didn't run
  is left out.
- `flags`: the ordered list for the analyst, one ROI report per flag (`CaseReport`),
  with its case ID once cases are recorded.

The raw step results stay in the session's `agent_results`.
"""

from __future__ import annotations

from pydantic import JsonValue

from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.platform import SessionView
from mdx_assessment.usecases.mdx.agents.escalation import (
    RecordCasesResult,
    case_reports,
    latest_result,
)
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    ESCALATION_REPORTING,
    INPUT_VALIDATION,
    MARKET_DATA_ASSESSMENT,
    NEAR_ZERO_CHECK,
    RECORD_CASES,
)
from mdx_assessment.usecases.mdx.agents.input_validation import IntakeResult
from mdx_assessment.usecases.mdx.agents.market_data import MarketAssessmentResult
from mdx_assessment.usecases.mdx.agents.teacher import validated_drafts
from mdx_assessment.usecases.mdx.checks.completeness import FileStatus
from mdx_assessment.usecases.mdx.checks.near_zero import Classification
from mdx_assessment.usecases.mdx.domain import Unit


def _input_validation(intake: IntakeResult) -> dict[str, JsonValue]:
    return {
        "datasets": [
            {
                "file": f.file_name,
                "present": f.status is not FileStatus.MISSING,
                "record_count": f.received_records,
                "expected": f.expected_records,
                "complete": f.status is FileStatus.OK,
                "status": f.status.value,
            }
            for f in intake.completeness.files
        ],
        "dataset_complete": intake.dataset_complete,
        "flags_read": intake.flags_read,
        "normalized_flags": [
            {
                "flag_id": f.flag_id,
                "item": f.item_key,
                "prev": f.previous,
                "curr": f.current,
                "change": f.change,
                "unit": f.measure_unit.value,
                "change_pct": f.relative_change_pct,
                "evidence_ids": list(f.evidence_ids),
            }
            for f in intake.normalized_flags
        ],
        "not_normalized": [i.model_dump(mode="json") for i in intake.flag_issues],
    }


def _market_data(state: RunState, assessed: MarketAssessmentResult) -> dict[str, JsonValue]:
    outcomes = validated_drafts(state.execution, state.agent_results, state.validations)
    items: list[JsonValue] = []
    for i in assessed.items:  # the near-zero step's review order
        a, outcome = i.assessment, outcomes.get(i.assessment.flag_id)
        item: dict[str, JsonValue] = {
            "flag_id": a.flag_id,
            "curve": a.item_key.split(" ")[0],
            "tenor": a.item_key.split(" ")[-1],
            "prev": a.previous,
            "curr": a.current,
            "change": a.change,
            "unit": i.finding.measures[0].unit if i.finding.measures else None,
            "change_pct": a.relative_change_pct,
            "near_zero": a.near_zero,
            "zscore_60d": a.z,
            "label": a.classification.value,
            "threshold_artifact": a.classification is Classification.THRESHOLD_ARTIFACT,
            "severity": a.severity.value,
            "explanation": i.explanation.text,
            "explanation_source": i.explanation.source.value,
            "evidence_ids": list(i.finding.evidence_ids),
        }
        if item["unit"] == Unit.BPS.value:
            item["change_bps"] = a.change
        if outcome is not None:
            item |= {
                "verdict": outcome.verdict.value,
                "validated": outcome.validated,
                "draft_source": outcome.draft.source.value,
                "tools_called": [c.tool for c in outcome.draft.tool_calls],
                "revision": outcome.draft.revision,
            }
        items.append(item)
    return {
        "items": items,
        "labels": {c.value: n for c, n in assessed.counts.items()},
        "explanation_sources": {s.value: n for s, n in assessed.explanation_sources.items()},
        "investigation": {
            "rounds": assessed.investigation_rounds,
            "tool_calls": assessed.tool_calls,
        },
    }


def _escalation(recorded: RecordCasesResult) -> dict[str, JsonValue]:
    report: dict[str, JsonValue] = recorded.report.model_dump(mode="json")
    # §2.4's names for the same figures.
    report["files_checked"] = recorded.report.files_received
    report["items_assessed"] = recorded.report.flags_assessed
    return {"case_ids": [line.case_id for line in recorded.cases], "report": report}


def session_view(state: RunState) -> SessionView | None:
    agents: dict[str, dict[str, JsonValue]] = {}
    intake_raw = latest_result(state, CHECK_FILES_AND_INTAKE)
    if intake_raw is not None:
        agents[INPUT_VALIDATION] = _input_validation(IntakeResult.model_validate(intake_raw))

    assessed_raw = latest_result(state, NEAR_ZERO_CHECK)
    if assessed_raw is None or intake_raw is None:
        return SessionView(agents=agents) if agents else None
    agents[MARKET_DATA_ASSESSMENT] = _market_data(
        state, MarketAssessmentResult.model_validate(assessed_raw)
    )

    recorded_raw = latest_result(state, RECORD_CASES)
    recorded = RecordCasesResult.model_validate(recorded_raw) if recorded_raw else None
    if recorded is not None:
        agents[ESCALATION_REPORTING] = _escalation(recorded)
    case_ids = {line.item: line.case_id for line in recorded.cases} if recorded else {}
    flags = tuple(
        {"case_id": case_ids.get(r.item), **r.model_dump(mode="json")} for r in case_reports(state)
    )
    return SessionView(agents=agents, flags=flags)

"""Review checks: what code verifies about a step's output at its review point (E10-01).

Deterministic, no LLM. Each reads the shared state after the step and returns a pass or
fail with details in plain words. When one fails, the Master's LLM review decides what
happens to the step (option C, `core.master.review`).

| Check | After                    | Passes when                                                   |
| ----- | ------------------------ | ------------------------------------------------------------- |
| C-1   | Market Data Assessment   | every normalized flag was assessed, none invented, and the    |
|       |                          | dataset they came from is complete (plan §2.4)                |
| C-2   | Input Validation         | every expected file arrived with its expected record count   |
| C-3   | Input Validation         | every flag MDX raised was normalized                          |
"""

from __future__ import annotations

from pydantic import JsonValue

from mdx_assessment.core.domain import ReviewCheckResult, TaskStatus
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState
from mdx_assessment.core.orchestration.tasks import ReviewCheckDefinition
from mdx_assessment.usecases.mdx.agents.ids import CHECK_FILES_AND_INTAKE

FLAG_RECONCILIATION = "C-1"
DATASET_COMPLETENESS = "C-2"
FLAG_NORMALIZATION = "C-3"
_RAN = (TaskStatus.DONE, TaskStatus.DONE_WITH_FALLBACK)
_LISTED = 5


def _result_of(state: RunState, entry: ExecutionEntry) -> dict[str, JsonValue]:
    result = state.agent_results.get(entry.step_id)
    if not isinstance(result, dict):
        raise ValueError(f"step {entry.step_id!r} has no result to check")
    return result


def _latest_intake(state: RunState) -> dict[str, JsonValue] | None:
    for e in reversed(state.execution):
        if e.task == CHECK_FILES_AND_INTAKE and e.status in _RAN:
            result = state.agent_results.get(e.step_id)
            return result if isinstance(result, dict) else None
    return None


def _ids(items: object, key: str) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(i[key]) for i in items if isinstance(i, dict) and key in i]


def _listed(ids: list[str]) -> str:
    shown = ", ".join(ids[:_LISTED])
    return shown + (f" and {len(ids) - _LISTED} more" if len(ids) > _LISTED else "")


def _finding_evidence(result: dict[str, JsonValue]) -> tuple[str, ...]:
    finding = result.get("finding")
    if isinstance(finding, dict) and isinstance(finding.get("evidence_ids"), list):
        return tuple(str(i) for i in finding["evidence_ids"])  # type: ignore[union-attr]
    return ()


def dataset_completeness(state: RunState, entry: ExecutionEntry) -> ReviewCheckResult:
    result = _result_of(state, entry)
    finding = result.get("finding")
    summary = finding.get("summary") if isinstance(finding, dict) else None
    return ReviewCheckResult(
        check_id=DATASET_COMPLETENESS,
        name="dataset completeness",
        passed=result.get("dataset_complete") is True,
        details=str(summary or "no completeness finding"),
        evidence_ids=_finding_evidence(result),
    )


def flag_normalization(state: RunState, entry: ExecutionEntry) -> ReviewCheckResult:
    result = _result_of(state, entry)
    normalized = _ids(result.get("normalized_flags"), "flag_id")
    issues = result.get("flag_issues")
    problems = (
        [f"{i['flag_id']} ({i['reason']})" for i in issues if isinstance(i, dict)]
        if isinstance(issues, list)
        else []
    )
    read = result.get("flags_read")
    if read is None:
        details, passed = "MDX had no flags to read for the date", False
    elif problems:
        details = (
            f"{len(normalized)} of {read} flags normalized; not normalized: {_listed(problems)}"
        )
        passed = False
    else:
        details, passed = f"{len(normalized)} of {read} flags normalized", True
    return ReviewCheckResult(
        check_id=FLAG_NORMALIZATION, name="flag normalization", passed=passed, details=details
    )


def flag_reconciliation(state: RunState, entry: ExecutionEntry) -> ReviewCheckResult:
    """C-1: the flags reconcile between Input Validation and Market Data Assessment."""
    result = _result_of(state, entry)
    intake = _latest_intake(state)
    if intake is None:
        return ReviewCheckResult(
            check_id=FLAG_RECONCILIATION,
            name="flag reconciliation",
            passed=False,
            details="no completed intake step to reconcile against",
        )
    normalized = _ids(intake.get("normalized_flags"), "flag_id")
    items = result.get("items")
    assessed = _ids(
        [i.get("assessment") for i in items if isinstance(i, dict)]
        if isinstance(items, list)
        else [],
        "flag_id",
    )
    missing = [f for f in normalized if f not in set(assessed)]
    invented = [f for f in assessed if f not in set(normalized)]
    twice = sorted({f for f in assessed if assessed.count(f) > 1})
    complete = intake.get("dataset_complete") is True

    problems = []
    if missing:
        problems.append(f"not assessed: {_listed(missing)}")
    if invented:
        problems.append(f"assessed but never normalized: {_listed(invented)}")
    if twice:
        problems.append(f"assessed more than once: {_listed(twice)}")
    if not complete:
        problems.append("the dataset they came from is incomplete")
    head = f"{len(set(assessed) & set(normalized))} of {len(normalized)} normalized flags assessed"
    details = head + (
        "; " + "; ".join(problems) if problems else ", none invented, dataset complete"
    )
    return ReviewCheckResult(
        check_id=FLAG_RECONCILIATION,
        name="flag reconciliation",
        passed=not problems,
        details=details,
        evidence_ids=_finding_evidence(intake),
    )


CHECKS = (
    ReviewCheckDefinition(FLAG_RECONCILIATION, "flag reconciliation", flag_reconciliation),
    ReviewCheckDefinition(DATASET_COMPLETENESS, "dataset completeness", dataset_completeness),
    ReviewCheckDefinition(FLAG_NORMALIZATION, "flag normalization", flag_normalization),
)

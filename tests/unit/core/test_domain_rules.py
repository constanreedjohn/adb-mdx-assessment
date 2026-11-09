"""Invariants the core models enforce on construction."""

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from mdx_assessment.core.domain import (
    Decision,
    DecisionKind,
    Evidence,
    Finding,
    Measure,
    Plan,
    PlanSource,
    PlanStep,
    SessionResult,
    SessionStatus,
    Task,
    TaskStatus,
)

T0 = datetime(2026, 10, 15, 17, 20, tzinfo=UTC)
STEP = PlanStep(step_id="s1", task="input_validation.check_files_and_intake")


def test_plan_rejects_duplicate_step_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate step_id"):
        Plan(source=PlanSource.MASTER, steps=(STEP,), follow_ups=(STEP,))


def test_plan_needs_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        Plan(source=PlanSource.DEFAULT, steps=())


def test_follow_up_decision_must_name_its_step() -> None:
    with pytest.raises(ValidationError, match="must name the follow-up step"):
        Decision(step_id="s1", kind=DecisionKind.FOLLOW_UP, reason="history too short")


def test_accept_decision_must_not_carry_a_follow_up() -> None:
    with pytest.raises(ValidationError, match="must not carry a follow-up"):
        Decision(step_id="s1", kind=DecisionKind.ACCEPT, reason="ok", follow_up=STEP)


def test_failed_task_needs_an_error() -> None:
    with pytest.raises(ValidationError, match="needs an error"):
        Task(task_id="t1", session_id="x", step_id="s1", task="a.b", status=TaskStatus.FAILED)


def test_task_duration() -> None:
    task = Task(
        task_id="t1",
        session_id="x",
        step_id="s1",
        task="a.b",
        status=TaskStatus.DONE,
        started_at=T0,
        finished_at=T0 + timedelta(seconds=2.5),
    )
    assert task.duration_seconds == 2.5


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        Task(
            task_id="t1", session_id="x", step_id="s1", task="a.b", started_at=datetime(2026, 1, 1)
        )


def test_finding_must_cite_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(finding_id="f1", kind="k", subject="JPY 3M", summary="s", evidence_ids=())


def test_measure_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Measure(name="change_bps", value=float("inf"), unit="bps")


def test_finding_rejects_malformed_evidence_id() -> None:
    with pytest.raises(ValidationError):
        Finding(finding_id="f1", kind="k", subject="s", summary="s", evidence_ids=("ev-xyz",))


@pytest.mark.parametrize(
    "status", [SessionStatus.COMPLETED_DEGRADED, SessionStatus.DEFAULT_COMPLETE]
)
def test_fallback_statuses_must_list_fallbacks(status: SessionStatus) -> None:
    with pytest.raises(ValidationError, match="fallbacks"):
        SessionResult(
            session_id="x",
            business_date=date(2026, 10, 15),
            data_type="linear",
            status=status,
            started_at=T0,
            finished_at=T0,
        )


def test_running_session_has_no_finish_time() -> None:
    with pytest.raises(ValidationError, match="no finished_at"):
        SessionResult(
            session_id="x",
            business_date=date(2026, 10, 15),
            data_type="linear",
            started_at=T0,
            finished_at=T0,
        )


def test_unknown_fields_are_rejected() -> None:
    ev = Evidence.create("k", "s")
    with pytest.raises(ValidationError, match="Extra inputs"):
        Measure(name="m", value=1.0, unit="bps", evidence_ids=(ev.evidence_id,), extra=1)  # type: ignore[call-arg]

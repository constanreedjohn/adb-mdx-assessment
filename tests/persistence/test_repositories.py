"""Acceptance (E4-01): repository round-trips against PostgreSQL."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mdx_assessment.core.domain import (
    Evidence,
    Finding,
    Plan,
    PlanSource,
    PlanStep,
    SessionResult,
    SessionStatus,
    Versions,
)
from mdx_assessment.core.persistence import (
    CaseRecord,
    CaseRepo,
    NotFoundError,
    SessionRepo,
    create_session_factory,
    session_scope,
)
from mdx_assessment.core.persistence.models import SessionRow

pytestmark = pytest.mark.db

T0 = datetime(2026, 10, 15, 9, 20, 0, 123456, tzinfo=UTC)
EV = Evidence.create("point_change", "mdx_data.get_flags", {"curve": "JPY", "tenor": "3M"})


def a_session(session_id: str = "sess-1", **overrides: object) -> SessionResult:
    data: dict[str, object] = {
        "session_id": session_id,
        "business_date": date(2026, 10, 15),
        "data_type": "linear",
        "started_at": T0,
        "versions": Versions(config="sha256:abc", prompts={"master_plan": "v1"}),
        "plan": Plan(
            source=PlanSource.DEFAULT,
            steps=(PlanStep(step_id="s1", task="input_validation.check_files_and_intake"),),
        ),
        "findings": (
            Finding(
                finding_id="f1",
                kind="threshold_artifact",
                subject="JPY 3M",
                summary="+75% is +3 bps — near zero",
                evidence_ids=(EV.evidence_id,),
            ),
        ),
    }
    data.update(overrides)
    return SessionResult.model_validate(data)


def a_case(
    case_id: str, session_id: str = "sess-1", minutes: int = 0, item_key: str = "JPY 3M"
) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        session_id=session_id,
        item_key=item_key,
        status="open",
        payload={"curve_id": "JPY", "tenor": "3M", "measures": [{"change_bps": 3.0}], "note": "ü"},
        created_at=T0 + timedelta(minutes=minutes),
    )


# --- sessions --------------------------------------------------------------------------


def test_session_round_trip(db: Session) -> None:
    original = a_session()
    SessionRepo(db).add(original)
    db.expire_all()
    assert SessionRepo(db).get("sess-1") == original


def test_session_query_columns_are_filled(db: Session) -> None:
    SessionRepo(db).add(a_session())
    row = db.scalars(select(SessionRow)).one()
    assert (row.status, row.data_type, row.config_version) == ("running", "linear", "sha256:abc")
    assert row.started_at == T0
    assert row.created_at is not None


def test_session_save_replaces_the_record(db: Session) -> None:
    repo = SessionRepo(db)
    repo.add(a_session())
    finished = a_session(status=SessionStatus.COMPLETED, finished_at=T0 + timedelta(minutes=3))
    repo.save(finished)
    db.expire_all()
    assert repo.get("sess-1") == finished
    assert db.scalars(select(SessionRow.status)).one() == "completed"


def test_missing_session(db: Session) -> None:
    assert SessionRepo(db).get("nope") is None
    with pytest.raises(NotFoundError):
        SessionRepo(db).save(a_session("nope"))


def test_duplicate_session_id_rejected(db: Session) -> None:
    SessionRepo(db).add(a_session())
    with pytest.raises(IntegrityError):
        SessionRepo(db).add(a_session())


# --- cases -----------------------------------------------------------------------------


def test_case_round_trip(db: Session) -> None:
    SessionRepo(db).add(a_session())
    record = a_case("case-1")
    CaseRepo(db).add(record)
    db.expire_all()
    assert CaseRepo(db).get("case-1") == record
    assert CaseRepo(db).get("missing") is None


def test_cases_listed_per_session_in_creation_order(db: Session) -> None:
    SessionRepo(db).add(a_session("sess-1"))
    SessionRepo(db).add(a_session("sess-2"))
    repo = CaseRepo(db)
    repo.add(a_case("case-b", minutes=2, item_key="GBP 2Y"))
    repo.add(a_case("case-a", minutes=1, item_key="JPY 3M"))
    repo.add(a_case("case-other", session_id="sess-2"))
    assert [c.case_id for c in repo.list_for_session("sess-1")] == ["case-a", "case-b"]
    assert repo.list_for_session("sess-3") == []


def test_one_case_per_item_per_session(db: Session) -> None:
    SessionRepo(db).add(a_session("sess-1"))
    CaseRepo(db).add(a_case("case-1"))
    with pytest.raises(IntegrityError, match="uq_cases_session_id"):
        CaseRepo(db).add(a_case("case-2"))


def test_find_by_item(db: Session) -> None:
    SessionRepo(db).add(a_session("sess-1"))
    CaseRepo(db).add(a_case("case-1"))
    found = CaseRepo(db).find("sess-1", "JPY 3M")
    assert found is not None and found.case_id == "case-1"
    assert CaseRepo(db).find("sess-1", "GBP 2Y") is None


def test_case_needs_an_existing_session(db: Session) -> None:
    with pytest.raises(IntegrityError, match="fk_cases_session_id_sessions"):
        CaseRepo(db).add(a_case("case-1", session_id="no-such-session"))


# --- unit of work ----------------------------------------------------------------------


def test_session_scope_rolls_back_on_error(migrated_engine: Engine) -> None:
    factory = create_session_factory(migrated_engine)
    with pytest.raises(RuntimeError), session_scope(factory) as s:
        SessionRepo(s).add(a_session("sess-rollback"))
        raise RuntimeError("boom")
    with session_scope(factory) as s:
        assert SessionRepo(s).get("sess-rollback") is None


def test_session_scope_commits(migrated_engine: Engine) -> None:
    factory = create_session_factory(migrated_engine)
    with session_scope(factory) as s:
        SessionRepo(s).add(a_session("sess-commit"))
    try:
        with session_scope(factory) as s:
            assert SessionRepo(s).get("sess-commit") is not None
    finally:  # this test commits for real, so clean up
        with session_scope(factory) as s:
            s.delete(s.get(SessionRow, "sess-commit"))

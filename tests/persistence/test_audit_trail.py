"""Acceptance (E4-02): the Postgres audit trail detects tampering and deletion.

These tests commit for real (each append is its own transaction), so a fixture
empties `audit_events` before and after each test. Emptying it, and the tampering
in the tests, use `session_replication_role = replica`, which disables triggers for
one transaction: what a database superuser could do to bypass the append-only rule.
The hash chain must still catch it.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from mdx_assessment.core.domain import GENESIS_HASH, NewAuditEvent
from mdx_assessment.core.persistence import AuditTrail, PostgresAuditTrail

pytestmark = pytest.mark.db


def as_superuser_bypassing_triggers(engine: Engine, sql: str, **params: Any) -> None:
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL session_replication_role = replica"))
        conn.execute(text(sql), params)


@pytest.fixture
def trail(migrated_engine: Engine) -> Iterator[PostgresAuditTrail]:
    as_superuser_bypassing_triggers(migrated_engine, "DELETE FROM audit_events")
    yield PostgresAuditTrail(migrated_engine)
    as_superuser_bypassing_triggers(migrated_engine, "DELETE FROM audit_events")


def ev(session: str = "sess-1", **payload: Any) -> NewAuditEvent:
    return NewAuditEvent(
        session_id=session, actor="master_agent", event_type="decision", payload=payload
    )


def fill(trail: PostgresAuditTrail, n: int = 5) -> None:
    for i in range(n):
        trail.append(ev(step=i, value=0.1 + 0.2, note="évidence ✓", nested={"a": [1, None, True]}))


def test_postgres_trail_satisfies_the_protocol(trail: PostgresAuditTrail) -> None:
    assert isinstance(trail, AuditTrail)


def test_valid_chain(trail: PostgresAuditTrail) -> None:
    fill(trail, 5)
    events = trail.read()
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert events[0].prev_hash == GENESIS_HASH
    result = trail.verify()
    assert (result.valid, result.events_checked) == (True, 5)


def test_stored_events_round_trip_exactly(trail: PostgresAuditTrail) -> None:
    appended = trail.append(ev(value=0.30000000000000004, big=2**60, text="ü"))
    assert trail.read() == [appended]  # JSONB and timestamptz preserve every hashed field


def test_tampered_payload_is_detected(trail: PostgresAuditTrail, migrated_engine: Engine) -> None:
    fill(trail, 5)
    as_superuser_bypassing_triggers(
        migrated_engine,
        """UPDATE audit_events SET payload = jsonb_set(payload, '{step}', '99') WHERE seq = 3""",
    )
    result = trail.verify()
    assert (result.valid, result.first_broken_seq) == (False, 3)
    assert result.reason == "hash does not match the event's content"


def test_deleted_row_is_detected(trail: PostgresAuditTrail, migrated_engine: Engine) -> None:
    fill(trail, 5)
    as_superuser_bypassing_triggers(migrated_engine, "DELETE FROM audit_events WHERE seq = 3")
    result = trail.verify()
    assert (result.valid, result.first_broken_seq) == (False, 4)
    assert "gap" in (result.reason or "")


def test_session_verification_detects_a_deleted_event_of_another_session(
    trail: PostgresAuditTrail, migrated_engine: Engine
) -> None:
    for i in range(6):
        trail.append(ev(session="sess-a" if i % 2 == 0 else "sess-b", step=i))
    assert trail.verify("sess-b").valid
    as_superuser_bypassing_triggers(migrated_engine, "DELETE FROM audit_events WHERE seq = 3")
    result = trail.verify("sess-b")
    assert (result.valid, result.first_broken_seq) == (False, 4)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE audit_events SET actor = 'forged' WHERE seq = 1",
        "DELETE FROM audit_events WHERE seq = 1",
        "TRUNCATE audit_events",
    ],
)
def test_append_only_is_enforced_by_the_database(
    trail: PostgresAuditTrail, migrated_engine: Engine, sql: str
) -> None:
    fill(trail, 2)
    with (
        pytest.raises(DBAPIError, match="audit_events is append-only"),
        migrated_engine.begin() as conn,
    ):
        conn.execute(text(sql))
    assert trail.verify().events_checked == 2


def test_concurrent_appends_keep_one_valid_chain(trail: PostgresAuditTrail) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: trail.append(ev(session=f"sess-{i % 3}", step=i)), range(40)))
    events = trail.read()
    assert [e.seq for e in events] == list(range(1, 41))
    assert trail.verify().valid
    assert all(trail.verify(f"sess-{k}").valid for k in range(3))

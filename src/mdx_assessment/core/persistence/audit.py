"""Audit trail stores: PostgreSQL (the real one) and in-memory (tests, fakes).

Both implement `AuditTrail` and share the chain logic in `core.domain.audit`.

PostgreSQL specifics:
- Each `append` is its own short transaction, independent of the caller's unit of
  work: an event about a step that later fails or rolls back is still recorded.
- Appends are serialized with a transaction-level advisory lock, so `seq` has no
  gaps and the chain stays a single line even with concurrent writers.
- The table is append-only at the database level: triggers reject UPDATE, DELETE
  and TRUNCATE (migration 0002). In ADB's tenant the application role should also
  have only SELECT and INSERT on it (E13-04).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from mdx_assessment.core.domain.audit import (
    GENESIS_HASH,
    AuditEvent,
    ChainVerification,
    NewAuditEvent,
    chain_event,
    verify_chain,
    verify_linked_events,
)
from mdx_assessment.core.persistence.models import AuditEventRow

Clock = Callable[[], datetime]

# Any fixed 64-bit key; it only has to be unique among this database's advisory locks.
AUDIT_APPEND_LOCK_KEY = 7_203_118_421


def utc_now() -> datetime:
    return datetime.now(UTC)


@runtime_checkable
class AuditTrail(Protocol):
    def append(self, event: NewAuditEvent) -> AuditEvent: ...

    def read(self, session_id: str | None = None) -> list[AuditEvent]:
        """Events in `seq` order: one session's, or all when `session_id` is None."""
        ...

    def verify(self, session_id: str | None = None) -> ChainVerification:
        """Verify one session's events, or the whole chain when `session_id` is None."""
        ...


class PostgresAuditTrail:
    def __init__(self, engine: Engine, *, clock: Clock = utc_now) -> None:
        self._engine = engine
        self._clock = clock

    def append(self, event: NewAuditEvent) -> AuditEvent:
        with Session(self._engine) as db, db.begin():
            db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": AUDIT_APPEND_LOCK_KEY})
            last = db.execute(
                select(AuditEventRow.seq, AuditEventRow.hash)
                .order_by(AuditEventRow.seq.desc())
                .limit(1)
            ).first()
            seq, prev_hash = (last.seq + 1, last.hash) if last else (1, GENESIS_HASH)
            chained = chain_event(event, seq=seq, ts=self._clock(), prev_hash=prev_hash)
            db.add(AuditEventRow(**chained.model_dump()))
        return chained

    def read(self, session_id: str | None = None) -> list[AuditEvent]:
        query = select(AuditEventRow).order_by(AuditEventRow.seq)
        if session_id is not None:
            query = query.where(AuditEventRow.session_id == session_id)
        with Session(self._engine) as db:
            return [_to_event(row) for row in db.scalars(query)]

    def verify(self, session_id: str | None = None) -> ChainVerification:
        if session_id is None:
            return verify_chain(self.read())
        events = self.read(session_id)
        wanted = [e.seq - 1 for e in events if e.seq > 1]
        with Session(self._engine) as db:
            rows = db.execute(
                select(AuditEventRow.seq, AuditEventRow.hash).where(AuditEventRow.seq.in_(wanted))
            )
            predecessors = {row.seq: row.hash for row in rows}
        return verify_linked_events(events, predecessors)


class InMemoryAuditTrail:
    """Same behaviour as the Postgres trail, held in a list. Thread-safe."""

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()
        self._clock = clock

    def append(self, event: NewAuditEvent) -> AuditEvent:
        with self._lock:
            prev_hash = self._events[-1].hash if self._events else GENESIS_HASH
            chained = chain_event(
                event, seq=len(self._events) + 1, ts=self._clock(), prev_hash=prev_hash
            )
            self._events.append(chained)
            return chained

    def read(self, session_id: str | None = None) -> list[AuditEvent]:
        with self._lock:
            return [e for e in self._events if session_id is None or e.session_id == session_id]

    def verify(self, session_id: str | None = None) -> ChainVerification:
        with self._lock:
            events = list(self._events)
        if session_id is None:
            return verify_chain(events)
        by_seq = {e.seq: e.hash for e in events}
        return verify_linked_events([e for e in events if e.session_id == session_id], by_seq)


def _to_event(row: AuditEventRow) -> AuditEvent:
    return AuditEvent(
        seq=row.seq,
        ts=row.ts,
        session_id=row.session_id,
        actor=row.actor,
        event_type=row.event_type,
        payload=row.payload,
        prev_hash=row.prev_hash,
        hash=row.hash,
    )

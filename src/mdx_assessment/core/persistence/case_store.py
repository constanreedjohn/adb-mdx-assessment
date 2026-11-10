"""Where exception cases are kept: `CaseRepo` in its own transactions, or memory in tests.

Core stores a case without knowing what it describes: the use-case pack decides the
item key (e.g. "JPY 3M") and the payload. Creating a case is idempotent on
(session_id, item_key), so a resumed step gets its earlier case back instead of a
duplicate.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from mdx_assessment.core.persistence.audit import Clock, utc_now
from mdx_assessment.core.persistence.db import session_scope
from mdx_assessment.core.persistence.repositories import CaseRecord, CaseRepo

OPEN = "open"


class UnknownSessionError(LookupError):
    """The case names a session that doesn't exist."""


@dataclass(frozen=True)
class CreatedCase:
    case: CaseRecord
    created: bool
    """False when the session already had a case for this item; `case` is that one."""


class CaseStore(Protocol):
    def create(
        self, session_id: str, item_key: str, payload: dict[str, JsonValue]
    ) -> CreatedCase: ...

    def get(self, case_id: str) -> CaseRecord | None: ...

    def list_for_session(self, session_id: str) -> list[CaseRecord]: ...


def new_case_id() -> str:
    return f"case-{uuid.uuid4().hex[:16]}"


class PostgresCaseStore:
    def __init__(self, factory: sessionmaker[Session], *, clock: Clock = utc_now) -> None:
        self._factory = factory
        self._clock = clock

    def create(self, session_id: str, item_key: str, payload: dict[str, JsonValue]) -> CreatedCase:
        record = CaseRecord(
            case_id=new_case_id(),
            session_id=session_id,
            item_key=item_key,
            status=OPEN,
            payload=payload,
            created_at=self._clock(),
        )
        try:
            with session_scope(self._factory) as db:
                repo = CaseRepo(db)
                existing = repo.find(session_id, item_key)
                if existing is not None:
                    return CreatedCase(existing, created=False)
                repo.add(record)
        except IntegrityError as exc:
            # Either another writer recorded this item first, or the session is unknown.
            with session_scope(self._factory) as db:
                existing = CaseRepo(db).find(session_id, item_key)
            if existing is not None:
                return CreatedCase(existing, created=False)
            raise UnknownSessionError(f"unknown session {session_id!r}") from exc
        return CreatedCase(record, created=True)

    def get(self, case_id: str) -> CaseRecord | None:
        with session_scope(self._factory) as db:
            return CaseRepo(db).get(case_id)

    def list_for_session(self, session_id: str) -> list[CaseRecord]:
        with session_scope(self._factory) as db:
            return CaseRepo(db).list_for_session(session_id)


class InMemoryCaseStore:
    """For tests and offline runs. Any session ID is accepted."""

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self._clock = clock
        self._cases: dict[str, CaseRecord] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, item_key: str, payload: dict[str, JsonValue]) -> CreatedCase:
        with self._lock:
            for case in self._cases.values():
                if (case.session_id, case.item_key) == (session_id, item_key):
                    return CreatedCase(case, created=False)
            record = CaseRecord(
                case_id=new_case_id(),
                session_id=session_id,
                item_key=item_key,
                status=OPEN,
                payload=payload,
                created_at=self._clock(),
            )
            self._cases[record.case_id] = record
            return CreatedCase(record, created=True)

    def get(self, case_id: str) -> CaseRecord | None:
        return self._cases.get(case_id)

    def list_for_session(self, session_id: str) -> list[CaseRecord]:
        found = [c for c in self._cases.values() if c.session_id == session_id]
        return sorted(found, key=lambda c: c.created_at)  # stable: ties keep insertion order

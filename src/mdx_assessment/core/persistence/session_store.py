"""Where finished and running sessions are kept: `SessionRepo` in its own transactions."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from mdx_assessment.core.domain import SessionResult
from mdx_assessment.core.persistence.db import session_scope
from mdx_assessment.core.persistence.repositories import NotFoundError, SessionRepo


class SessionStore(Protocol):
    def add(self, result: SessionResult) -> None: ...

    def save(self, result: SessionResult) -> None: ...

    def get(self, session_id: str) -> SessionResult | None: ...


class PostgresSessionStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def add(self, result: SessionResult) -> None:
        with session_scope(self._factory) as db:
            SessionRepo(db).add(result)

    def save(self, result: SessionResult) -> None:
        with session_scope(self._factory) as db:
            SessionRepo(db).save(result)

    def get(self, session_id: str) -> SessionResult | None:
        with session_scope(self._factory) as db:
            return SessionRepo(db).get(session_id)


class InMemorySessionStore:
    """For tests and offline runs."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionResult] = {}

    def add(self, result: SessionResult) -> None:
        self._sessions[result.session_id] = result

    def save(self, result: SessionResult) -> None:
        if result.session_id not in self._sessions:
            raise NotFoundError(f"session {result.session_id!r} not found")
        self._sessions[result.session_id] = result

    def get(self, session_id: str) -> SessionResult | None:
        return self._sessions.get(session_id)

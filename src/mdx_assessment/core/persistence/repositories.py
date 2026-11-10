"""Repositories: the only code that reads and writes session and case rows.

They work inside a caller-provided SQLAlchemy `Session` (one unit of work, see
`db.session_scope`) and never commit themselves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, Field, JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from mdx_assessment.core.domain import DomainModel, SessionResult
from mdx_assessment.core.persistence.models import CaseRow, SessionRow


class NotFoundError(LookupError):
    pass


class SessionRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(self, result: SessionResult) -> None:
        self._db.add(SessionRow(id=result.session_id, **self._columns(result)))
        self._db.flush()

    def save(self, result: SessionResult) -> None:
        """Replace the stored session with `result` (e.g. after each step or at the end)."""
        row = self._db.get(SessionRow, result.session_id)
        if row is None:
            raise NotFoundError(f"session {result.session_id!r} not found")
        for key, value in self._columns(result).items():
            setattr(row, key, value)
        self._db.flush()

    def get(self, session_id: str) -> SessionResult | None:
        row = self._db.get(SessionRow, session_id)
        return SessionResult.model_validate(row.result) if row else None

    @staticmethod
    def _columns(result: SessionResult) -> dict[str, Any]:
        return {
            "business_date": result.business_date,
            "data_type": result.data_type,
            "status": result.status.value,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "config_version": result.versions.config,
            "result": result.model_dump(mode="json"),
        }


class CaseRecord(DomainModel):
    """A stored case, as core sees it. The use-case pack maps its own model to `payload`."""

    case_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    item_key: str = Field(min_length=1, max_length=200)
    status: str = Field(min_length=1, max_length=32)
    payload: dict[str, JsonValue]
    created_at: AwareDatetime


class CaseRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(self, record: CaseRecord) -> None:
        self._db.add(
            CaseRow(
                id=record.case_id,
                session_id=record.session_id,
                item_key=record.item_key,
                status=record.status,
                payload=record.payload,
                created_at=record.created_at,
            )
        )
        self._db.flush()

    def get(self, case_id: str) -> CaseRecord | None:
        row = self._db.get(CaseRow, case_id)
        return self._to_record(row) if row else None

    def find(self, session_id: str, item_key: str) -> CaseRecord | None:
        row = self._db.scalar(
            select(CaseRow).where(CaseRow.session_id == session_id, CaseRow.item_key == item_key)
        )
        return self._to_record(row) if row else None

    def list_for_session(self, session_id: str) -> list[CaseRecord]:
        rows = self._db.scalars(
            select(CaseRow)
            .where(CaseRow.session_id == session_id)
            .order_by(CaseRow.created_at, CaseRow.id)
        )
        return [self._to_record(r) for r in rows]

    @staticmethod
    def _to_record(row: CaseRow) -> CaseRecord:
        created: datetime = row.created_at
        return CaseRecord(
            case_id=row.id,
            session_id=row.session_id,
            item_key=row.item_key,
            status=row.status,
            payload=row.payload,
            created_at=created,
        )

"""Table definitions (SQLAlchemy 2). The schema itself is created by Alembic migrations.

Use-case agnostic: a few columns for querying, the full record as JSONB. A test
checks that these models and the migrations describe the same schema.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit constraint names, so migrations are stable and readable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONB}  # noqa: RUF012 (SQLAlchemy class config)


class SessionRow(Base):
    """One assessment session (a run of the graph for one business date and data type)."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_date: Mapped[date] = mapped_column(Date)
    data_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    config_version: Mapped[str | None] = mapped_column(String(80))
    result: Mapped[dict[str, Any]]
    """The full SessionResult as JSON (what `GET /sessions/{id}` returns)."""
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index(None, "business_date", "data_type"),)


class CaseRow(Base):
    """One exception case. The use-case pack defines the payload; core stores it."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="RESTRICT"), index=True
    )
    item_key: Mapped[str] = mapped_column(String(200))
    """Human-readable subject for listing and filtering, e.g. "JPY 3M"."""
    status: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # One case per item per session, so a step that is resumed can't record an item twice.
    __table_args__ = (UniqueConstraint("session_id", "item_key"),)


class AuditEventRow(Base):
    """Append-only audit trail with a SHA-256 hash chain (behaviour added in E4-02)."""

    __tablename__ = "audit_events"

    seq: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    """Not a foreign key: the audit trail must outlive and never block other tables."""
    actor: Mapped[str] = mapped_column(String(100))
    event_type: Mapped[str] = mapped_column(String(100))
    payload: Mapped[dict[str, Any]]
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)

    __table_args__ = (
        CheckConstraint("seq > 0", name="seq_positive"),
        CheckConstraint("hash ~ '^[0-9a-f]{64}$'", name="hash_hex"),
        CheckConstraint("prev_hash ~ '^[0-9a-f]{64}$'", name="prev_hash_hex"),
    )

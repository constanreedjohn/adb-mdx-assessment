"""Initial schema: sessions, cases, audit_events.

Revision ID: 0001
Revises:
Create Date: 2026-11-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("business_date", sa.Date(), nullable=False),
        sa.Column("data_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("config_version", sa.String(80), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
    )
    op.create_index(
        "ix_sessions_business_date", "sessions", ["business_date", "data_type"], unique=False
    )

    op.create_table(
        "cases",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("item_key", sa.String(200), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_cases_session_id_sessions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cases"),
    )
    op.create_index("ix_cases_session_id", "cases", ["session_id"], unique=False)

    op.create_table(
        "audit_events",
        sa.Column("seq", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("actor", sa.String(100), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("prev_hash", sa.String(64), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.CheckConstraint("seq > 0", name=op.f("ck_audit_events_seq_positive")),
        sa.CheckConstraint("hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_audit_events_hash_hex")),
        sa.CheckConstraint(
            "prev_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_audit_events_prev_hash_hex")
        ),
        sa.PrimaryKeyConstraint("seq", name="pk_audit_events"),
        sa.UniqueConstraint("hash", name="uq_audit_events_hash"),
    )
    op.create_index("ix_audit_events_session_id", "audit_events", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_events_session_id", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_cases_session_id", table_name="cases")
    op.drop_table("cases")
    op.drop_index("ix_sessions_business_date", table_name="sessions")
    op.drop_table("sessions")

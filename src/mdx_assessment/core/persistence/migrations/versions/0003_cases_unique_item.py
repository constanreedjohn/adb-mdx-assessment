"""cases: one case per item per session.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-10

`create_case` is idempotent on (session_id, item_key): a step resumed after a crash
gets the case it already recorded back instead of a duplicate. The constraint makes
that hold even if two writers race.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint("uq_cases_session_id", "cases", ["session_id", "item_key"])


def downgrade() -> None:
    op.drop_constraint("uq_cases_session_id", "cases", type_="unique")

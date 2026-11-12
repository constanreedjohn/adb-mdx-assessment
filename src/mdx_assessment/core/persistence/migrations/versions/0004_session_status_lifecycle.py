"""sessions: the status is the run's lifecycle only (adb-mdx-agent ADR-0001).

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-10

`completed`, `completed_degraded` and `default_complete` all become `succeeded`. What
told them apart is still in the stored result: the plan's `source`, each step's status,
`review` and `fallbacks`. The status is rewritten in the column and in `result`.

The downgrade can't tell the three apart again, so it maps `succeeded` back to
`completed` and leaves `pending` (never stored while sessions run inline) as it is.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = ("completed", "completed_degraded", "default_complete")


def _set_status(new: str, old: Sequence[str]) -> None:
    op.get_bind().execute(
        sa.text(
            "UPDATE sessions"
            " SET status = CAST(:new AS varchar),"
            " result = jsonb_set(result, '{status}', to_jsonb(CAST(:new AS varchar)))"
            " WHERE status IN :old"
        ).bindparams(sa.bindparam("old", expanding=True)),
        {"new": new, "old": list(old)},
    )


def upgrade() -> None:
    _set_status("succeeded", _OLD)


def downgrade() -> None:
    _set_status("completed", ("succeeded",))

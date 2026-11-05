"""Migrations: they apply, match the ORM models, name constraints as intended, and reverse."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect, text

from mdx_assessment.core.persistence.migrate import (
    current_revision,
    downgrade,
    head_revision,
    upgrade,
)
from mdx_assessment.core.persistence.models import Base

pytestmark = pytest.mark.db

TABLES = {"sessions", "cases", "audit_events"}


def test_database_is_at_head(migrated_engine: Engine) -> None:
    assert current_revision(migrated_engine) == head_revision() == "0001"
    assert set(inspect(migrated_engine).get_table_names()) >= TABLES


def test_models_and_migrations_describe_the_same_schema(migrated_engine: Engine) -> None:
    # If this fails, a model changed without a migration: run `make migration MSG="..."`.
    with migrated_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_constraint_names_follow_the_convention(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as conn:
        names = set(
            conn.scalars(
                text(
                    "SELECT conname FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
                    "WHERE t.relname IN ('sessions', 'cases', 'audit_events')"
                )
            )
        )
    assert names == {
        "pk_sessions",
        "pk_cases",
        "fk_cases_session_id_sessions",
        "pk_audit_events",
        "uq_audit_events_hash",
        "ck_audit_events_seq_positive",
        "ck_audit_events_hash_hex",
        "ck_audit_events_prev_hash_hex",
    }


def test_downgrade_and_upgrade_again(migrated_engine: Engine) -> None:
    downgrade(migrated_engine, "base")
    assert not TABLES & set(inspect(migrated_engine).get_table_names())
    upgrade(migrated_engine)
    assert current_revision(migrated_engine) == "0001"
    assert set(inspect(migrated_engine).get_table_names()) >= TABLES

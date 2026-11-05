"""Database fixtures for persistence tests.

- Test database: `<POSTGRES_DB>_test` on the configured server (created if missing),
  or `MDX_TEST_DATABASE_URL` if set. Its name must end in `_test`: the schema is
  dropped and rebuilt, so the fixtures refuse any other database.
- Postgres unreachable: tests are skipped locally (start it with `make dev`) but
  fail when `MDX_REQUIRE_DB=1` (CI).
- The schema is rebuilt once per test run by running the real migrations.
- Each test runs inside a transaction that is rolled back afterwards, so tests
  don't see each other's rows, even when the code under test commits.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

import pytest
from sqlalchemy import URL, Engine, create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from mdx_assessment.core.persistence.db import database_url
from mdx_assessment.core.persistence.migrate import upgrade
from mdx_assessment.settings import get_settings

REQUIRE_DB = os.environ.get("MDX_REQUIRE_DB") == "1"
_SAFE_DB_NAME = re.compile(r"^[a-z0-9_]+_test$")


def _unavailable(reason: str) -> None:
    message = f"PostgreSQL unavailable: {reason}. Start it with `make dev`."
    if REQUIRE_DB:
        pytest.fail(message + " (MDX_REQUIRE_DB=1)", pytrace=False)
    pytest.skip(message)


def _ensure_database(url: URL) -> None:
    """Create the test database if it doesn't exist (connects to the main database)."""
    name = url.database or ""
    admin = create_engine(
        url.set(database=get_settings().postgres.db), isolation_level="AUTOCOMMIT"
    )
    try:
        with admin.connect() as conn:
            exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name})
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{name}"'))  # name checked by _SAFE_DB_NAME
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def test_db_url() -> URL:
    override = os.environ.get("MDX_TEST_DATABASE_URL")
    url = (
        make_url(override)
        if override
        else database_url(database=get_settings().postgres.db + "_test")
    )
    if not _SAFE_DB_NAME.match(url.database or ""):
        pytest.fail(f"refusing to use {url.database!r} for tests: name must end in '_test'")
    try:
        _ensure_database(url)
    except OperationalError as exc:
        _unavailable(str(exc.orig).strip().splitlines()[0])
    return url


@pytest.fixture(scope="session")
def migrated_engine(test_db_url: URL) -> Iterator[Engine]:
    engine = create_engine(test_db_url, pool_pre_ping=True)
    with engine.begin() as conn:  # start from an empty schema every run
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    upgrade(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db(migrated_engine: Engine) -> Iterator[Session]:
    """A session whose work is rolled back after the test (commits become savepoints)."""
    connection = migrated_engine.connect()
    outer = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        connection.close()

"""Database engine and sessions.

The URL is built with `sqlalchemy.URL.create`, so the password (from the secrets
provider) is escaped correctly whatever characters it contains, and it never
appears when the URL is printed or logged (SQLAlchemy masks it as ***).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from mdx_assessment.secret_store import SecretName, SecretsProvider, get_secrets_provider
from mdx_assessment.settings import PostgresSettings, get_settings

DRIVER = "postgresql+psycopg"


def database_url(
    postgres: PostgresSettings | None = None,
    secrets: SecretsProvider | None = None,
    *,
    database: str | None = None,
) -> URL:
    """URL for the configured Postgres; `database` overrides the database name."""
    pg = postgres or get_settings().postgres
    password = (secrets or get_secrets_provider()).require(SecretName.POSTGRES_PASSWORD)
    return URL.create(
        DRIVER,
        username=pg.user,
        password=password.get_secret_value(),
        host=pg.host,
        port=pg.port,
        database=database or pg.db,
    )


def create_db_engine(url: URL | None = None) -> Engine:
    """Engine with connection checks (a dropped connection is replaced, not reused)."""
    return create_engine(url or database_url(), pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One unit of work: commit on success, roll back on any error."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()

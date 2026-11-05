"""Alembic environment.

Connection, in order of preference:
1. `config.attributes["connection"]`: an open connection passed in by code
   (`mdx_assessment.core.persistence.migrate`, used by tests).
2. Otherwise an engine built from settings and the secrets provider (`make migrate`).
No URL is ever written to alembic.ini.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from mdx_assessment.core.persistence.db import create_db_engine, database_url
from mdx_assessment.core.persistence.models import Base

config = context.config
target_metadata = Base.metadata

if config.config_file_name:  # CLI run: use alembic.ini's logging settings
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_offline() -> None:
    """`alembic upgrade head --sql`: print the SQL instead of running it."""
    context.configure(
        url=database_url().render_as_string(hide_password=True),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = config.attributes.get("connection")
    if isinstance(connection, Connection):
        _run(connection)
        return
    engine = create_db_engine()
    try:
        with engine.connect() as conn:
            _run(conn)
            conn.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()

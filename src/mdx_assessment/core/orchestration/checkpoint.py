"""LangGraph checkpointer on PostgreSQL (PDF §2.1: shared state, checkpointed after each step).

- **Separate schema.** LangGraph creates and migrates its own tables (`checkpoints`,
  `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`). They live in the
  `langgraph` schema, so they never mix with the Alembic-managed `public` schema.
- **Strict deserialization.** A checkpoint can only rebuild classes on an explicit
  allow-list. The list is derived from `RunState` (every model and enum it can
  contain), so it can't drift from the state, and an unknown class in the database
  is refused instead of instantiated.
- **One thread per session.** `thread_config(session_id)` is the LangGraph config
  for a session; resuming a session means invoking the graph again with it.
"""

from __future__ import annotations

import types
import typing
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel
from sqlalchemy import URL

from mdx_assessment.core.orchestration.state import RunState

CHECKPOINT_SCHEMA = "langgraph"


def model_types(root: type[BaseModel]) -> set[type]:
    """Every Pydantic model and Enum reachable from `root`'s field annotations."""
    found: set[type] = set()

    def visit(annotation: Any) -> None:
        if isinstance(annotation, type):
            if annotation in found:
                return
            if issubclass(annotation, BaseModel):
                found.add(annotation)
                for field in annotation.model_fields.values():
                    visit(field.annotation)
                return
            if issubclass(annotation, Enum):
                found.add(annotation)
                return
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            visit(typing.get_args(annotation)[0])
            return
        if origin is not None or isinstance(annotation, types.UnionType):
            for arg in typing.get_args(annotation):
                visit(arg)

    visit(root)
    return found


def checkpoint_serializer(extra_types: Iterable[type] = ()) -> JsonPlusSerializer:
    """Serializer that may rebuild only `RunState`'s own types (plus `extra_types`)."""
    allowed = sorted(model_types(RunState) | set(extra_types), key=lambda t: t.__qualname__)
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def checkpoint_conninfo(url: URL) -> str:
    """psycopg connection string for the checkpointer, with `search_path` = its schema."""
    plain = url.set(
        drivername="postgresql", query={"options": f"-csearch_path={CHECKPOINT_SCHEMA}"}
    )
    return plain.render_as_string(hide_password=False)


def thread_config(session_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": session_id}}


@asynccontextmanager
async def open_checkpointer(url: URL) -> AsyncIterator[AsyncPostgresSaver]:
    """A ready checkpointer: schema created, LangGraph's own migrations applied."""
    base = url.set(drivername="postgresql").render_as_string(hide_password=False)
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{CHECKPOINT_SCHEMA}"')
    async with AsyncPostgresSaver.from_conn_string(
        checkpoint_conninfo(url), serde=checkpoint_serializer()
    ) as saver:
        await saver.setup()
        yield saver

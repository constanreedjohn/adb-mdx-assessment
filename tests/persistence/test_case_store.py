"""Acceptance (E3-04) against PostgreSQL: the cases MCP over the real case store."""

import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest
from mcp import Client
from mcp.types import TextContent
from sqlalchemy import Engine, delete

from mdx_assessment.core.domain import SessionResult, Versions
from mdx_assessment.core.persistence import (
    PostgresCaseStore,
    SessionRepo,
    UnknownSessionError,
    create_session_factory,
    session_scope,
)
from mdx_assessment.core.persistence.models import CaseRow, SessionRow
from mdx_assessment.core.tools.cases import (
    CREATE_CASE,
    GET_CASE,
    LIST_CASES,
    CreateCaseResult,
    ListCasesResult,
    create_cases_server,
)

pytestmark = [pytest.mark.db, pytest.mark.anyio]
T0 = datetime(2026, 10, 15, 9, 20, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StepClock:
    """One minute later on every call, so creation order is unambiguous."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> datetime:
        self._n += 1
        return T0 + timedelta(minutes=self._n)


@pytest.fixture
def session_id(migrated_engine: Engine) -> Iterator[str]:
    """A fresh session row, removed afterwards with its cases: other tests expect empty tables."""
    sid = f"sess-{uuid.uuid4().hex[:12]}"
    factory = create_session_factory(migrated_engine)
    with session_scope(factory) as db:
        SessionRepo(db).add(
            SessionResult(
                session_id=sid,
                business_date=date(2026, 10, 15),
                data_type="linear",
                started_at=T0,
                versions=Versions(config="sha256:abc"),
            )
        )
    yield sid
    with session_scope(factory) as db:
        db.execute(delete(CaseRow).where(CaseRow.session_id == sid))
        db.execute(delete(SessionRow).where(SessionRow.id == sid))


@pytest.fixture
def store(migrated_engine: Engine) -> PostgresCaseStore:
    return PostgresCaseStore(create_session_factory(migrated_engine), clock=StepClock())


def test_create_is_idempotent_per_item(store: PostgresCaseStore, session_id: str) -> None:
    first = store.create(session_id, "JPY 3M", {"change_bps": 3.0})
    again = store.create(session_id, "JPY 3M", {"change_bps": 99.0})
    assert (first.created, again.created) == (True, False)
    assert again.case == first.case
    store.create(session_id, "GBP 2Y", {})
    assert [c.item_key for c in store.list_for_session(session_id)] == ["JPY 3M", "GBP 2Y"]
    assert store.get(first.case.case_id) == first.case
    assert store.get("case-nope") is None


def test_a_case_needs_an_existing_session(store: PostgresCaseStore) -> None:
    with pytest.raises(UnknownSessionError, match="unknown session 'sess-nope'"):
        store.create("sess-nope", "JPY 3M", {})


async def test_through_mcp(store: PostgresCaseStore, session_id: str) -> None:
    async with Client(create_cases_server(store)) as client:
        result = await client.call_tool(
            CREATE_CASE,
            {"session_id": session_id, "item_key": "JPY 3M", "payload": {"note": "ü"}},
        )
        created = CreateCaseResult.model_validate(result.structured_content)
        case_id = created.case.case_id

        got = await client.call_tool(GET_CASE, {"case_id": case_id})
        assert got.structured_content is not None
        assert got.structured_content["payload"] == {"note": "ü"}

        listed = ListCasesResult.model_validate(
            (await client.call_tool(LIST_CASES, {"session_id": session_id})).structured_content
        )
        assert [c.case_id for c in listed.cases] == [case_id]

        unknown = await client.call_tool(
            CREATE_CASE, {"session_id": "sess-nope", "item_key": "JPY 3M"}
        )
        assert unknown.is_error
        text = " ".join(c.text for c in unknown.content if isinstance(c, TextContent))
        assert "unknown session" in text

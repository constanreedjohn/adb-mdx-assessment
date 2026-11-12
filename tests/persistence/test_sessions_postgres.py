"""Acceptance (E12-01) on PostgreSQL: the row and the final state are persisted."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import URL, Engine, text
from tests.support import explain_from_inputs, investigate_from_inputs

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.api import create_app
from mdx_assessment.core.gateway import FakeProvider, LLMGateway, create_llm_gateway
from mdx_assessment.core.observability import Tracer
from mdx_assessment.core.orchestration.checkpoint import open_checkpointer
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.session_service import SessionService
from mdx_assessment.core.persistence import (
    PostgresAuditTrail,
    PostgresCaseStore,
    create_session_factory,
)
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.persistence.session_store import PostgresSessionStore
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.db


def test_a_session_survives_in_postgres(
    migrated_engine: Engine, test_db_url: URL, tmp_path: Path
) -> None:
    config = load_config(PROJECT_ROOT / "config")
    write_day(build_linear_day(config, date(2026, 10, 15)), tmp_path, config_version=config.version)
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=tmp_path)}
    )
    tasks = resolve_playbook([], PlaybookStore.from_config(config)).content.mandatory_steps
    plan = {
        "steps": [
            {"step_id": f"s{i}", "task": t, "rationale": "Needed."} for i, t in enumerate(tasks)
        ],
        "follow_ups": [],
        "rationale": "Normal day.",
        "cited_fact_ids": ["fact-flag-count"],
    }

    @asynccontextmanager
    async def checkpointer() -> AsyncIterator[BaseCheckpointSaver[str]]:
        async with open_checkpointer(test_db_url) as saver:
            yield saver

    def make_gateway(audit: AuditTrail, tracer: Tracer) -> LLMGateway:
        return create_llm_gateway(
            audit,
            tracer,
            settings,
            FakeProvider(
                {
                    "master_plan": plan,
                    "mda_explain": explain_from_inputs,
                    "mda_investigate": investigate_from_inputs,
                }
            ),
        )

    store = PostgresSessionStore(create_session_factory(migrated_engine))
    service = SessionService(
        registries=build_registries(),
        config=config,
        settings=settings,
        store=store,
        audit=PostgresAuditTrail(migrated_engine),
        cases=PostgresCaseStore(create_session_factory(migrated_engine)),
        checkpointer=checkpointer,
        make_gateway=make_gateway,
        make_tracer=lambda: Tracer(None),
    )
    client = TestClient(create_app(sessions=service))
    body = {"business_date": "2026-10-15", "data_type": "linear"}
    session_id = client.post("/sessions", json=body).json()["session_id"]

    stored = store.get(session_id)
    assert stored is not None and stored.status.value == "succeeded"
    assert client.get(f"/sessions/{session_id}").json()["plan"]["source"] == "master"
    with migrated_engine.connect() as conn:
        status = conn.scalar(text("SELECT status FROM sessions WHERE id = :i"), {"i": session_id})
        checkpoints = conn.scalar(
            text("SELECT count(*) FROM langgraph.checkpoints WHERE thread_id = :i"),
            {"i": session_id},
        )
    assert status == "succeeded"
    assert checkpoints and checkpoints > 0

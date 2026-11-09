"""Acceptance (E12-01): both endpoints on the simulated day, with Langfuse spans captured
in memory. Everything real except the model and the database (see tests/persistence)."""

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langfuse import LangfuseOtelSpanAttributes
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.api import create_app
from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.gateway import (
    FakeProvider,
    LLMGateway,
    LLMUnavailableError,
    create_llm_gateway,
)
from mdx_assessment.core.gateway.provider import LLMRequest
from mdx_assessment.core.observability import Tracer, build_langfuse
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.session_service import SessionService
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.persistence.session_store import InMemorySessionStore
from mdx_assessment.secret_store import SecretName
from mdx_assessment.settings import (
    PROJECT_ROOT,
    AppSettings,
    LangfuseSettings,
    Settings,
    get_settings,
)
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

DAY = date(2026, 10, 15)
BODY = {"business_date": "2026-10-15", "data_type": "linear"}


@pytest.fixture(scope="module")
def config() -> LoadedConfig:
    return load_config(PROJECT_ROOT / "config")


@pytest.fixture(scope="module")
def settings(config: LoadedConfig, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    data_dir = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(config, DAY), data_dir, config_version=config.version)
    base = get_settings()
    return base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )


@asynccontextmanager
async def memory_checkpointer() -> AsyncIterator[InMemorySaver]:
    yield InMemorySaver(serde=checkpoint_serializer())


def valid_plan(config: LoadedConfig) -> dict[str, Any]:
    tasks = resolve_playbook([], PlaybookStore.from_config(config)).content.mandatory_steps
    return {
        "steps": [
            {"step_id": f"s{i}", "task": t, "rationale": "Needed."} for i, t in enumerate(tasks)
        ],
        "follow_ups": [],
        "rationale": "Normal day, see fact-flag-count.",
        "cited_fact_ids": ["fact-flag-count"],
    }


def client_for(
    config: LoadedConfig, settings: Settings, script: Any, tracer: Tracer | None = None
) -> TestClient:
    audit = InMemoryAuditTrail()

    def make_gateway(trail: AuditTrail, active: Tracer) -> LLMGateway:
        if isinstance(script, Exception):
            raise script
        fake = FakeProvider({"master_plan": script})
        return create_llm_gateway(trail, active, settings, fake)

    service = SessionService(
        registries=build_registries(),
        config=config,
        settings=settings,
        store=InMemorySessionStore(),
        audit=audit,
        checkpointer=memory_checkpointer,
        make_gateway=make_gateway,
        make_tracer=lambda: tracer or Tracer(None),
    )
    return TestClient(create_app(sessions=service))


def down(request: LLMRequest) -> dict[str, Any]:
    raise LLMUnavailableError("connection refused")


@pytest.fixture
def client(config: LoadedConfig, settings: Settings) -> TestClient:
    return client_for(config, settings, valid_plan(config))


def test_post_then_get_on_the_simulated_day(client: TestClient, config: LoadedConfig) -> None:
    started = client.post("/sessions", json=BODY)
    assert started.status_code == 201
    assert started.json()["status"] == "completed"

    session = client.get(f"/sessions/{started.json()['session_id']}").json()
    assert session["status"] == "completed"
    assert session["finished_at"] is not None
    facts = {f["id"]: f["value"] for f in session["run_context"]["facts"]}
    assert facts["fact-flag-count"] == 28
    assert session["playbook"]["id"] == "normal_eod"
    assert session["playbook"]["selection"]["rules"][0]["met"] is True
    assert session["plan"]["source"] == "master"
    assert session["plan"]["cited_fact_ids"] == ["fact-flag-count"]
    assert session["plan"]["validation"]["checks"]["P-1"] is True
    assert [e["status"] for e in session["execution"]] == ["done", "done", "done"]
    assert session["agent_results"]["s0"]["flags_read"] == 28
    assert session["versions"] == {
        "config": config.version,
        "prompts": {"master_plan": "v1"},
        "llm_deployment": "fake",
        "ml_model": None,
    }
    assert session["trace"]["langfuse_trace_id"]


def test_a_rejected_plan_shows_the_marked_default_plan(
    config: LoadedConfig, settings: Settings
) -> None:
    client = client_for(config, settings, {**valid_plan(config), "steps": []})
    session_id = client.post("/sessions", json=BODY).json()["session_id"]
    session = client.get(f"/sessions/{session_id}").json()
    assert session["status"] == "default_complete"
    assert session["plan"]["source"] == "default"
    assert session["fallbacks"] == ["plan_default:plan_invalid"]


def test_an_unavailable_model_still_completes_on_the_default_plan(
    config: LoadedConfig, settings: Settings
) -> None:
    client = client_for(config, settings, down)
    started = client.post("/sessions", json=BODY).json()
    assert started["status"] == "default_complete"
    assert "developer" in started["analyst_notice"]
    session = client.get(f"/sessions/{started['session_id']}").json()
    assert session["plan"]["source"] == "default"
    assert session["fallbacks"] == ["plan_default:model_unavailable"]
    assert [e["status"] for e in session["execution"]] == ["done", "done", "done"]
    assert session["agent_results"]["validate_inputs"]["flags_read"] == 28


def test_with_the_llm_off_a_run_needs_no_gateway(config: LoadedConfig, settings: Settings) -> None:
    # This gateway factory raises if it is called, so a completed run proves it wasn't.
    client = client_for(config, settings, RuntimeError("no model credentials"))
    started = client.post("/sessions", json={**BODY, "llm_enabled": False}).json()
    assert started["status"] == "default_complete"
    assert "turned off" in started["analyst_notice"]
    session = client.get(f"/sessions/{started['session_id']}").json()
    assert session["llm_enabled"] is False
    assert session["plan"]["source"] == "default"
    assert session["fallbacks"] == ["plan_default:llm_off"]
    assert session["versions"]["llm_deployment"] is None
    assert [e["status"] for e in session["execution"]] == ["done", "done", "done"]


def test_an_unknown_data_type_stops_the_run(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="mdx_assessment.core.orchestration.graph"):
        started = client.post("/sessions", json={**BODY, "data_type": "equity"}).json()
    assert started["status"] == "failed"
    assert "source system" in started["analyst_notice"]
    assert "could not gather the facts" in caplog.text
    assert "Traceback" in caplog.text


def test_a_crash_is_stored_as_a_failed_session(
    config: LoadedConfig, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    client = client_for(config, settings, RuntimeError("boom"))
    with caplog.at_level(logging.ERROR):
        started = client.post("/sessions", json=BODY).json()
    assert started["status"] == "failed"
    assert "unexpected" in started["analyst_notice"]
    assert "session" in caplog.text
    assert "crashed" in caplog.text
    assert client.get(f"/sessions/{started['session_id']}").status_code == 200


def test_unknown_session_is_404(client: TestClient) -> None:
    assert client.get("/sessions/sess-nope").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"business_date": "not-a-date", "data_type": "linear"},
        {"business_date": "2026-10-15"},
        {"business_date": "2026-10-15", "data_type": "Bad Type"},
        {**BODY, "surprise": 1},
    ],
)
def test_bad_requests_are_rejected(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/sessions", json=body).status_code == 422


# --- the Langfuse trace ---------------------------------------------------------------------


class FakeSecrets:
    backend = "fake"

    def get(self, name: SecretName) -> SecretStr | None:
        return SecretStr("sk-lf-test")

    def require(self, name: SecretName) -> SecretStr:
        raise NotImplementedError


@pytest.fixture
def traced() -> Iterator[tuple[Tracer, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    base = get_settings()
    settings = base.model_copy(
        update={
            "langfuse": LangfuseSettings(
                host="http://127.0.0.1:9", public_key=f"pk-lf-{uuid.uuid4().hex}", enabled=True
            )
        }
    )
    client = build_langfuse(settings, FakeSecrets(), span_exporter=exporter)
    assert client is not None
    yield Tracer(client), exporter
    client.shutdown()


def test_one_trace_shows_every_node_tool_and_model_call(
    config: LoadedConfig, settings: Settings, traced: tuple[Tracer, InMemorySpanExporter]
) -> None:
    tracer, exporter = traced
    client = client_for(config, settings, valid_plan(config), tracer)
    client.post("/sessions", json=BODY)
    tracer.flush()
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    for node in ("run_context", "select_playbook", "master_plan", "master_synthesize"):
        assert names.count(f"node:{node}") == 1
    assert names.count("node:dispatch") == names.count("node:review_point") == 3
    assert "llm:master_plan" in names
    assert "plan_validation" in names
    assert any(n.startswith("task:") for n in names)
    assert any("get_flags" in n for n in names)
    assert len({s.context.trace_id for s in spans}) == 1
    assert names.count("assessment_session") == 1
    ids = {s.context.span_id for s in spans}
    children = [s for s in spans if s.name != "assessment_session"]
    assert all(s.parent is not None and s.parent.span_id in ids for s in children)


def test_a_stopped_run_is_marked_as_an_error_in_the_trace(
    config: LoadedConfig, settings: Settings, traced: tuple[Tracer, InMemorySpanExporter]
) -> None:
    tracer, exporter = traced
    client = client_for(config, settings, valid_plan(config), tracer)
    client.post("/sessions", json={**BODY, "data_type": "equity"})  # no facts can be read
    tracer.flush()
    level = LangfuseOtelSpanAttributes.OBSERVATION_LEVEL
    errors = {
        s.name for s in exporter.get_finished_spans() if (s.attributes or {}).get(level) == "ERROR"
    }
    assert {"assessment_session", "node:run_context"} <= errors

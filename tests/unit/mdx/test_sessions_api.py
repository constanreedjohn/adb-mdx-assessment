"""Acceptance (E12-01): both endpoints on the simulated day, with MLflow traces captured
in memory. Everything real except the model and the database (see tests/persistence)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from tests.support import CapturedTraces, accept_all, explain_from_inputs, investigate_from_inputs

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
from mdx_assessment.core.observability import Tracer
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.session_service import SessionService
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.persistence.session_store import InMemorySessionStore
from mdx_assessment.settings import (
    PROJECT_ROOT,
    AppSettings,
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
        fake = FakeProvider(
            {
                "master_plan": script,
                "mda_explain": explain_from_inputs,
                "mda_investigate": investigate_from_inputs,
                "master_validate": accept_all,
            }
        )
        return create_llm_gateway(trail, active, settings, fake)

    service = SessionService(
        registries=build_registries(),
        config=config,
        settings=settings,
        store=InMemorySessionStore(),
        audit=audit,
        cases=InMemoryCaseStore(),
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
    assert started.json()["status"] == "succeeded"

    session = client.get(f"/sessions/{started.json()['session_id']}").json()
    assert session["status"] == "succeeded"
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
        "prompts": {
            "master_plan": "v1",
            "master_review": "v1",
            "mda_explain": "v1",
            "mda_investigate": "v1",
            "master_validate": "v1",
        },
        "llm_deployment": "fake",
        "ml_model": None,
    }
    assert session["trace"] is None  # no tracing in this client


def test_a_rejected_plan_shows_the_marked_default_plan(
    config: LoadedConfig, settings: Settings
) -> None:
    client = client_for(config, settings, {**valid_plan(config), "steps": []})
    session_id = client.post("/sessions", json=BODY).json()["session_id"]
    session = client.get(f"/sessions/{session_id}").json()
    assert session["status"] == "succeeded"
    assert session["plan"]["source"] == "default"
    assert session["fallbacks"] == ["plan_default:plan_invalid"]


def test_an_unavailable_model_still_completes_on_the_default_plan(
    config: LoadedConfig, settings: Settings
) -> None:
    client = client_for(config, settings, down)
    started = client.post("/sessions", json=BODY).json()
    assert started["status"] == "succeeded"
    assert "developer" in started["analyst_notice"]
    session = client.get(f"/sessions/{started['session_id']}").json()
    assert session["plan"]["source"] == "default"
    assert session["fallbacks"] == ["plan_default:model_unavailable"]
    assert [e["status"] for e in session["execution"]] == ["done", "done", "done"]
    assert session["agent_results"]["validate_inputs"]["flags_read"] == 28


class ClosingProvider(FakeProvider):
    def __init__(self, script: Any) -> None:
        super().__init__(
            {
                "master_plan": script,
                "mda_explain": explain_from_inputs,
                "mda_investigate": investigate_from_inputs,
                "master_validate": accept_all,
            }
        )
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("data_type", ["linear", "equity"])  # completed, and stopped
def test_every_session_closes_its_model_connections(
    config: LoadedConfig, settings: Settings, data_type: str
) -> None:
    providers: list[ClosingProvider] = []

    def make_gateway(trail: AuditTrail, active: Tracer) -> LLMGateway:
        providers.append(ClosingProvider(valid_plan(config)))
        return create_llm_gateway(trail, active, settings, providers[-1])

    service = SessionService(
        registries=build_registries(),
        config=config,
        settings=settings,
        store=InMemorySessionStore(),
        audit=InMemoryAuditTrail(),
        cases=InMemoryCaseStore(),
        checkpointer=memory_checkpointer,
        make_gateway=make_gateway,
        make_tracer=lambda: Tracer(None),
    )
    client = TestClient(create_app(sessions=service))
    client.post("/sessions", json={**BODY, "data_type": data_type})
    assert [p.closed for p in providers] == [True]


def test_with_the_llm_off_a_run_needs_no_gateway(config: LoadedConfig, settings: Settings) -> None:
    # This gateway factory raises if it is called, so a completed run proves it wasn't.
    client = client_for(config, settings, RuntimeError("no model credentials"))
    started = client.post("/sessions", json={**BODY, "llm_enabled": False}).json()
    assert started["status"] == "succeeded"
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


# --- the MLflow trace -----------------------------------------------------------------------


def test_one_trace_shows_every_node_tool_and_model_call(
    config: LoadedConfig, settings: Settings, captured_traces: CapturedTraces
) -> None:
    tracer, traces = captured_traces
    client = client_for(config, settings, valid_plan(config), tracer)
    started = client.post("/sessions", json=BODY).json()
    assert started["trace_url"].startswith("http://mlflow.test/#/experiments/0/traces")

    [trace] = traces
    names = [s.name for s in trace.data.spans]
    for node in ("run_context", "select_playbook", "master_plan", "master_synthesize"):
        assert names.count(f"node:{node}") == 1
    assert names.count("node:dispatch") == names.count("node:review_point") == 3
    assert "llm:master_plan" in names
    assert "plan_validation" in names
    assert any(n.startswith("task:") for n in names)
    assert any("get_flags" in n for n in names)
    assert names.count("assessment_session") == 1
    ids = {s.span_id for s in trace.data.spans}
    children = [s for s in trace.data.spans if s.name != "assessment_session"]
    assert all(s.parent_id in ids for s in children)
    assert trace.info.trace_metadata["mlflow.trace.session"] == started["session_id"]
    assert trace.info.trace_id in started["trace_url"]


def test_a_stopped_run_is_marked_as_an_error_in_the_trace(
    config: LoadedConfig, settings: Settings, captured_traces: CapturedTraces
) -> None:
    tracer, traces = captured_traces
    client = client_for(config, settings, valid_plan(config), tracer)
    client.post("/sessions", json={**BODY, "data_type": "equity"})
    [trace] = traces
    errors = {s.name for s in trace.data.spans if s.status.status_code == "ERROR"}
    assert {"assessment_session", "node:run_context"} <= errors
    assert trace.info.state == "ERROR"

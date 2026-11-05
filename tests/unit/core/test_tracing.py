"""Tracing, offline: spans are captured in memory instead of sent to Langfuse."""

import re
import uuid
from collections.abc import Iterator

import pytest
from langfuse import Langfuse, LangfuseOtelSpanAttributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from mdx_assessment.core.observability import Tracer, build_langfuse, trace_id_for_session
from mdx_assessment.secret_store import SecretName
from mdx_assessment.settings import AppSettings, LangfuseSettings, Settings, get_settings


class FakeSecrets:
    backend = "fake"

    def __init__(self, langfuse_secret: str | None) -> None:
        self._secret = langfuse_secret

    def get(self, name: SecretName) -> SecretStr | None:
        return SecretStr(self._secret) if self._secret else None

    def require(self, name: SecretName) -> SecretStr:
        raise NotImplementedError


def settings_with(**langfuse: object) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "langfuse": LangfuseSettings.model_validate(
                {"host": "http://127.0.0.1:9", "public_key": "", "enabled": True, **langfuse}
            ),
            "app": AppSettings(environment="test"),
        }
    )


# --- configuration ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("langfuse", "secret"),
    [
        ({"public_key": ""}, "sk-lf-x"),  # no public key
        ({"public_key": "pk-lf-x", "enabled": False}, "sk-lf-x"),  # switched off
        ({"public_key": "pk-lf-x"}, None),  # no secret key
    ],
)
def test_tracing_is_off_unless_fully_configured(
    langfuse: dict[str, object], secret: str | None
) -> None:
    assert build_langfuse(settings_with(**langfuse), FakeSecrets(secret)) is None


def test_disabled_tracer_is_a_working_no_op() -> None:
    tracer = Tracer(None)
    assert not tracer.enabled
    with tracer.session_trace("sess-1") as handle, tracer.span("step") as span:
        assert span is None
    assert handle.trace_id == trace_id_for_session("sess-1")
    assert handle.url is None
    assert tracer.auth_check() is False
    tracer.flush()
    tracer.shutdown()


def test_trace_id_is_stable_per_session() -> None:
    a = trace_id_for_session("sess-1")
    assert re.fullmatch(r"[0-9a-f]{32}", a)
    assert trace_id_for_session("sess-1") == a
    assert trace_id_for_session("sess-2") != a


# --- spans, captured in memory ------------------------------------------------------------


@pytest.fixture
def captured() -> Iterator[tuple[Tracer, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    client = build_langfuse(
        settings_with(
            public_key=f"pk-lf-test-{uuid.uuid4()}"
        ),  # unique: Langfuse caches clients per key
        FakeSecrets("sk-lf-test"),
        span_exporter=exporter,
    )
    assert isinstance(client, Langfuse)
    tracer = Tracer(client)
    yield tracer, exporter
    tracer.shutdown()


def test_one_session_is_one_trace(captured: tuple[Tracer, InMemorySpanExporter]) -> None:
    tracer, exporter = captured
    with tracer.session_trace("sess-42", input={"business_date": "2026-10-15"}) as handle:
        with tracer.span("validate_inputs"):
            pass
        with tracer.span("assess_movements"):
            pass
    tracer.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"assessment_session", "validate_inputs", "assess_movements"}
    trace_ids = {format(s.context.trace_id, "032x") for s in spans.values() if s.context}
    assert trace_ids == {handle.trace_id} == {trace_id_for_session("sess-42")}

    root = spans["assessment_session"]
    for child in ("validate_inputs", "assess_movements"):
        parent = spans[child].parent
        assert parent is not None and root.context is not None
        assert parent.span_id == root.context.span_id


def test_every_span_carries_the_session_and_environment(
    captured: tuple[Tracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = captured
    with tracer.session_trace("sess-7", tags=["skeleton"]), tracer.span("step"):
        pass
    tracer.flush()
    for span in exporter.get_finished_spans():
        attributes = span.attributes or {}
        assert attributes.get(LangfuseOtelSpanAttributes.TRACE_SESSION_ID) == "sess-7"
        assert attributes.get(LangfuseOtelSpanAttributes.ENVIRONMENT) == "test"


def test_a_resumed_session_continues_the_same_trace(
    captured: tuple[Tracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = captured
    for _ in range(2):  # e.g. interrupted, then resumed from the checkpoint
        with tracer.session_trace("sess-resume"), tracer.span("step"):
            pass
    tracer.flush()
    assert {
        format(s.context.trace_id, "032x") for s in exporter.get_finished_spans() if s.context
    } == {trace_id_for_session("sess-resume")}

"""Tracing, offline: finished traces are captured instead of sent to an MLflow server."""

import pytest
from tests.support import CapturedTraces

from mdx_assessment.core.observability import MlflowTarget, Tracer, build_target
from mdx_assessment.core.observability import tracing as tracing_module
from mdx_assessment.settings import AppSettings, MLflowSettings, Settings, get_settings


def settings_with(*, enabled: bool = True, uri: str = "http://127.0.0.1:9") -> Settings:
    return get_settings().model_copy(
        update={
            "mlflow": MLflowSettings(tracking_uri=uri),
            "app": AppSettings(environment="test", tracing_enabled=enabled),
        }
    )


# --- configuration ---------------------------------------------------------------------


def test_tracing_can_be_switched_off() -> None:
    assert build_target(settings_with(enabled=False)) is None


def test_an_unreachable_server_disables_tracing(caplog: pytest.LogCaptureFixture) -> None:
    assert build_target(settings_with()) is None
    assert "not reachable" in caplog.text


def test_set_up_failure_disables_tracing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(name: str) -> None:
        raise RuntimeError("no such thing")

    monkeypatch.setattr(tracing_module, "_server_is_healthy", lambda uri: True)
    monkeypatch.setattr(tracing_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(tracing_module.mlflow, "set_experiment", broken)
    assert build_target(settings_with()) is None
    assert "tracing disabled" in caplog.text


def test_disabled_tracer_is_a_working_no_op() -> None:
    tracer = Tracer(None)
    assert not tracer.enabled
    with tracer.session_trace("sess-1") as handle, tracer.span("step") as span:
        assert span is None
    with tracer.generation("llm", model="m") as generation:
        assert generation is None
    assert (handle.trace_id, handle.url, handle.root) == (None, None, None)
    tracer.flush()


# --- spans, captured ------------------------------------------------------------------------


def test_one_session_is_one_trace(captured_traces: CapturedTraces) -> None:
    tracer, traces = captured_traces
    with tracer.session_trace("sess-42", input={"business_date": "2026-10-15"}) as handle:
        with tracer.span("validate_inputs"):
            pass
        with tracer.span("assess_movements"):
            pass

    [trace] = traces
    assert trace.info.trace_id == handle.trace_id
    spans = {s.name: s for s in trace.data.spans}
    assert set(spans) == {"assessment_session", "validate_inputs", "assess_movements"}
    root = spans["assessment_session"]
    assert root.span_type == "AGENT"
    assert root.inputs == {"business_date": "2026-10-15"}
    assert spans["validate_inputs"].parent_id == root.span_id
    assert spans["assess_movements"].parent_id == root.span_id


def test_trace_carries_the_session_environment_and_tags(captured_traces: CapturedTraces) -> None:
    tracer, traces = captured_traces
    with (
        tracer.session_trace("sess-7", metadata={"config": "abc"}, tags={"data_type": "linear"}),
        tracer.span("step"),
    ):
        pass
    [trace] = traces
    metadata = trace.info.trace_metadata
    assert metadata["mlflow.trace.session"] == "sess-7"
    assert metadata["environment"] == "test"
    assert metadata["config"] == "abc"
    assert trace.info.tags["data_type"] == "linear"


def test_a_session_run_twice_is_grouped_by_session(captured_traces: CapturedTraces) -> None:
    tracer, traces = captured_traces
    for _ in range(2):  # e.g. interrupted, then resumed from the checkpoint
        with tracer.session_trace("sess-resume"), tracer.span("step"):
            pass
    assert len(traces) == 2
    assert {t.info.trace_metadata["mlflow.trace.session"] for t in traces} == {"sess-resume"}


def test_the_handle_links_to_the_trace(captured_traces: CapturedTraces) -> None:
    tracer, _ = captured_traces
    with tracer.session_trace("sess-1") as handle:
        pass
    assert handle.url == (
        f"http://mlflow.test/#/experiments/0/traces?selectedEvaluationId={handle.trace_id}"
    )


def test_span_updates_are_recorded(captured_traces: CapturedTraces) -> None:
    tracer, traces = captured_traces
    with tracer.session_trace("sess-1") as handle:
        with tracer.span("step", kind="chain", metadata={"step_id": "s1"}) as span:
            assert span is not None
            span.update(output={"n": 1})
        with tracer.span("broken", kind="tool") as span:
            assert span is not None
            span.update(error="tool said no")
        assert handle.root is not None
        handle.root.update(error="the run stopped")
    [trace] = traces
    spans = {s.name: s for s in trace.data.spans}
    assert spans["step"].span_type == "CHAIN"
    assert spans["step"].outputs == {"n": 1}
    assert spans["step"].attributes["step_id"] == "s1"
    assert spans["step"].status.status_code == "OK"
    assert spans["broken"].span_type == "TOOL"
    assert spans["broken"].status.status_code == "ERROR"
    assert spans["broken"].status.description == "tool said no"
    assert trace.info.state == "ERROR"


def test_a_model_call_records_model_prompt_and_tokens(captured_traces: CapturedTraces) -> None:
    tracer, traces = captured_traces
    with (
        tracer.session_trace("sess-1"),
        tracer.generation("llm:plan", model="gpt-x", version="v3", input={"q": 1}) as generation,
    ):
        assert generation is not None
        generation.update(output={"a": 2}, model="gpt-x-2026", input_tokens=10, output_tokens=5)
    [trace] = traces
    span = next(s for s in trace.data.spans if s.name == "llm:plan")
    assert span.span_type == "LLM"
    assert span.attributes["mlflow.llm.model"] == "gpt-x-2026"
    assert span.attributes["prompt_version"] == "v3"
    assert span.attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert trace.info.token_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_a_crash_inside_a_span_marks_it_as_an_error(captured_traces: CapturedTraces) -> None:
    tracer, traces = captured_traces
    with pytest.raises(RuntimeError), tracer.session_trace("sess-1"), tracer.span("step"):
        raise RuntimeError("boom")
    [trace] = traces
    assert next(s for s in trace.data.spans if s.name == "step").status.status_code == "ERROR"


def test_target_is_plain_configuration() -> None:
    assert Tracer(MlflowTarget(experiment_id="7", ui_url="http://x")).experiment_id == "7"

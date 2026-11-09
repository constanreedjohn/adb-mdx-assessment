"""Tracing with self-hosted MLflow (ADR-0004): one end-to-end trace per session.

- One run of a session = one trace. MLflow assigns the trace ID, so every trace carries
  the session ID in `mlflow.trace.session`, which groups a session's traces in the UI.
- Tracing never breaks a run. With tracing off, or the server unreachable at start-up,
  every method is a no-op. MLflow exports in the background, so losing the server
  mid-run only costs traces.
- Callers use `Tracer` and `TraceSpan`, never the MLflow SDK.

Graph nodes and tool calls are spans, and LLM calls are "LLM" spans with model and token
usage, all opened inside `session_trace()`.
"""

from __future__ import annotations

import logging
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import mlflow
from mlflow.entities import SpanStatus, SpanStatusCode, SpanType
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID, LiveSpan
from mlflow.tracing.constant import SpanAttributeKey

from mdx_assessment.core.meta import app_version
from mdx_assessment.settings import Settings, get_settings

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_SECONDS = 2
SpanKind = Literal["span", "tool", "chain"]
_SPAN_TYPES: dict[SpanKind, str] = {
    "span": SpanType.UNKNOWN,
    "tool": SpanType.TOOL,
    "chain": SpanType.CHAIN,
}


class TraceSpan:
    """One open span. Everything is optional, so a caller records only what it knows."""

    def __init__(self, span: LiveSpan) -> None:
        self._span = span

    def update(
        self,
        *,
        output: Any = None,
        error: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if output is not None:
            self._span.set_outputs(output)
        if error is not None:
            # The SDK's annotation omits SpanStatus, which its docstring and code accept.
            status = SpanStatus(SpanStatusCode.ERROR, description=error)
            self._span.set_status(status)  # type: ignore[arg-type]
        if model is not None:
            self._span.set_attribute(SpanAttributeKey.MODEL, model)
        if input_tokens is not None and output_tokens is not None:
            self._span.set_attribute(
                SpanAttributeKey.CHAT_USAGE,
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            )


@dataclass(frozen=True)
class TraceHandle:
    """What a session needs to know about its trace. Empty when tracing is off."""

    trace_id: str | None = None
    url: str | None = None
    root: TraceSpan | None = None


@dataclass(frozen=True)
class MlflowTarget:
    """Where traces go. The SDK itself is configured once, in `build_target`."""

    experiment_id: str
    ui_url: str
    environment: str = "local"


class Tracer:
    """Tracing behind one small interface. `Tracer(None)` is a working no-op tracer."""

    def __init__(self, target: MlflowTarget | None) -> None:
        self._target = target

    @property
    def enabled(self) -> bool:
        return self._target is not None

    @property
    def experiment_id(self) -> str | None:
        return self._target.experiment_id if self._target else None

    @contextmanager
    def session_trace(
        self,
        session_id: str,
        *,
        name: str = "assessment_session",
        input: Any = None,
        metadata: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[TraceHandle]:
        """Root span of the session's trace. Everything traced inside joins this trace."""
        if self._target is None:
            yield TraceHandle()
            return
        with mlflow.start_span(name=name, span_type=SpanType.AGENT) as span:
            if span.trace_id == NO_OP_SPAN_TRACE_ID:  # MLflow couldn't start the span
                yield TraceHandle()
                return
            if input is not None:
                span.set_inputs(input)
            mlflow.update_current_trace(
                session_id=session_id,
                metadata={
                    **(metadata or {}),
                    "environment": self._target.environment,
                    "app_version": app_version(),
                },
                tags=tags,
            )
            yield TraceHandle(
                trace_id=span.trace_id,
                url=self._trace_url(span.trace_id),
                root=TraceSpan(span),
            )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = "span",
    ) -> Iterator[TraceSpan | None]:
        """A child span of whatever is current. None is yielded when tracing is off."""
        if self._target is None:
            yield None
            return
        with self._open(name, _SPAN_TYPES[kind], input, metadata) as span:
            yield span

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan | None]:
        """A model call. Update it with output and token usage when done.

        `version` is the prompt version, so traces can be compared across prompt versions.
        """
        if self._target is None:
            yield None
            return
        attributes = {**(metadata or {}), SpanAttributeKey.MODEL: model}
        if version is not None:
            attributes["prompt_version"] = version
        with self._open(name, SpanType.LLM, input, attributes) as span:
            yield span

    @staticmethod
    @contextmanager
    def _open(
        name: str, span_type: str, input: Any, attributes: dict[str, Any] | None
    ) -> Iterator[TraceSpan]:
        with mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
            if input is not None:
                span.set_inputs(input)
            yield TraceSpan(span)

    def flush(self) -> None:
        """Wait for buffered traces to be sent (end of a session, end of a CLI command)."""
        if self._target is not None:
            mlflow.flush_trace_async_logging()

    def _trace_url(self, trace_id: str) -> str | None:
        if self._target is None:
            return None
        base = self._target.ui_url.rstrip("/")
        experiment = self._target.experiment_id
        return f"{base}/#/experiments/{experiment}/traces?selectedEvaluationId={trace_id}"


def _server_is_healthy(tracking_uri: str) -> bool:
    """A quick check first: MLflow's own client retries for minutes against a dead server."""
    url = tracking_uri.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_SECONDS) as response:  # noqa: S310
            return bool(response.status == 200)
    except (OSError, ValueError):
        return False


def build_target(settings: Settings | None = None) -> MlflowTarget | None:
    """Point the SDK at the server, or None when tracing is off or the server is down."""
    s = settings or get_settings()
    if not s.app.tracing_enabled:
        return None
    if not _server_is_healthy(s.mlflow.tracking_uri):
        logger.warning("MLflow is not reachable at %s; tracing disabled", s.mlflow.tracking_uri)
        return None
    try:
        mlflow.set_tracking_uri(s.mlflow.tracking_uri)
        experiment = mlflow.set_experiment(s.app.trace_experiment)
    except Exception:
        logger.warning("could not set up MLflow tracing; tracing disabled", exc_info=True)
        return None
    return MlflowTarget(
        experiment_id=experiment.experiment_id,
        ui_url=s.mlflow.ui_url or s.mlflow.tracking_uri,
        environment=s.app.environment,
    )


def build_tracer(settings: Settings | None = None) -> Tracer:
    return Tracer(build_target(settings))

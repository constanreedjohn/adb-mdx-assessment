"""Tracing with Langfuse (PDF §2.1 observability: one end-to-end trace per session).

- One session = one trace. The trace ID is derived from the session ID
  (`trace_id_for_session`), so a session resumed from a checkpoint keeps adding
  to the same trace, and the session JSON can carry the trace ID before anything
  is exported.
- Every span in a session also carries the Langfuse `session_id`, so sessions can
  be filtered in the Langfuse UI.
- Tracing never breaks a run: when Langfuse isn't configured (CI, tests, a missing
  key) every method is a no-op. The SDK exports in the background, so a Langfuse
  outage only costs traces, not the session.

E4-04 adds spans for every graph node and tool call, and generations for LLM calls,
using `span()` inside `session_trace()`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from langfuse import Langfuse, LangfuseChain, LangfuseSpan, LangfuseTool, propagate_attributes
from opentelemetry.sdk.trace.export import SpanExporter

from mdx_assessment.core.meta import app_version
from mdx_assessment.secret_store import SecretName, SecretsProvider, get_secrets_provider
from mdx_assessment.settings import Settings, get_settings

logger = logging.getLogger(__name__)

APP_NAME = "mdx-assessment"

SpanKind = Literal["span", "tool", "chain"]


def trace_id_for_session(session_id: str) -> str:
    """Deterministic 32-hex trace ID for a session (W3C trace-context format)."""
    return Langfuse.create_trace_id(seed=f"{APP_NAME}:session:{session_id}")


@dataclass(frozen=True)
class TraceHandle:
    """What a session needs to know about its trace."""

    trace_id: str
    url: str | None  # link to the trace in Langfuse; None when tracing is disabled


class Tracer:
    """Thin wrapper over the Langfuse client. `Tracer(None)` is a working no-op tracer."""

    def __init__(self, client: Langfuse | None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def session_trace(
        self,
        session_id: str,
        *,
        name: str = "assessment_session",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[TraceHandle]:
        """Root span of the session's trace. Everything traced inside joins this trace."""
        trace_id = trace_id_for_session(session_id)
        if self._client is None:
            yield TraceHandle(trace_id=trace_id, url=None)
            return
        with (
            self._client.start_as_current_observation(
                trace_context={"trace_id": trace_id},
                name=name,
                as_type="agent",
                input=input,
                metadata=metadata,
            ),
            propagate_attributes(session_id=session_id, trace_name=name, tags=tags),
        ):
            yield TraceHandle(trace_id=trace_id, url=self._trace_url(trace_id))

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = "span",
    ) -> Iterator[LangfuseSpan | LangfuseTool | LangfuseChain | None]:
        """A child span of whatever is current (e.g. a graph node, a tool call).

        `kind` sets how Langfuse displays it: "span", "tool" (MCP tool calls) or
        "chain" (a sequence of steps). None is yielded when tracing is disabled.
        """
        if self._client is None:
            yield None
            return
        with self._client.start_as_current_observation(
            name=name, as_type=kind, input=input, metadata=metadata
        ) as span:
            yield span

    def auth_check(self) -> bool:
        """True if Langfuse accepts the keys (a network call; for smoke tests, not per run).

        False on rejected keys, a wrong host/region or a network error; the reason is logged.
        """
        if self._client is None:
            return False
        try:
            return self._client.auth_check()
        except Exception as exc:
            logger.warning("Langfuse auth check failed: %s", type(exc).__name__)
            return False

    @property
    def client(self) -> Langfuse | None:
        """The underlying client, for API reads (e.g. confirming a trace arrived)."""
        return self._client

    def flush(self) -> None:
        """Send buffered spans now (end of a session, end of a CLI command)."""
        if self._client is not None:
            self._client.flush()

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()

    def _trace_url(self, trace_id: str) -> str | None:
        try:
            return self._client.get_trace_url(trace_id=trace_id) if self._client else None
        except Exception:  # a link is a convenience; never fail the run for it
            logger.debug("could not build Langfuse trace URL", exc_info=True)
            return None


def build_langfuse(
    settings: Settings | None = None,
    secrets: SecretsProvider | None = None,
    *,
    span_exporter: SpanExporter | None = None,
) -> Langfuse | None:
    """A Langfuse client from settings, or None when tracing is off or not configured.

    `span_exporter` replaces the network exporter (tests capture spans in memory).
    """
    s = settings or get_settings()
    lf = s.langfuse
    if not (lf.enabled and lf.public_key):
        return None
    secret = (secrets or get_secrets_provider()).get(SecretName.LANGFUSE_SECRET_KEY)
    if secret is None:
        logger.warning("Langfuse public key is set but the secret key isn't; tracing disabled")
        return None
    return Langfuse(
        public_key=lf.public_key,
        secret_key=secret.get_secret_value(),
        base_url=lf.host,
        environment=s.app.environment,
        release=app_version(),
        span_exporter=span_exporter,
    )


def build_tracer(
    settings: Settings | None = None, secrets: SecretsProvider | None = None
) -> Tracer:
    return Tracer(build_langfuse(settings, secrets))

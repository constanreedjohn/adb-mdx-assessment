"""`make smoke-trace`: send one test trace to Langfuse and confirm it was stored.

1. Check the keys (auth).
2. Open a session trace with one child span, then flush.
3. Poll the Langfuse API until the trace's observations are visible (ingestion is
   asynchronous), and print the trace URL.

Exit code 0 on success, 1 otherwise. Never prints keys.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime

from mdx_assessment.core.observability.tracing import build_tracer
from mdx_assessment.settings import get_settings

POLL_SECONDS = 60.0
POLL_INTERVAL = 3.0


def main() -> int:
    settings = get_settings()
    tracer = build_tracer(settings)
    print(f"Langfuse host : {settings.langfuse.host}")
    print(f"Environment   : {settings.app.environment}")
    if not tracer.enabled or tracer.client is None:
        print("FAILED: Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).")
        return 1
    if not tracer.auth_check():
        print("FAILED: Langfuse rejected the keys, or the host/region is wrong.")
        return 1
    print("Auth          : ok")

    session_id = "smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    with (
        tracer.session_trace(
            session_id,
            name="smoke_test",
            input={"purpose": "E4-03 tracing smoke test"},
            tags=["smoke"],
        ) as trace,
        tracer.span("smoke_child_step", input={"step": 1}) as span,
    ):
        if span is not None:
            span.update(output={"ok": True})
    tracer.flush()
    print(f"Session       : {session_id}")
    print(f"Trace ID      : {trace.trace_id}")

    deadline = time.monotonic() + POLL_SECONDS
    names: list[str] = []
    while time.monotonic() < deadline:
        response = tracer.client.api.observations.get_many(trace_id=trace.trace_id, limit=10)
        names = sorted(o.name or "" for o in response.data)
        if {"smoke_test", "smoke_child_step"} <= set(names):
            break
        time.sleep(POLL_INTERVAL)
    tracer.shutdown()

    if {"smoke_test", "smoke_child_step"} <= set(names):
        print(f"Stored        : yes ({', '.join(names)})")
        print(f"Open          : {trace.url}")
        return 0
    print(f"FAILED: trace not visible in Langfuse after {POLL_SECONDS:.0f}s (seen: {names}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""`make smoke-trace`: send one test trace to MLflow and confirm it was stored.

1. Check the server responds.
2. Open a session trace with one child span, then flush.
3. Search the server for the session until the trace's spans are visible.

Exit code 0 on success, 1 otherwise.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime

import mlflow

from mdx_assessment.core.observability.tracing import build_tracer
from mdx_assessment.settings import get_settings

POLL_SECONDS = 30.0
POLL_INTERVAL = 1.0


def main() -> int:
    settings = get_settings()
    tracer = build_tracer(settings)
    print(f"MLflow server : {settings.mlflow.tracking_uri}")
    print(f"Environment   : {settings.app.environment}")
    if not tracer.enabled or tracer.experiment_id is None:
        print("FAILED: MLflow is not reachable, or MDX_TRACING_ENABLED is false.")
        return 1

    session_id = "smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    with (
        tracer.session_trace(
            session_id,
            name="smoke_test",
            input={"purpose": "tracing smoke test"},
            tags={"purpose": "smoke"},
        ) as trace,
        tracer.span("smoke_child_step", input={"step": 1}) as span,
    ):
        if span is not None:
            span.update(output={"ok": True})
    tracer.flush()
    print(f"Session       : {session_id}")
    print(f"Trace ID      : {trace.trace_id}")

    expected = {"smoke_test", "smoke_child_step"}
    names: set[str] = set()
    deadline = time.monotonic() + POLL_SECONDS
    while time.monotonic() < deadline and not expected <= names:
        found = mlflow.search_traces(
            locations=[tracer.experiment_id],
            filter_string=f"metadata.`mlflow.trace.session` = '{session_id}'",
            return_type="list",
        )
        names = {s.name for t in found for s in t.data.spans}
        if not expected <= names:
            time.sleep(POLL_INTERVAL)

    if expected <= names:
        print(f"Stored        : yes ({', '.join(sorted(names))})")
        print(f"Open          : {trace.url}")
        return 0
    print(f"FAILED: trace not visible in MLflow after {POLL_SECONDS:.0f}s (seen: {sorted(names)}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

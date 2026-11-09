"""Suite-wide test configuration.

Tests marked `live` call real external services (Azure OpenAI, the MLflow server).
They are skipped unless `MDX_RUN_LIVE=1`, so `make test` stays offline and
deterministic; run them with `make test-live`. CI excludes them as well.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import mlflow
import pytest
from mlflow.entities import Trace
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from tests.support import CapturedTraces

from mdx_assessment.core.observability import MlflowTarget, Tracer

RUN_LIVE = os.environ.get("MDX_RUN_LIVE") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if RUN_LIVE:
        return
    skip_live = pytest.mark.skip(reason="live test: set MDX_RUN_LIVE=1 (make test-live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def captured_traces(monkeypatch: pytest.MonkeyPatch) -> Iterator[CapturedTraces]:
    """A real tracer whose finished traces land in a list instead of on a server."""
    traces: list[Trace] = []

    def capture(self: Any, trace: Trace, *args: Any, **kwargs: Any) -> None:
        traces.append(trace)

    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.setattr(MlflowV3SpanExporter, "_log_trace", capture)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", "0")
    mlflow.set_tracking_uri("http://127.0.0.1:9")
    target = MlflowTarget(experiment_id="0", ui_url="http://mlflow.test", environment="test")
    yield Tracer(target), traces

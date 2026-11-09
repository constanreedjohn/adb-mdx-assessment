"""Types shared by tests."""

from mlflow.entities import Trace

from mdx_assessment.core.observability import Tracer

CapturedTraces = tuple[Tracer, list[Trace]]

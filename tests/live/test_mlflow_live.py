"""Live: a test trace reaches the MLflow server (start it with `make dev`)."""

import pytest

from mdx_assessment.core.observability.smoke import main as smoke_trace

pytestmark = pytest.mark.live


def test_trace_is_stored_in_mlflow() -> None:
    assert smoke_trace() == 0

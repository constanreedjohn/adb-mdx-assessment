"""Live (E4-03 acceptance): a test trace reaches the Langfuse Cloud project."""

import pytest

from mdx_assessment.core.observability.smoke import main as smoke_trace

pytestmark = pytest.mark.live


def test_trace_is_stored_in_langfuse() -> None:
    assert smoke_trace() == 0

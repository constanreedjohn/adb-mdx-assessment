"""Live (E5-01 acceptance): the Azure OpenAI deployment returns a parsed structured reply."""

import pytest

from mdx_assessment.core.gateway.smoke import main as smoke_llm

pytestmark = pytest.mark.live


def test_structured_output_from_the_deployment() -> None:
    assert smoke_llm() == 0

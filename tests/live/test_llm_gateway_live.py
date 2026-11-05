"""Live (E5-02 acceptance): the same gateway call works against Azure OpenAI."""

import secrets

import pytest

from mdx_assessment.core.gateway import create_llm_gateway
from mdx_assessment.core.gateway.smoke import SmokeReply
from mdx_assessment.core.persistence import InMemoryAuditTrail

pytestmark = [pytest.mark.live, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_gateway_call_against_azure_openai() -> None:
    audit = InMemoryAuditTrail()
    gateway = create_llm_gateway(audit)  # default provider: Azure OpenAI
    token = "mdx-" + secrets.token_hex(4)
    result = await gateway.call(
        "smoke_test",
        "v1",
        inputs={"token": token},
        output_model=SmokeReply,
        actor="system",
        session_id="live-test",
    )
    assert result.provider == "azure_openai"
    assert result.output.status == "ok"
    assert result.output.echo == token
    [event] = audit.read()
    assert event.payload["outcome"] == "ok"
    assert event.payload["prompt_version"] == "v1"
    assert isinstance(event.payload["prompt_tokens"], int) and event.payload["prompt_tokens"] > 0

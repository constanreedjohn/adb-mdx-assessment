"""LLM gateway with the fake provider (and stub providers for edge cases). No network."""

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field
from tests.support import CapturedTraces

from mdx_assessment.core.gateway import (
    FakeProvider,
    LLMGateway,
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
    PromptStore,
    create_provider,
)
from mdx_assessment.core.observability import Tracer
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.settings import AppSettings, LLMProviderKind, get_settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["likely_valid", "monitor", "likely_data_issue"]
    confidence: float = Field(ge=0, le=1)


@pytest.fixture
def prompts(tmp_path: Path) -> PromptStore:
    folder = tmp_path / "prompts" / "classify"
    folder.mkdir(parents=True)
    (folder / "v1.md").write_text("Classify the movement.\n")
    (folder / "v2.md").write_text("Classify the movement, carefully.\n")
    return PromptStore(tmp_path)


@pytest.fixture
def audit() -> InMemoryAuditTrail:
    return InMemoryAuditTrail()


def gateway(
    provider: object, prompts: PromptStore, audit: InMemoryAuditTrail, tracer: Tracer | None = None
) -> LLMGateway:
    return LLMGateway(provider, prompts, audit, tracer)  # type: ignore[arg-type]


async def classify(gw: LLMGateway, version: str = "v1") -> Verdict:
    result = await gw.call(
        "classify",
        version,
        inputs={"curve": "JPY", "tenor": "3M", "change_bps": 3.0},
        output_model=Verdict,
        actor="market_data_assessment",
        session_id="s1",
    )
    return result.output


# --- the call path ------------------------------------------------------------------------


async def test_structured_output_and_what_the_provider_received(
    prompts: PromptStore, audit: InMemoryAuditTrail
) -> None:
    fake = FakeProvider({"classify": {"label": "likely_valid", "confidence": 0.9}})
    result = await gateway(fake, prompts, audit).call(
        "classify",
        "v2",
        inputs={"curve": "JPY", "change_bps": 3.0},
        output_model=Verdict,
        actor="market_data_assessment",
        session_id="s1",
    )
    assert result.output == Verdict(label="likely_valid", confidence=0.9)
    assert (result.prompt_id, result.prompt_version, result.provider) == ("classify", "v2", "fake")
    [request] = fake.requests
    assert request.system_prompt == "Classify the movement, carefully.\n"  # the v2 file
    assert json.loads(request.user_message.removeprefix("Inputs (JSON):\n")) == {
        "curve": "JPY",
        "change_bps": 3.0,
    }
    assert request.output_model is Verdict


async def test_one_audit_event_per_call_with_versions_and_usage(
    prompts: PromptStore, audit: InMemoryAuditTrail
) -> None:
    fake = FakeProvider({"classify": {"label": "monitor", "confidence": 0.5}})
    await classify(gateway(fake, prompts, audit))
    [event] = audit.read()
    assert (event.event_type, event.actor, event.session_id) == (
        "llm_call",
        "market_data_assessment",
        "s1",
    )
    p = event.payload
    assert (
        p["prompt_id"],
        p["prompt_version"],
        p["provider"],
        p["deployment"],
        p["model"],
        p["outcome"],
    ) == (
        "classify",
        "v1",
        "fake",
        "fake",
        "fake-model",
        "ok",
    )
    assert isinstance(p["prompt_tokens"], int) and p["prompt_tokens"] > 0
    assert isinstance(p["latency_ms"], float)


async def test_scripted_sequences_and_computed_replies(
    prompts: PromptStore, audit: InMemoryAuditTrail
) -> None:
    fake = FakeProvider(
        {
            "classify": [
                {"label": "monitor", "confidence": 0.4},
                {"label": "likely_data_issue", "confidence": 0.8},
            ]
        }
    )
    gw = gateway(fake, prompts, audit)
    assert [(await classify(gw)).label for _ in range(3)] == [
        "monitor",
        "likely_data_issue",
        "likely_data_issue",
    ]

    def by_size(request: LLMRequest) -> dict[str, object]:
        big = json.loads(request.user_message.split("\n", 1)[1])["change_bps"] > 10
        return {"label": "likely_data_issue" if big else "likely_valid", "confidence": 0.7}

    fake.set_reply("classify", by_size)
    assert (await classify(gw)).label == "likely_valid"


# --- allowed values and validation -----------------------------------------------------------


async def test_a_value_outside_the_allowed_set_is_rejected(
    prompts: PromptStore, audit: InMemoryAuditTrail
) -> None:
    fake = FakeProvider({"classify": {"label": "definitely_fine", "confidence": 0.9}})
    with pytest.raises(LLMOutputError, match="did not match Verdict"):
        await classify(gateway(fake, prompts, audit))
    assert audit.read()[-1].payload["outcome"] == "output_error"


class TrustingProvider:
    """Mimics a provider that hands back a parsed object without validating it."""

    name = "stub"
    deployment = "stub"

    def __init__(self, parsed: BaseModel | None, refusal: str | None = None) -> None:
        self._parsed, self._refusal = parsed, refusal

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(self._parsed, self._refusal, "", "stub-model", 10, 5)

    async def aclose(self) -> None:
        pass


async def test_gateway_revalidates_whatever_the_provider_returns(
    prompts: PromptStore, audit: InMemoryAuditTrail
) -> None:
    unvalidated = Verdict.model_construct(label="nonsense", confidence=7.0)  # skips validation
    with pytest.raises(LLMOutputError, match="invalid Verdict: confidence, label"):
        await classify(gateway(TrustingProvider(unvalidated), prompts, audit))


async def test_refusal_is_an_output_error(prompts: PromptStore, audit: InMemoryAuditTrail) -> None:
    with pytest.raises(LLMOutputError, match="refused: not appropriate"):
        await classify(gateway(TrustingProvider(None, refusal="not appropriate"), prompts, audit))


async def test_missing_structured_output(prompts: PromptStore, audit: InMemoryAuditTrail) -> None:
    with pytest.raises(LLMOutputError, match="no structured output"):
        await classify(gateway(TrustingProvider(None), prompts, audit))


async def test_malformed_json_is_an_output_error(
    prompts: PromptStore, audit: InMemoryAuditTrail
) -> None:
    fake = FakeProvider({"classify": '{"label": "monitor", "confidence": '})
    with pytest.raises(LLMOutputError):
        await classify(gateway(fake, prompts, audit))


# --- prompts and availability ----------------------------------------------------------------


async def test_unknown_prompt_version(prompts: PromptStore, audit: InMemoryAuditTrail) -> None:
    fake = FakeProvider({"classify": {"label": "monitor", "confidence": 0.5}})
    with pytest.raises(LLMOutputError, match="prompt not found"):
        await classify(gateway(fake, prompts, audit), version="v9")
    assert audit.read()[-1].payload["outcome"] == "prompt_error"
    assert fake.requests == []  # never reached the model


async def test_unavailable_provider(prompts: PromptStore, audit: InMemoryAuditTrail) -> None:
    with pytest.raises(LLMUnavailableError, match="no reply for 'classify'"):
        await classify(gateway(FakeProvider(), prompts, audit))
    assert audit.read()[-1].payload["outcome"] == "unavailable"


async def test_prompts_are_loaded_once(prompts: PromptStore) -> None:
    assert prompts.get("classify", "v1") is prompts.get("classify", "v1")


# --- provider selection and tracing -------------------------------------------------------------


def test_fake_provider_selected_by_setting() -> None:
    base = get_settings()
    settings = base.model_copy(update={"app": AppSettings(llm_provider=LLMProviderKind.FAKE)})
    assert isinstance(create_provider(settings), FakeProvider)


async def test_each_call_is_an_llm_span(
    prompts: PromptStore,
    audit: InMemoryAuditTrail,
    captured_traces: CapturedTraces,
) -> None:
    tracer, traces = captured_traces
    fake = FakeProvider({"classify": {"label": "monitor", "confidence": 0.5}})
    with tracer.session_trace("s1"):
        await classify(gateway(fake, prompts, audit, tracer))
    [span] = [s for t in traces for s in t.data.spans if s.name == "llm:classify"]
    assert span.span_type == "LLM"
    assert span.attributes["prompt_version"] == "v1"
    assert span.attributes["mlflow.llm.model"] == "fake-model"
    assert span.attributes["mlflow.chat.tokenUsage"]["total_tokens"] > 0
    assert span.outputs == {"label": "monitor", "confidence": 0.5}

"""Governed LLM gateway: providers, structured output, guardrails, retry and template fallback."""

from mdx_assessment.core.gateway.factory import create_llm_gateway, create_provider
from mdx_assessment.core.gateway.fake import FakeProvider
from mdx_assessment.core.gateway.gateway import GatewayResult, LLMGateway, PromptStore
from mdx_assessment.core.gateway.grounding import (
    EvidenceBundle,
    EvidenceItem,
    Grounded,
    GroundingReport,
    check_grounding,
)
from mdx_assessment.core.gateway.numbers import NumberTolerance, Quantity, Unit
from mdx_assessment.core.gateway.provider import (
    LLMError,
    LLMGroundingError,
    LLMOutputError,
    LLMPromptError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
)

__all__ = [
    "EvidenceBundle",
    "EvidenceItem",
    "FakeProvider",
    "GatewayResult",
    "Grounded",
    "GroundingReport",
    "LLMError",
    "LLMGateway",
    "LLMGroundingError",
    "LLMOutputError",
    "LLMPromptError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMUnavailableError",
    "NumberTolerance",
    "PromptStore",
    "Quantity",
    "Unit",
    "check_grounding",
    "create_llm_gateway",
    "create_provider",
]

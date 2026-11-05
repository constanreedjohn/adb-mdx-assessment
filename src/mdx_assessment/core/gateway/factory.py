"""Build the LLM gateway for the configured provider (`MDX_LLM_PROVIDER`)."""

from __future__ import annotations

from mdx_assessment.core.gateway.azure_provider import (
    AzureOpenAIProvider,
    create_async_azure_openai_client,
)
from mdx_assessment.core.gateway.fake import FakeProvider
from mdx_assessment.core.gateway.gateway import LLMGateway, PromptStore
from mdx_assessment.core.gateway.provider import LLMProvider
from mdx_assessment.core.observability.tracing import Tracer
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.settings import LLMProviderKind, Settings, get_settings


def create_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.app.llm_provider is LLMProviderKind.FAKE:
        return FakeProvider()  # empty script: tests and dev runs register replies
    return AzureOpenAIProvider(
        create_async_azure_openai_client(s.azure_openai), s.azure_openai.deployment
    )


def create_llm_gateway(
    audit: AuditTrail,
    tracer: Tracer | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
) -> LLMGateway:
    s = settings or get_settings()
    return LLMGateway(provider or create_provider(s), PromptStore(s.app.config_dir), audit, tracer)

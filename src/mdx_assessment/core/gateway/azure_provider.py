"""Azure OpenAI provider: structured output via `chat.completions.parse` (async client).

Structured output uses the model's strict JSON-schema mode, so output models should
keep to what it supports: every field required (use `X | None` rather than defaults),
`extra="forbid"`, and allowed values as `Literal`/`Enum`. The gateway re-validates
the parsed result through the model either way.
"""

from __future__ import annotations

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncAzureOpenAI
from pydantic import ValidationError

from mdx_assessment.core.gateway.azure_openai import (
    COGNITIVE_SERVICES_SCOPE,
    LLMNotConfiguredError,
    missing_settings,
)
from mdx_assessment.core.gateway.provider import (
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
)
from mdx_assessment.secret_store import SecretName, SecretsProvider, get_secrets_provider
from mdx_assessment.settings import AzureOpenAIAuthMode, AzureOpenAISettings, get_settings


def create_async_azure_openai_client(
    azure: AzureOpenAISettings | None = None, secrets: SecretsProvider | None = None
) -> AsyncAzureOpenAI:
    """Async twin of `create_azure_openai_client` (same settings, same auth modes)."""
    settings = azure or get_settings().azure_openai
    provider = secrets or get_secrets_provider()
    missing = missing_settings(settings, provider)
    if missing:
        raise LLMNotConfiguredError(missing)
    if settings.auth_mode is AzureOpenAIAuthMode.ENTRA:
        from azure.identity import (
            DefaultAzureCredential,
            get_bearer_token_provider,
        )

        return AsyncAzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_version=settings.api_version,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
            ),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
    return AsyncAzureOpenAI(
        azure_endpoint=settings.endpoint,
        api_version=settings.api_version,
        api_key=provider.require(SecretName.AZURE_OPENAI_API_KEY).get_secret_value(),
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


class AzureOpenAIProvider:
    name = "azure_openai"

    def __init__(self, client: AsyncAzureOpenAI, deployment: str) -> None:
        self._client = client
        self._deployment = deployment

    @property
    def deployment(self) -> str:
        return self._deployment

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            completion = await self._client.chat.completions.parse(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_message},
                ],
                response_format=request.output_model,
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMUnavailableError(f"Azure OpenAI unreachable: {type(exc).__name__}") from exc
        except APIStatusError as exc:
            raise LLMUnavailableError(f"Azure OpenAI returned HTTP {exc.status_code}") from exc
        except ValidationError as exc:  # the reply didn't fit the output model
            raise LLMOutputError(f"output did not match {request.output_model.__name__}") from exc

        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            parsed=choice.message.parsed,
            refusal=choice.message.refusal,
            raw_text=choice.message.content or "",
            model=completion.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

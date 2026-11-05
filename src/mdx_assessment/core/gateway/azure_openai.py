"""Azure OpenAI client construction (PDF §2.1 model access; plan D-06).

The only place that creates an Azure OpenAI client. SP2's LLM gateway (E5-02)
wraps it with prompts, structured output, guardrails and logging; nothing else
in the application calls the model directly (PDF §3.2).

Authentication (`AZURE_OPENAI_AUTH_MODE`):
- `api_key`: key from the secrets provider (`azure-openai-api-key`).
- `entra`: Microsoft Entra ID token via `DefaultAzureCredential` (Managed Identity
  in ADB's tenant, `az login` locally). No key is stored at all.
"""

from __future__ import annotations

from openai import AzureOpenAI

from mdx_assessment.secret_store import SecretName, SecretsProvider, get_secrets_provider
from mdx_assessment.settings import AzureOpenAIAuthMode, AzureOpenAISettings, get_settings

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


class LLMNotConfiguredError(RuntimeError):
    """Azure OpenAI settings are incomplete. Lists what's missing, never any value."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("Azure OpenAI is not configured; set in .env: " + ", ".join(missing))


def missing_settings(
    azure: AzureOpenAISettings, secrets: SecretsProvider | None = None
) -> list[str]:
    missing = []
    if not azure.endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not azure.deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT")
    if not azure.api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if azure.auth_mode is AzureOpenAIAuthMode.API_KEY:
        provider = secrets or get_secrets_provider()
        if provider.get(SecretName.AZURE_OPENAI_API_KEY) is None:
            missing.append("AZURE_OPENAI_API_KEY (or AZURE_OPENAI_AUTH_MODE=entra)")
    return missing


def create_azure_openai_client(
    azure: AzureOpenAISettings | None = None, secrets: SecretsProvider | None = None
) -> AzureOpenAI:
    """Client for the configured endpoint, API version and auth mode.

    Calls take `model=<deployment name>` (Azure routes by deployment).
    Raises LLMNotConfiguredError if anything required is missing.
    """
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

        return AzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_version=settings.api_version,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
            ),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    return AzureOpenAI(
        azure_endpoint=settings.endpoint,
        api_version=settings.api_version,
        api_key=provider.require(SecretName.AZURE_OPENAI_API_KEY).get_secret_value(),
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )

"""Azure OpenAI client factory and smoke-test logic, offline (no network, no keys)."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from openai import AzureOpenAI
from pydantic import SecretStr

from mdx_assessment.core.gateway import azure_openai as factory
from mdx_assessment.core.gateway.azure_openai import (
    LLMNotConfiguredError,
    create_azure_openai_client,
)
from mdx_assessment.core.gateway.smoke import SmokeError, SmokeReply, run_smoke
from mdx_assessment.secret_store import MissingSecretError, SecretName
from mdx_assessment.settings import AzureOpenAISettings


class FakeSecrets:
    backend = "fake"

    def __init__(self, api_key: str | None) -> None:
        self._key = api_key

    def get(self, name: SecretName) -> SecretStr | None:
        return SecretStr(self._key) if self._key else None

    def require(self, name: SecretName) -> SecretStr:
        value = self.get(name)
        if value is None:
            raise MissingSecretError(name, self.backend)
        return value


def azure_settings(**overrides: Any) -> AzureOpenAISettings:
    values = {
        "endpoint": "https://example.openai.azure.com/",
        "deployment": "gpt-test",
        "api_version": "2025-01-01-preview",
        "auth_mode": "api_key",
        **overrides,
    }
    return AzureOpenAISettings.model_validate(values)


# --- client factory --------------------------------------------------------------------


def test_api_key_client() -> None:
    client = create_azure_openai_client(azure_settings(), FakeSecrets("key-123"))
    assert isinstance(client, AzureOpenAI)
    assert client.api_key == "key-123"
    assert str(client.base_url).startswith("https://example.openai.azure.com/openai")


def test_missing_settings_are_listed_without_values() -> None:
    with pytest.raises(LLMNotConfiguredError) as exc:
        create_azure_openai_client(azure_settings(endpoint="", deployment=""), FakeSecrets(None))
    assert exc.value.missing == [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_KEY (or AZURE_OPENAI_AUTH_MODE=entra)",
    ]


def test_entra_mode_needs_no_key_and_uses_a_token_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fake_token_provider(credential: object, scope: str) -> Callable[[], str]:
        requested.append(scope)
        return lambda: "entra-token"

    import azure.identity

    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda: object())
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider", fake_token_provider)

    client = create_azure_openai_client(azure_settings(auth_mode="entra"), FakeSecrets(None))
    assert isinstance(client, AzureOpenAI)
    assert requested == [factory.COGNITIVE_SERVICES_SCOPE]


# --- smoke-test logic with a fake client ----------------------------------------------


def fake_client(reply: Callable[[str], SmokeReply | None], refusal: str | None = None) -> Any:
    """Mimics `client.chat.completions.parse` and records what it was called with."""
    calls: list[dict[str, Any]] = []

    def parse(**kwargs: Any) -> Any:
        calls.append(kwargs)
        token = kwargs["messages"][-1]["content"].removeprefix("Token: ")
        message = SimpleNamespace(parsed=reply(token), refusal=refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=20),
            model="gpt-test-2025",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    client.calls = calls
    return client


def test_smoke_parses_the_reply_and_checks_the_echo() -> None:
    client = fake_client(lambda token: SmokeReply(status="ok", model_family="gpt", echo=token))
    result = run_smoke(client, "gpt-test", "system prompt")
    assert result.echo_matches
    assert (result.prompt_tokens, result.completion_tokens, result.model) == (
        120,
        20,
        "gpt-test-2025",
    )
    call = client.calls[0]
    assert call["model"] == "gpt-test"  # Azure routes by deployment name
    assert call["response_format"] is SmokeReply
    assert call["messages"][0] == {"role": "system", "content": "system prompt"}


def test_smoke_detects_a_wrong_echo() -> None:
    client = fake_client(lambda token: SmokeReply(status="ok", model_family="gpt", echo="other"))
    assert not run_smoke(client, "gpt-test", "p").echo_matches


def test_smoke_reports_a_refusal() -> None:
    client = fake_client(lambda token: None, refusal="cannot comply")
    with pytest.raises(SmokeError, match="refused: cannot comply"):
        run_smoke(client, "gpt-test", "p")


def test_smoke_reports_a_missing_structured_reply() -> None:
    client = fake_client(lambda token: None)
    with pytest.raises(SmokeError, match="no structured reply"):
        run_smoke(client, "gpt-test", "p")


def test_reply_schema_is_strict() -> None:
    schema = SmokeReply.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "model_family", "echo"}

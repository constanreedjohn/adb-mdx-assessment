import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from mdx_assessment import secret_store
from mdx_assessment import settings as settings_mod
from mdx_assessment.secret_store import (
    EnvSecretsProvider,
    KeyVaultSecretsProvider,
    MissingSecretError,
    SecretName,
    SecretsProvider,
    create_secrets_provider,
    get_secrets_provider,
)

PREFIXES = ("AZURE_", "POSTGRES_", "LANGFUSE_", "MLFLOW_", "APPLICATIONINSIGHTS_", "MDX_")


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / ".env"
    path.write_text(
        "AZURE_OPENAI_API_KEY=azure-secret\n"
        "POSTGRES_PASSWORD=pg-secret\n"
        "LANGFUSE_SECRET_KEY=\n"  # present but empty: treated as not set
    )
    for key in list(os.environ):
        if key.startswith(PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MDX_ENV_FILE", str(path))
    settings_mod.get_settings.cache_clear()
    get_secrets_provider.cache_clear()
    yield path
    settings_mod.get_settings.cache_clear()
    get_secrets_provider.cache_clear()


def test_env_backend_is_the_default(env_file: Path) -> None:
    provider = get_secrets_provider()
    assert isinstance(provider, EnvSecretsProvider)
    assert isinstance(provider, SecretsProvider)  # satisfies the protocol
    assert provider.backend == "env"


def test_reads_secrets_by_logical_name(env_file: Path) -> None:
    provider = get_secrets_provider()
    assert provider.require(SecretName.AZURE_OPENAI_API_KEY).get_secret_value() == "azure-secret"
    assert provider.require(SecretName.POSTGRES_PASSWORD).get_secret_value() == "pg-secret"


def test_unset_and_empty_secrets_are_none(env_file: Path) -> None:
    provider = get_secrets_provider()
    assert provider.get(SecretName.LANGFUSE_SECRET_KEY) is None  # empty in .env
    assert provider.get(SecretName.APPINSIGHTS_CONNECTION_STRING) is None  # absent


def test_require_names_the_secret_but_never_a_value(env_file: Path) -> None:
    with pytest.raises(MissingSecretError, match="'langfuse-secret-key' is not set"):
        get_secrets_provider().require(SecretName.LANGFUSE_SECRET_KEY)


def test_environment_variable_overrides_env_file(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "from-environment")
    settings_mod.get_settings.cache_clear()
    provider = create_secrets_provider(settings_mod.get_settings())
    assert (
        provider.require(SecretName.AZURE_OPENAI_API_KEY).get_secret_value() == "from-environment"
    )


def test_secret_values_never_appear_in_repr(env_file: Path) -> None:
    secret = get_secrets_provider().require(SecretName.AZURE_OPENAI_API_KEY)
    assert "azure-secret" not in repr(secret)
    assert "azure-secret" not in str(secret)


def test_every_secret_has_an_env_mapping(env_file: Path) -> None:
    provider = EnvSecretsProvider(settings_mod.get_settings())
    for name in SecretName:
        provider.get(name)  # no KeyError for any declared secret


@pytest.mark.parametrize("name", list(SecretName))
def test_secret_names_are_valid_key_vault_names(name: SecretName) -> None:
    # Key Vault: 1-127 characters, letters, digits and dashes only.
    assert 1 <= len(name.value) <= 127
    assert all(c.isalnum() or c == "-" for c in name.value)


def test_key_vault_backend_is_a_clear_stub(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDX_SECRETS_BACKEND", "key_vault")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://example.vault.azure.net/")
    settings_mod.get_settings.cache_clear()
    with pytest.raises(NotImplementedError, match="E13-04"):
        get_secrets_provider()


def test_key_vault_backend_needs_a_vault_url() -> None:
    with pytest.raises(ValueError, match="AZURE_KEY_VAULT_URL"):
        KeyVaultSecretsProvider("")


def test_unknown_backend_is_rejected(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDX_SECRETS_BACKEND", "vaultish")
    settings_mod.get_settings.cache_clear()
    with pytest.raises(ValueError, match="secrets_backend"):
        settings_mod.get_settings()


def test_module_exports_protocol() -> None:
    assert secret_store.SecretsProvider is SecretsProvider

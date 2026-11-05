"""Secrets behind one interface, so moving to Azure Key Vault changes no calling code.

Code that needs a credential asks for it by logical name instead of reading a
settings field:

    from mdx_assessment.secret_store import SecretName, get_secrets_provider
    key = get_secrets_provider().require(SecretName.AZURE_OPENAI_API_KEY)
    client = AzureOpenAI(api_key=key.get_secret_value(), ...)

Backends (`MDX_SECRETS_BACKEND`):
- `env` (default): `.env` / environment variables, through `settings.py`.
- `key_vault`: Azure Key Vault in ADB's tenant. **Stub** until deployment hardening
  (E13-04). Secret names are already valid Key Vault names (letters, digits,
  dashes), so each logical name maps 1:1 to a vault secret.

Values are always `SecretStr`: they don't appear in `repr`, logs or tracebacks.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import lru_cache
from typing import Protocol, runtime_checkable

from pydantic import SecretStr

from mdx_assessment.settings import SecretsBackend, Settings, get_settings


class SecretName(StrEnum):
    """Every secret the application uses. Values double as Key Vault secret names.

    These are names, not values (hence the S105 suppressions below).
    """

    AZURE_OPENAI_API_KEY = "azure-openai-api-key"
    POSTGRES_PASSWORD = "postgres-password"  # noqa: S105
    LANGFUSE_SECRET_KEY = "langfuse-secret-key"  # noqa: S105
    APPINSIGHTS_CONNECTION_STRING = "appinsights-connection-string"


class MissingSecretError(LookupError):
    def __init__(self, name: SecretName, backend: str) -> None:
        super().__init__(f"secret {name.value!r} is not set ({backend} backend)")
        self.name = name


@runtime_checkable
class SecretsProvider(Protocol):
    @property
    def backend(self) -> str: ...

    def get(self, name: SecretName) -> SecretStr | None:
        """The secret, or None if it isn't set."""
        ...

    def require(self, name: SecretName) -> SecretStr:
        """The secret, or MissingSecretError naming it (never its value)."""
        ...


class EnvSecretsProvider:
    """Reads secrets from the loaded settings (environment variables, then `.env`)."""

    backend = SecretsBackend.ENV.value

    def __init__(self, settings: Settings) -> None:
        self._lookup: dict[SecretName, Callable[[], SecretStr | None]] = {
            SecretName.AZURE_OPENAI_API_KEY: lambda: settings.azure_openai.api_key,
            SecretName.POSTGRES_PASSWORD: lambda: settings.postgres.password,
            SecretName.LANGFUSE_SECRET_KEY: lambda: settings.langfuse.secret_key,
            SecretName.APPINSIGHTS_CONNECTION_STRING: lambda: (
                settings.app_insights.connection_string
            ),
        }
        missing = set(SecretName) - set(self._lookup)
        if missing:  # every declared secret must have a source
            raise RuntimeError(f"no env mapping for secrets: {sorted(missing)}")

    def get(self, name: SecretName) -> SecretStr | None:
        value = self._lookup[name]()
        return value if value is not None and value.get_secret_value() else None

    def require(self, name: SecretName) -> SecretStr:
        value = self.get(name)
        if value is None:
            raise MissingSecretError(name, self.backend)
        return value


class KeyVaultSecretsProvider:
    """Azure Key Vault backend. STUB: implemented with deployment hardening (E13-04).

    Intended implementation: `azure.keyvault.secrets.SecretClient(vault_url,
    credential=DefaultAzureCredential())` (Managed Identity in ADB's tenant), with
    `get_secret(name.value)` and a short in-process cache. Construction fails now,
    so selecting this backend can't silently run without secrets.
    """

    backend = SecretsBackend.KEY_VAULT.value

    def __init__(self, vault_url: str) -> None:
        if not vault_url:
            raise ValueError("the key_vault backend needs AZURE_KEY_VAULT_URL")
        self.vault_url = vault_url
        raise NotImplementedError(
            "Key Vault secrets backend is not implemented yet (E13-04). "
            "Use MDX_SECRETS_BACKEND=env for local development."
        )

    def get(self, name: SecretName) -> SecretStr | None:  # pragma: no cover - stub
        raise NotImplementedError

    def require(self, name: SecretName) -> SecretStr:  # pragma: no cover - stub
        raise NotImplementedError


def create_secrets_provider(settings: Settings) -> SecretsProvider:
    match settings.app.secrets_backend:
        case SecretsBackend.ENV:
            return EnvSecretsProvider(settings)
        case SecretsBackend.KEY_VAULT:
            return KeyVaultSecretsProvider(settings.azure.key_vault_url)


@lru_cache(maxsize=1)
def get_secrets_provider() -> SecretsProvider:
    """Process-wide provider for the configured backend. `cache_clear()` in tests."""
    return create_secrets_provider(get_settings())

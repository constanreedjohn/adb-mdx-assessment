"""Third-party service settings, read from the project `.env` file.

pydantic-settings parses `.env` directly: nothing calls `load_dotenv()` and
`os.environ` is never mutated. Precedence (highest first):

1. Real environment variables (e.g. set by CI, Docker, Azure Container Apps)
2. Values in the `.env` file at the project root (or `MDX_ENV_FILE` if set)
3. Defaults declared below

Usage:
    from mdx_assessment.settings import get_settings
    settings = get_settings()
    settings.azure_openai.endpoint
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_file() -> Path:
    """`.env` at the project root, independent of the current working directory."""
    override = os.environ.get("MDX_ENV_FILE")
    return Path(override) if override else PROJECT_ROOT / ".env"


def _config(prefix: str) -> SettingsConfigDict:
    # env_file is resolved at load time (see `_load`), not import time.
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file_encoding="utf-8",
        extra="ignore",  # each group only picks its own prefixed keys
        case_sensitive=False,
    )


class AzureOpenAIAuthMode(StrEnum):
    API_KEY = "api_key"
    ENTRA = "entra"


class AzureOpenAISettings(BaseSettings):
    model_config = _config("AZURE_OPENAI_")

    endpoint: str = ""
    auth_mode: AzureOpenAIAuthMode = AzureOpenAIAuthMode.API_KEY
    api_key: SecretStr | None = None
    api_version: str = "2024-10-21"
    deployment: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 2

    @property
    def is_configured(self) -> bool:
        if not (self.endpoint and self.deployment):
            return False
        if self.auth_mode is AzureOpenAIAuthMode.API_KEY:
            return bool(self.api_key and self.api_key.get_secret_value())
        return True


class PostgresSettings(BaseSettings):
    model_config = _config("POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    db: str = "mdx_assessment"
    user: str = "mdx"
    password: SecretStr = SecretStr("mdx")

    def url(self, driver: str = "postgresql+psycopg") -> str:
        """SQLAlchemy URL. Contains the password: never log it."""
        return (
            f"{driver}://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class LangfuseSettings(BaseSettings):
    model_config = _config("LANGFUSE_")

    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: SecretStr | None = None
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(
            self.enabled
            and self.public_key
            and self.secret_key
            and self.secret_key.get_secret_value()
        )


class MLflowSettings(BaseSettings):
    model_config = _config("MLFLOW_")

    tracking_uri: str = "http://localhost:5001"


class AzurePlatformSettings(BaseSettings):
    model_config = _config("AZURE_")

    tenant_id: str = ""
    client_id: str = ""
    key_vault_url: str = ""


class AppInsightsSettings(BaseSettings):
    # Uses the standard Azure Monitor variable name: APPLICATIONINSIGHTS_CONNECTION_STRING
    model_config = _config("APPLICATIONINSIGHTS_")

    connection_string: SecretStr | None = None


class Settings(BaseModel):
    """All third-party settings, grouped by service."""

    azure_openai: AzureOpenAISettings
    postgres: PostgresSettings
    langfuse: LangfuseSettings
    mlflow: MLflowSettings
    azure: AzurePlatformSettings
    app_insights: AppInsightsSettings


def _load[T: BaseSettings](cls: type[T]) -> T:
    return cls(_env_file=_env_file())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load once per process. Call `get_settings.cache_clear()` in tests to reload."""
    return Settings(
        azure_openai=_load(AzureOpenAISettings),
        postgres=_load(PostgresSettings),
        langfuse=_load(LangfuseSettings),
        mlflow=_load(MLflowSettings),
        azure=_load(AzurePlatformSettings),
        app_insights=_load(AppInsightsSettings),
    )

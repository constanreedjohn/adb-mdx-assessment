import os
from pathlib import Path

import pytest

from mdx_assessment import settings as settings_mod


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/\n"
        "AZURE_OPENAI_API_KEY=secret-key\n"
        "AZURE_OPENAI_DEPLOYMENT=gpt-test\n"
        "POSTGRES_HOST=db.local\n"
        "POSTGRES_PASSWORD=pw\n"
    )
    monkeypatch.setenv("MDX_ENV_FILE", str(path))
    for key in list(os.environ):
        if key.startswith(("AZURE_", "POSTGRES_", "MLFLOW_")):
            monkeypatch.delenv(key)
    settings_mod.get_settings.cache_clear()
    yield path
    settings_mod.get_settings.cache_clear()


def test_reads_values_from_env_file(env_file: Path) -> None:
    s = settings_mod.get_settings()
    assert s.azure_openai.endpoint == "https://example.openai.azure.com/"
    assert s.azure_openai.deployment == "gpt-test"
    assert s.azure_openai.is_configured
    assert s.postgres.url() == "postgresql+psycopg://mdx:pw@db.local:5432/mdx_assessment"


def test_env_file_does_not_mutate_os_environ(env_file: Path) -> None:
    settings_mod.get_settings()
    assert "AZURE_OPENAI_ENDPOINT" not in os.environ


def test_real_env_var_overrides_env_file(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "override-host")
    assert settings_mod.get_settings().postgres.host == "override-host"


def test_secrets_are_masked(env_file: Path) -> None:
    s = settings_mod.get_settings()
    assert "secret-key" not in repr(s)
    assert "pw" not in str(s.postgres.password)

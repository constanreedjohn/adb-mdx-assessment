from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mdx_assessment.app import build_app
from mdx_assessment.core.api import create_app
from mdx_assessment.core.config import ConfigError, config_version
from mdx_assessment.settings import PROJECT_ROOT


def test_health_returns_ok() -> None:
    response = TestClient(build_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reports_the_config_hash() -> None:
    body = TestClient(build_app()).get("/version").json()
    assert body["app"] == "mdx-assessment"
    assert body["version"] == "0.1.0"
    assert body["config_version"] == config_version(PROJECT_ROOT / "config")


def test_create_app_passes_config_version_through() -> None:
    body = TestClient(create_app(config_version="sha256:abc")).get("/version").json()
    assert body["config_version"] == "sha256:abc"


def test_app_refuses_to_start_on_invalid_config(tmp_path: Path) -> None:
    (tmp_path / "unexpected.yaml").write_text("a: 1\n")
    with pytest.raises(ConfigError):
        build_app(tmp_path)

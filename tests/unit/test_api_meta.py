from fastapi.testclient import TestClient

from mdx_assessment.app import build_app
from mdx_assessment.core.api import create_app


def test_health_returns_ok() -> None:
    response = TestClient(build_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reports_app_and_config_version() -> None:
    client = TestClient(create_app(config_version="sha256:abc"))
    body = client.get("/version").json()
    assert body["app"] == "mdx-assessment"
    assert body["version"] == "0.1.0"
    assert body["config_version"] == "sha256:abc"

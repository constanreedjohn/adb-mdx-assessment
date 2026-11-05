"""FastAPI application factory.

The app is built by the composition root (`mdx_assessment.app`), which passes in
whatever core must not compute itself (for example the config version, which
depends on the use-case packs that registered config schemas).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

DIST_NAME = "mdx-assessment"


def app_version() -> str:
    try:
        return version(DIST_NAME)
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+unknown"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    app: str
    version: str
    config_version: str | None
    """Content hash of `config/` (E1-06). None until config versioning is wired in."""


def create_app(*, config_version: str | None = None) -> FastAPI:
    app = FastAPI(
        title="MDX Daily Validation Agent",
        version=app_version(),
        summary="Supervised multi-agent exception assessment (PoC). Raw JSON endpoints.",
    )

    @app.get("/health", tags=["meta"])
    def health() -> HealthResponse:
        """Liveness. Dependency checks (database, Langfuse) are added with those stories."""
        return HealthResponse()

    @app.get("/version", tags=["meta"])
    def get_version() -> VersionResponse:
        return VersionResponse(app=DIST_NAME, version=app_version(), config_version=config_version)

    return app

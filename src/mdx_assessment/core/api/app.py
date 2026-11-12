"""FastAPI application factory.

The app is built by the composition root (`mdx_assessment.app`), which passes in
whatever core must not compute itself (for example the config version, which
depends on the use-case packs that registered config schemas).
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from mdx_assessment.core.api.cases import create_cases_router
from mdx_assessment.core.api.evaluation import create_eval_router
from mdx_assessment.core.api.sessions import create_sessions_router
from mdx_assessment.core.meta import DIST_NAME, app_version
from mdx_assessment.core.observability.log_file import Lifespan
from mdx_assessment.core.orchestration.session_service import SessionService


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    app: str
    version: str
    config_version: str | None
    """Content hash of `config/` (E1-06), as recorded by every session."""


def create_app(
    *,
    config_version: str | None = None,
    sessions: SessionService | None = None,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    app = FastAPI(
        title="MDX Daily Validation Agent",
        version=app_version(),
        summary="Supervised multi-agent exception assessment (PoC). Raw JSON endpoints.",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["meta"])
    def health() -> HealthResponse:
        """Liveness. Dependency checks (database, MLflow) are added with those stories."""
        return HealthResponse()

    @app.get("/version", tags=["meta"])
    def get_version() -> VersionResponse:
        return VersionResponse(app=DIST_NAME, version=app_version(), config_version=config_version)

    app.include_router(create_eval_router())
    if sessions is not None:
        app.include_router(create_sessions_router(sessions))
        app.include_router(create_cases_router(sessions))
    return app

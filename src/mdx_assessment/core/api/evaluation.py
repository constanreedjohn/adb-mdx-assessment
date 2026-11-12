"""Evaluation endpoints (SP4). Declared now so the API's shape is known: 501 until then."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status


def create_eval_router() -> APIRouter:
    router = APIRouter(prefix="/eval", tags=["evaluation"])

    @router.get("/master", status_code=status.HTTP_501_NOT_IMPLEMENTED)
    async def master_metrics() -> None:
        """The Master Agent's metrics over the evaluation set (E10-03, SP4)."""
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "the Master Agent's evaluation arrives in SP4 (E10-03)",
        )

    return router

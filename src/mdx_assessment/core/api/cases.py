"""Case endpoints: a session's exception cases as raw JSON (E11-01, E12-02).

Cases are listed in the order they were recorded. The Escalation and Reporting Agent
records them in the order the analyst should read them, so that is the list order.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from mdx_assessment.core.orchestration.session_service import SessionService
from mdx_assessment.core.persistence.repositories import CaseRecord


class CaseList(BaseModel):
    session_id: str
    count: int
    cases: list[CaseRecord]


def create_cases_router(service: SessionService) -> APIRouter:
    router = APIRouter(prefix="/cases", tags=["cases"])

    @router.get("")
    async def list_cases(
        session_id: Annotated[str, Query(min_length=1, max_length=64)],
    ) -> CaseList:
        """A session's cases, in list order."""
        cases = await asyncio.to_thread(service.list_cases, session_id)
        if cases is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"session {session_id!r} not found")
        return CaseList(session_id=session_id, count=len(cases), cases=cases)

    return router

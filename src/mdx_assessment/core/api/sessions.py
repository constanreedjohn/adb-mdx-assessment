"""Session endpoints: start a run for a business date, then read it as JSON (plan §2.4)."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from mdx_assessment.core.domain import AuditEvent, ChainVerification, SessionResult, SessionStatus
from mdx_assessment.core.orchestration.session_service import SessionService

DataType = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=32)]


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_date: date
    data_type: DataType
    llm_enabled: bool = True
    """False runs the playbook's default plan with no AI model."""


class SessionAudit(BaseModel):
    session_id: str
    chain: ChainVerification
    """The session's events verified now: each hash recomputed, each link to the one before."""
    count: int
    events: list[AuditEvent]
    """In `seq` order."""


class StartSessionResponse(BaseModel):
    session_id: str
    status: SessionStatus
    analyst_notice: str | None
    trace_url: str | None


def create_sessions_router(service: SessionService) -> APIRouter:
    router = APIRouter(prefix="/sessions", tags=["sessions"])

    @router.post("", status_code=status.HTTP_201_CREATED)
    async def start_session(request: StartSessionRequest) -> StartSessionResponse:
        """Run a session to its end. A run that stopped is still a session: read its status."""
        result = await service.start(
            request.business_date, request.data_type, llm_enabled=request.llm_enabled
        )
        return StartSessionResponse(
            session_id=result.session_id,
            status=result.status,
            analyst_notice=result.analyst_notice,
            trace_url=result.trace.url if result.trace else None,
        )

    @router.get("/{session_id}")
    async def get_session(session_id: str) -> SessionResult:
        result = await asyncio.to_thread(service.get, session_id)
        if result is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"session {session_id!r} not found")
        return result

    @router.get("/{session_id}/audit")
    async def get_session_audit(session_id: str) -> SessionAudit:
        """The session's audit trail, with its hash chain verified."""
        found = await asyncio.to_thread(service.audit, session_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"session {session_id!r} not found")
        events, chain = found
        return SessionAudit(session_id=session_id, chain=chain, count=len(events), events=events)

    return router

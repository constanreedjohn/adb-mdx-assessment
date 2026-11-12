"""Runs one session end to end and keeps its result (plan §2.4).

Everything a run needs is opened for that run: a tool hub, a checkpointer, and one
MLflow trace around the whole graph. The session row is written when the run starts
and replaced with the final state when it ends.

A crash inside the run is logged in full and stored as a `failed` session, so a caller
always gets a session to look at.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import (
    SessionResult,
    SessionStatus,
    Task,
    TraceRef,
    Versions,
)
from mdx_assessment.core.gateway import LLMGateway, create_llm_gateway
from mdx_assessment.core.gateway.grounding import GUARDRAILS_PATH, GuardrailsConfig
from mdx_assessment.core.master.opinion import PROMPT_ID as OPINION_PROMPT_ID
from mdx_assessment.core.master.opinion import PROMPT_VERSION as OPINION_PROMPT_VERSION
from mdx_assessment.core.master.opinion import master_opinion_hook
from mdx_assessment.core.master.planning import PROMPT_ID, PROMPT_VERSION, master_plan_hook
from mdx_assessment.core.master.review import PROMPT_ID as REVIEW_PROMPT_ID
from mdx_assessment.core.master.review import PROMPT_VERSION as REVIEW_PROMPT_VERSION
from mdx_assessment.core.master.review import master_review_hook
from mdx_assessment.core.observability.tracing import TraceHandle, Tracer, build_tracer
from mdx_assessment.core.orchestration.checkpoint import thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.playbooks import PlaybookStore
from mdx_assessment.core.orchestration.run_context import context_hook, playbook_hook
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.persistence.case_store import CaseStore
from mdx_assessment.core.persistence.repositories import CaseRecord
from mdx_assessment.core.persistence.session_store import SessionStore
from mdx_assessment.core.platform import PlatformRegistries
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import Settings

logger = logging.getLogger(__name__)

TRACE_FLUSH_SECONDS = 10
UNEXPECTED_FAILURE_NOTICE = (
    "This run stopped because of an unexpected error. A developer needs to look at it, "
    "then the run can be started again."
)
CheckpointerFactory = Callable[[], AbstractAsyncContextManager[BaseCheckpointSaver[Any]]]
GatewayFactory = Callable[[AuditTrail, Tracer], LLMGateway]


def _default_gateway(audit: AuditTrail, tracer: Tracer) -> LLMGateway:
    return create_llm_gateway(audit, tracer)


class SessionService:
    def __init__(
        self,
        *,
        registries: PlatformRegistries,
        config: LoadedConfig,
        settings: Settings,
        store: SessionStore,
        audit: AuditTrail,
        cases: CaseStore,
        checkpointer: CheckpointerFactory,
        make_gateway: GatewayFactory = _default_gateway,
        make_tracer: Callable[[], Tracer] = build_tracer,
    ) -> None:
        self._registries = registries
        self._config = config
        self._settings = settings
        self._store = store
        self._audit = audit
        self._cases = cases
        self._checkpointer = checkpointer
        self._make_gateway = make_gateway
        self._make_tracer = make_tracer
        self._tracer: Tracer | None = None

    async def start(
        self, business_date: date, data_type: str, *, llm_enabled: bool = True
    ) -> SessionResult:
        """Run a session to its end (synchronously for now) and return the stored result."""
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(UTC)
        prompts = {
            PROMPT_ID: PROMPT_VERSION,
            REVIEW_PROMPT_ID: REVIEW_PROMPT_VERSION,
            OPINION_PROMPT_ID: OPINION_PROMPT_VERSION,
        }
        prompts.update(p for task in self._registries.tasks for p in task.prompts)
        versions = Versions(config=self._config.version, prompts=prompts)
        initial = RunState(
            session_id=session_id,
            business_date=business_date,
            data_type=data_type,
            versions=versions,
            llm_enabled=llm_enabled,
        )
        await asyncio.to_thread(
            self._store.add,
            SessionResult(
                session_id=session_id,
                business_date=business_date,
                data_type=data_type,
                llm_enabled=llm_enabled,
                started_at=started_at,
                versions=versions,
            ),
        )
        if self._tracer is None or not self._tracer.enabled:  # MLflow may be back by now
            self._tracer = self._make_tracer()
        tracer = self._tracer

        with tracer.session_trace(
            session_id,
            input={
                "business_date": business_date.isoformat(),
                "data_type": data_type,
                "llm_enabled": llm_enabled,
            },
            metadata={"config_version": self._config.version},
            tags={"data_type": data_type},
        ) as trace:
            try:
                final = await self._run(initial, tracer)
            except Exception:
                logger.exception("session %s crashed", session_id)
                final = initial.model_copy(
                    update={
                        "status": SessionStatus.FAILED,
                        "analyst_notice": UNEXPECTED_FAILURE_NOTICE,
                    }
                )
            result = self._result(final, started_at, trace)
            self._close_trace(trace, result)
        await asyncio.to_thread(self._store.save, result)
        await self._flush(tracer)
        logger.info("session %s finished: %s", session_id, result.status.value)
        return result

    @staticmethod
    async def _flush(tracer: Tracer) -> None:
        """Have the trace visible when the caller gets its answer, without waiting on a sick server."""
        try:
            await asyncio.wait_for(asyncio.to_thread(tracer.flush), TRACE_FLUSH_SECONDS)
        except Exception:
            logger.warning("could not flush traces", exc_info=True)

    def get(self, session_id: str) -> SessionResult | None:
        return self._store.get(session_id)

    def list_cases(self, session_id: str) -> list[CaseRecord] | None:
        """A session's cases, in the order they were recorded; None if no such session."""
        if self._store.get(session_id) is None:
            return None
        return self._cases.list_for_session(session_id)

    async def _run(self, initial: RunState, tracer: Tracer) -> RunState:
        registries = self._registries
        if registries.facts is None:
            raise ValueError("no use-case pack registered a facts builder")
        # With the LLM off no gateway exists, so a run needs no model credentials.
        gateway = self._make_gateway(self._audit, tracer) if initial.llm_enabled else None
        versions = initial.versions.model_copy(
            update={"llm_deployment": gateway.provider.deployment if gateway else None}
        )
        context = ToolContext(
            config=self._config, settings=self._settings, audit=self._audit, cases=self._cases
        )
        permissions = self._config.get(PERMISSIONS_PATH, PermissionMatrix)
        async with (
            self._closing(gateway),
            ToolHub(registries.tools, permissions, context, tracer) as hub,
            self._checkpointer() as checkpointer,
        ):
            runtime = SessionRuntime(
                tasks=registries.tasks,
                checks=registries.checks,
                audit=self._audit,
                hub=hub,
                gateway=gateway,
                tracer=tracer,
                config=self._config,
            )
            tolerance = self._config.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching
            hooks = GraphHooks(
                plan=master_plan_hook(runtime, tolerance),
                decide=master_review_hook(runtime, tolerance),
                gather_context=context_hook(registries.facts, hub, self._config),
                select_playbook=playbook_hook(PlaybookStore.from_config(self._config)),
                synthesize=master_opinion_hook(runtime, registries.opinion),
            )
            app = build_session_graph(runtime, hooks).compile(checkpointer=checkpointer)
            output = await app.ainvoke(
                initial.model_copy(update={"versions": versions}),
                thread_config(initial.session_id),
            )
        return RunState.model_validate(output)

    @staticmethod
    @asynccontextmanager
    async def _closing(gateway: LLMGateway | None) -> AsyncIterator[None]:
        """Close the session's model connections on the loop that opened them."""
        try:
            yield
        finally:
            if gateway is not None:
                await gateway.aclose()

    @staticmethod
    def _result(state: RunState, started_at: datetime, trace: TraceHandle) -> SessionResult:
        execution = tuple(
            Task(
                task_id=f"{state.session_id}/{entry.step_id}",
                session_id=state.session_id,
                step_id=entry.step_id,
                task=entry.task,
                status=entry.status,
                started_at=entry.started_at,
                finished_at=entry.finished_at,
                error=entry.error,
            )
            for entry in state.execution
        )
        return SessionResult(
            session_id=state.session_id,
            business_date=state.business_date,
            data_type=state.data_type,
            llm_enabled=state.llm_enabled,
            status=state.status,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            versions=state.versions,
            run_context=state.run_context,
            playbook=state.playbook.model_dump(mode="json") if state.playbook else None,
            plan=state.plan,
            execution=execution,
            agent_results=state.agent_results,
            review_checks=tuple(state.review_checks),
            decisions=tuple(state.decisions),
            validations=tuple(state.validations),
            run_opinion=state.run_opinion,
            review=state.review,
            fallbacks=tuple(state.fallbacks),
            analyst_notice=state.analyst_notice,
            trace=TraceRef(mlflow_trace_id=trace.trace_id, url=trace.url)
            if trace.trace_id
            else None,
        )

    @staticmethod
    def _close_trace(trace: TraceHandle, result: SessionResult) -> None:
        if trace.root is None:
            return
        trace.root.update(
            output={
                "status": result.status.value,
                "plan_source": result.plan.source.value if result.plan else None,
                "steps": [{"step": t.step_id, "status": t.status.value} for t in result.execution],
                "fallbacks": list(result.fallbacks),
                "analyst_notice": result.analyst_notice,
            }
        )
        if result.status is SessionStatus.FAILED:
            trace.root.update(error=result.analyst_notice)

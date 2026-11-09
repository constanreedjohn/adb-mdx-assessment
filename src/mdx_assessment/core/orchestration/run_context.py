"""The two code-only steps at session start: gather the facts, select the playbook.

A use-case pack registers a `FactsBuilder`. These helpers turn it, and the playbook
store, into the graph's `gather_context` and `select_playbook` hooks. Neither involves
an LLM, so both still work when the model is down (plan §2.5).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import RunContext
from mdx_assessment.core.orchestration.graph import MASTER_AGENT, ContextHook, PlaybookHook
from mdx_assessment.core.orchestration.playbooks import (
    PlaybookStore,
    ResolvedPlaybook,
    resolve_playbook,
)
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.tools.client import AgentToolClient, TaskBudget, ToolHub

CONTEXT_BUDGET = TaskBudget(max_calls=10, max_seconds=30)


@dataclass(frozen=True)
class FactsRequest:
    state: RunState
    tools: AgentToolClient
    config: LoadedConfig


FactsBuilder = Callable[[FactsRequest], Awaitable[RunContext]]


def context_hook(builder: FactsBuilder, hub: ToolHub, config: LoadedConfig) -> ContextHook:
    async def gather_context(state: RunState) -> RunContext:
        tools = hub.client(
            MASTER_AGENT,
            session_id=state.session_id,
            task_id=f"{state.session_id}/run_context",
            budget=CONTEXT_BUDGET,
        )
        return await builder(FactsRequest(state=state, tools=tools, config=config))

    return gather_context


def playbook_hook(store: PlaybookStore) -> PlaybookHook:
    async def select_playbook(state: RunState) -> ResolvedPlaybook:
        return resolve_playbook(state.run_context.facts, store)

    return select_playbook

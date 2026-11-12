"""E6-05 with the MDX pack: the teacher's send-back re-runs the Market Data Assessment
for the named flags only, in revise mode, with their remaining tool budget.

Acceptance: a scripted send-back re-runs the MDA for those flags once; a second
send-back is refused.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from tests.support import conclude, explain_from_inputs, investigation_flags

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.domain import (
    Decision,
    DecisionKind,
    DecisionSource,
    Plan,
    PlanStep,
    ReviewCheckResult,
)
from mdx_assessment.core.gateway import (
    FakeProvider,
    LLMGateway,
    LLMRequest,
    LLMUnavailableError,
    PromptStore,
)
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import GraphHooks, SessionRuntime, build_session_graph
from mdx_assessment.core.orchestration.playbooks import (
    PlaybookStore,
    ResolvedPlaybook,
    resolve_playbook,
)
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.ids import REVISE_DRAFTS
from mdx_assessment.usecases.mdx.agents.market_data import RevisionResult, current_drafts
from mdx_assessment.usecases.mdx.checks.verdict import Verdict
from mdx_assessment.usecases.mdx.config import INVESTIGATION_PATH, InvestigationSettings
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
CONFIG = load_config(PROJECT_ROOT / "config")
BOOK = resolve_playbook([], PlaybookStore.from_config(CONFIG))
BUDGET = CONFIG.get(INVESTIGATION_PATH, InvestigationSettings).max_tool_calls_per_flag
GBP_2Y, JPY_3M = "20261015-GBP-2Y", "20261015-JPY-3M"
FEEDBACK = {
    JPY_3M: "The report doesn't say whether this point is flagged often.",
    GBP_2Y: "Say whether other curves moved the same way.",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(CONFIG, DAY), path, config_version=CONFIG.version)
    return path


def tool(flag: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "flag_id": flag["flag_id"],
        "action": "investigate",
        "tool_calls": [{"tool": name, "days": None, "reason": f"check {name}"}],
        "verdict": None,
        "report": None,
        "evidence_ids": [],
    }


def student(request: LLMRequest) -> dict[str, Any]:
    """First pass: JPY 3M checks its neighbours, then all conclude. Revision: JPY 3M
    checks its flag frequency and ends can't tell; GBP 2Y concludes at once."""
    steps = []
    for f in investigation_flags(request):
        done = {r["tool"] for r in f["tool_results"]}
        if f["flag_id"] == JPY_3M and "neighbour_moves" not in done:
            steps.append(tool(f, "neighbour_moves"))
        elif f["flag_id"] == JPY_3M and "teacher_feedback" in f and "flag_frequency" not in done:
            steps.append(tool(f, "flag_frequency"))
        elif f["flag_id"] == JPY_3M and "teacher_feedback" in f:
            steps.append(conclude(f, "cant_tell"))
        else:
            steps.append(conclude(f))
    return {"steps": steps}


def teacher(after: set[str], flags: dict[str, str] | None = None) -> Any:
    """Send back the named flags after each step in `after`; accept everything else."""

    async def decide(
        state: RunState, entry: ExecutionEntry, checks: list[ReviewCheckResult]
    ) -> Decision:
        if entry.step_id not in after:
            return Decision(step_id=entry.step_id, kind=DecisionKind.ACCEPT, reason="fine")
        n = sum(d.kind is DecisionKind.FOLLOW_UP for d in state.decisions) + 1
        return Decision(
            step_id=entry.step_id,
            kind=DecisionKind.FOLLOW_UP,
            reason="two drafts need another look",
            source=DecisionSource.LLM_REVIEW,
            follow_up=PlanStep(
                step_id=f"revise_{n}",
                task=REVISE_DRAFTS,
                arguments={"of_step": "assess_movements", "flags": flags or FEEDBACK},  # type: ignore[dict-item]
            ),
        )

    return decide


@dataclass
class Outcome:
    final: RunState
    audit: InMemoryAuditTrail
    fake: FakeProvider

    def revision_rounds(self) -> list[list[dict[str, Any]]]:
        rounds = [
            investigation_flags(r) for r in self.fake.requests if r.prompt_id == "mda_investigate"
        ]
        return [flags for flags in rounds if any("teacher_feedback" in f for f in flags)]

    def revision(self, step_id: str = "revise_1") -> RevisionResult:
        return RevisionResult.model_validate(self.final.agent_results[step_id])


async def session(
    data_dir: Path, decide: Any, script: Callable[[LLMRequest], dict[str, Any]] = student
) -> Outcome:
    registries = build_registries()
    audit = InMemoryAuditTrail()
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    fake = FakeProvider({"mda_explain": explain_from_inputs, "mda_investigate": script})
    gateway = LLMGateway(fake, PromptStore(PROJECT_ROOT / "config"), audit)
    hub = ToolHub(
        registries.tools,
        CONFIG.get(PERMISSIONS_PATH, PermissionMatrix),
        ToolContext(config=CONFIG, settings=settings, audit=audit, cases=InMemoryCaseStore()),
    )

    async def default_plan(state: RunState) -> Plan:
        return Plan(source="default", steps=BOOK.content.default_plan)  # type: ignore[arg-type]

    async def select(state: RunState) -> ResolvedPlaybook:
        return BOOK

    async with hub:
        runtime = SessionRuntime(
            tasks=registries.tasks,
            checks=registries.checks,
            audit=audit,
            hub=hub,
            gateway=gateway,
            config=CONFIG,
        )
        hooks = GraphHooks(plan=default_plan, select_playbook=select, decide=decide)
        app = build_session_graph(runtime, hooks).compile(
            checkpointer=InMemorySaver(serde=checkpoint_serializer())
        )
        initial = RunState(
            session_id="sess-sb", business_date=DAY, data_type="linear", playbook=BOOK
        )
        final = RunState.model_validate(await app.ainvoke(initial, thread_config("sess-sb")))
    return Outcome(final, audit, fake)


# --- acceptance ---------------------------------------------------------------------------------


async def test_a_send_back_re_runs_the_assessment_for_the_named_flags_once(data_dir: Path) -> None:
    outcome = await session(data_dir, teacher({"assess_movements"}))
    final = outcome.final
    assert [(e.step_id, e.status.value) for e in final.execution] == [
        ("validate_inputs", "done"),
        ("assess_movements", "done"),
        ("revise_1", "done"),
        ("record_cases", "done"),
    ]
    # Only the named flags go back, each with the teacher's reason and its remaining budget.
    first = outcome.revision_rounds()[0]
    assert {f["flag_id"]: f["teacher_feedback"] for f in first} == FEEDBACK
    assert {f["flag_id"]: f["tool_calls_left"] for f in first} == {
        JPY_3M: BUDGET - 1,
        GBP_2Y: BUDGET,
    }

    revision = outcome.revision()
    assert [d.flag_id for d in revision.drafts] == [GBP_2Y, JPY_3M]  # review order
    jpy = next(d for d in revision.drafts if d.flag_id == JPY_3M)
    assert (jpy.revision, jpy.teacher_feedback, jpy.verdict) == (
        1,
        FEEDBACK[JPY_3M],
        Verdict.CANT_TELL,
    )
    assert [(c.tool, c.revision) for c in jpy.tool_calls] == [
        ("neighbour_moves", 0),
        ("flag_frequency", 1),
    ]
    assert jpy.tool_calls_left == BUDGET - 2  # the budget covers both rounds
    assert revision.tool_calls == 1 and not revision.used_fallback

    drafts = current_drafts(final.execution, final.agent_results)
    assert len(drafts) == 28
    assert {f for f, d in drafts.items() if d.revision == 1} == {GBP_2Y, JPY_3M}
    assert drafts[JPY_3M] == jpy


async def test_a_second_send_back_is_refused(data_dir: Path) -> None:
    outcome = await session(data_dir, teacher({"assess_movements", "revise_1"}))
    final = outcome.final
    assert [e.step_id for e in final.execution] == [
        "validate_inputs",
        "assess_movements",
        "revise_1",
        "record_cases",
    ]
    refused = next(d for d in final.decisions if d.step_id == "revise_1")
    assert refused.kind is DecisionKind.FLAG_FOR_HUMAN
    assert "P-3" in refused.reason and "1 already ran" in refused.reason
    assert [
        e.payload["step_id"] for e in outcome.audit.read() if e.event_type == "send_back_refused"
    ] == ["revise_1"]
    assert len(outcome.revision_rounds()) == 2  # revise_1's rounds only: nothing re-ran


# --- revise mode, failing safe ----------------------------------------------------------------------


async def test_a_revision_the_llm_cant_complete_ends_cant_tell_or_more_cautious(
    data_dir: Path,
) -> None:
    def down_on_revision(request: LLMRequest) -> dict[str, Any]:
        if any("teacher_feedback" in f for f in investigation_flags(request)):
            raise LLMUnavailableError("model down")
        return student(request)

    outcome = await session(data_dir, teacher({"assess_movements"}), down_on_revision)
    revision = outcome.revision()
    verdicts = {d.flag_id: (d.verdict, d.fallback_reason) for d in revision.drafts}
    # GBP 2Y keeps likely real; JPY 3M can't stand at harmless once challenged.
    assert verdicts == {
        GBP_2Y: (Verdict.LIKELY_REAL, "model_unavailable"),
        JPY_3M: (Verdict.CANT_TELL, "model_unavailable"),
    }
    assert revision.used_fallback
    assert outcome.final.execution[2].status.value == "done_with_fallback"
    assert all("could not be resolved" in d.report for d in revision.drafts)


async def test_a_send_back_naming_an_unknown_flag_fails_that_step_only(data_dir: Path) -> None:
    outcome = await session(data_dir, teacher({"assess_movements"}, {"20261015-XXX-1Y": "?"}))
    revise = outcome.final.execution[2]
    assert (revise.step_id, revise.status.value) == ("revise_1", "failed")
    assert "no draft for flag(s) ['20261015-XXX-1Y']" in (revise.error or "")
    assert outcome.final.execution[3].step_id == "record_cases"
    drafts = current_drafts(outcome.final.execution, outcome.final.agent_results)
    assert all(d.revision == 0 for d in drafts.values())  # the original drafts stand

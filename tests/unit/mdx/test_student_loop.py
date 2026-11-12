"""E8-04: the student loop and draft verdicts, with a scripted fake model.

Acceptance: happy day gives 28 drafts with numbers matching evidence; a scripted
borderline flag calls tools; a scripted lowered verdict is rejected; the budget stops
at 4.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from tests.support import (
    conclude,
    explain_from_inputs,
    investigate_from_inputs,
    investigation_flags,
)

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.domain import Evidence, PlanStep
from mdx_assessment.core.gateway import (
    FakeProvider,
    LLMGateway,
    LLMRequest,
    PromptStore,
    check_grounding,
)
from mdx_assessment.core.gateway.grounding import GUARDRAILS_PATH, GuardrailsConfig
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState, TaskStatus
from mdx_assessment.core.orchestration.tasks import TaskContext, TaskInput
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.explanations import (
    FALLBACK_LLM_OFF,
    FALLBACK_REJECTED,
    FALLBACK_UNAVAILABLE,
    evidence_bundle,
)
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    INPUT_VALIDATION,
    MARKET_DATA_ASSESSMENT,
    NEAR_ZERO_CHECK,
)
from mdx_assessment.usecases.mdx.agents.input_validation import check_files_and_intake
from mdx_assessment.usecases.mdx.agents.investigate import (
    FALLBACK_BUDGET_SPENT,
    FALLBACK_DRAFT_REJECTED,
    Draft,
    StudentStep,
    ToolRequest,
)
from mdx_assessment.usecases.mdx.agents.market_data import (
    MarketAssessmentResult,
    measure_units,
    near_zero_check,
)
from mdx_assessment.usecases.mdx.checks.near_zero import Classification
from mdx_assessment.usecases.mdx.checks.verdict import (
    Verdict,
    at_least,
    lowers,
    starting_verdict,
)
from mdx_assessment.usecases.mdx.config import (
    DATA_TYPES_PATH,
    INVESTIGATION_PATH,
    DataTypeProfiles,
    InvestigationSettings,
)
from mdx_assessment.usecases.mdx.domain import ExplanationSource, MarketDataKind
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
CONFIG = load_config(PROJECT_ROOT / "config")
TOLERANCE = CONFIG.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching
RATE_UNITS = measure_units(
    CONFIG.get(DATA_TYPES_PATH, DataTypeProfiles).for_kind(MarketDataKind.RATE)
)
BUDGET = CONFIG.get(INVESTIGATION_PATH, InvestigationSettings).max_tool_calls_per_flag
GBP_2Y, JPY_3M = "20261015-GBP-2Y", "20261015-JPY-3M"
Reply = Callable[[LLMRequest], dict[str, Any]]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(CONFIG, DAY), path, config_version=CONFIG.version)
    return path


@dataclass
class Run:
    result: MarketAssessmentResult
    fake: FakeProvider
    evidence: dict[str, Evidence]

    def draft(self, flag_id: str) -> Draft:
        return next(i.draft for i in self.result.items if i.assessment.flag_id == flag_id)

    @property
    def rounds(self) -> list[LLMRequest]:
        return [r for r in self.fake.requests if r.prompt_id == "mda_investigate"]

    def grounded(self, draft: Draft) -> list[str]:
        bundle = evidence_bundle(
            [self.evidence[i] for i in draft.evidence_ids], RATE_UNITS, TOLERANCE
        )
        step = StudentStep(
            flag_id=draft.flag_id,
            action="conclude",
            tool_calls=(),
            verdict=draft.verdict,
            report=draft.report,
            evidence_ids=draft.evidence_ids,
        )
        return list(check_grounding(step, bundle).errors)


async def run(
    data_dir: Path, student: Any = investigate_from_inputs, *, llm_enabled: bool = True
) -> Run:
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    audit = InMemoryAuditTrail()
    script: dict[str, Any] = {"mda_explain": explain_from_inputs}
    if student is not None:
        script["mda_investigate"] = student
    fake = FakeProvider(script)
    gateway = LLMGateway(fake, PromptStore(PROJECT_ROOT / "config"), audit)
    context = ToolContext(config=CONFIG, settings=settings, audit=audit, cases=InMemoryCaseStore())
    registries = build_registries()
    budgets = {t.id: t.budget for t in registries.tasks}
    playbook = resolve_playbook([], PlaybookStore.from_config(CONFIG))
    state = RunState(
        session_id="s1",
        business_date=DAY,
        data_type="linear",
        playbook=playbook,
        llm_enabled=llm_enabled,
    )
    async with ToolHub(
        registries.tools, CONFIG.get(PERMISSIONS_PATH, PermissionMatrix), context
    ) as hub:

        def ctx(state: RunState, step_id: str, task: str, agent: str) -> TaskContext:
            return TaskContext(
                inputs=TaskInput(
                    session_id="s1", business_date=DAY, data_type="linear", step_id=step_id
                ),
                state=state,
                step=PlanStep(step_id=step_id, task=task),
                tools_client=hub.client(
                    agent, session_id="s1", task_id=f"s1/{step_id}", budget=budgets[task]
                ),
                gateway=gateway,
                loaded_config=CONFIG,
            )

        intake = await check_files_and_intake(
            ctx(state, "v", CHECK_FILES_AND_INTAKE, INPUT_VALIDATION)
        )
        entry = ExecutionEntry(
            step_id="v",
            task=CHECK_FILES_AND_INTAKE,
            status=TaskStatus.DONE,
            started_at=f"{DAY}T09:00:00Z",  # type: ignore[arg-type]
        )
        state = state.model_copy(
            update={"execution": [entry], "agent_results": {"v": intake.model_dump(mode="json")}}
        )
        result = await near_zero_check(ctx(state, "a", NEAR_ZERO_CHECK, MARKET_DATA_ASSESSMENT))
    evidence = {e.evidence_id: e for e in [*intake.evidence, *result.evidence]}
    return Run(result, fake, evidence)


def student(on: str, act: Callable[[dict[str, Any]], dict[str, Any]]) -> Reply:
    """A fake student that concludes every flag at once, except `on`, which `act` plays."""

    def answer(request: LLMRequest) -> dict[str, Any]:
        return {
            "steps": [
                act(f) if f["flag_id"] == on else conclude(f) for f in investigation_flags(request)
            ]
        }

    return answer


def investigate(*tools: str, days: int | None = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def act(flag: dict[str, Any]) -> dict[str, Any]:
        return {
            "flag_id": flag["flag_id"],
            "action": "investigate",
            "tool_calls": [
                {"tool": t, "days": days if t == "point_history" else None, "reason": f"check {t}"}
                for t in tools
            ],
            "verdict": None,
            "report": None,
            "evidence_ids": [],
        }

    return act


def neighbours(flag: dict[str, Any]) -> dict[str, Any] | None:
    return next((e for e in flag["evidence"] if e["kind"] == "neighbour_moves"), None)


# --- acceptance ---------------------------------------------------------------------------------


async def test_happy_day_gives_28_grounded_drafts(data_dir: Path) -> None:
    outcome = await run(data_dir)
    result = outcome.result
    assert len(result.items) == 28
    assert result.draft_sources == {ExplanationSource.LLM: 28, ExplanationSource.TEMPLATE: 0}
    assert result.verdicts == {
        Verdict.LIKELY_REAL: 1,
        Verdict.CANT_TELL: 0,
        Verdict.LIKELY_HARMLESS: 27,
    }
    assert (result.investigation_rounds, result.tool_calls, result.used_fallback) == (1, 0, False)
    [request] = outcome.rounds  # one batched round for every flag
    assert len(investigation_flags(request)) == 28
    for item in result.items:
        assert item.draft.verdict is item.draft.starting_verdict
        assert outcome.grounded(item.draft) == [], item.assessment.item_key
    assert outcome.draft(GBP_2Y).verdict is Verdict.LIKELY_REAL
    assert outcome.draft(JPY_3M).tool_calls_left == BUDGET


async def test_a_borderline_flag_calls_tools_and_cites_them(data_dir: Path) -> None:
    def act(flag: dict[str, Any]) -> dict[str, Any]:
        found = neighbours(flag)
        if found is None:
            return investigate("neighbour_moves", "flag_frequency")(flag)
        moves = " and ".join(
            f"{n['tenor']} {n['change']:+.2f} bps" for n in found["data"]["neighbours"]
        )
        return {
            **conclude(flag),
            "report": f"{flag['explanation']['text']} Its neighbours moved with it: {moves}.",
            "evidence_ids": [*flag["explanation"]["evidence_ids"], found["id"]],
        }

    outcome = await run(data_dir, student(JPY_3M, act))
    draft = outcome.draft(JPY_3M)
    assert [c.tool for c in draft.tool_calls] == ["neighbour_moves", "flag_frequency"]
    assert all(c.evidence_id and c.error is None and c.round == 1 for c in draft.tool_calls)
    assert draft.tool_calls_left == BUDGET - 2
    assert (draft.source, draft.verdict) == (ExplanationSource.LLM, Verdict.LIKELY_HARMLESS)
    assert "1M +2.50 bps and 6M +2.50 bps" in draft.report
    assert outcome.grounded(draft) == []
    # Round 2 asks only about the flag still open, with what its tools returned.
    first, second = outcome.rounds
    assert len(investigation_flags(first)) == 28
    [open_flag] = investigation_flags(second)
    assert open_flag["flag_id"] == JPY_3M and open_flag["tool_calls_left"] == BUDGET - 2
    assert {r["tool"] for r in open_flag["tool_results"]} == {"neighbour_moves", "flag_frequency"}
    assert {c.evidence_id for c in draft.tool_calls} <= {
        e.evidence_id for e in outcome.result.evidence
    }
    assert (outcome.result.tool_calls, outcome.result.investigation_rounds) == (2, 2)


async def test_a_lowered_verdict_is_sent_back_then_replaced_by_a_template(data_dir: Path) -> None:
    outcome = await run(data_dir, student(GBP_2Y, lambda f: conclude(f, "likely_harmless")))
    draft = outcome.draft(GBP_2Y)
    assert (draft.verdict, draft.source) == (Verdict.LIKELY_REAL, ExplanationSource.TEMPLATE)
    assert draft.fallback_reason == FALLBACK_DRAFT_REJECTED
    assert len(draft.rejections) == 2 and "only move towards caution" in draft.rejections[0]
    [sent_back] = investigation_flags(outcome.rounds[1])
    assert "less cautious" in sent_back["draft_sent_back"]
    assert outcome.result.used_fallback  # the LLM's draft wasn't usable


async def test_a_sent_back_draft_that_is_fixed_is_kept(data_dir: Path) -> None:
    def act(flag: dict[str, Any]) -> dict[str, Any]:
        return conclude(flag, "likely_real" if "draft_sent_back" in flag else "likely_harmless")

    outcome = await run(data_dir, student(GBP_2Y, act))
    draft = outcome.draft(GBP_2Y)
    assert (draft.verdict, draft.source, draft.fallback_reason) == (
        Verdict.LIKELY_REAL,
        ExplanationSource.LLM,
        None,
    )
    assert len(draft.rejections) == 1
    assert not outcome.result.used_fallback


async def test_moving_towards_caution_is_allowed(data_dir: Path) -> None:
    outcome = await run(data_dir, student(JPY_3M, lambda f: conclude(f, "cant_tell")))
    draft = outcome.draft(JPY_3M)
    assert (draft.starting_verdict, draft.verdict) == (Verdict.LIKELY_HARMLESS, Verdict.CANT_TELL)
    assert draft.source is ExplanationSource.LLM and draft.rejections == ()


async def test_the_budget_stops_at_four_tool_calls(data_dir: Path) -> None:
    keep_going = investigate("neighbour_moves", "flag_frequency", "same_tenor_other_curves")
    outcome = await run(data_dir, student(JPY_3M, keep_going))
    draft = outcome.draft(JPY_3M)
    assert len(draft.tool_calls) == BUDGET == 4
    assert [c.round for c in draft.tool_calls] == [1, 1, 1, 2]  # the 5th and 6th never ran
    assert draft.tool_calls_left == 0
    assert (draft.verdict, draft.fallback_reason) == (Verdict.CANT_TELL, FALLBACK_BUDGET_SPENT)
    assert "4 tool calls were spent" in draft.report
    assert len(outcome.rounds) == 3
    assert [f["tool_calls_left"] for f in investigation_flags(outcome.rounds[2])] == [0]
    assert not outcome.result.used_fallback  # a spent budget is a verdict, not a failure


async def test_a_spent_budget_never_lowers_a_large_move(data_dir: Path) -> None:
    outcome = await run(data_dir, student(GBP_2Y, investigate("neighbour_moves", "flag_frequency")))
    assert outcome.draft(GBP_2Y).verdict is Verdict.LIKELY_REAL


async def test_a_tool_that_cant_run_still_costs_a_call_and_says_why(data_dir: Path) -> None:
    def act(flag: dict[str, Any]) -> dict[str, Any]:
        return (
            conclude(flag) if flag["tool_results"] else investigate("point_history", days=61)(flag)
        )

    outcome = await run(data_dir, student(JPY_3M, act))
    [call] = outcome.draft(JPY_3M).tool_calls
    assert call.evidence_id is None and call.error is not None and "at most 60 days" in call.error
    assert outcome.draft(JPY_3M).tool_calls_left == BUDGET - 1


async def test_an_invented_number_is_sent_back(data_dir: Path) -> None:
    def act(flag: dict[str, Any]) -> dict[str, Any]:
        reply = conclude(flag)
        if "draft_sent_back" not in flag:
            reply["report"] += " The curve moved 45 bps."
        return reply

    outcome = await run(data_dir, student(JPY_3M, act))
    draft = outcome.draft(JPY_3M)
    assert draft.source is ExplanationSource.LLM and "45" not in draft.report
    assert "does not match the cited evidence" in draft.rejections[0]


# --- fallbacks ------------------------------------------------------------------------------------


async def test_llm_off_runs_no_tools_and_keeps_the_starting_verdicts(data_dir: Path) -> None:
    outcome = await run(data_dir, llm_enabled=False)
    assert outcome.rounds == []
    for item in outcome.result.items:
        d = item.draft
        assert (d.source, d.fallback_reason, d.tool_calls) == (
            ExplanationSource.TEMPLATE,
            FALLBACK_LLM_OFF,
            (),
        )
        assert d.verdict is starting_verdict(item.assessment.classification)
        assert outcome.grounded(d) == []
    assert "Verdict: likely real" in outcome.draft(GBP_2Y).report
    assert not outcome.result.used_fallback


async def test_an_unreachable_model_gives_template_drafts(data_dir: Path) -> None:
    outcome = await run(data_dir, student=None)  # no reply scripted: the model is unavailable
    assert {i.draft.fallback_reason for i in outcome.result.items} == {FALLBACK_UNAVAILABLE}
    assert outcome.result.used_fallback


async def test_a_round_missing_a_flag_twice_gives_template_drafts(data_dir: Path) -> None:
    def drop_first(request: LLMRequest) -> dict[str, Any]:
        return {"steps": [conclude(f) for f in investigation_flags(request)[1:]]}

    outcome = await run(data_dir, drop_first)
    assert {i.draft.fallback_reason for i in outcome.result.items} == {FALLBACK_REJECTED}
    assert any("no answer was given" in r for r in outcome.result.investigation_rejections)


# --- the verdict rule and the output schema -------------------------------------------------------


def test_the_starting_verdict_comes_from_codes_label() -> None:
    assert starting_verdict(Classification.GENUINE_LARGE_MOVE) is Verdict.LIKELY_REAL
    assert starting_verdict(Classification.THRESHOLD_ARTIFACT) is Verdict.LIKELY_HARMLESS
    assert starting_verdict(Classification.ORDINARY) is Verdict.LIKELY_HARMLESS
    assert starting_verdict(Classification.INSUFFICIENT_HISTORY) is Verdict.CANT_TELL


@pytest.mark.parametrize(
    ("start", "proposed", "lowered"),
    [
        (Verdict.LIKELY_REAL, Verdict.CANT_TELL, True),
        (Verdict.LIKELY_REAL, Verdict.LIKELY_HARMLESS, True),
        (Verdict.CANT_TELL, Verdict.LIKELY_HARMLESS, True),
        (Verdict.LIKELY_HARMLESS, Verdict.CANT_TELL, False),
        (Verdict.LIKELY_HARMLESS, Verdict.LIKELY_REAL, False),
        (Verdict.CANT_TELL, Verdict.CANT_TELL, False),
    ],
)
def test_a_verdict_may_only_move_towards_caution(
    start: Verdict, proposed: Verdict, lowered: bool
) -> None:
    assert lowers(start, proposed) is lowered


def test_at_least_keeps_the_more_cautious() -> None:
    assert at_least(Verdict.LIKELY_REAL, Verdict.CANT_TELL) is Verdict.LIKELY_REAL
    assert at_least(Verdict.LIKELY_HARMLESS, Verdict.CANT_TELL) is Verdict.CANT_TELL


def test_the_output_has_no_field_for_a_label_a_severity_or_a_measure() -> None:
    assert set(StudentStep.model_fields) == {
        "flag_id",
        "action",
        "tool_calls",
        "verdict",
        "report",
        "evidence_ids",
    }
    assert set(ToolRequest.model_fields) == {"tool", "days", "reason"}

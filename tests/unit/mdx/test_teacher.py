"""E10-05: the teacher validates every flag's draft, with a scripted fake model.

Acceptance: happy day: 28 accepted; a scripted send-back produces one revised draft; a
rejected revision ends can't tell.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from tests.support import (
    accept_all,
    conclude,
    explain_from_inputs,
    investigate_from_inputs,
    investigation_flags,
    review_flags,
)

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.domain import (
    DecisionKind,
    DecisionSource,
    ItemDecisionKind,
    Plan,
    PlanSource,
)
from mdx_assessment.core.gateway import (
    FakeProvider,
    LLMGateway,
    LLMRequest,
    LLMUnavailableError,
    PromptStore,
)
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer, thread_config
from mdx_assessment.core.orchestration.graph import (
    GraphHooks,
    SessionRuntime,
    build_session_graph,
    run_review,
)
from mdx_assessment.core.orchestration.playbooks import (
    PlaybookStore,
    ResolvedPlaybook,
    resolve_playbook,
)
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.orchestration.tasks import ValidationContext
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.teacher import FlagReview, validate_drafts, validated_drafts
from mdx_assessment.usecases.mdx.checks.verdict import Verdict
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
CONFIG = load_config(PROJECT_ROOT / "config")
BOOK = resolve_playbook([], PlaybookStore.from_config(CONFIG))
GBP_2Y, JPY_3M = "20261015-GBP-2Y", "20261015-JPY-3M"
REVISE_STEP = "assess_movements_revise"
Reply = Callable[[LLMRequest], dict[str, Any]]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(CONFIG, DAY), path, config_version=CONFIG.version)
    return path


def round_of(request: LLMRequest) -> str:
    text = request.user_message.removeprefix("Inputs (JSON):\n").split("\n\n")[0]
    return str(json.loads(text)["round"])


def teacher(send_back: dict[str, str], *, on_revision: dict[str, str] | None = None) -> Reply:
    """Send back the named flags in the first round; in the revision round, send back
    the flags in `on_revision` (ending them unresolved) and accept the rest."""

    def answer(request: LLMRequest) -> dict[str, Any]:
        names = send_back if round_of(request) == "first_drafts" else (on_revision or {})
        return {
            "reviews": [
                {
                    "flag_id": f["flag_id"],
                    "decision": "send_back" if f["flag_id"] in names else "accept",
                    "reason": names.get(f["flag_id"], "follows from the evidence"),
                }
                for f in review_flags(request)
            ]
        }

    return answer


def student_revising_to_cant_tell(request: LLMRequest) -> dict[str, Any]:
    flags = investigation_flags(request)
    return {"steps": [conclude(f, "cant_tell" if "teacher_feedback" in f else None) for f in flags]}


@dataclass
class Outcome:
    final: RunState
    audit: InMemoryAuditTrail
    fake: FakeProvider

    def requests(self, prompt_id: str) -> list[LLMRequest]:
        return [r for r in self.fake.requests if r.prompt_id == prompt_id]

    @property
    def outcomes(self) -> dict[str, Any]:
        return validated_drafts(
            self.final.execution, self.final.agent_results, self.final.validations
        )


async def session(
    data_dir: Path,
    teach: Reply | None = accept_all,
    student: Reply = student_revising_to_cant_tell,
    *,
    llm_enabled: bool = True,
    book: ResolvedPlaybook = BOOK,
) -> Outcome:
    registries = build_registries()
    audit = InMemoryAuditTrail()
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    script: dict[str, Any] = {"mda_explain": explain_from_inputs, "mda_investigate": student}
    if teach is not None:
        script["master_validate"] = teach
    fake = FakeProvider(script)
    gateway = LLMGateway(fake, PromptStore(PROJECT_ROOT / "config"), audit)
    hub = ToolHub(
        registries.tools,
        CONFIG.get(PERMISSIONS_PATH, PermissionMatrix),
        ToolContext(config=CONFIG, settings=settings, audit=audit, cases=InMemoryCaseStore()),
    )

    async def default_plan(state: RunState) -> Plan:
        return Plan(source=PlanSource.DEFAULT, steps=book.content.default_plan)

    async def select(state: RunState) -> ResolvedPlaybook:
        return book

    async with hub:
        runtime = SessionRuntime(
            tasks=registries.tasks,
            checks=registries.checks,
            audit=audit,
            hub=hub,
            gateway=gateway,
            config=CONFIG,
        )
        app = build_session_graph(
            runtime, GraphHooks(plan=default_plan, select_playbook=select)
        ).compile(checkpointer=InMemorySaver(serde=checkpoint_serializer()))
        initial = RunState(
            session_id="sess-t", business_date=DAY, data_type="linear", llm_enabled=llm_enabled
        )
        final = RunState.model_validate(await app.ainvoke(initial, thread_config("sess-t")))
    return Outcome(final, audit, fake)


# --- acceptance -----------------------------------------------------------------------------------


async def test_happy_day_all_28_drafts_are_accepted(data_dir: Path) -> None:
    outcome = await session(data_dir)
    [validation] = outcome.final.validations
    assert (validation.step_id, validation.validated) == ("assess_movements", True)
    assert [d.kind for d in validation.decisions] == [ItemDecisionKind.ACCEPT] * 28
    assert {d.source for d in validation.decisions} == {DecisionSource.LLM_REVIEW}
    [request] = outcome.requests("master_validate")  # one batched review
    assert len(review_flags(request)) == 28
    assert [e.step_id for e in outcome.final.execution] == [
        "validate_inputs",
        "assess_movements",
        "record_cases",
    ]
    outcomes = outcome.outcomes
    assert {o.decision for o in outcomes.values()} == {ItemDecisionKind.ACCEPT}
    assert all(o.validated for o in outcomes.values())
    assert outcomes[GBP_2Y].verdict is Verdict.LIKELY_REAL
    assert sum(o.verdict is Verdict.LIKELY_HARMLESS for o in outcomes.values()) == 27
    [event] = [e for e in outcome.audit.read() if e.event_type == "item_decisions"]
    assert len(event.payload["decisions"]) == 28  # type: ignore[arg-type]
    # Step-level review points are as before: every step accepted by rule.
    assert [(d.kind, d.source) for d in outcome.final.decisions] == [
        (DecisionKind.ACCEPT, DecisionSource.RULE)
    ] * 3
    assert run_review(outcome.final).reliability.value == "high"


async def test_a_send_back_produces_one_revised_draft(data_dir: Path) -> None:
    why = "The report doesn't say whether the move is shared along the curve."
    outcome = await session(data_dir, teacher({JPY_3M: why}))
    final = outcome.final
    assert [e.step_id for e in final.execution] == [
        "validate_inputs",
        "assess_movements",
        REVISE_STEP,
        "record_cases",
    ]
    assert final.decisions[1].kind is DecisionKind.FOLLOW_UP
    assert final.decisions[1].follow_up is not None
    assert final.decisions[1].follow_up.arguments == {
        "of_step": "assess_movements",
        "flags": {JPY_3M: why},
    }
    first, second = final.validations
    assert [d.item_id for d in first.decisions if d.kind is ItemDecisionKind.SEND_BACK] == [JPY_3M]
    assert [(d.item_id, d.kind, d.revision) for d in second.decisions] == [
        (JPY_3M, ItemDecisionKind.ACCEPT, 1)
    ]
    jpy = outcome.outcomes[JPY_3M]
    assert (jpy.draft.revision, jpy.draft.teacher_feedback) == (1, why)
    assert (jpy.decision, jpy.verdict, jpy.validated) == (
        ItemDecisionKind.ACCEPT,
        Verdict.CANT_TELL,
        True,
    )
    assert sum(o.draft.revision for o in outcome.outcomes.values()) == 1  # one revised draft
    # The revision round tells the teacher it can't send back again.
    revision_request = outcome.requests("master_validate")[1]
    assert (
        round_of(revision_request) == "revision"
        and '"send_back_available": false' in revision_request.user_message
    )


async def test_a_rejected_revision_ends_cant_tell(data_dir: Path) -> None:
    why = "Is this point flagged often?"
    outcome = await session(
        data_dir, teacher({JPY_3M: why}, on_revision={JPY_3M: "Still unclear."})
    )

    jpy = outcome.outcomes[JPY_3M]
    assert (jpy.decision, jpy.verdict) == (ItemDecisionKind.UNRESOLVED, Verdict.CANT_TELL)
    assert [e.step_id for e in outcome.final.execution].count(REVISE_STEP) == 1  # no second round
    assert (
        "1 item(s) ended unresolved after the Master's review" in run_review(outcome.final).reasons
    )


async def test_a_rejected_revision_never_lowers_a_large_move(data_dir: Path) -> None:
    outcome = await session(
        data_dir,
        teacher(
            {GBP_2Y: "Say whether other curves moved."}, on_revision={GBP_2Y: "Not convincing."}
        ),
        investigate_from_inputs,
    )
    gbp = outcome.outcomes[GBP_2Y]
    assert (gbp.decision, gbp.verdict) == (ItemDecisionKind.UNRESOLVED, Verdict.LIKELY_REAL)


# --- the reason names no tool ---------------------------------------------------------------------


async def test_a_send_back_reason_that_names_a_tool_is_retried(data_dir: Path) -> None:
    calls: list[int] = []

    def names_a_tool_once(request: LLMRequest) -> dict[str, Any]:
        calls.append(1)
        reason = (
            "Run neighbour_moves." if len(calls) == 1 else "Is the move shared along the curve?"
        )
        return (
            teacher({JPY_3M: reason})(request)
            if round_of(request) == "first_drafts"
            else accept_all(request)
        )

    outcome = await session(data_dir, names_a_tool_once)
    retry = outcome.requests("master_validate")[1].user_message
    assert "names tool(s) ['neighbour_moves']" in retry
    assert outcome.outcomes[JPY_3M].draft.teacher_feedback == "Is the move shared along the curve?"


def test_the_teachers_output_has_no_field_for_a_verdict_or_a_tool() -> None:
    assert set(FlagReview.model_fields) == {"flag_id", "decision", "reason"}


# --- not validated, and send-backs that can't run -------------------------------------------------


async def test_llm_off_keeps_the_drafts_not_validated(data_dir: Path) -> None:
    outcome = await session(data_dir, llm_enabled=False)
    [validation] = outcome.final.validations
    assert validation.validated is False
    assert {(d.kind, d.source) for d in validation.decisions} == {
        (ItemDecisionKind.NOT_VALIDATED, DecisionSource.RULE_FALLBACK)
    }
    assert outcome.requests("master_validate") == []
    assert not any(o.validated for o in outcome.outcomes.values())
    assert outcome.outcomes[GBP_2Y].verdict is Verdict.LIKELY_REAL  # the drafts stand


async def test_an_unreachable_teacher_keeps_the_drafts_not_validated(data_dir: Path) -> None:
    outcome = await session(data_dir, teach=None)
    [validation] = outcome.final.validations
    assert validation.validated is False and "unreachable" in validation.decisions[0].reason
    assert (
        "step 'assess_movements': its items were not reviewed by the Master"
        in run_review(outcome.final).reasons
    )


async def test_a_send_back_the_playbook_cant_run_ends_unresolved(data_dir: Path) -> None:
    limits = BOOK.content.limits.model_copy(update={"max_send_backs": 0})
    book = BOOK.model_copy(update={"content": BOOK.content.model_copy(update={"limits": limits})})
    outcome = await session(data_dir, teacher({JPY_3M: "Unclear."}), book=book)
    [validation] = outcome.final.validations
    jpy = next(d for d in validation.decisions if d.item_id == JPY_3M)
    assert jpy.kind is ItemDecisionKind.UNRESOLVED and "no send-back: P-3" in jpy.reason
    assert REVISE_STEP not in [e.step_id for e in outcome.final.execution]
    assert '"send_back_available": false' in outcome.requests("master_validate")[0].user_message


# --- code checks first ----------------------------------------------------------------------------


async def test_a_draft_that_fails_code_checks_never_reaches_the_teacher(data_dir: Path) -> None:
    outcome = await session(data_dir)
    state = outcome.final
    result = dict(state.agent_results["assess_movements"])
    items = [dict(i) for i in result["items"]]  # type: ignore[union-attr]
    for item in items:
        draft = dict(item["draft"])  # type: ignore[arg-type]
        if draft["flag_id"] == GBP_2Y:
            draft["verdict"] = "likely_harmless"  # lowered after the loop
        if draft["flag_id"] == JPY_3M:
            draft["report"] += " It moved 45 bps."  # an invented number
        item["draft"] = draft
    tampered = state.model_copy(
        update={
            "agent_results": {**state.agent_results, "assess_movements": {**result, "items": items}}
        }
    )
    fake = FakeProvider({"master_validate": accept_all})
    gateway = LLMGateway(fake, PromptStore(PROJECT_ROOT / "config"), InMemoryAuditTrail())
    ctx = ValidationContext(
        state=tampered,
        entry=state.execution[1],
        gateway=gateway,
        loaded_config=CONFIG,
        send_back_refusal=lambda s: None,
    )
    validation = (await validate_drafts(ctx)).validation
    by_flag = {d.item_id: d for d in validation.decisions}
    for flag_id, message in ((GBP_2Y, "less cautious"), (JPY_3M, "does not match")):
        assert (by_flag[flag_id].kind, by_flag[flag_id].source) == (
            ItemDecisionKind.UNRESOLVED,
            DecisionSource.RULE,
        )
        assert message in by_flag[flag_id].reason
    sent = {f["flag_id"] for f in review_flags(fake.requests[0])}
    assert len(sent) == 26 and not sent & {GBP_2Y, JPY_3M}


async def test_a_revision_that_fails_ends_cant_tell(data_dir: Path) -> None:
    def student_down_on_revision(request: LLMRequest) -> dict[str, Any]:
        if any("teacher_feedback" in f for f in investigation_flags(request)):
            raise LLMUnavailableError("down")
        return investigate_from_inputs(request)

    outcome = await session(data_dir, teacher({JPY_3M: "Unclear."}), student_down_on_revision)
    jpy = outcome.outcomes[JPY_3M]
    assert jpy.verdict is Verdict.CANT_TELL and jpy.draft.revision == 1

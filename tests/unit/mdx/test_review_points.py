"""E10-01: review points and the Master's decisions, through the real session service.

Acceptance: a happy day gives `accept` on every step; a test day with a dropped flag
triggers the LLM review and `flag_for_human`; a failed step skips its dependents.
"""

import dataclasses
import json
import shutil
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from tests.support import explain_from_inputs, investigate_from_inputs

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.api import create_app
from mdx_assessment.core.gateway import FakeProvider, LLMRequest, create_llm_gateway
from mdx_assessment.core.observability import Tracer
from mdx_assessment.core.orchestration.checkpoint import checkpoint_serializer
from mdx_assessment.core.orchestration.playbooks import PlaybookStore, resolve_playbook
from mdx_assessment.core.orchestration.session_service import SessionService
from mdx_assessment.core.orchestration.tasks import TaskContext
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.persistence.audit import AuditTrail
from mdx_assessment.core.persistence.session_store import InMemorySessionStore
from mdx_assessment.core.platform import PlatformRegistries
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, Settings, get_settings
from mdx_assessment.usecases.mdx.agents.ids import CHECK_FILES_AND_INTAKE, NEAR_ZERO_CHECK
from mdx_assessment.usecases.mdx.agents.market_data import MarketAssessmentResult, near_zero_check
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

DAY = date(2026, 10, 15)
BODY = {"business_date": "2026-10-15", "data_type": "linear"}
CONFIG = load_config(PROJECT_ROOT / "config")
Script = Callable[[LLMRequest], dict[str, Any]]


# --- fakes ----------------------------------------------------------------------------------


def plan(request: LLMRequest) -> dict[str, Any]:
    tasks = resolve_playbook([], PlaybookStore.from_config(CONFIG)).content.mandatory_steps
    return {
        "steps": [
            {"step_id": f"s{i}", "task": t, "rationale": "Needed."} for i, t in enumerate(tasks)
        ],
        "follow_ups": [],
        "rationale": "Normal day, see fact-flag-count.",
        "cited_fact_ids": ["fact-flag-count"],
    }


def checks_of(request: LLMRequest) -> list[dict[str, Any]]:
    text = request.user_message.removeprefix("Inputs (JSON):\n").split("\n\n")[0]
    checks: list[dict[str, Any]] = json.loads(text)["checks"]
    return checks


def review(
    decision: str = "flag_for_human", reason: Callable[[dict[str, Any]], str] | None = None
) -> Script:
    """A Master review that cites every failed check and quotes its details."""

    def answer(request: LLMRequest) -> dict[str, Any]:
        failed = [c for c in checks_of(request) if not c["passed"]]
        text = reason(failed[0]) if reason else f"{failed[0]['id']} failed: {failed[0]['details']}."
        return {"decision": decision, "reason": text, "cited_check_ids": [c["id"] for c in failed]}

    return answer


# --- the service ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def happy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(CONFIG, DAY), path, config_version=CONFIG.version)
    return path


def settings_for(data_dir: Path) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )


@asynccontextmanager
async def memory_checkpointer() -> AsyncIterator[InMemorySaver]:
    yield InMemorySaver(serde=checkpoint_serializer())


class Session:
    def __init__(
        self,
        data_dir: Path,
        script: dict[str, Any],
        registries: PlatformRegistries | None = None,
    ) -> None:
        settings = settings_for(data_dir)
        self.fake = FakeProvider(
            {
                "master_plan": plan,
                "mda_explain": explain_from_inputs,
                "mda_investigate": investigate_from_inputs,
                **script,
            }
        )
        self.audit = InMemoryAuditTrail()

        def make_gateway(trail: AuditTrail, tracer: Tracer) -> Any:
            return create_llm_gateway(trail, tracer, settings, self.fake)

        service = SessionService(
            registries=registries or build_registries(),
            config=CONFIG,
            settings=settings,
            store=InMemorySessionStore(),
            audit=self.audit,
            cases=InMemoryCaseStore(),
            checkpointer=memory_checkpointer,
            make_gateway=make_gateway,
            make_tracer=lambda: Tracer(None),
        )
        self.client = TestClient(create_app(sessions=service))

    def run(self, **body: Any) -> dict[str, Any]:
        started = self.client.post("/sessions", json={**BODY, **body}).json()
        result: dict[str, Any] = self.client.get(f"/sessions/{started['session_id']}").json()
        return result

    def requests(self, prompt_id: str) -> list[LLMRequest]:
        return [r for r in self.fake.requests if r.prompt_id == prompt_id]


def decisions(session: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(d["step_id"], d["kind"], d["source"]) for d in session["decisions"]]


def with_handler(task_id: str, handler: Callable[[TaskContext], Any]) -> PlatformRegistries:
    registries = build_registries()
    tasks = registries.tasks._tasks  # test seam: swap one handler, keep everything else
    tasks[task_id] = dataclasses.replace(tasks[task_id], handler=handler)
    return registries


async def drop_one_flag(ctx: TaskContext) -> MarketAssessmentResult:
    result = await near_zero_check(ctx)
    return result.model_copy(update={"items": result.items[:-1]})


def short_file_day(happy_dir: Path, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(happy_dir, data_dir)
    path = data_dir / "sim" / "2026-10-15" / "file_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[0]["record_count"] = 79
    path.write_text(json.dumps(metadata))
    return data_dir


# --- acceptance -----------------------------------------------------------------------------


def test_a_happy_day_accepts_every_step_without_an_llm_review(happy_dir: Path) -> None:
    session = Session(happy_dir, {})
    result = session.run()
    assert decisions(result) == [
        ("s0", "accept", "rule"),
        ("s1", "accept", "rule"),
        ("s2", "accept", "rule"),
    ]
    checks = {(c["step_id"], c["check_id"]): c for c in result["review_checks"]}
    assert set(checks) == {("s0", "C-2"), ("s0", "C-3"), ("s1", "C-1")}
    assert all(c["passed"] for c in checks.values())
    assert checks[("s1", "C-1")]["details"] == (
        "28 of 28 normalized flags assessed, none invented, dataset complete"
    )
    assert session.requests("master_review") == []
    assert result["review"] == {
        "reliability": "high",
        "reasons": ["every step done, every check passed"],
        "human_review_required": [],
        "llm_reviews": 0,
    }
    assert result["status"] == "succeeded"


def test_a_dropped_flag_triggers_the_llm_review_and_flag_for_human(happy_dir: Path) -> None:
    session = Session(
        happy_dir, {"master_review": review()}, with_handler(NEAR_ZERO_CHECK, drop_one_flag)
    )
    result = session.run()
    c1 = next(c for c in result["review_checks"] if c["check_id"] == "C-1")
    assert c1["passed"] is False
    assert c1["details"].startswith("27 of 28 normalized flags assessed; not assessed: 20261015-")
    [request] = session.requests("master_review")
    assert [c["id"] for c in checks_of(request) if not c["passed"]] == ["C-1"]
    assert decisions(result)[1] == ("s1", "flag_for_human", "llm_review")
    assert decisions(result)[2] == ("s2", "accept", "rule")  # nothing failed, nothing skipped
    review_ = result["review"]
    assert (review_["reliability"], review_["human_review_required"], review_["llm_reviews"]) == (
        "low",
        ["s1"],
        1,
    )


def test_a_failed_step_skips_its_dependents(happy_dir: Path) -> None:
    async def crash(ctx: TaskContext) -> Any:
        raise RuntimeError("MDX metadata unreadable")

    session = Session(happy_dir, {}, with_handler(CHECK_FILES_AND_INTAKE, crash))
    result = session.run()
    steps = [(e["step_id"], e["status"]) for e in result["execution"]]
    assert steps == [("s0", "failed"), ("s1", "skipped"), ("s2", "skipped")]
    assert result["execution"][1]["error"] == "skipped: it needs step 's0', which failed"
    assert result["execution"][2]["error"] == "skipped: it needs step 's1', which skipped"
    assert decisions(result) == [
        ("s0", "flag_for_human", "rule"),
        ("s1", "flag_for_human", "rule"),
        ("s2", "flag_for_human", "rule"),
    ]
    assert result["review_checks"] == []  # nothing ran to check
    assert result["status"] == "succeeded"
    assert result["fallbacks"] == ["step_failed:s0", "step_skipped:s1", "step_skipped:s2"]
    assert result["review"]["reliability"] == "low"
    assert session.requests("mda_explain") == []  # the skipped step made no call


# --- guard cases ----------------------------------------------------------------------------


def test_a_short_file_is_flagged_where_it_shows_up(happy_dir: Path, tmp_path: Path) -> None:
    session = Session(short_file_day(happy_dir, tmp_path), {"master_review": review()})
    result = session.run()
    failed = [(c["step_id"], c["check_id"]) for c in result["review_checks"] if not c["passed"]]
    assert failed == [("s0", "C-2"), ("s1", "C-1")]
    assert decisions(result)[:2] == [
        ("s0", "flag_for_human", "llm_review"),
        ("s1", "flag_for_human", "llm_review"),
    ]
    c1 = next(c for c in result["review_checks"] if c["check_id"] == "C-1")
    assert c1["details"].endswith("the dataset they came from is incomplete")
    assert result["review"]["human_review_required"] == ["s0", "s1"]


def test_templates_after_bad_explanations_mark_the_step_done_with_fallback(happy_dir: Path) -> None:
    def invent(request: LLMRequest) -> dict[str, Any]:
        reply = explain_from_inputs(request)
        reply["explanations"][0]["explanation"] += " It moved 45 bps."
        return reply

    result = Session(happy_dir, {"mda_explain": invent}).run()
    assert [e["status"] for e in result["execution"]] == ["done", "done_with_fallback", "done"]
    assert decisions(result)[1] == ("s1", "accept", "rule")
    assert "template fallback" in result["decisions"][1]["reason"]
    assert result["status"] == "succeeded"
    assert result["fallbacks"] == ["step_template:s1"]
    assert result["review"]["reliability"] == "medium"


def test_an_accepted_failed_check_lowers_reliability_to_medium(happy_dir: Path) -> None:
    session = Session(
        happy_dir,
        {"master_review": review("accept")},
        with_handler(NEAR_ZERO_CHECK, drop_one_flag),
    )
    result = session.run()
    assert decisions(result)[1] == ("s1", "accept", "llm_review")
    assert result["review"]["reliability"] == "medium"
    assert "accepted despite a failed check" in " ".join(result["review"]["reasons"])


# --- when the review can't be trusted -------------------------------------------------------


def test_with_the_llm_off_a_failed_check_is_flagged_without_a_call(happy_dir: Path) -> None:
    session = Session(happy_dir, {}, with_handler(NEAR_ZERO_CHECK, drop_one_flag))
    result = session.run(llm_enabled=False)
    # With the LLM off the playbook's default plan runs, so its own step IDs.
    assert decisions(result)[1] == ("assess_movements", "flag_for_human", "rule_fallback")
    assert "the LLM is off" in result["decisions"][1]["reason"]
    assert session.fake.requests == []


def test_an_unusable_review_flags_the_step(happy_dir: Path) -> None:
    def invented(failed: dict[str, Any]) -> str:
        return "Only 12 flags are affected, so this is fine."  # 12 isn't in the details

    session = Session(
        happy_dir,
        {"master_review": review("accept", invented)},
        with_handler(NEAR_ZERO_CHECK, drop_one_flag),
    )
    result = session.run()
    assert len(session.requests("master_review")) == 2  # retried once with the reason
    assert "'12' does not match" in session.requests("master_review")[1].user_message
    assert decisions(result)[1] == ("s1", "flag_for_human", "rule_fallback")


def test_the_review_may_cite_only_failed_checks(happy_dir: Path) -> None:
    calls = {"n": 0}

    def cites_a_passed_check(request: LLMRequest) -> dict[str, Any]:
        calls["n"] += 1
        reply = review()(request)
        if calls["n"] == 1:
            reply["cited_check_ids"] = ["C-1", "C-2"]  # C-2 isn't checked at this point
        return reply

    session = Session(
        happy_dir,
        {"master_review": cites_a_passed_check},
        with_handler(NEAR_ZERO_CHECK, drop_one_flag),
    )
    result = session.run()
    retry = session.requests("master_review")[1].user_message
    assert "- cited ID 'C-2' does not exist" in retry  # only this point's checks are evidence
    assert decisions(result)[1] == ("s1", "flag_for_human", "llm_review")


def test_a_flag_that_cannot_be_normalized_fails_c3(happy_dir: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(happy_dir, data_dir)
    path = data_dir / "sim" / "2026-10-15" / "flags.json"
    flags = json.loads(path.read_text())
    flags[0]["current_value"] = flags[0]["current_value"] + 1  # disagrees with MDX's stored value
    path.write_text(json.dumps(flags))

    result = Session(data_dir, {"master_review": review()}).run()
    c3 = next(c for c in result["review_checks"] if c["check_id"] == "C-3")
    assert c3["passed"] is False
    assert c3["details"] == (
        f"27 of 28 flags normalized; not normalized: {flags[0]['flag_id']} (value_mismatch)"
    )
    assert decisions(result)[0] == ("s0", "flag_for_human", "llm_review")
    c1 = next(c for c in result["review_checks"] if c["check_id"] == "C-1")
    assert c1["passed"] is True  # C-1 reconciles normalized flags; C-3 caught the lost one

"""E11-01: the Escalation and Reporting Agent records one case per flag, in list order.

Acceptance: cases are listed via `GET /cases?session_id=` in order, GBP 2Y first.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.mdx.test_sessions_api import BODY, client_for, config, settings, valid_plan  # noqa: F401
from tests.unit.mdx.test_teacher import GBP_2Y, JPY_3M, data_dir, session, teacher  # noqa: F401

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.settings import Settings
from mdx_assessment.usecases.mdx.agents.escalation import CaseReport, RecordCasesResult
from mdx_assessment.usecases.mdx.checks.verdict import Verdict

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def client(config: LoadedConfig, settings: Settings) -> TestClient:  # noqa: F811
    return client_for(config, settings, valid_plan(config))


def start(client: TestClient, **body: Any) -> str:
    started = client.post("/sessions", json={**BODY, **body})
    assert started.status_code == 201
    return str(started.json()["session_id"])


# --- acceptance -----------------------------------------------------------------------------------


def test_cases_are_listed_in_order_gbp_2y_first(client: TestClient) -> None:
    session_id = start(client)
    listed = client.get("/cases", params={"session_id": session_id})
    assert listed.status_code == 200
    body = listed.json()
    assert (body["session_id"], body["count"]) == (session_id, 28)
    cases = body["cases"]
    reports = [CaseReport.model_validate(c["payload"]) for c in cases]
    assert [r.rank for r in reports] == list(range(1, 29))
    assert (cases[0]["item_key"], reports[0].verdict, reports[0].suggested_action.value) == (
        "GBP 2Y",
        Verdict.LIKELY_REAL,
        "review_first",
    )
    assert all(c["status"] == "open" for c in cases)
    # Then by verdict, and within a verdict in the near-zero step's review order.
    session = client.get(f"/sessions/{session_id}").json()
    review_order = [i["assessment"]["item_key"] for i in session["agent_results"]["s1"]["items"]]
    harmless = [
        c["item_key"]
        for c, r in zip(cases, reports, strict=True)
        if r.verdict is Verdict.LIKELY_HARMLESS
    ]
    assert harmless == [i for i in review_order if i in harmless]
    assert [c["item_key"] for c in cases[-4:]] == review_order[-4:]  # the four artifacts last
    recorded = RecordCasesResult.model_validate(session["agent_results"]["s2"])
    assert [line.case_id for line in recorded.cases] == [c["case_id"] for c in cases]
    assert (recorded.cases_created, recorded.cases_existing, recorded.not_validated) == (28, 0, 0)


def test_each_case_is_the_flags_roi_report(client: TestClient) -> None:
    session_id = start(client)
    first = client.get("/cases", params={"session_id": session_id}).json()["cases"][0]
    report = CaseReport.model_validate(first["payload"])
    assert (
        report.flag_id == GBP_2Y and report.validated and report.teacher_decision.value == "accept"
    )  # type: ignore[union-attr]
    assert (report.change, report.unit, report.severity) == (pytest.approx(18.0), "bps", "high")
    assert report.mdx_shows is not None and report.mdx_shows > 2
    assert report.report.text and report.report.source.value == "llm"
    trail = report.trail
    assert trail.label == "genuine_large_move" and trail.starting_verdict is Verdict.LIKELY_REAL
    assert trail.rule_trace and trail.explanation
    assert [d.kind.value for d in trail.teacher_decisions] == ["accept"]
    session = client.get(f"/sessions/{session_id}").json()
    known = {
        e["evidence_id"] for r in session["agent_results"].values() for e in r.get("evidence", [])
    }
    assert report.evidence_ids and set(report.evidence_ids) <= known


def test_unknown_session_is_404_and_session_id_is_required(client: TestClient) -> None:
    assert client.get("/cases", params={"session_id": "sess-nope"}).status_code == 404
    assert client.get("/cases").status_code == 422


def test_with_the_llm_off_every_case_is_recorded_not_validated(client: TestClient) -> None:
    session_id = start(client, llm_enabled=False)
    cases = client.get("/cases", params={"session_id": session_id}).json()["cases"]
    reports = [CaseReport.model_validate(c["payload"]) for c in cases]
    assert len(reports) == 28 and reports[0].item == "GBP 2Y"
    assert not any(r.validated for r in reports)
    assert {r.report.source.value for r in reports} == {"template"}


# --- order and trail after a send-back --------------------------------------------------------------


async def test_a_cant_tell_flag_comes_right_after_the_likely_real_ones(data_dir: Path) -> None:  # noqa: F811
    why = "Is the move shared along the curve?"
    outcome = await session(data_dir, teacher({JPY_3M: why}))
    recorded = RecordCasesResult.model_validate(outcome.final.agent_results["record_cases"])
    assert [(line.item, line.verdict) for line in recorded.cases[:2]] == [
        ("GBP 2Y", Verdict.LIKELY_REAL),
        ("JPY 3M", Verdict.CANT_TELL),
    ]
    assert recorded.verdicts == {
        Verdict.LIKELY_REAL: 1,
        Verdict.CANT_TELL: 1,
        Verdict.LIKELY_HARMLESS: 26,
    }
    jpy = CaseReport.model_validate(outcome.cases.list_for_session("sess-t")[1].payload)
    assert (jpy.trail.revision, jpy.trail.teacher_feedback) == (1, why)
    assert [d.kind.value for d in jpy.trail.teacher_decisions] == ["send_back", "accept"]
    assert jpy.suggested_action.value == "review"

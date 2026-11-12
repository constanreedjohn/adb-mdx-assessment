"""E11-02: the raw run report, counted by code.

Acceptance: figures match the simulated day's ground truth (read here only, never by
the solution).
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.mdx.test_send_back_round import student as investigating_student
from tests.unit.mdx.test_sessions_api import BODY, client_for, config, settings, valid_plan  # noqa: F401
from tests.unit.mdx.test_teacher import JPY_3M, data_dir, session, teacher  # noqa: F401

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import ItemDecisionKind
from mdx_assessment.settings import Settings
from mdx_assessment.usecases.mdx.agents.escalation import RecordCasesResult, RunReport
from mdx_assessment.usecases.mdx.checks.verdict import Verdict
from mdx_assessment.usecases.mdx.config import EXPECTED_FILES_PATH, ExpectedFiles
from mdx_assessment.usecases.mdx.domain import DataType

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def client(config: LoadedConfig, settings: Settings) -> TestClient:  # noqa: F811
    return client_for(config, settings, valid_plan(config))


def truth(app: Settings) -> dict[str, Any]:
    path = app.app.data_dir / "sim" / "2026-10-15" / "ground_truth.json"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def report_of(client: TestClient, **body: Any) -> RunReport:
    started = client.post("/sessions", json={**BODY, **body}).json()
    session = client.get(f"/sessions/{started['session_id']}").json()
    [step] = [
        e["step_id"]
        for e in session["execution"]
        if e["task"] == "escalation_reporting.record_cases"
    ]
    return RecordCasesResult.model_validate(session["agent_results"][step]).report


# --- acceptance -----------------------------------------------------------------------------------


def test_the_figures_match_the_ground_truth(
    client: TestClient,
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
) -> None:
    gt = truth(settings)
    report = report_of(client)
    expected_files = config.get(EXPECTED_FILES_PATH, ExpectedFiles).for_data_type(DataType.LINEAR)

    assert report.business_date == gt["business_date"]
    assert (report.files_expected, report.files_received, report.dataset_complete) == (
        len(expected_files),
        1,
        True,
    )
    [file] = report.files
    assert (file.status, file.received_records) == ("ok", expected_files[0].expected_record_count)

    assert report.flags_read == report.flags_normalized == report.flags_assessed == gt["flag_count"]
    assert report.flags_not_normalized == 0
    classes = Counter(f["expected_class"] for f in gt["flags"])
    assert report.labels == {
        "genuine_large_move": classes["genuine_large_move"],
        "threshold_artifact": classes["threshold_artifact"],
        "ordinary": classes["ordinary"],
        "insufficient_history": 0,
    }
    assert (report.threshold_artifacts, report.large_moves) == (4, 1)

    by_id = {f["flag_id"]: f for f in gt["flags"]}
    assert report.review_first == tuple(
        f"{by_id[i]['curve_id']} {by_id[i]['tenor']}" for i in gt["review_first"]
    )
    assert report.verdicts == {
        Verdict.LIKELY_REAL: len(gt["review_first"]),
        Verdict.CANT_TELL: 0,
        Verdict.LIKELY_HARMLESS: gt["flag_count"] - len(gt["review_first"]),
    }
    assert report.teacher_decisions[ItemDecisionKind.ACCEPT] == gt["flag_count"]
    assert (report.cases_created, report.cases_existing) == (gt["flag_count"], 0)
    # A happy day the student settles from the crucial results alone.
    assert (
        report.flags_investigated,
        report.tools_called,
        report.send_backs,
        report.send_back_rounds,
    ) == (0, 0, 0, 0)
    assert report.templates == {"explanations": 0, "reports": 0}


def test_with_the_llm_off_the_figures_still_hold(client: TestClient, settings: Settings) -> None:  # noqa: F811
    gt = truth(settings)
    report = report_of(client, llm_enabled=False)
    assert (report.flags_read, report.threshold_artifacts, report.large_moves) == (
        gt["flag_count"],
        4,
        1,
    )
    assert report.review_first == ("GBP 2Y",)
    assert report.templates == {"explanations": 28, "reports": 28}
    assert report.teacher_decisions[ItemDecisionKind.NOT_VALIDATED] == 28
    assert report.cases_created == 28


async def test_tools_and_send_backs_are_counted(data_dir: Path) -> None:  # noqa: F811
    outcome = await session(
        data_dir, teacher({JPY_3M: "Is this point flagged often?"}), investigating_student
    )
    report = RecordCasesResult.model_validate(outcome.final.agent_results["record_cases"]).report
    assert (report.flags_investigated, report.tools_called) == (1, 2)
    assert report.tools_by_name == {"flag_frequency": 1, "neighbour_moves": 1}
    assert (report.send_backs, report.send_back_rounds, report.tool_errors) == (1, 1, 0)
    assert report.verdicts[Verdict.CANT_TELL] == 1
    assert report.teacher_decisions[ItemDecisionKind.ACCEPT] == 28  # JPY 3M's revision accepted

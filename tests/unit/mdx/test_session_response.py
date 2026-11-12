"""E12-02: the full session response, the audit endpoint and the evaluation stub.

Acceptance: the response matches the §2.4 shape (docs/implementation-sprint-plan.md).
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.support import CapturedTraces
from tests.unit.mdx.test_sessions_api import BODY, client_for, config, settings, valid_plan  # noqa: F401
from tests.unit.mdx.test_teacher import GBP_2Y, JPY_3M

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.settings import Settings

# The §2.4 shape, as keys each object must have. Lists give the keys of their items.
SHAPE: dict[str, Any] = {
    "session_id": None,
    "business_date": None,
    "data_type": None,
    "versions": {"config": None, "prompts": None, "llm_deployment": None},
    "run_context": {"facts": [{"id": None, "name": None, "value": None}], "observations": None},
    "playbook": {
        "id": None,
        "version": None,
        "selected_by": None,
        "selection": {
            "rules": None,
            "conditions_met_not_selected": None,
            "multi_condition_day": None,
        },
        "content": {"mandatory_steps": None, "allowed_follow_ups": None, "limits": None},
    },
    "plan": {
        "source": None,
        "steps": None,
        "follow_ups": None,
        "rationale": None,
        "cited_fact_ids": None,
        "validation": {"valid": None, "checks": None, "attempts": None},
    },
    "agents": {
        "input_validation": {
            "datasets": [
                {
                    "file": None,
                    "present": None,
                    "record_count": None,
                    "expected": None,
                    "complete": None,
                }
            ],
            "flags_read": None,
            "normalized_flags": None,
        },
        "market_data_assessment": {
            "items": [
                {
                    "curve": None,
                    "tenor": None,
                    "prev": None,
                    "curr": None,
                    "change_bps": None,
                    "change_pct": None,
                    "near_zero": None,
                    "zscore_60d": None,
                    "threshold_artifact": None,
                    "explanation": None,
                    "evidence_ids": None,
                    "explanation_source": None,
                }
            ]
        },
        "escalation_reporting": {
            "case_ids": None,
            "report": {
                "files_checked": None,
                "flags_read": None,
                "items_assessed": None,
                "threshold_artifacts": None,
                "large_moves": None,
                "cases_created": None,
            },
        },
    },
    "execution": [
        {
            "step_id": None,  # §2.4 writes "step"; the API keeps the name used everywhere else
            "task": None,
            "status": None,
            "review_point": {"checks": None, "llm_review": None, "decision": None},
        }
    ],
    "review": {"reliability": None, "human_review_required": None, "llm_reviews": None},
    "run_opinion": {
        "summary": None,
        "review_first": None,
        "uncertainties": None,
        "evidence_ids": None,
    },
    "fallbacks": None,
    "llm_usage": {"calls": None, "tokens_in": None, "tokens_out": None, "latency_ms": None},
    "trace": {"mlflow_trace_id": None, "url": None},
    "audit": {"events": None, "chain_valid": None},
}


def missing(shape: Any, value: Any, path: str = "") -> list[str]:
    """Every key in `shape` that `value` lacks, as a dotted path."""
    if isinstance(shape, dict):
        if not isinstance(value, dict):
            return [f"{path or '.'}: not an object"]
        found = []
        for key, inner in shape.items():
            if key not in value:
                found.append(f"{path}.{key}")
            else:
                found += missing(inner, value[key], f"{path}.{key}")
        return found
    if isinstance(shape, list):
        if not isinstance(value, list) or not value:
            return [f"{path}: not a non-empty list"]
        return [p for item in value for p in missing(shape[0], item, f"{path}[]")]
    return []


@pytest.fixture
def traced(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
    captured_traces: CapturedTraces,
) -> TestClient:
    tracer, _ = captured_traces
    return client_for(config, settings, valid_plan(config), tracer)


def start(client: TestClient) -> dict[str, Any]:
    started = client.post("/sessions", json=BODY).json()
    session: dict[str, Any] = client.get(f"/sessions/{started['session_id']}").json()
    return session


# --- acceptance -----------------------------------------------------------------------------------


def test_the_response_matches_the_section_2_4_shape(traced: TestClient) -> None:
    session = start(traced)
    assert missing(SHAPE, session) == []


def test_the_response_carries_the_days_answers(traced: TestClient) -> None:
    session = start(traced)
    agents = session["agents"]
    [dataset] = agents["input_validation"]["datasets"]
    assert (
        dataset["present"],
        dataset["record_count"],
        dataset["expected"],
        dataset["complete"],
    ) == (True, 80, 80, True)
    items = {f"{i['curve']} {i['tenor']}": i for i in agents["market_data_assessment"]["items"]}
    jpy = items["JPY 3M"]
    assert (jpy["prev"], jpy["curr"], jpy["change_bps"], jpy["change_pct"]) == (
        0.04,
        0.07,
        3.0,
        75.0,
    )
    assert (jpy["near_zero"], jpy["threshold_artifact"]) == (True, True)
    assert jpy["zscore_60d"] == pytest.approx(0.85, abs=0.25)
    assert (
        items["GBP 2Y"]["threshold_artifact"] is False
        and items["GBP 2Y"]["verdict"] == "likely_real"
    )
    report = agents["escalation_reporting"]["report"]
    assert (report["files_checked"], report["flags_read"], report["items_assessed"]) == (1, 28, 28)
    assert (report["threshold_artifacts"], report["large_moves"], report["cases_created"]) == (
        4,
        1,
        28,
    )

    # The ordered flag list, each with its ROI report and case.
    flags = session["flags"]
    assert [f["rank"] for f in flags] == list(range(1, 29))
    assert (flags[0]["flag_id"], flags[0]["verdict"], flags[0]["suggested_action"]) == (
        GBP_2Y,
        "likely_real",
        "review_first",
    )
    assert [f["case_id"] for f in flags] == agents["escalation_reporting"]["case_ids"]
    assert (
        next(f for f in flags if f["flag_id"] == JPY_3M)["trail"]["label"] == "threshold_artifact"
    )

    # Each step's review point, and the run's usage and audit.
    steps = session["execution"]
    assert [s["review_point"]["decision"] for s in steps] == ["accept"] * 3
    assert [c["check_id"] for c in steps[1]["review_point"]["checks"]] == ["C-1"]
    assert steps[1]["review_point"]["validation"] == "28 draft(s) reviewed: 28 accept"
    usage = session["llm_usage"]
    assert usage["calls"] == 5
    assert usage["by_prompt"] == {
        p: 1
        for p in (
            "master_opinion",
            "master_plan",
            "master_validate",
            "mda_explain",
            "mda_investigate",
        )
    }
    assert session["audit"]["chain_valid"] is True and session["audit"]["events"] > 0


# --- the audit endpoint ---------------------------------------------------------------------------


def test_the_audit_endpoint_returns_the_verified_trail(traced: TestClient) -> None:
    session = start(traced)
    audit = traced.get(f"/sessions/{session['session_id']}/audit").json()
    assert audit["chain"]["valid"] is True
    assert (
        audit["count"]
        == len(audit["events"])
        == audit["chain"]["events_checked"]
        == session["audit"]["events"]
    )
    seqs = [e["seq"] for e in audit["events"]]
    assert seqs == sorted(seqs)
    types = [e["event_type"] for e in audit["events"]]
    assert types[-1] == "session_completed"
    assert types.index("run_context") < types.index("plan")  # facts read through MDX come first
    assert {
        "plan",
        "task_completed",
        "decision",
        "item_decisions",
        "llm_call",
        "tool_call",
        "run_opinion",
    } <= set(types)
    assert {e["session_id"] for e in audit["events"]} == {session["session_id"]}


def test_unknown_sessions_are_404(traced: TestClient) -> None:
    assert traced.get("/sessions/sess-nope/audit").status_code == 404


def test_the_master_evaluation_is_not_implemented_yet(traced: TestClient) -> None:
    response = traced.get("/eval/master")
    assert response.status_code == 501
    assert "SP4" in response.json()["detail"]

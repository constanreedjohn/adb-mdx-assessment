"""E4-04: one session gives one complete trace, with the real agents in it.

The session runs every kind of work: Master planning, the crucial steps, explanations,
a student who calls tools, a teacher who sends a flag back, the send-back round, the
cases and the run opinion. Every graph node, task, tool call (MCP and investigation),
review and model call must be a span of the session's one trace, and every model call
a generation with model, prompt version and token usage.
"""

from collections import Counter
from typing import Any

from mlflow.tracing.constant import SpanAttributeKey
from tests.support import CapturedTraces
from tests.unit.mdx.test_send_back_round import student as investigating_student
from tests.unit.mdx.test_sessions_api import BODY, client_for, config, settings, valid_plan  # noqa: F401
from tests.unit.mdx.test_teacher import JPY_3M, teacher

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.settings import Settings
from mdx_assessment.usecases.mdx.agents.escalation import RecordCasesResult

PROMPTS = ("master_plan", "mda_explain", "mda_investigate", "master_validate", "master_opinion")


def run_traced(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
    captured: CapturedTraces,
) -> tuple[Any, dict[str, Any]]:
    tracer, traces = captured
    extra = {
        "mda_investigate": investigating_student,
        "master_validate": teacher({JPY_3M: "Is this point flagged often?"}),
    }
    client = client_for(config, settings, valid_plan(config), tracer, extra=extra)
    started = client.post("/sessions", json=BODY).json()
    session = client.get(f"/sessions/{started['session_id']}").json()
    [trace] = traces  # one session, one trace
    return trace, session


def test_one_session_gives_one_complete_trace(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
    captured_traces: CapturedTraces,
) -> None:
    trace, session = run_traced(config, settings, captured_traces)
    spans = trace.data.spans
    names = Counter(s.name for s in spans)
    steps = [(e["step_id"], e["task"]) for e in session["execution"]]
    assert [t for _, t in steps] == [
        "input_validation.check_files_and_intake",
        "market_data.near_zero_check",
        "market_data.revise_drafts",
        "escalation_reporting.record_cases",
    ]

    # Every graph node, every task and every review of the drafts.
    for node in ("run_context", "select_playbook", "master_plan", "master_synthesize"):
        assert names[f"node:{node}"] == 1
    assert names["node:dispatch"] == names["node:review_point"] == len(steps)
    for _, task in steps:
        assert names[f"task:{task}"] == 1
    assert (
        names["validate:market_data.near_zero_check"]
        == names["validate:market_data.revise_drafts"]
        == 1
    )

    # Every tool call: MCP tools, and the investigation tools the student chose.
    report = RecordCasesResult.model_validate(session["agent_results"]["s2"]).report
    assert names["exception_cases.create_case"] == report.cases_created == 28
    investigation = {n: c for n, c in names.items() if n.startswith("investigation.")}
    assert investigation == {f"investigation.{t}": c for t, c in report.tools_by_name.items()}
    assert sum(investigation.values()) == report.tools_called == 2
    assert names["mdx_data.get_flags"] >= 1 and names["mdx_data.get_history"] >= 28

    # Every model call, as a generation with model, prompt version and token usage.
    generations = [s for s in spans if s.name.startswith("llm:")]
    assert {s.name.removeprefix("llm:") for s in generations} == set(PROMPTS)
    assert names["llm:master_validate"] == 2  # the first drafts, then the revision
    for span in generations:
        assert span.span_type == "LLM"
        assert (
            span.attributes["prompt_version"]
            == session["versions"]["prompts"][span.name.removeprefix("llm:")]
        )
        assert span.attributes[SpanAttributeKey.MODEL]
        assert span.attributes.get(SpanAttributeKey.CHAT_USAGE), span.name

    # One tree: every span hangs off the session's root, and the session links to it.
    ids = {s.span_id for s in spans}
    [root] = [s for s in spans if s.parent_id is None]
    assert root.name == "assessment_session"
    assert all(s.parent_id in ids for s in spans if s is not root)
    assert session["trace"]["mlflow_trace_id"] == trace.info.trace_id
    # The only error is expected: flag_frequency asks MDX for a past day's flags, and the
    # simulated MDX keeps none (E8-03), so it re-applies the configured rule instead.
    errors = [(s.name, s.inputs) for s in spans if s.status.status_code == "ERROR"]
    assert errors == [
        ("mdx_data.get_flags", {"business_date": "2026-10-14", "data_type": "linear"})
    ]

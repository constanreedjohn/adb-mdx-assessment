"""E10-02: the Master's run opinion, with a scripted fake model.

Acceptance: the summary names the GBP move as the item to look at first and the JPY
item as an artifact, with citations.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.support import opinion_from_inputs
from tests.unit.mdx.test_sessions_api import BODY, client_for, config, settings, valid_plan  # noqa: F401
from tests.unit.mdx.test_teacher import CONFIG, GBP_2Y, JPY_3M, data_dir, session, teacher  # noqa: F401

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.gateway import LLMRequest, check_grounding
from mdx_assessment.core.master.opinion import OpinionOutput
from mdx_assessment.settings import Settings
from mdx_assessment.usecases.mdx.agents.opinion import opinion_material

pytestmark = pytest.mark.anyio
COUNTS = (
    "28 flags assessed: 1 likely real, 0 can't tell, 27 likely harmless "
    "(4 near-zero threshold artifacts, 23 ordinary moves)."
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def run(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
    opinion: Any = None,
    **body: Any,
) -> dict[str, Any]:
    extra = {"master_opinion": opinion} if opinion is not None else None
    client: TestClient = client_for(config, settings, valid_plan(config), extra=extra)
    started = client.post("/sessions", json={**BODY, **body}).json()
    result: dict[str, Any] = client.get(f"/sessions/{started['session_id']}").json()
    return result


def evidence_of(session: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        e["evidence_id"]: e
        for r in session["agent_results"].values()
        for e in r.get("evidence", [])
    }


# --- acceptance -----------------------------------------------------------------------------------


def test_the_opinion_names_gbp_first_and_jpy_as_an_artifact_with_citations(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
) -> None:
    session = run(config, settings)
    opinion = session["run_opinion"]
    assert opinion["source"] == "llm" and opinion["fallback_reason"] is None
    assert opinion["review_first"] == [GBP_2Y]
    summary = opinion["summary"]
    assert summary.startswith("Review GBP 2Y first: it moved +18.00 bps.")
    assert "JPY 3M is a near-zero threshold artifact." in summary
    cited = [evidence_of(session)[i] for i in opinion["evidence_ids"]]
    assert {(e["kind"], e["data"]["flag_id"]) for e in cited} >= {
        ("near_zero_measures", GBP_2Y),
        ("near_zero_measures", JPY_3M),
    }
    assert session["versions"]["prompts"]["master_opinion"] == "v2"
    assert opinion["counts"] == COUNTS
    assert "opinion_template" not in " ".join(session["fallbacks"])


async def test_the_master_sees_figures_and_flags_in_codes_order(data_dir: Path) -> None:  # noqa: F811
    outcome = await session(data_dir)
    material = opinion_material(outcome.final, CONFIG)
    assert material is not None
    flags = material.inputs["flags"]
    assert isinstance(flags, list) and len(flags) == 28
    assert [f["item"] for f in flags[:1]] == ["GBP 2Y"]  # type: ignore[index]
    with_evidence = {f["item"] for f in flags if "evidence" in f}  # type: ignore[operator,index]
    assert with_evidence == {"GBP 2Y", "JPY 3M", "JPY 1M", "CHF 1M", "CHF 6M"}  # notable only
    assert material.must_review == (GBP_2Y,)
    figures = material.inputs["figures"]
    assert isinstance(figures, dict) and figures["flags_assessed"] == 28


# --- the template ---------------------------------------------------------------------------------


def test_with_the_llm_off_code_writes_the_opinion(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
) -> None:
    session = run(config, settings, llm_enabled=False)
    opinion = session["run_opinion"]
    assert (opinion["source"], opinion["fallback_reason"]) == ("template_fallback", "llm_off")
    assert opinion["review_first"] == [GBP_2Y]
    summary = opinion["summary"]
    assert "Review first: GBP 2Y (+18.00 bps, z +4.80, likely real)." in summary
    assert "JPY 3M (MDX +75.0%, actually +3.00 bps)" in summary
    assert "The other flags are ordinary moves for their curves." in summary
    assert "Some verdicts were not validated by the Master." in opinion["uncertainties"]
    assert opinion["counts"] == COUNTS + " 28 not validated by the Master."


async def test_the_template_is_grounded_and_lists_cant_tell_flags(data_dir: Path) -> None:  # noqa: F811
    outcome = await session(data_dir, teacher({JPY_3M: "Is the move shared along the curve?"}))
    material = opinion_material(outcome.final, CONFIG)
    assert material is not None
    template = material.template
    assert template.review_first == (GBP_2Y, JPY_3M)
    assert "Can't tell, to review: JPY 3M" in template.summary
    assert any(u.startswith("JPY 3M: can't tell.") for u in template.uncertainties)
    assert check_grounding(template, material.bundle).ok


# --- guardrails -----------------------------------------------------------------------------------


def edited(edit: Any, seen: list[str]) -> Any:
    """Breaks the first reply with `edit`; records every request."""

    def answer(request: LLMRequest) -> dict[str, Any]:
        seen.append(request.user_message)
        reply = opinion_from_inputs(request)
        return edit(reply) if len(seen) == 1 else reply

    return answer


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda r: {**r, "review_first": []}, "must include '20261015-GBP-2Y'"),
        (lambda r: {**r, "review_first": [GBP_2Y, "20261015-XXX-1Y"]}, "unknown item"),
        (
            lambda r: {**r, "summary": r["summary"] + " The curve moved 45 bps."},
            "does not match",
        ),
        (lambda r: {**r, "evidence_ids": ["ev-0000000000000000"]}, "does not exist"),
        # v2, from the first live run:
        (
            lambda r: {**r, "summary": r["summary"] + " The other 27 flags are ordinary."},
            "does not match",
        ),
        (
            lambda r: {**r, "evidence_ids": r["evidence_ids"][:1]},
            "JPY 3M is named but its evidence isn't cited",
        ),
        (
            lambda r: {**r, "summary": r["summary"] + " EUR 3M moved a little."},
            "EUR 3M has no evidence to cite",
        ),
        (
            lambda r: {**r, "summary": r["summary"] + " Analysts should focus on GBP 2Y."},
            "gives the analyst advice",
        ),
        (
            lambda r: {**r, "uncertainties": ["You need to check the CHF curve."]},
            "'need to' gives the analyst advice",
        ),
    ],
    ids=[
        "drops-the-large-move",
        "invents-an-item",
        "invents-a-number",
        "cites-nothing-real",
        "quotes-a-count",
        "names-a-flag-uncited",
        "names-a-flag-without-evidence",
        "advises-in-the-summary",
        "advises-in-the-uncertainties",
    ],
)
def test_an_opinion_that_breaks_a_guardrail_is_retried(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
    edit: Any,
    reason: str,
) -> None:
    seen: list[str] = []
    session = run(config, settings, edited(edit, seen))
    opinion = session["run_opinion"]
    assert (opinion["source"], opinion["attempts"]) == ("llm", 2)
    assert opinion["review_first"] == [GBP_2Y]
    assert reason in seen[1]  # the retry says what was wrong


def test_twice_unusable_falls_back_to_the_template(
    config: LoadedConfig,  # noqa: F811
    settings: Settings,  # noqa: F811
) -> None:
    seen: list[str] = []

    def always_drops(request: LLMRequest) -> dict[str, Any]:
        seen.append(request.user_message)
        return {**opinion_from_inputs(request), "review_first": []}

    session = run(config, settings, always_drops)
    opinion = session["run_opinion"]
    assert (opinion["source"], opinion["fallback_reason"]) == (
        "template_fallback",
        "llm_output_rejected",
    )
    assert opinion["review_first"] == [GBP_2Y]
    assert "must include '20261015-GBP-2Y'" in seen[1]
    assert "opinion_template:llm_output_rejected" in session["fallbacks"]


def test_the_opinions_output_has_no_field_for_a_verdict_or_a_rank() -> None:
    assert set(OpinionOutput.model_fields) == {
        "summary",
        "review_first",
        "uncertainties",
        "evidence_ids",
    }

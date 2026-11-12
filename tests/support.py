"""Types and fakes shared by tests."""

import json
from typing import Any

from mlflow.entities import Trace

from mdx_assessment.core.gateway import LLMRequest
from mdx_assessment.core.observability import Tracer

CapturedTraces = tuple[Tracer, list[Trace]]


def explanation_items(request: LLMRequest) -> list[dict[str, Any]]:
    """The items an `mda_explain` request asks about, read back from its inputs."""
    text = request.user_message.removeprefix("Inputs (JSON):\n").split("\n\n")[0]
    items: list[dict[str, Any]] = json.loads(text)["items"]
    return items


def explain_from_inputs(request: LLMRequest) -> dict[str, Any]:
    """A fake `mda_explain` reply that quotes each item's own measures: always grounded."""
    out = []
    for item in explanation_items(request):
        m = next(e for e in item["evidence"] if e["kind"] == "near_zero_measures")
        data = m["data"]
        out.append(
            {
                "flag_id": item["flag_id"],
                "explanation": (
                    f"{item['item']} moved {data['change']:+.2f} bps against a normal daily "
                    f"move of {data['normal_daily_move']:.2f} bps (z {data['z']:+.2f})."
                ),
                "evidence_ids": [m["id"]],
            }
        )
    return {"explanations": out}


def investigation_flags(request: LLMRequest) -> list[dict[str, Any]]:
    """The flags an `mda_investigate` round asks about, read back from its inputs."""
    text = request.user_message.removeprefix("Inputs (JSON):\n").split("\n\n")[0]
    flags: list[dict[str, Any]] = json.loads(text)["flags"]
    return flags


def conclude(flag: dict[str, Any], verdict: str | None = None) -> dict[str, Any]:
    """A grounded conclusion: the explanation as the report, at the starting verdict."""
    return {
        "flag_id": flag["flag_id"],
        "action": "conclude",
        "tool_calls": [],
        "verdict": verdict or flag["starting_verdict"],
        "report": flag["explanation"]["text"],
        "evidence_ids": flag["explanation"]["evidence_ids"],
    }


def investigate_from_inputs(request: LLMRequest) -> dict[str, Any]:
    """A fake `mda_investigate` reply that concludes every flag at once: always valid."""
    return {"steps": [conclude(f) for f in investigation_flags(request)]}


def review_flags(request: LLMRequest) -> list[dict[str, Any]]:
    """The drafts a `master_validate` request asks about, read back from its inputs."""
    text = request.user_message.removeprefix("Inputs (JSON):\n").split("\n\n")[0]
    flags: list[dict[str, Any]] = json.loads(text)["flags"]
    return flags


def accept_all(request: LLMRequest) -> dict[str, Any]:
    """A fake `master_validate` reply that accepts every draft."""
    return {
        "reviews": [
            {"flag_id": f["flag_id"], "decision": "accept", "reason": "follows from the evidence"}
            for f in review_flags(request)
        ]
    }

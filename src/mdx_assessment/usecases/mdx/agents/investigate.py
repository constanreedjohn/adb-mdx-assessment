"""The student loop: investigate each flag, then draft its verdict (E8-04).

After the crucial steps (E8-01) and the explanations (E8-02), the Market Data
Assessment Agent (student) decides per flag whether the crucial results settle it or
whether to investigate further with the tools of E8-03.

- **Batched rounds** (prompt `mda_investigate/v1`). Each round is one call for every
  flag still open. Per flag the student either asks for tools, saying why, or concludes
  with a verdict and a report. Code runs the tools it asked for and feeds the evidence
  back in the next round.
- **Tool budget**: `max_tool_calls_per_flag` (4) per flag over every round, the
  send-back round of E6-05 included. Asking for tools after the budget is spent ends the
  flag at can't tell, or its starting verdict if that's more cautious.
- **Verdict rule** (`checks.verdict`): the starting verdict comes from code's label, and
  the student may move it only towards caution. Code checks every draft before anyone
  else sees it: the verdict rule, citations from that flag's own evidence, and every
  number in the report against that evidence. A draft that fails goes back once with
  the reasons. A second failure ends in a template.
- **Template fallback**: written by code, keeping the starting verdict. Used when the
  LLM is off (no tools are run), beyond `max_ai_explanations`, when the model can't be
  reached, or when a round's output is unusable twice.

The LLM's output has no field for a label, a severity, a measure or the review order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, JsonValue

from mdx_assessment.core.domain import DomainModel, Evidence
from mdx_assessment.core.gateway import LLMGateway, NumberTolerance, check_grounding
from mdx_assessment.core.tools.client import BudgetExceededError, ToolCallError
from mdx_assessment.usecases.mdx.agents.explanations import (
    FALLBACK_LLM_OFF,
    FALLBACK_OVER_LIMIT,
    FALLBACK_REJECTED,
    FALLBACK_UNAVAILABLE,
    Explanation,
    ItemToExplain,
    evidence_bundle,
)
from mdx_assessment.usecases.mdx.agents.ids import MARKET_DATA_ASSESSMENT
from mdx_assessment.usecases.mdx.checks.verdict import (
    Verdict,
    at_least,
    lowers,
    starting_verdict,
)
from mdx_assessment.usecases.mdx.config import InvestigationSettings
from mdx_assessment.usecases.mdx.domain import ExplanationSource
from mdx_assessment.usecases.mdx.investigation.toolkit import (
    POINT_HISTORY,
    TOOLS,
    Investigation,
    InvestigationError,
    catalogue,
)

PROMPT_ID = "mda_investigate"
PROMPT_VERSION = "v1"

FALLBACK_DRAFT_REJECTED = "draft_rejected"
FALLBACK_BUDGET_SPENT = "tool_budget_spent"
UNUSABLE = (FALLBACK_REJECTED, FALLBACK_UNAVAILABLE, FALLBACK_DRAFT_REJECTED)
"""Fallbacks that mean the LLM's work wasn't usable, so the step is `done_with_fallback`."""

_TOOL_IDS = {t.id for t in TOOLS}


# --- what the model returns ------------------------------------------------------------------


class ToolRequest(DomainModel):
    tool: str
    days: int | None
    """For `point_history` only; null otherwise."""
    reason: str


class StudentStep(DomainModel):
    """One flag's move in a round. No defaults: strict output schema."""

    flag_id: str
    action: Literal["investigate", "conclude"]
    tool_calls: tuple[ToolRequest, ...]
    verdict: Verdict | None
    report: str | None
    evidence_ids: tuple[str, ...]

    def cited_ids(self) -> tuple[str, ...]:
        return self.evidence_ids

    def grounded_texts(self) -> tuple[str, ...]:
        return (self.report or "",)


class StudentRound(DomainModel):
    steps: tuple[StudentStep, ...]


# --- what the step keeps ---------------------------------------------------------------------


class ToolCallRecord(DomainModel):
    round: int = Field(ge=1)
    tool: str
    arguments: dict[str, JsonValue]
    reason: str
    evidence_id: str | None = None
    error: str | None = None
    """Why the tool couldn't run; the call still counts against the budget."""


class Draft(DomainModel):
    """The student's draft for one flag: what the teacher (E10-05) validates."""

    flag_id: str
    starting_verdict: Verdict
    verdict: Verdict
    report: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    source: ExplanationSource
    fallback_reason: str | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    """The investigation trail: every tool asked for, why, and what it returned."""
    tool_calls_left: int = Field(ge=0)
    """What remains of the budget, for the send-back round (E6-05)."""
    rejections: tuple[str, ...] = ()
    """Why earlier drafts for this flag were sent back by code."""


@dataclass
class _Open:
    """A flag still being investigated."""

    item: ItemToExplain
    explanation: Explanation
    start: Verdict
    calls_left: int
    evidence: list[Evidence] = field(default_factory=list)
    calls: list[ToolCallRecord] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    @property
    def flag_id(self) -> str:
        return self.item.assessment.flag_id

    def all_evidence(self) -> list[Evidence]:
        return [*self.item.evidence, *self.evidence]


@dataclass(frozen=True)
class InvestigateOutcome:
    drafts: dict[str, Draft]
    """By flag ID; one per item."""
    tool_evidence: tuple[Evidence, ...] = ()
    """Every evidence record the tools returned."""
    rounds: int = 0
    llm_attempts: int = 0
    rejections: tuple[str, ...] = ()
    """Why rounds were rejected by the gateway's structural check."""


# --- templates -------------------------------------------------------------------------------

_VERDICT_WORDS = {
    Verdict.LIKELY_REAL: "likely real",
    Verdict.CANT_TELL: "can't tell",
    Verdict.LIKELY_HARMLESS: "likely harmless",
}


def template_report(explanation: Explanation, verdict: Verdict, why: str) -> str:
    return f"{explanation.text} Verdict: {_VERDICT_WORDS[verdict]} ({why})."


def _template(
    o: _Open, reason: str, verdict: Verdict | None = None, why: str | None = None
) -> Draft:
    v = verdict or o.start
    return Draft(
        flag_id=o.flag_id,
        starting_verdict=o.start,
        verdict=v,
        report=template_report(o.explanation, v, why or "from code's label, not investigated"),
        evidence_ids=o.explanation.evidence_ids,
        source=ExplanationSource.TEMPLATE,
        fallback_reason=reason,
        tool_calls=tuple(o.calls),
        tool_calls_left=o.calls_left,
        rejections=tuple(o.rejections),
    )


# --- one round -------------------------------------------------------------------------------


def round_inputs(
    open_flags: Sequence[_Open], settings: InvestigationSettings
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    untrusted: dict[str, str] = {}
    rows: list[JsonValue] = []
    for n, o in enumerate(open_flags, start=1):
        a = o.item.assessment
        row: dict[str, JsonValue] = {
            "flag_id": a.flag_id,
            "item": a.item_key,
            "label": a.classification.value,
            "starting_verdict": o.start.value,
            "rule_trace": list(a.reasons),
            "explanation": {
                "text": o.explanation.text,
                "evidence_ids": list(o.explanation.evidence_ids),
            },
            "evidence": [
                {"id": e.evidence_id, "kind": e.kind, "data": e.data} for e in o.all_evidence()
            ],
            "tool_calls_left": o.calls_left,
            "tool_results": [
                {
                    "tool": c.tool,
                    "arguments": c.arguments,
                    "reason": c.reason,
                    "evidence_id": c.evidence_id,
                    "error": c.error,
                }
                for c in o.calls
            ],
        }
        if o.rejections:
            row["draft_sent_back"] = o.rejections[-1]
        if o.item.comment:
            name = f"comment_{n}"
            untrusted[name] = o.item.comment
            row["mdx_comment"] = f"see the untrusted block {name!r}"
        rows.append(row)
    return {"tools": catalogue(settings), "flags": rows}, untrusted  # type: ignore[dict-item]


def check_round(output: StudentRound, open_flags: Sequence[_Open]) -> list[str]:
    """Structure only: every open flag answered once, and each answer complete. What a
    conclusion says is checked per flag afterwards, so one flag can't sink the round."""
    wanted = {o.flag_id for o in open_flags}
    seen = Counter(s.flag_id for s in output.steps)
    problems = [f"{f}: no answer was given" for f in sorted(wanted - set(seen))]
    problems += [f"{f}: not a flag to investigate" for f in sorted(set(seen) - wanted)]
    problems += [f"{f}: answered {n} times" for f, n in seen.items() if n > 1 and f in wanted]
    for s in output.steps:
        if s.action == "investigate":
            if not s.tool_calls:
                problems.append(f"{s.flag_id}: investigate needs at least one tool call")
            for t in s.tool_calls:
                if t.tool not in _TOOL_IDS:
                    problems.append(
                        f"{s.flag_id}: unknown tool {t.tool!r}; choose from {sorted(_TOOL_IDS)}"
                    )
                elif t.tool == POINT_HISTORY and t.days is None:
                    problems.append(f"{s.flag_id}: {POINT_HISTORY} needs `days`")
        elif s.verdict is None or not (s.report or "").strip():
            problems.append(f"{s.flag_id}: a conclusion needs a verdict and a report")
    return problems


def check_conclusion(step: StudentStep, o: _Open, tolerance: NumberTolerance) -> list[str]:
    """The verdict rule, then citations and numbers against the flag's own evidence."""
    assert step.verdict is not None
    problems = []
    if lowers(o.start, step.verdict):
        problems.append(
            f"verdict {step.verdict.value} is less cautious than the starting verdict "
            f"{o.start.value}: a verdict may only move towards caution"
        )
    bundle = evidence_bundle(o.all_evidence(), o.item.units, tolerance)
    problems += list(check_grounding(step, bundle).errors)
    return problems


async def _run_tools(
    step: StudentStep, o: _Open, investigation: Investigation, round_no: int
) -> list[Evidence]:
    added: list[Evidence] = []
    for request in step.tool_calls:
        arguments: dict[str, JsonValue] = {"flag_id": o.flag_id}
        if request.tool == POINT_HISTORY:
            arguments["days"] = request.days
        if o.calls_left == 0:
            break  # the rest are over the budget: never run
        o.calls_left -= 1
        record = ToolCallRecord(
            round=round_no, tool=request.tool, arguments=arguments, reason=request.reason
        )
        try:
            evidence = await investigation.run(request.tool, arguments)
        except (InvestigationError, ToolCallError, BudgetExceededError) as exc:
            record = record.model_copy(update={"error": str(exc)})
        else:
            record = record.model_copy(update={"evidence_id": evidence.evidence_id})
            if evidence.evidence_id not in {e.evidence_id for e in o.all_evidence()}:
                o.evidence.append(evidence)
            added.append(evidence)
        o.calls.append(record)
    return added


# --- the loop --------------------------------------------------------------------------------


async def investigate(
    items: Sequence[ItemToExplain],
    explanations: Mapping[str, Explanation],
    *,
    gateway: LLMGateway | None,
    llm_enabled: bool,
    session_id: str,
    investigation: Investigation | None,
    settings: InvestigationSettings,
    max_ai_explanations: int,
    tolerance: NumberTolerance,
) -> InvestigateOutcome:
    flags = [
        _Open(
            item=i,
            explanation=explanations[i.assessment.flag_id],
            start=starting_verdict(i.assessment.classification),
            calls_left=settings.max_tool_calls_per_flag,
        )
        for i in items
    ]
    if not llm_enabled or gateway is None or investigation is None:
        return InvestigateOutcome({o.flag_id: _template(o, FALLBACK_LLM_OFF) for o in flags})

    drafts = {o.flag_id: _template(o, FALLBACK_OVER_LIMIT) for o in flags[max_ai_explanations:]}
    open_flags = flags[:max_ai_explanations]
    tool_evidence: list[Evidence] = []
    rejections: list[str] = []
    attempts = rounds = 0
    # Every round either spends a tool call or a flag's one send-back, so this always ends.
    max_rounds = settings.max_tool_calls_per_flag + 2

    while open_flags and rounds < max_rounds:
        rounds += 1
        inputs, untrusted = round_inputs(open_flags, settings)
        current = list(open_flags)
        result = await gateway.call(
            PROMPT_ID,
            PROMPT_VERSION,
            inputs=inputs,
            output_model=StudentRound,
            actor=MARKET_DATA_ASSESSMENT,
            session_id=session_id,
            untrusted=untrusted,
            check=lambda r, flags=current: check_round(r, flags),  # type: ignore[misc]
            fallback=lambda: StudentRound(steps=()),
        )
        attempts += result.attempts
        rejections += result.rejections
        if result.source != "llm":
            reason = FALLBACK_REJECTED if result.attempts > 1 else FALLBACK_UNAVAILABLE
            drafts |= {o.flag_id: _template(o, reason) for o in current}
            open_flags = []
            break

        by_flag = {s.flag_id: s for s in result.output.steps}
        still_open: list[_Open] = []
        for o in current:
            step = by_flag[o.flag_id]
            if step.action == "investigate":
                if o.calls_left == 0:
                    drafts[o.flag_id] = _template(
                        o,
                        FALLBACK_BUDGET_SPENT,
                        at_least(o.start, Verdict.CANT_TELL),
                        f"the {settings.max_tool_calls_per_flag} tool calls were spent "
                        "without a conclusion",
                    )
                    continue
                tool_evidence += await _run_tools(step, o, investigation, rounds)
                still_open.append(o)
                continue
            problems = check_conclusion(step, o, tolerance)
            if not problems:
                assert step.verdict is not None and step.report is not None
                drafts[o.flag_id] = Draft(
                    flag_id=o.flag_id,
                    starting_verdict=o.start,
                    verdict=step.verdict,
                    report=step.report.strip(),
                    evidence_ids=step.evidence_ids,
                    source=ExplanationSource.LLM,
                    tool_calls=tuple(o.calls),
                    tool_calls_left=o.calls_left,
                    rejections=tuple(o.rejections),
                )
            elif not o.rejections:
                o.rejections.append("; ".join(problems))
                still_open.append(o)  # sent back once, with the reasons
            else:
                o.rejections.append("; ".join(problems))
                drafts[o.flag_id] = _template(o, FALLBACK_DRAFT_REJECTED)
        open_flags = still_open

    for o in open_flags:  # out of rounds: treated as a spent budget
        drafts[o.flag_id] = _template(
            o,
            FALLBACK_BUDGET_SPENT,
            at_least(o.start, Verdict.CANT_TELL),
            "the investigation did not conclude",
        )
    unique = tuple({e.evidence_id: e for e in tool_evidence}.values())
    return InvestigateOutcome(drafts, unique, rounds, attempts, tuple(rejections))

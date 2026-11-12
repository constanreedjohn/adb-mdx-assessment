"""The teacher: the Master Agent validates every flag's draft verdict (E10-05).

Runs at the review point after the Market Data Assessment step, and again after its
send-back round (`market_data.revise_drafts`), registered as those tasks' validator.

1. **Code first.** Every draft is checked again: the verdict rule (never less cautious
   than code's starting verdict), citations from that flag's own evidence only, and
   every number in the report against that evidence. A draft that fails never reaches
   the teacher: it ends unresolved.
2. **The teacher** (prompt `master_validate/v1`, one batched call) decides per flag:
   `accept`, or `send_back` with a reason. The reason says what is missing or doubtful
   and names no tool: the student chooses how to address it.
3. **Send-back.** Flags sent back after the first drafts go back to the student once
   (E6-05). After the send-back round there is no second one: the teacher accepts a
   revised draft, or the flag ends unresolved. A send-back the playbook can't run also
   ends its flags unresolved.
4. **No usable review** (LLM off, model unreachable, output unusable twice): the drafts
   are kept and marked not validated.

`validated_drafts` gives each flag's outcome: an unresolved flag ends at can't tell, or
its draft's verdict if that is more cautious, so a likely real flag is never lowered.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from mdx_assessment.core.domain import (
    DecisionSource,
    DomainModel,
    Evidence,
    ItemDecision,
    ItemDecisionKind,
    PlanStep,
    StepValidation,
)
from mdx_assessment.core.gateway import NumberTolerance, check_grounding
from mdx_assessment.core.gateway.grounding import GUARDRAILS_PATH, GuardrailsConfig
from mdx_assessment.core.orchestration.graph import MASTER_AGENT, RAN
from mdx_assessment.core.orchestration.state import ExecutionEntry
from mdx_assessment.core.orchestration.tasks import ValidationContext, ValidationOutcome
from mdx_assessment.usecases.mdx.agents.explanations import ItemToExplain, evidence_bundle
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    NEAR_ZERO_CHECK,
    REVISE_DRAFTS,
)
from mdx_assessment.usecases.mdx.agents.input_validation import IntakeResult
from mdx_assessment.usecases.mdx.agents.investigate import Draft, StudentStep
from mdx_assessment.usecases.mdx.agents.market_data import (
    MarketAssessmentResult,
    RevisionResult,
    current_drafts,
    item_to_explain,
)
from mdx_assessment.usecases.mdx.checks.verdict import Verdict, at_least, lowers
from mdx_assessment.usecases.mdx.investigation.toolkit import TOOLS

PROMPT_ID = "master_validate"
PROMPT_VERSION = "v1"
_TOOL_IDS = tuple(t.id for t in TOOLS)


class FlagReview(DomainModel):
    """The teacher's decision on one draft. No defaults: strict output schema."""

    flag_id: str
    decision: Literal["accept", "send_back"]
    reason: str


class TeacherReview(DomainModel):
    reviews: tuple[FlagReview, ...]


@dataclass(frozen=True)
class _Flag:
    item: ItemToExplain
    draft: Draft
    evidence: tuple[Evidence, ...]
    """The crucial evidence and everything the draft's tools returned."""

    @property
    def flag_id(self) -> str:
        return self.draft.flag_id


# --- what is under review --------------------------------------------------------------------


def _records(results: Mapping[str, Mapping[str, JsonValue]]) -> dict[str, Evidence]:
    """Every evidence record the run's results hold, by ID."""
    records: dict[str, Evidence] = {}
    for result in results.values():
        for raw in result.get("evidence") or []:  # type: ignore[union-attr]
            e = Evidence.model_validate(raw)
            records[e.evidence_id] = e
    return records


def _under_review(ctx: ValidationContext) -> list[_Flag]:
    state, entry = ctx.state, ctx.entry
    result = state.agent_results.get(entry.step_id)
    intake_raw = next(
        (
            state.agent_results.get(e.step_id)
            for e in reversed(state.execution)
            if e.task == CHECK_FILES_AND_INTAKE and e.status in RAN
        ),
        None,
    )
    if result is None or intake_raw is None:
        return []
    intake = IntakeResult.model_validate(intake_raw)
    records = _records(state.agent_results)
    if entry.task == NEAR_ZERO_CHECK:
        assessed = MarketAssessmentResult.model_validate(result)
        items = [(i, i.draft) for i in assessed.items]
    else:
        revision = RevisionResult.model_validate(result)
        original = MarketAssessmentResult.model_validate(state.agent_results[revision.of_step])
        by_flag = {i.assessment.flag_id: i for i in original.items}
        items = [(by_flag[d.flag_id], d) for d in revision.drafts]
    flags = []
    for assessed_item, draft in items:
        item = item_to_explain(assessed_item, intake, records)
        tools = tuple(records[c.evidence_id] for c in draft.tool_calls if c.evidence_id)
        flags.append(_Flag(item, draft, (*item.evidence, *tools)))
    return flags


def code_check(flag: _Flag, tolerance: NumberTolerance) -> list[str]:
    """The verdict rule, citations and numbers, before the teacher sees the draft."""
    d = flag.draft
    problems = []
    if lowers(d.starting_verdict, d.verdict):
        problems.append(
            f"verdict {d.verdict.value} is less cautious than the starting verdict "
            f"{d.starting_verdict.value}"
        )
    known = {e.evidence_id for e in flag.evidence}
    problems += [f"cites {i!r}, not this flag's evidence" for i in d.evidence_ids if i not in known]
    step = StudentStep(
        flag_id=d.flag_id,
        action="conclude",
        tool_calls=(),
        verdict=d.verdict,
        report=d.report,
        evidence_ids=tuple(i for i in d.evidence_ids if i in known),
    )
    bundle = evidence_bundle(flag.evidence, flag.item.units, tolerance)
    problems += list(check_grounding(step, bundle).errors)
    return problems


def review_inputs(
    flags: Sequence[_Flag], revision: bool, can_send_back: bool
) -> dict[str, JsonValue]:
    rows: list[JsonValue] = []
    for f in flags:
        d, a = f.draft, f.item.assessment
        cited = [e for e in f.evidence if e.evidence_id in set(d.evidence_ids)]
        rows.append(
            {
                "flag_id": d.flag_id,
                "item": a.item_key,
                "label": a.classification.value,
                "starting_verdict": d.starting_verdict.value,
                "verdict": d.verdict.value,
                "report": d.report,
                "report_source": d.source.value,
                "evidence": [{"id": e.evidence_id, "kind": e.kind, "data": e.data} for e in cited],
                "tools_called": [
                    {"tool": c.tool, "reason": c.reason, "error": c.error} for c in d.tool_calls
                ],
                "tool_calls_left": d.tool_calls_left,
                **({"earlier_feedback": d.teacher_feedback} if d.teacher_feedback else {}),
            }
        )
    return {
        "round": "revision" if revision else "first_drafts",
        "send_back_available": can_send_back,
        "flags": rows,
    }


def check_review(output: TeacherReview, flags: Sequence[_Flag]) -> list[str]:
    wanted = {f.flag_id for f in flags}
    seen = Counter(r.flag_id for r in output.reviews)
    problems = [f"{f}: no decision was given" for f in sorted(wanted - set(seen))]
    problems += [f"{f}: not a draft under review" for f in sorted(set(seen) - wanted)]
    problems += [f"{f}: decided {n} times" for f, n in seen.items() if n > 1 and f in wanted]
    for r in output.reviews:
        if r.decision == "send_back":
            if not r.reason.strip():
                problems.append(f"{r.flag_id}: a send-back needs a reason")
            named = [t for t in _TOOL_IDS if t in r.reason]
            if named:
                problems.append(
                    f"{r.flag_id}: the reason names tool(s) {named}; say what is missing or "
                    "doubtful, and leave the choice of tools to the agent"
                )
    return problems


# --- the validator ---------------------------------------------------------------------------


def _summary(decisions: Sequence[ItemDecision]) -> str:
    counts = Counter(d.kind for d in decisions)
    parts = [f"{counts[k]} {k.value.replace('_', ' ')}" for k in ItemDecisionKind if counts[k]]
    return f"{len(decisions)} draft(s) reviewed: " + ", ".join(parts) if decisions else "no drafts"


async def validate_drafts(ctx: ValidationContext) -> ValidationOutcome:
    entry = ctx.entry
    revision = entry.task == REVISE_DRAFTS
    flags = _under_review(ctx)
    tolerance = ctx.config.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching

    decisions: dict[str, ItemDecision] = {}
    to_teacher: list[_Flag] = []
    for f in flags:
        problems = code_check(f, tolerance)
        if problems:
            decisions[f.flag_id] = ItemDecision(
                item_id=f.flag_id,
                kind=ItemDecisionKind.UNRESOLVED,
                reason="code check failed: " + "; ".join(problems),
                source=DecisionSource.RULE,
                revision=f.draft.revision,
            )
        else:
            to_teacher.append(f)

    def ordered(validated: bool, attempts: int = 0) -> StepValidation:
        listed = tuple(decisions[f.flag_id] for f in flags)
        return StepValidation(
            step_id=entry.step_id,
            validated=validated,
            summary=_summary(listed),
            decisions=listed,
            llm_attempts=attempts,
        )

    def not_validated(why: str) -> None:
        for f in to_teacher:
            decisions[f.flag_id] = ItemDecision(
                item_id=f.flag_id,
                kind=ItemDecisionKind.NOT_VALIDATED,
                reason=why,
                source=DecisionSource.RULE_FALLBACK,
                revision=f.draft.revision,
            )

    gateway = ctx.gateway
    if not to_teacher:
        return ValidationOutcome(ordered(validated=bool(flags)))
    if not ctx.state.llm_enabled or gateway is None:
        not_validated("the LLM is off for this run: the drafts stand, not validated")
        return ValidationOutcome(ordered(validated=False))

    probe = PlanStep(
        step_id=f"{entry.step_id}_revise",
        task=REVISE_DRAFTS,
        arguments={"of_step": entry.step_id, "flags": {}},
    )
    refusal = "the send-back round has already run" if revision else ctx.send_back_refusal(probe)
    result = await gateway.call(
        PROMPT_ID,
        PROMPT_VERSION,
        inputs=review_inputs(to_teacher, revision, refusal is None),
        output_model=TeacherReview,
        actor=MASTER_AGENT,
        session_id=ctx.state.session_id,
        check=lambda r: check_review(r, to_teacher),
        fallback=lambda: TeacherReview(reviews=()),
    )
    if result.source != "llm":
        why = "rejected twice" if result.attempts > 1 else "unreachable"
        not_validated(f"the Master's review was {why}: the drafts stand, not validated")
        return ValidationOutcome(ordered(validated=False, attempts=result.attempts))

    sent_back: dict[str, JsonValue] = {}
    for r in result.output.reviews:
        f = next(x for x in to_teacher if x.flag_id == r.flag_id)
        if r.decision == "accept":
            kind, reason = ItemDecisionKind.ACCEPT, r.reason
        elif refusal is None:
            kind, reason = ItemDecisionKind.SEND_BACK, r.reason
            sent_back[r.flag_id] = r.reason
        else:
            kind, reason = ItemDecisionKind.UNRESOLVED, f"{r.reason} (no send-back: {refusal})"
        decisions[r.flag_id] = ItemDecision(
            item_id=r.flag_id,
            kind=kind,
            reason=reason,
            source=DecisionSource.LLM_REVIEW,
            revision=f.draft.revision,
        )
    validation = ordered(validated=True, attempts=result.attempts)
    if not sent_back:
        return ValidationOutcome(validation)
    step = probe.model_copy(
        update={
            "arguments": {"of_step": entry.step_id, "flags": sent_back},
            "rationale": validation.summary,
        }
    )
    return ValidationOutcome(validation, send_back=step)


# --- the outcome per flag --------------------------------------------------------------------


class ValidatedDraft(DomainModel):
    """A flag's latest draft and what the teacher made of it."""

    draft: Draft
    decision: ItemDecisionKind | None
    """The latest decision on the flag; None if no review ran (e.g. the step failed)."""
    verdict: Verdict
    """The verdict the flag ends with."""
    validated: bool


def validated_drafts(
    execution: Sequence[ExecutionEntry],
    results: Mapping[str, Mapping[str, JsonValue]],
    validations: Sequence[StepValidation],
) -> dict[str, ValidatedDraft]:
    """Each flag's outcome, in review order. Unresolved, or sent back with no revision to
    review, ends at can't tell or the draft's verdict if that is more cautious."""
    latest: dict[str, ItemDecisionKind] = {}
    for v in validations:
        for d in v.decisions:
            latest[d.item_id] = d.kind
    out = {}
    for flag_id, draft in current_drafts(execution, results).items():
        kind = latest.get(flag_id)
        if kind in (ItemDecisionKind.UNRESOLVED, ItemDecisionKind.SEND_BACK):
            verdict = at_least(draft.verdict, Verdict.CANT_TELL)
        else:
            verdict = draft.verdict
        out[flag_id] = ValidatedDraft(
            draft=draft,
            decision=kind,
            verdict=verdict,
            validated=kind is not None and kind is not ItemDecisionKind.NOT_VALIDATED,
        )
    return out

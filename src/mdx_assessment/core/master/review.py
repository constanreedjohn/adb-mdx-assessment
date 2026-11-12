"""The Master Agent's review point: decide after every step (plan §2.6, option C; E10-01).

Code decides whenever it can; the LLM only when a check fails:

| The step                        | Decision           | Source          |
| ------------------------------- | ------------------ | --------------- |
| failed, or skipped              | `flag_for_human`   | `rule`          |
| ran, every check passed         | `accept`           | `rule`          |
| ran, a check failed             | the Master decides | `llm_review`    |
| … and the review isn't usable   | `flag_for_human`   | `rule_fallback` |

The Master's review (prompt `master_review/v1`) sees the step and every check result,
and may only `accept` or `flag_for_human`: Normal EOD allows no follow-ups in Phase 1.
Its reason must cite the failed checks, and any number in it must match their details.
When the review can't be used (LLM off, unreachable, or invalid twice), the step is
flagged: a human looks whenever the Master can't vouch for it.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import JsonValue

from mdx_assessment.core.domain import (
    Decision,
    DecisionKind,
    DecisionSource,
    DomainModel,
    ReviewCheckResult,
    TaskStatus,
)
from mdx_assessment.core.gateway import EvidenceBundle, EvidenceItem, LLMError, LLMGateway
from mdx_assessment.core.gateway.numbers import NumberTolerance, Quantity, extract_numbers
from mdx_assessment.core.orchestration.graph import MASTER_AGENT, DecideHook, SessionRuntime
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState

logger = logging.getLogger(__name__)

PROMPT_ID = "master_review"
PROMPT_VERSION = "v1"


class MasterReviewOutput(DomainModel):
    """What the Master returns. `follow_up` is deliberately absent until SP9 (E10-08)."""

    decision: Literal["accept", "flag_for_human"]
    reason: str
    cited_check_ids: tuple[str, ...]

    def cited_ids(self) -> tuple[str, ...]:
        return self.cited_check_ids

    def grounded_texts(self) -> tuple[str, ...]:
        return (self.reason,)


def _decision(
    entry: ExecutionEntry,
    kind: DecisionKind,
    reason: str,
    source: DecisionSource,
    checks: list[ReviewCheckResult] | None = None,
) -> Decision:
    evidence = tuple(dict.fromkeys(i for c in checks or [] for i in c.evidence_ids))
    return Decision(
        step_id=entry.step_id, kind=kind, reason=reason, source=source, evidence_ids=evidence
    )


def review_inputs(
    state: RunState, entry: ExecutionEntry, checks: list[ReviewCheckResult], description: str
) -> dict[str, JsonValue]:
    return {
        "business_date": state.business_date.isoformat(),
        "data_type": state.data_type,
        "step": {
            "step_id": entry.step_id,
            "task": entry.task,
            "description": description,
            "status": entry.status.value,
        },
        "checks": [
            {"id": c.check_id, "name": c.name, "passed": c.passed, "details": c.details}
            for c in checks
        ],
    }


def review_evidence(
    checks: list[ReviewCheckResult], tolerance: NumberTolerance | None
) -> EvidenceBundle:
    """The checks are the evidence: their IDs are citable, the numbers in their details quotable."""
    return EvidenceBundle.of(
        [
            EvidenceItem(
                c.check_id, tuple(Quantity(n.value, n.unit) for n in extract_numbers(c.details))
            )
            for c in checks
        ],
        tolerance=tolerance,
    )


def master_review_hook(
    runtime: SessionRuntime, tolerance: NumberTolerance | None = None
) -> DecideHook:
    """The graph's `decide` hook. `tolerance` is `config/guardrails.yaml`'s `number_matching`."""
    gateway: LLMGateway | None = runtime.gateway

    async def decide(
        state: RunState, entry: ExecutionEntry, checks: list[ReviewCheckResult]
    ) -> Decision:
        if entry.status is TaskStatus.FAILED:
            reason = f"the step failed: {entry.error}"
            return _decision(entry, DecisionKind.FLAG_FOR_HUMAN, reason, DecisionSource.RULE)
        if entry.status is TaskStatus.SKIPPED:
            reason = f"the step was {entry.error}"
            return _decision(entry, DecisionKind.FLAG_FOR_HUMAN, reason, DecisionSource.RULE)

        failed = [c for c in checks if not c.passed]
        if not failed:
            reason = "every check passed" if checks else "the step ran; it has no review checks"
            if entry.status is TaskStatus.DONE_WITH_FALLBACK:
                reason += "; part of its output is a template fallback"
            return _decision(entry, DecisionKind.ACCEPT, reason, DecisionSource.RULE, checks)

        names = ", ".join(f"{c.check_id} ({c.details})" for c in failed)
        if not state.llm_enabled or gateway is None:
            reason = f"failed checks: {names}. No Master review: the LLM is off."
            return _decision(
                entry, DecisionKind.FLAG_FOR_HUMAN, reason, DecisionSource.RULE_FALLBACK, failed
            )

        definition = runtime.tasks.get(entry.task)
        try:
            result = await gateway.call(
                PROMPT_ID,
                PROMPT_VERSION,
                inputs=review_inputs(
                    state, entry, checks, definition.description if definition else ""
                ),
                output_model=MasterReviewOutput,
                actor=MASTER_AGENT,
                session_id=state.session_id,
                evidence=review_evidence(checks, tolerance),
                check=lambda out: [
                    f"{i} is not a failed check"
                    for i in out.cited_check_ids
                    if i not in {c.check_id for c in failed}
                ],
            )
        except LLMError as exc:
            logger.warning(
                "session %s step %s: Master review not usable (%s); flagged for a human",
                state.session_id,
                entry.step_id,
                exc,
            )
            reason = f"failed checks: {names}. The Master's review wasn't usable: {exc}"[:500]
            return _decision(
                entry, DecisionKind.FLAG_FOR_HUMAN, reason, DecisionSource.RULE_FALLBACK, failed
            )

        output = result.output
        kind = DecisionKind(output.decision)
        logger.info(
            "session %s step %s: Master review after failed %s: %s (%s)",
            state.session_id,
            entry.step_id,
            [c.check_id for c in failed],
            kind.value,
            output.reason,
        )
        return _decision(entry, kind, output.reason, DecisionSource.LLM_REVIEW, failed)

    return decide

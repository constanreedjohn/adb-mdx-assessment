"""The Master Agent's run opinion: one summary of the run for the analyst (E10-02).

The graph's `synthesize` hook, run at the end of every run that got past planning.

- **Inputs** (prompt `master_opinion/v1`): the execution log, the step decisions and the
  run's reliability, from core; the validated verdicts and the day's figures, from the
  use-case pack (`OpinionMaterial`, registered as `PlatformRegistries.opinion`).
- **Output**: `summary`, `review_first`, `uncertainties`, `evidence_ids`.
- **Guardrails**, through the gateway (v2, after the first live run):
  - every cited ID exists, and every number in the summary and the uncertainties
    matches the cited evidence. The day's counts are not quotable: code writes them
    (`OpinionMaterial.counts`, the opinion's `counts`), so a count can't be quoted with
    the wrong meaning ("27 ordinary" when 27 is the likely harmless total);
  - every item the text names cites its own evidence, and an item with none to cite
    isn't named (`OpinionMaterial.named_items`);
  - no advice to the analyst ("should", "recommend", "focus on", ...): the analyst decides;
  - `review_first` lists only known items and includes every item code ranked as a must
    (`OpinionMaterial.must_review`): code ranks, the Master explains.
- **Template fallback**: written by the pack from the same material, with the run
  review's reasons as uncertainties when reliability isn't high. Used with the LLM off,
  when the model is unreachable, or when the output is unusable twice.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import DomainModel, Reliability, RunOpinion
from mdx_assessment.core.gateway import EvidenceBundle, LLMGateway
from mdx_assessment.core.orchestration.graph import (
    MASTER_AGENT,
    SessionRuntime,
    SynthesizeHook,
    run_review,
)
from mdx_assessment.core.orchestration.state import RunState

PROMPT_ID = "master_opinion"
PROMPT_VERSION = "v2"
FALLBACK_LLM_OFF = "llm_off"
FALLBACK_REJECTED = "llm_output_rejected"
FALLBACK_UNAVAILABLE = "model_unavailable"


class OpinionOutput(DomainModel):
    """What the Master returns. No defaults: strict output schema."""

    summary: str
    review_first: tuple[str, ...]
    uncertainties: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def cited_ids(self) -> tuple[str, ...]:
        return self.evidence_ids

    def grounded_texts(self) -> tuple[str, ...]:
        return (self.summary, *self.uncertainties)


@dataclass(frozen=True)
class OpinionMaterial:
    """What the use-case pack gives the Master to write the opinion from."""

    inputs: dict[str, JsonValue]
    """The validated verdicts and the day's figures, as data."""
    bundle: EvidenceBundle
    """What the opinion may cite and quote: evidence records only, no counts."""
    known_items: frozenset[str]
    named_items: Mapping[str, tuple[str, ...]]
    """Each item's name as the text would write it (e.g. "JPY 3M") → the evidence IDs
    that may back it. Empty for an item with nothing to cite."""
    counts: str
    """The day's counts in one sentence, written by code."""
    must_review: tuple[str, ...]
    """Items code ranked to look at first: `review_first` must include them."""
    template: OpinionOutput
    """Code's own opinion, used when the LLM's isn't."""


OpinionBuilder = Callable[[RunState, LoadedConfig], OpinionMaterial | None]


def check_opinion(output: OpinionOutput, material: OpinionMaterial) -> list[str]:
    problems = [
        f"review_first: unknown item {i!r}"
        for i in output.review_first
        if i not in material.known_items
    ]
    problems += [
        f"review_first must include {i!r}: code ranks it to look at first"
        for i in material.must_review
        if i not in output.review_first
    ]
    dupes = sorted({i for i in output.review_first if output.review_first.count(i) > 1})
    problems += [f"review_first lists {i!r} more than once" for i in dupes]
    if not output.summary.strip():
        problems.append("the summary is empty")
    texts = (output.summary, *output.uncertainties)
    cited = set(output.evidence_ids)
    for name, evidence_ids in material.named_items.items():
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
        if not any(pattern.search(t) for t in texts):
            continue
        if not evidence_ids:
            problems.append(f"{name} has no evidence to cite: describe it without naming it")
        elif not cited & set(evidence_ids):
            problems.append(f"{name} is named but its evidence isn't cited: cite {evidence_ids[0]}")
    for text in texts:
        advice = _ADVICE.search(text)
        if advice:
            problems.append(
                f"{advice.group(0)!r} gives the analyst advice; state what the run found, "
                "and leave the decision to the analyst"
            )
    return problems


_ADVICE = re.compile(
    r"\b(should|must|recommend\w*|advis\w*|focus (?:on|their)|"
    r"need(?:s)? to|ought to)\b",
    re.IGNORECASE,
)


def opinion_inputs(state: RunState, material: OpinionMaterial) -> dict[str, JsonValue]:
    review = run_review(state)
    return {
        "business_date": state.business_date.isoformat(),
        "data_type": state.data_type,
        "plan_source": state.plan.source.value if state.plan else None,
        "execution": [
            {"step_id": e.step_id, "task": e.task, "status": e.status.value, "error": e.error}
            for e in state.execution
        ],
        "step_decisions": [
            {
                "step_id": d.step_id,
                "decision": d.kind.value,
                "source": d.source.value,
                "reason": d.reason,
            }
            for d in state.decisions
        ],
        "reliability": {"level": review.reliability.value, "reasons": list(review.reasons)},
        **material.inputs,
    }


def _with_run_reasons(template: OpinionOutput, state: RunState) -> OpinionOutput:
    review = run_review(state)
    if review.reliability is Reliability.HIGH:
        return template
    extra: Sequence[str] = [r for r in review.reasons if r not in template.uncertainties]
    return template.model_copy(update={"uncertainties": (*template.uncertainties, *extra)})


def master_opinion_hook(runtime: SessionRuntime, builder: OpinionBuilder | None) -> SynthesizeHook:
    gateway: LLMGateway | None = runtime.gateway

    async def synthesize(state: RunState) -> RunOpinion | None:
        if builder is None or runtime.config is None:
            return None
        material = builder(state, runtime.config)
        if material is None:
            return None
        template = _with_run_reasons(material.template, state)

        def as_opinion(
            out: OpinionOutput, source: str, reason: str | None, attempts: int
        ) -> RunOpinion:
            return RunOpinion(
                summary=out.summary.strip(),
                counts=material.counts,
                review_first=out.review_first,
                uncertainties=out.uncertainties,
                evidence_ids=out.evidence_ids,
                source=source,  # type: ignore[arg-type]
                fallback_reason=reason,
                attempts=attempts,
            )

        if not state.llm_enabled or gateway is None:
            return as_opinion(template, "template_fallback", FALLBACK_LLM_OFF, 0)
        result = await gateway.call(
            PROMPT_ID,
            PROMPT_VERSION,
            inputs=opinion_inputs(state, material),
            output_model=OpinionOutput,
            actor=MASTER_AGENT,
            session_id=state.session_id,
            evidence=material.bundle,
            check=lambda out: check_opinion(out, material),
            fallback=lambda: template,
        )
        if result.source == "llm":
            return as_opinion(result.output, "llm", None, result.attempts)
        reason = FALLBACK_REJECTED if result.attempts > 1 else FALLBACK_UNAVAILABLE
        return as_opinion(result.output, "template_fallback", reason, result.attempts)

    return synthesize

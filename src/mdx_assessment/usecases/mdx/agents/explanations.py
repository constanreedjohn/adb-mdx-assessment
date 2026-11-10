"""Explanations of the near-zero check, in plain words (E8-02).

The deterministic core has already labelled every item (E8-01). Here the LLM writes one
or two sentences per item saying why, and nothing else: its output has no field for a
label, a severity or a measure, so it can't change them.

- **One batched call** for every item (prompt `mda_explain/v1`), through the gateway.
- **Checked per item**: each explanation may cite only that item's evidence, and every
  number it states must match that evidence (unit-aware, `config/guardrails.yaml`).
  Every item must be explained exactly once. Any problem rejects the batch, which the
  gateway retries once with the reasons.
- **Template fallback**: if the retry fails too, or the model can't be reached, every item
  gets a template written by code from its measures. A template restates the code's
  label; it never decides one. Templates are used without a call when the LLM is off,
  and for items beyond the playbook's `max_ai_explanations`.
- **MDX comments** go to the model as untrusted content, never as instructions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import Field, JsonValue

from mdx_assessment.core.domain import DomainModel, Evidence, Finding
from mdx_assessment.core.gateway import (
    EvidenceBundle,
    EvidenceItem,
    LLMGateway,
    NumberTolerance,
    check_grounding,
)
from mdx_assessment.core.gateway import Unit as GroundingUnit
from mdx_assessment.usecases.mdx.agents.ids import MARKET_DATA_ASSESSMENT
from mdx_assessment.usecases.mdx.checks.near_zero import Classification, NearZeroAssessment
from mdx_assessment.usecases.mdx.domain import ExplanationSource, Unit

PROMPT_ID = "mda_explain"
PROMPT_VERSION = "v1"

FALLBACK_LLM_OFF = "llm_off"
FALLBACK_OVER_LIMIT = "over_ai_explanation_limit"
FALLBACK_REJECTED = "llm_output_rejected"
FALLBACK_UNAVAILABLE = "model_unavailable"

_GROUNDING_UNITS = {Unit.PERCENT: GroundingUnit.PERCENT, Unit.BPS: GroundingUnit.BPS}


class ItemExplanation(DomainModel):
    """What the model returns per item. No defaults or limits: strict output schema."""

    flag_id: str
    explanation: str
    evidence_ids: tuple[str, ...]

    def cited_ids(self) -> tuple[str, ...]:
        return self.evidence_ids

    def grounded_texts(self) -> tuple[str, ...]:
        return (self.explanation,)


class ExplanationBatch(DomainModel):
    explanations: tuple[ItemExplanation, ...]

    def cited_ids(self) -> tuple[str, ...]:
        return tuple(i for e in self.explanations for i in e.evidence_ids)

    def grounded_texts(self) -> tuple[str, ...]:
        return tuple(e.explanation for e in self.explanations)


class Explanation(DomainModel):
    text: str = Field(min_length=1)
    source: ExplanationSource
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    fallback_reason: str | None = None
    """Why a template was used, when one was."""


@dataclass(frozen=True)
class ItemToExplain:
    assessment: NearZeroAssessment
    finding: Finding
    evidence: tuple[Evidence, ...]
    """Every evidence record the finding cites."""
    units: Mapping[str, Unit]
    """Unit of each number field in the evidence data, e.g. `change` → bps."""
    measure_unit: Unit
    comment: str | None = None


def evidence_bundle(
    records: Sequence[Evidence], units: Mapping[str, Unit], tolerance: NumberTolerance
) -> EvidenceBundle:
    """Evidence records as a bundle the gateway can check numbers against."""
    grounding = {k: _GROUNDING_UNITS[u] for k, u in units.items() if u in _GROUNDING_UNITS}
    return EvidenceBundle.of(
        [EvidenceItem.from_data(e.evidence_id, e.data, grounding) for e in records],
        tolerance=tolerance,
    )


def _bundle(item: ItemToExplain, tolerance: NumberTolerance) -> EvidenceBundle:
    return evidence_bundle(item.evidence, item.units, tolerance)


def template_text(item: ItemToExplain) -> str:
    """Code's own words for an item, from its measures only."""
    a, unit = item.assessment, item.measure_unit.value
    change = f"{a.change:+.2f} {unit}"
    if a.classification is Classification.GENUINE_LARGE_MOVE and a.z and a.history.stdev:
        return (
            f"{a.item_key} moved {change}, {abs(a.z):.2f} times its normal daily move of "
            f"{a.history.stdev:.2f} {unit}: a genuine large move, to review first."
        )
    if a.classification is Classification.THRESHOLD_ARTIFACT and a.z is not None:
        rel = f"{a.relative_change_pct:+.1f}%" if a.relative_change_pct is not None else "a large"
        return (
            f"{a.item_key}: MDX's {rel} change comes from a near-zero base of {a.previous}%. "
            f"The actual move is {change}, z {a.z:+.2f}, a normal day: a threshold artifact."
        )
    if a.classification is Classification.ORDINARY and a.z is not None:
        return f"{a.item_key} moved {change} (z {a.z:+.2f}), within its normal range: ordinary."
    return f"{a.item_key} moved {change}; there isn't enough history to judge how unusual it is."


def _templates(items: Sequence[ItemToExplain], reason: str) -> dict[str, Explanation]:
    return {
        i.assessment.flag_id: Explanation(
            text=template_text(i),
            source=ExplanationSource.TEMPLATE,
            evidence_ids=i.finding.evidence_ids,
            fallback_reason=reason,
        )
        for i in items
    }


def explanation_inputs(
    items: Sequence[ItemToExplain],
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    """The model's inputs, and the untrusted comment texts they refer to by name."""
    untrusted: dict[str, str] = {}
    rows: list[JsonValue] = []
    for n, item in enumerate(items, start=1):
        a = item.assessment
        row: dict[str, JsonValue] = {
            "flag_id": a.flag_id,
            "item": a.item_key,
            "label": a.classification.value,
            "severity": a.severity.value,
            "rule_trace": list(a.reasons),
            "evidence": [
                {"id": e.evidence_id, "kind": e.kind, "data": e.data} for e in item.evidence
            ],
        }
        if item.comment:
            name = f"comment_{n}"
            untrusted[name] = item.comment
            row["mdx_comment"] = f"see the untrusted block {name!r}"
        rows.append(row)
    return {"items": rows}, untrusted


def check_batch(
    batch: ExplanationBatch, items: Sequence[ItemToExplain], tolerance: NumberTolerance
) -> list[str]:
    """Every item explained once, citing only its own evidence, with matching numbers."""
    wanted = {i.assessment.flag_id: i for i in items}
    seen = Counter(e.flag_id for e in batch.explanations)
    problems = [f"{f}: no explanation was given" for f in wanted if f not in seen]
    problems += [f"{f}: not an item to explain" for f in seen if f not in wanted]
    problems += [f"{f}: explained {n} times" for f, n in seen.items() if n > 1 and f in wanted]
    for e in batch.explanations:
        item = wanted.get(e.flag_id)
        if item is None:
            continue
        if not e.explanation.strip():
            problems.append(f"{e.flag_id}: the explanation is empty")
            continue
        report = check_grounding(e, _bundle(item, tolerance))
        problems += [f"{e.flag_id}: {p}" for p in report.errors]
    return problems


@dataclass(frozen=True)
class ExplainOutcome:
    explanations: dict[str, Explanation]
    """By flag ID; one per item."""
    llm_attempts: int = 0
    rejections: tuple[str, ...] = ()


async def explain(
    items: Sequence[ItemToExplain],
    *,
    gateway: LLMGateway | None,
    llm_enabled: bool,
    session_id: str,
    max_ai_explanations: int,
    tolerance: NumberTolerance,
) -> ExplainOutcome:
    if not items:
        return ExplainOutcome({})
    if not llm_enabled or gateway is None:
        return ExplainOutcome(_templates(items, FALLBACK_LLM_OFF))

    to_llm, beyond = list(items[:max_ai_explanations]), list(items[max_ai_explanations:])
    explanations = _templates(beyond, FALLBACK_OVER_LIMIT)
    if not to_llm:
        return ExplainOutcome(explanations)

    inputs, untrusted = explanation_inputs(to_llm)

    def as_templates() -> ExplanationBatch:
        return ExplanationBatch(
            explanations=tuple(
                ItemExplanation(
                    flag_id=i.assessment.flag_id,
                    explanation=template_text(i),
                    evidence_ids=i.finding.evidence_ids,
                )
                for i in to_llm
            )
        )

    # A missing prompt file raises LLMPromptError and fails the step: a configuration
    # fault for a developer, never hidden behind templates.
    result = await gateway.call(
        PROMPT_ID,
        PROMPT_VERSION,
        inputs=inputs,
        output_model=ExplanationBatch,
        actor=MARKET_DATA_ASSESSMENT,
        session_id=session_id,
        untrusted=untrusted,
        # Per item rather than the gateway's whole-output `evidence=` check: an item may
        # cite and quote only its own evidence, and every reason names its item.
        check=lambda batch: check_batch(batch, to_llm, tolerance),
        fallback=as_templates,
    )
    llm = result.source == "llm"
    reason = None
    if not llm:
        reason = FALLBACK_REJECTED if result.attempts > 1 else FALLBACK_UNAVAILABLE
    for e in result.output.explanations:
        explanations[e.flag_id] = Explanation(
            text=e.explanation.strip(),
            source=ExplanationSource.LLM if llm else ExplanationSource.TEMPLATE,
            evidence_ids=e.evidence_ids,
            fallback_reason=reason,
        )
    return ExplainOutcome(explanations, result.attempts, result.rejections)

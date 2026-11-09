"""Master Agent planning: one LLM call within the selected playbook, checked by code (plan §2.5).

The Master receives the facts, observations, selected playbook and task catalogue as
data, and returns a plan. Code then validates it:

- **P-1 Mandatory steps:** every mandatory step, exactly once, in the playbook's order.
- **P-2 Allowed actions:** nothing in `steps` beyond the mandatory tasks, and only
  follow-ups the playbook allows.
- **P-3 Limits:** follow-ups within the round limit (one per round), and the steps'
  combined time budgets within the run limit. The AI-explanation limit can't be judged
  from a plan: it applies when explanations are generated.
- **P-4 Evidence:** at least one citation, and every cited ID, or ID named in a
  rationale, exists. Number matching joins this check with E5-03.

Invalid output is retried once with the reasons attached. If it still fails, or there
are no facts to cite, the selected playbook's default plan runs, marked `source: default`.
An unavailable model is not handled here: `LLMError` propagates and the graph stops the
run, because that needs a developer. Each attempt is logged to the terminal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import JsonValue

from mdx_assessment.core.domain import (
    DomainModel,
    Plan,
    PlanSource,
    PlanStep,
    PlanValidation,
)
from mdx_assessment.core.gateway.gateway import LLMGateway
from mdx_assessment.core.gateway.provider import LLMOutputError
from mdx_assessment.core.orchestration.graph import MASTER_AGENT, PlanHook, SessionRuntime
from mdx_assessment.core.orchestration.playbooks import ResolvedPlaybook
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.orchestration.tasks import TaskRegistry

logger = logging.getLogger(__name__)

PROMPT_ID = "master_plan"
PROMPT_VERSION = "v1"
MAX_ATTEMPTS = 2

FALLBACK_INVALID = "plan_invalid"
FALLBACK_NO_FACTS = "no_facts"

CHECK_IDS = ("P-1", "P-2", "P-3", "P-4")
_MENTIONED_ID = re.compile(r"\b(?:fact|obs)-[a-z0-9]+(?:-[a-z0-9]+)*")


class PlannedStep(DomainModel):
    """No defaults or length limits: the model's strict output schema allows neither."""

    step_id: str
    task: str
    rationale: str


class MasterPlanOutput(DomainModel):
    """What the Master Agent returns."""

    steps: tuple[PlannedStep, ...]
    follow_ups: tuple[PlannedStep, ...]
    rationale: str
    cited_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlanCheckReport:
    checks: dict[str, bool]
    errors: tuple[str, ...] = field(default=())

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def validate_plan(
    output: MasterPlanOutput,
    playbook: ResolvedPlaybook,
    citable_ids: frozenset[str],
    tasks: TaskRegistry,
) -> PlanCheckReport:
    """P-1 to P-4 against the selected playbook. Pure: no LLM, no I/O."""
    by_check = {
        "P-1": _mandatory_steps(output, playbook),
        "P-2": _allowed_actions(output, playbook),
        "P-3": _limits(output, playbook, tasks),
        "P-4": _evidence(output, citable_ids),
    }
    return PlanCheckReport(
        checks={check: not errors for check, errors in by_check.items()},
        errors=tuple(f"{check}: {e}" for check, errors in by_check.items() for e in errors),
    )


def _mandatory_steps(output: MasterPlanOutput, playbook: ResolvedPlaybook) -> list[str]:
    if not output.steps:
        return ["the plan has no steps"]
    planned = [s.task for s in output.steps]
    mandatory = playbook.content.mandatory_steps
    errors = [f"mandatory step {t!r} is missing" for t in mandatory if t not in planned]
    errors += [
        f"mandatory step {t!r} appears {planned.count(t)} times"
        for t in mandatory
        if planned.count(t) > 1
    ]
    if not errors and [t for t in planned if t in mandatory] != list(mandatory):
        errors.append(f"mandatory steps must run in this order: {list(mandatory)}")
    return errors


def _allowed_actions(output: MasterPlanOutput, playbook: ResolvedPlaybook) -> list[str]:
    content = playbook.content
    errors = [
        f"task {s.task!r} is not a mandatory step of this playbook"
        for s in output.steps
        if s.task not in content.mandatory_steps
    ]
    errors += [
        f"follow-up {s.task!r} is not allowed by this playbook"
        for s in output.follow_ups
        if s.task not in content.allowed_follow_ups
    ]
    ids = [s.step_id.strip() for s in (*output.steps, *output.follow_ups)]
    if not all(ids):
        errors.append("every step needs a step_id")
    errors += [
        f"step_id {i!r} is used more than once" for i in sorted(set(ids)) if ids.count(i) > 1
    ]
    return errors


def _limits(output: MasterPlanOutput, playbook: ResolvedPlaybook, tasks: TaskRegistry) -> list[str]:
    limits = playbook.content.limits
    errors: list[str] = []
    if len(output.follow_ups) > limits.max_follow_up_rounds:
        errors.append(
            f"{len(output.follow_ups)} follow-up(s) exceed the limit of "
            f"{limits.max_follow_up_rounds}"
        )
    definitions = [tasks.get(s.task) for s in (*output.steps, *output.follow_ups)]
    budgeted = sum(d.budget.max_seconds for d in definitions if d is not None)
    if budgeted > limits.max_run_seconds:
        errors.append(
            f"the steps' time budgets total {budgeted:g}s, over the run limit of "
            f"{limits.max_run_seconds}s"
        )
    return errors


def _evidence(output: MasterPlanOutput, citable_ids: frozenset[str]) -> list[str]:
    errors = [] if output.cited_fact_ids else ["the plan cites no facts or observations"]
    errors += [
        f"cited ID {i!r} does not exist" for i in output.cited_fact_ids if i not in citable_ids
    ]
    texts = [output.rationale, *(s.rationale for s in (*output.steps, *output.follow_ups))]
    mentioned = {match for text in texts for match in _MENTIONED_ID.findall(text)}
    errors += [f"rationale names unknown ID {i!r}" for i in sorted(mentioned - citable_ids)]
    return errors


def planning_inputs(
    state: RunState,
    playbook: ResolvedPlaybook,
    tasks: TaskRegistry,
    previous_attempt: dict[str, JsonValue] | None,
) -> dict[str, JsonValue]:
    """Everything the Master sees, as data. Code never names a task or playbook here."""
    content = playbook.content
    inputs: dict[str, JsonValue] = {
        "business_date": state.business_date.isoformat(),
        "data_type": state.data_type,
        "facts": [
            f.model_dump(mode="json", exclude={"evidence_id"}) for f in state.run_context.facts
        ],
        "observations": [o.model_dump(mode="json") for o in state.run_context.observations],
        "playbook": {
            "id": playbook.id,
            "title": content.title,
            "description": content.description,
            "mandatory_steps": list(content.mandatory_steps),
            "allowed_follow_ups": list(content.allowed_follow_ups),
            "limits": content.limits.model_dump(mode="json"),
            "other_conditions_met": list(playbook.selection.conditions_met_not_selected),
        },
        "task_catalogue": [dict(entry) for entry in tasks.catalogue()],
    }
    if previous_attempt is not None:
        inputs["previous_attempt"] = previous_attempt
    return inputs


def _master_plan(output: MasterPlanOutput, report: PlanCheckReport, attempts: int) -> Plan:
    def steps(planned: tuple[PlannedStep, ...]) -> tuple[PlanStep, ...]:
        return tuple(
            PlanStep(step_id=s.step_id, task=s.task, rationale=s.rationale) for s in planned
        )

    return Plan(
        source=PlanSource.MASTER,
        steps=steps(output.steps),
        follow_ups=steps(output.follow_ups),
        rationale=output.rationale,
        cited_fact_ids=output.cited_fact_ids,
        validation=PlanValidation(valid=True, checks=report.checks, attempts=attempts),
    )


def _default_plan(
    playbook: ResolvedPlaybook,
    reason: str,
    attempts: int,
    checks: dict[str, bool],
    errors: tuple[str, ...],
) -> Plan:
    return Plan(
        source=PlanSource.DEFAULT,
        steps=playbook.content.default_plan,
        rationale=f"The playbook's default plan: Master planning did not succeed ({reason}).",
        validation=PlanValidation(
            valid=False, checks=checks, attempts=attempts, errors=errors, fallback_reason=reason
        ),
    )


def master_plan_hook(runtime: SessionRuntime) -> PlanHook:
    """The graph's `plan` hook. Needs the runtime's LLM gateway."""
    if runtime.gateway is None:
        raise ValueError("Master planning needs an LLM gateway")
    gateway: LLMGateway = runtime.gateway

    async def plan(state: RunState) -> Plan:
        playbook = state.playbook
        if playbook is None:
            raise ValueError("Master planning needs a selected playbook")
        citable = state.run_context.citable_ids
        if not citable:  # a plan must cite facts, so a call could only fail
            logger.warning("session %s: using the default plan (no facts)", state.session_id)
            return _default_plan(playbook, FALLBACK_NO_FACTS, 0, {}, ())

        previous: dict[str, JsonValue] | None = None
        report = PlanCheckReport(checks={})
        for attempt in range(1, MAX_ATTEMPTS + 1):
            output: MasterPlanOutput | None = None
            try:
                result = await gateway.call(
                    PROMPT_ID,
                    PROMPT_VERSION,
                    inputs=planning_inputs(state, playbook, runtime.tasks, previous),
                    output_model=MasterPlanOutput,
                    actor=MASTER_AGENT,
                    session_id=state.session_id,
                )
            except LLMOutputError as exc:
                report = PlanCheckReport(checks={}, errors=(f"output: {exc}",))
            else:
                output = result.output
                report = validate_plan(output, playbook, citable, runtime.tasks)

            dumped = output.model_dump(mode="json") if output is not None else None
            logger.info(
                "session %s plan attempt %d: valid=%s checks=%s errors=%s output=%s",
                state.session_id,
                attempt,
                report.valid,
                report.checks,
                list(report.errors),
                dumped,
            )
            if output is not None and report.valid:
                return _master_plan(output, report, attempt)
            previous = {"output": dumped, "rejected_because": list(report.errors)}

        logger.warning(
            "session %s: using the default plan (%s)", state.session_id, FALLBACK_INVALID
        )
        return _default_plan(
            playbook, FALLBACK_INVALID, MAX_ATTEMPTS, dict(report.checks), report.errors
        )

    return plan

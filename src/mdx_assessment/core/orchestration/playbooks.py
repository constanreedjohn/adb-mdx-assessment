"""Playbooks: approved documents, one selected per session by code (plan §2.5, D-16).

    selected_when:                 # a rule over the run-context facts, as data
      any_of:
        - {fact: files_received, op: lt, compare_to: expected}
    precedence: 1                  # lower number wins

`resolve_playbook(facts, store)` evaluates every rule, picks the highest-precedence
playbook whose rule holds, and returns it with a selection record. The Master Agent
plans within the result; because code makes the selection, it stands even when the
LLM is down. **The interface is fixed**: the deferred "effective playbook" design
(IMP-01) would replace only this function.

A playbook may `extends: <parent>` and override only what differs. Its own
`selected_when` and `precedence` are always required: those are what make it a
different playbook. A condition on a fact that isn't available is simply not met,
so a missing value can never select a playbook by accident.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from mdx_assessment.core.config import ConfigError, LoadedConfig
from mdx_assessment.core.domain import DomainModel, Fact, PlanStep

PLAYBOOKS_PATTERN = "playbooks/*.yaml"

PlaybookId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class Op(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


_COMPARE: dict[Op, Callable[[Any, Any], bool]] = {
    Op.EQ: operator.eq,
    Op.NE: operator.ne,
    Op.GT: operator.gt,
    Op.GTE: operator.ge,
    Op.LT: operator.lt,
    Op.LTE: operator.le,
}
_SYMBOL = {Op.EQ: "==", Op.NE: "!=", Op.GT: ">", Op.GTE: ">=", Op.LT: "<", Op.LTE: "<="}


class Condition(DomainModel):
    """`<fact> <op> <value>`, where the right side is a literal or the fact's own limit."""

    fact: NonEmpty
    op: Op
    value: JsonValue = None
    compare_to: Literal["expected", "threshold"] | None = None

    @model_validator(mode="after")
    def _one_right_hand_side(self) -> Self:
        if (self.value is None) == (self.compare_to is None):
            raise ValueError("set exactly one of `value` or `compare_to`")
        return self

    def holds(self, facts: Mapping[str, Fact]) -> bool:
        fact = facts.get(self.fact)
        if fact is None or fact.value is None:
            return False
        right = self.value if self.compare_to is None else getattr(fact, self.compare_to)
        if right is None:
            return False
        try:
            return _COMPARE[self.op](fact.value, right)
        except TypeError:  # e.g. comparing text with a number: treat as not met
            return False

    def __str__(self) -> str:
        right = self.compare_to if self.compare_to is not None else repr(self.value)
        return f"{self.fact} {_SYMBOL[self.op]} {right}"


class SelectionRule(DomainModel):
    """Exactly one form: `default: true`, `any_of: [...]` or `all_of: [...]`."""

    default: bool = False
    any_of: tuple[Condition, ...] = ()
    all_of: tuple[Condition, ...] = ()

    @model_validator(mode="after")
    def _one_form(self) -> Self:
        if sum((self.default, bool(self.any_of), bool(self.all_of))) != 1:
            raise ValueError("set exactly one of `default: true`, `any_of` or `all_of`")
        return self

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return self.any_of or self.all_of

    def holds(self, facts: Mapping[str, Fact]) -> bool:
        if self.default:
            return True
        results = [c.holds(facts) for c in self.conditions]
        return any(results) if self.any_of else all(results)

    def __str__(self) -> str:
        if self.default:
            return "default"
        joiner = " or " if self.any_of else " and "
        return joiner.join(str(c) for c in self.conditions)


class PlaybookLimits(DomainModel):
    max_follow_up_rounds: int = Field(ge=0)
    max_run_seconds: int = Field(gt=0)
    max_ai_explanations: int = Field(ge=0)


class PlaybookDocument(DomainModel):
    """One `config/playbooks/<id>.yaml`. With `extends`, content fields are optional."""

    id: PlaybookId
    version: NonEmpty
    extends: PlaybookId | None = None
    selected_when: SelectionRule
    precedence: int = Field(ge=1)
    title: NonEmpty | None = None
    description: str | None = None
    mandatory_steps: tuple[NonEmpty, ...] | None = None
    """Task IDs that every plan under this playbook must include (P-1)."""
    allowed_follow_ups: tuple[NonEmpty, ...] | None = None
    """Task IDs the Master Agent may add as follow-ups (P-2)."""
    limits: PlaybookLimits | None = None
    default_plan: tuple[PlanStep, ...] | None = None
    """What runs if planning fails (PDF §3.3)."""

    @model_validator(mode="after")
    def _complete_unless_extending(self) -> Self:
        if self.extends is None:
            missing = [f for f in _CONTENT_FIELDS if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"missing {missing} (only a playbook with `extends` may omit them)"
                )
        return self


_CONTENT_FIELDS = (
    "title",
    "description",
    "mandatory_steps",
    "allowed_follow_ups",
    "limits",
    "default_plan",
)


class PlaybookContent(DomainModel):
    """A playbook after `extends` is resolved: what the Master plans within."""

    title: str
    description: str
    mandatory_steps: tuple[str, ...]
    allowed_follow_ups: tuple[str, ...]
    limits: PlaybookLimits
    default_plan: tuple[PlanStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_plan_covers_mandatory(self) -> Self:
        ids = [s.step_id for s in self.default_plan]
        if len(set(ids)) != len(ids):
            raise ValueError("default_plan has duplicate step ids")
        missing = sorted(set(self.mandatory_steps) - {s.task for s in self.default_plan})
        if missing:
            raise ValueError(f"default_plan lacks mandatory step(s) {missing}")
        return self


class Playbook(DomainModel):
    id: str
    version: str
    selected_when: SelectionRule
    precedence: int
    content: PlaybookContent


class RuleOutcome(DomainModel):
    playbook: str
    met: bool
    rule: str
    fact_ids: tuple[str, ...] = ()


class SelectionRecord(DomainModel):
    """How the playbook was chosen: every rule, and any other condition that also held."""

    rules: tuple[RuleOutcome, ...]
    conditions_met_not_selected: tuple[str, ...] = ()
    multi_condition_day: bool = False


class ResolvedPlaybook(DomainModel):
    id: str
    version: str
    selected_by: Literal["selector"] = "selector"
    selection: SelectionRecord
    content: PlaybookContent


class PlaybookStore:
    """The validated, `extends`-resolved playbooks. Raises ConfigError on any problem."""

    def __init__(self, documents: Mapping[str, PlaybookDocument]) -> None:
        """`documents` maps a config path (or any label) to its document."""
        problems: list[str] = []
        by_id: dict[str, PlaybookDocument] = {}
        for path, doc in documents.items():
            stem = PurePosixPath(path).stem
            if stem != doc.id:
                problems.append(f"{path}: id {doc.id!r} must match the file name {stem!r}")
            by_id[doc.id] = doc

        precedences: dict[int, str] = {}
        for doc in by_id.values():
            if doc.precedence in precedences:
                problems.append(
                    f"playbooks {precedences[doc.precedence]!r} and {doc.id!r} share "
                    f"precedence {doc.precedence}"
                )
            precedences[doc.precedence] = doc.id

        defaults = [d for d in by_id.values() if d.selected_when.default]
        if len(defaults) != 1:
            problems.append(
                f"exactly one playbook must be `default: true`, found {sorted(d.id for d in defaults)}"
            )
        elif by_id and defaults[0].precedence != max(d.precedence for d in by_id.values()):
            problems.append(f"default playbook {defaults[0].id!r} must have the lowest priority")

        playbooks: list[Playbook] = []
        for doc in by_id.values():
            try:
                playbooks.append(self._resolve(doc, by_id))
            except ValueError as exc:
                problems.append(f"playbook {doc.id!r}: {exc}")
        if problems:
            raise ConfigError(problems)
        self.playbooks: tuple[Playbook, ...] = tuple(sorted(playbooks, key=lambda p: p.precedence))

    @classmethod
    def from_config(cls, config: LoadedConfig) -> PlaybookStore:
        return cls(config.all_of(PlaybookDocument))

    @staticmethod
    def _resolve(doc: PlaybookDocument, by_id: Mapping[str, PlaybookDocument]) -> Playbook:
        chain = [doc]
        while chain[-1].extends is not None:
            parent = by_id.get(chain[-1].extends)
            if parent is None:
                raise ValueError(f"extends unknown playbook {chain[-1].extends!r}")
            if parent in chain:
                raise ValueError("`extends` forms a cycle")
            chain.append(parent)
        merged: dict[str, Any] = {}
        for link in reversed(chain):  # root first, so the child's values win
            for name in _CONTENT_FIELDS:
                value = getattr(link, name)
                if value is not None:
                    merged[name] = value
        return Playbook(
            id=doc.id,
            version=doc.version,
            selected_when=doc.selected_when,
            precedence=doc.precedence,
            content=PlaybookContent(**merged),
        )

    def unknown_tasks(self, known: Iterable[str]) -> list[str]:
        """Problems: tasks named by a playbook that nobody registered."""
        registered = set(known)
        problems = []
        for playbook in self.playbooks:
            content = playbook.content
            named = {
                *content.mandatory_steps,
                *content.allowed_follow_ups,
                *(s.task for s in content.default_plan),
            }
            problems += [
                f"playbook {playbook.id!r}: unknown task {task!r}"
                for task in sorted(named - registered)
            ]
        return problems


def resolve_playbook(facts: Sequence[Fact], store: PlaybookStore) -> ResolvedPlaybook:
    """Select the playbook for these facts: the highest-precedence one whose rule holds."""
    by_name = {f.name: f for f in facts}
    outcomes: list[RuleOutcome] = []
    eligible: list[Playbook] = []
    for playbook in store.playbooks:  # sorted by precedence
        rule = playbook.selected_when
        met = rule.holds(by_name)
        outcomes.append(
            RuleOutcome(
                playbook=playbook.id,
                met=met,
                rule=str(rule),
                fact_ids=tuple(by_name[c.fact].id for c in rule.conditions if c.fact in by_name),
            )
        )
        if met:
            eligible.append(playbook)
    selected = eligible[0]  # the store guarantees a default, which always holds
    others = tuple(p.id for p in eligible[1:] if not p.selected_when.default)
    return ResolvedPlaybook(
        id=selected.id,
        version=selected.version,
        selection=SelectionRecord(
            rules=tuple(outcomes),
            conditions_met_not_selected=others,
            multi_condition_day=bool(others),
        ),
        content=selected.content,
    )

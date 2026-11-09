"""Playbook rules, the store and `resolve_playbook`, with toy documents (no use-case pack)."""

from typing import Any

import pytest
from pydantic import ValidationError

from mdx_assessment.core.config import ConfigError
from mdx_assessment.core.domain import Fact
from mdx_assessment.core.orchestration.playbooks import (
    Condition,
    PlaybookDocument,
    PlaybookStore,
    SelectionRule,
    resolve_playbook,
)

SCOPE: dict[str, Any] = {"business_date": "2026-10-15"}


def facts(files: int | None = 1, flags: int | None = 25, snapshot: bool = True) -> list[Fact]:
    return [
        Fact.create("files_received", files, expected=1, source="t", scope=SCOPE),
        Fact.create("flag_count", flags, threshold=300, source="t", scope=SCOPE),
        Fact.create("snapshot_available", snapshot, source="t", scope=SCOPE),
    ]


def content(**overrides: Any) -> dict[str, Any]:
    return {
        "title": "Normal",
        "description": "d",
        "mandatory_steps": ["a.one"],
        "allowed_follow_ups": [],
        "limits": {"max_follow_up_rounds": 0, "max_run_seconds": 60, "max_ai_explanations": 10},
        "default_plan": [{"step_id": "s1", "task": "a.one"}],
        **overrides,
    }


def doc(
    id_: str, precedence: int, rule: dict[str, Any], **fields: Any
) -> tuple[str, PlaybookDocument]:
    data = {"id": id_, "version": "1", "selected_when": rule, "precedence": precedence, **fields}
    return f"playbooks/{id_}.yaml", PlaybookDocument.model_validate(data)


NORMAL = doc("normal_eod", 4, {"default": True}, **content())
MISSING = doc(
    "missing_input",
    1,
    {"any_of": [{"fact": "files_received", "op": "lt", "compare_to": "expected"}]},
    extends="normal_eod",
    title="Missing input",
    allowed_follow_ups=["a.recheck"],
)
FALLBACK = doc(
    "materiality_fallback",
    2,
    {"all_of": [{"fact": "snapshot_available", "op": "eq", "value": False}]},
    extends="normal_eod",
)
HIGH = doc(
    "high_volume",
    3,
    {"any_of": [{"fact": "flag_count", "op": "gt", "compare_to": "threshold"}]},
    extends="normal_eod",
)


@pytest.fixture
def store() -> PlaybookStore:
    return PlaybookStore(dict([NORMAL, MISSING, FALLBACK, HIGH]))


# --- rules ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"fact": "flag_count", "op": "gt", "compare_to": "threshold"}, False),
        ({"fact": "flag_count", "op": "lte", "compare_to": "threshold"}, True),
        ({"fact": "flag_count", "op": "eq", "value": 25}, True),
        ({"fact": "flag_count", "op": "ne", "value": 25}, False),
        ({"fact": "files_received", "op": "gte", "compare_to": "expected"}, True),
        ({"fact": "no_such_fact", "op": "eq", "value": 1}, False),
        ({"fact": "flag_count", "op": "gt", "value": "text"}, False),  # can't compare: not met
        ({"fact": "flag_count", "op": "gt", "compare_to": "expected"}, False),  # no expected value
    ],
)
def test_condition(condition: dict[str, Any], expected: bool) -> None:
    by_name = {f.name: f for f in facts()}
    assert Condition.model_validate(condition).holds(by_name) is expected


def test_an_unavailable_fact_never_meets_a_condition() -> None:
    by_name = {f.name: f for f in facts(flags=None)}
    for op in ("gt", "lt", "ne"):
        assert not Condition.model_validate({"fact": "flag_count", "op": op, "value": 1}).holds(
            by_name
        )


def test_any_of_and_all_of() -> None:
    true = {"fact": "flag_count", "op": "eq", "value": 25}
    false = {"fact": "flag_count", "op": "eq", "value": 99}
    by_name = {f.name: f for f in facts()}
    assert SelectionRule.model_validate({"any_of": [false, true]}).holds(by_name)
    assert not SelectionRule.model_validate({"all_of": [false, true]}).holds(by_name)


def test_rules_read_as_text() -> None:
    assert str(SelectionRule.model_validate({"default": True})) == "default"
    rule = {
        "any_of": [
            {"fact": "flag_count", "op": "gt", "compare_to": "threshold"},
            {"fact": "x", "op": "eq", "value": 1},
        ]
    }
    assert str(SelectionRule.model_validate(rule)) == "flag_count > threshold or x == 1"


@pytest.mark.parametrize(
    "rule",
    [{}, {"default": True, "any_of": [{"fact": "a", "op": "eq", "value": 1}]}, {"default": False}],
)
def test_a_rule_has_exactly_one_form(rule: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        SelectionRule.model_validate(rule)


def test_a_condition_has_exactly_one_right_hand_side() -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        Condition.model_validate({"fact": "a", "op": "eq"})


# --- selection -----------------------------------------------------------------------------


def test_default_when_no_rule_holds(store: PlaybookStore) -> None:
    resolved = resolve_playbook(facts(), store)
    assert resolved.id == "normal_eod"
    assert resolved.selected_by == "selector"
    assert not resolved.selection.multi_condition_day
    assert resolved.selection.conditions_met_not_selected == ()


def test_selection_record_covers_every_rule(store: PlaybookStore) -> None:
    rules = {r.playbook: r for r in resolve_playbook(facts(), store).selection.rules}
    assert set(rules) == {"missing_input", "materiality_fallback", "high_volume", "normal_eod"}
    assert rules["high_volume"].rule == "flag_count > threshold"
    assert rules["high_volume"].fact_ids == ("fact-flag-count",)
    assert not rules["high_volume"].met
    assert rules["normal_eod"].met


@pytest.mark.parametrize(
    ("kwargs", "selected"),
    [
        ({"files": 0}, "missing_input"),
        ({"snapshot": False}, "materiality_fallback"),
        ({"flags": 450}, "high_volume"),
    ],
)
def test_each_condition_selects_its_playbook(
    store: PlaybookStore, kwargs: dict[str, Any], selected: str
) -> None:
    assert resolve_playbook(facts(**kwargs), store).id == selected


def test_multi_condition_day_runs_the_highest_precedence_and_records_the_rest(
    store: PlaybookStore,
) -> None:
    resolved = resolve_playbook(facts(files=0, flags=450, snapshot=False), store)
    assert resolved.id == "missing_input"
    assert resolved.selection.multi_condition_day
    assert resolved.selection.conditions_met_not_selected == ("materiality_fallback", "high_volume")


def test_missing_facts_fall_back_to_the_default(store: PlaybookStore) -> None:
    assert resolve_playbook([], store).id == "normal_eod"


# --- extends -------------------------------------------------------------------------------


def test_child_inherits_and_overrides(store: PlaybookStore) -> None:
    resolved = resolve_playbook(facts(files=0), store)
    assert resolved.content.title == "Missing input"  # overridden
    assert resolved.content.allowed_follow_ups == ("a.recheck",)  # overridden
    assert resolved.content.mandatory_steps == ("a.one",)  # inherited
    assert resolved.content.limits.max_run_seconds == 60  # inherited


def test_chained_extends() -> None:
    grandchild = doc(
        "very_high",
        2,
        {"any_of": [{"fact": "flag_count", "op": "gt", "value": 900}]},
        extends="high_volume",
        description="very high",
    )
    store = PlaybookStore(dict([NORMAL, HIGH, grandchild]))
    resolved = resolve_playbook(facts(flags=1000), store)
    assert resolved.id == "very_high"
    assert resolved.content.description == "very high"
    assert resolved.content.title == "Normal"


def test_extending_does_not_inherit_the_rule_or_precedence() -> None:
    with pytest.raises(ValidationError):
        PlaybookDocument.model_validate({"id": "x", "version": "1", "extends": "normal_eod"})


# --- store validation ------------------------------------------------------------------------


def expect_problem(entries: list[tuple[str, PlaybookDocument]], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        PlaybookStore(dict(entries))


def test_a_complete_document_is_required_without_extends() -> None:
    with pytest.raises(ValidationError, match="only a playbook with `extends` may omit"):
        PlaybookDocument.model_validate(
            {"id": "x", "version": "1", "selected_when": {"default": True}, "precedence": 1}
        )


def test_no_default() -> None:
    expect_problem(
        [doc("only", 1, {"any_of": [{"fact": "a", "op": "eq", "value": 1}]}, **content())],
        "exactly one playbook",
    )


def test_two_defaults() -> None:
    expect_problem(
        [NORMAL, doc("other", 5, {"default": True}, **content())], "exactly one playbook"
    )


def test_default_must_have_the_lowest_priority() -> None:
    expect_problem(
        [doc("normal_eod", 1, {"default": True}, **content()), HIGH],
        "must have the lowest priority",
    )


def test_duplicate_precedence() -> None:
    twin = doc("twin", 3, {"any_of": [{"fact": "a", "op": "eq", "value": 1}]}, extends="normal_eod")
    expect_problem([NORMAL, HIGH, twin], "share precedence 3")


def test_unknown_parent() -> None:
    orphan = doc("orphan", 1, {"any_of": [{"fact": "a", "op": "eq", "value": 1}]}, extends="ghost")
    expect_problem([NORMAL, orphan], "extends unknown playbook 'ghost'")


def test_extends_cycle() -> None:
    rule = {"any_of": [{"fact": "a", "op": "eq", "value": 1}]}
    a = doc("a", 1, rule, extends="b")
    b = doc("b", 2, rule, extends="a")
    expect_problem([NORMAL, a, b], "cycle")


def test_file_name_must_match_id() -> None:
    expect_problem([("playbooks/wrong.yaml", NORMAL[1])], "must match the file name 'wrong'")


def test_default_plan_must_cover_mandatory_steps() -> None:
    expect_problem(
        [doc("normal_eod", 4, {"default": True}, **content(mandatory_steps=["a.other"]))],
        "default_plan lacks mandatory",
    )


def test_unknown_tasks_are_reported(store: PlaybookStore) -> None:
    assert store.unknown_tasks(["a.one", "a.recheck"]) == []
    assert store.unknown_tasks(["a.one"]) == ["playbook 'missing_input': unknown task 'a.recheck'"]

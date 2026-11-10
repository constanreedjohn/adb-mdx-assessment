"""Evidence citation and number matching (E5-03).

Acceptance: an invented number is rejected; a correct, unit-converted number is accepted.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from mdx_assessment.app import load_config
from mdx_assessment.core.gateway import (
    EvidenceBundle,
    EvidenceItem,
    FakeProvider,
    LLMGateway,
    LLMGroundingError,
    NumberTolerance,
    PromptStore,
    Quantity,
    Unit,
    check_grounding,
)
from mdx_assessment.core.gateway.grounding import GUARDRAILS_PATH, GuardrailsConfig
from mdx_assessment.core.gateway.numbers import UNICODE_MINUS, extract_numbers, matches
from mdx_assessment.core.persistence import InMemoryAuditTrail
from mdx_assessment.settings import PROJECT_ROOT

pytestmark = pytest.mark.anyio
TOL = NumberTolerance()
EV_JPY = "ev-" + "a" * 16
EV_GBP = "ev-" + "b" * 16

# What the near-zero check knows about JPY 3M and GBP 2Y on the example day.
BUNDLE = EvidenceBundle.of(
    [
        EvidenceItem.from_data(
            EV_JPY,
            {"prev": 0.04, "curr": 0.07, "change_bps": 3.0, "change_pct": 75.0, "z": 0.6},
            units={
                "prev": Unit.PERCENT,
                "curr": Unit.PERCENT,
                "change_bps": Unit.BPS,
                "change_pct": Unit.PERCENT,
            },
        ),
        EvidenceItem.from_data(
            EV_GBP, {"change_bps": 18.04, "z": 5.14}, units={"change_bps": Unit.BPS}
        ),
    ]
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(frozen=True)
class Claim:
    text: str
    cited: tuple[str, ...] = (EV_JPY,)

    def cited_ids(self) -> tuple[str, ...]:
        return self.cited

    def grounded_texts(self) -> tuple[str, ...]:
        return (self.text,)


def errors(text: str, cited: tuple[str, ...] = (EV_JPY,)) -> tuple[str, ...]:
    return check_grounding(Claim(text, cited), BUNDLE).errors


# --- extraction ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rose +3 bps (+75%)", [(3.0, Unit.BPS, True), (75.0, Unit.PERCENT, True)]),
        ("from 0.04% to 0.07%", [(0.04, Unit.PERCENT, False), (0.07, Unit.PERCENT, False)]),
        ("fell -3 basis points", [(-3.0, Unit.BPS, True)]),
        ("fell " + UNICODE_MINUS + "3bp", [(-3.0, Unit.BPS, True)]),
        ("1,250.5 records", [(1250.5, None, False)]),
        ("3 percent, 4 pct", [(3.0, Unit.PERCENT, False), (4.0, Unit.PERCENT, False)]),
        ("z = 5.14.", [(5.14, None, False)]),
    ],
)
def test_numbers_and_units_are_extracted(
    text: str, expected: list[tuple[float, Unit | None, bool]]
) -> None:
    assert [(n.value, n.unit, n.signed) for n in extract_numbers(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "JPY 3M and GBP 10Y",  # tenors
        "on 2026-10-15 at 09:20",  # dates and times
        "P-1 and P-4 passed; see fact-2 and sess-abc12",  # IDs
        "prompt v1, 2nd attempt",  # glued to letters
        f"cited {EV_JPY}",  # evidence IDs
    ],
)
def test_labels_are_not_numbers(text: str) -> None:
    assert extract_numbers(text) == []


# --- matching --------------------------------------------------------------------------------


def test_an_invented_number_is_rejected() -> None:
    assert errors("JPY 3M moved 45 bps.") == ("'45 bps' does not match the cited evidence",)


@pytest.mark.parametrize(
    "text",
    [
        "JPY 3M rose 0.03%.",  # 3 bps in percent
        "JPY 3M rose 3 bps.",
        "JPY 3M went from 4 bps to 7 bps.",  # 0.04% and 0.07% in bps
        "MDX shows +75%, but it is only +3 bps.",
        "Z-score 0.6.",
    ],
)
def test_a_correct_unit_converted_number_is_accepted(text: str) -> None:
    assert errors(text) == ()


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("GBP 2Y rose 18 bps", True),  # 18.04 rounded as written
        ("GBP 2Y rose 18.0 bps", True),
        ("GBP 2Y rose 18.1 bps", False),  # one decimal written: 18.04 isn't 18.1
        ("z = 5.1", True),
        ("z = 5.2", False),
        ("z = 5", True),
    ],
)
def test_rounding_follows_the_digits_written(text: str, ok: bool) -> None:
    assert (errors(text, (EV_GBP,)) == ()) is ok


def test_without_rounding_only_the_relative_tolerance_applies() -> None:
    [stated] = extract_numbers("18 bps")
    strict = NumberTolerance(accept_rounding=False)
    assert not matches(stated, Quantity(18.04, Unit.BPS), strict)
    assert matches(stated, Quantity(18.01, Unit.BPS), strict)  # within 0.1%


def test_signs() -> None:
    [unsigned] = extract_numbers("fell 3 bps")
    [plus] = extract_numbers("+3 bps")
    assert matches(unsigned, Quantity(-3.0, Unit.BPS), TOL)
    assert not matches(plus, Quantity(-3.0, Unit.BPS), TOL)


def test_a_number_with_a_unit_never_matches_a_plain_count() -> None:
    [percent] = extract_numbers("28%")
    [plain] = extract_numbers("28")
    assert not matches(percent, Quantity(28.0), TOL)
    assert matches(plain, Quantity(28.0, Unit.PERCENT), TOL)


def test_small_plain_integers_are_not_checked_but_units_always_are() -> None:
    assert errors("Three steps ran; 2 were quick.") == ()
    assert errors("It moved 2 bps.") == ("'2 bps' does not match the cited evidence",)
    report = check_grounding(Claim("2 bps and 11 flags"), BUNDLE)
    assert report.numbers_checked == 2


def test_numbers_must_come_from_the_cited_evidence() -> None:
    # 18 bps is GBP's: true in the bundle, but not in what this text cites.
    assert errors("JPY 3M moved 18 bps.", (EV_JPY,)) != ()
    assert errors("GBP 2Y moved 18 bps.", (EV_GBP,)) == ()


def test_context_numbers_may_be_quoted_but_not_cited() -> None:
    bundle = EvidenceBundle.of([EvidenceItem(EV_JPY)], context=(Quantity(300.0),))
    assert check_grounding(Claim("below the threshold of 300"), bundle).ok


# --- citations -------------------------------------------------------------------------------


def test_citation_errors() -> None:
    assert errors("fine", ()) == ("no evidence is cited",)
    assert check_grounding(Claim("fine", ()), BUNDLE, require_citation=False).ok
    unknown = "ev-" + "c" * 16
    assert errors("fine", (unknown,)) == (f"cited ID {unknown!r} does not exist",)
    assert errors(f"see {unknown}") == (f"the text names unknown ID {unknown!r}",)
    assert errors("see fact-made-up") == ("the text names unknown ID 'fact-made-up'",)


def test_a_bundle_holds_each_id_once() -> None:
    with pytest.raises(ValueError, match="twice"):
        EvidenceBundle.of([EvidenceItem(EV_JPY), EvidenceItem(EV_JPY)])


def test_the_tolerance_comes_from_config() -> None:
    config = load_config(PROJECT_ROOT / "config")
    assert config.get(GUARDRAILS_PATH, GuardrailsConfig).number_matching == NumberTolerance()


# --- through the gateway ---------------------------------------------------------------------


class Explanation(BaseModel):
    text: str
    evidence_ids: list[str]

    def cited_ids(self) -> list[str]:
        return self.evidence_ids

    def grounded_texts(self) -> list[str]:
        return [self.text]


class Plain(BaseModel):
    text: str


def gateway(tmp_path: Path, reply: dict[str, object]) -> tuple[LLMGateway, InMemoryAuditTrail]:
    folder = tmp_path / "prompts" / "explain"
    folder.mkdir(parents=True)
    (folder / "v1.md").write_text("Explain.\n")
    audit = InMemoryAuditTrail()
    return LLMGateway(FakeProvider({"explain": reply}), PromptStore(tmp_path), audit), audit


async def explain(gw: LLMGateway, output_model: type[BaseModel] = Explanation) -> BaseModel:
    result = await gw.call(
        "explain",
        "v1",
        inputs={},
        output_model=output_model,
        actor="market_data_assessment",
        session_id="s1",
        evidence=BUNDLE,
    )
    return result.output


async def test_the_gateway_rejects_an_invented_number(tmp_path: Path) -> None:
    gw, audit = gateway(tmp_path, {"text": "JPY 3M moved 45 bps.", "evidence_ids": [EV_JPY]})
    with pytest.raises(LLMGroundingError, match="'45 bps' does not match") as raised:
        await explain(gw)
    assert raised.value.problems == ("'45 bps' does not match the cited evidence",)
    events = audit.read()  # retried once (E5-04), rejected both times
    assert [e.payload["outcome"] for e in events] == ["ungrounded", "ungrounded"]
    assert all("45 bps" in str(e.payload["error"]) for e in events)


async def test_the_gateway_accepts_a_converted_number(tmp_path: Path) -> None:
    gw, audit = gateway(tmp_path, {"text": "Only 0.03%, i.e. +3 bps.", "evidence_ids": [EV_JPY]})
    output = await explain(gw)
    assert isinstance(output, Explanation)
    assert audit.read()[0].payload["outcome"] == "ok"


async def test_evidence_needs_a_grounded_output_model(tmp_path: Path) -> None:
    gw, audit = gateway(tmp_path, {"text": "x"})
    with pytest.raises(TypeError, match="Plain must implement Grounded"):
        await explain(gw, Plain)
    assert audit.read() == []  # rejected before any model call

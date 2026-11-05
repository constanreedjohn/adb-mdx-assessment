from fractions import Fraction

import pytest
from pydantic import BaseModel, ValidationError

from mdx_assessment.usecases.mdx.domain import Tenor


@pytest.mark.parametrize(
    ("label", "count", "unit"),
    [("1D", 1, "D"), ("1W", 1, "W"), ("3M", 3, "M"), ("2Y", 2, "Y"), ("30Y", 30, "Y")],
)
def test_parse(label: str, count: int, unit: str) -> None:
    tenor = Tenor.parse(label)
    assert (tenor.count, tenor.unit) == (count, unit)
    assert str(tenor) == label


@pytest.mark.parametrize("label", ["3m", " 3M ", "3 m"])
def test_parse_normalizes_case_and_spaces(label: str) -> None:
    assert str(Tenor.parse(label)) == "3M"


@pytest.mark.parametrize("label", ["", "M", "3", "3Q", "-1M", "0M", "1.5Y", "3MM"])
def test_parse_rejects_invalid(label: str) -> None:
    with pytest.raises(ValueError):
        Tenor.parse(label)


def test_ordering_by_length_of_time() -> None:
    labels = ["30Y", "1Y", "3M", "1W", "18M", "2Y", "1D", "6M", "10Y", "1M"]
    ordered = [str(t) for t in sorted(Tenor.parse(x) for x in labels)]
    assert ordered == ["1D", "1W", "1M", "3M", "6M", "1Y", "18M", "2Y", "10Y", "30Y"]


def test_same_length_different_label_is_ordered_deterministically_and_not_equal() -> None:
    m12, y1 = Tenor.parse("12M"), Tenor.parse("1Y")
    assert m12.years == y1.years == Fraction(1)
    assert m12 != y1
    assert m12 < y1  # month form first: a total, deterministic order
    assert sorted([y1, m12]) == [m12, y1]


def test_years() -> None:
    assert Tenor.parse("3M").years == Fraction(1, 4)
    assert Tenor.parse("1W").years == Fraction(7, 365)


def test_hashable_and_equal_by_label() -> None:
    assert {Tenor.parse("3M"), Tenor.parse("3m")} == {Tenor(3, "M")}


class _Holder(BaseModel):
    tenor: Tenor


def test_pydantic_reads_and_writes_the_label() -> None:
    holder = _Holder.model_validate({"tenor": "3m"})
    assert holder.tenor == Tenor(3, "M")
    assert holder.model_dump_json() == '{"tenor":"3M"}'
    assert _Holder.model_validate_json(holder.model_dump_json()) == holder


def test_pydantic_rejects_bad_labels_and_types() -> None:
    with pytest.raises(ValidationError):
        _Holder.model_validate({"tenor": "3Q"})
    with pytest.raises(ValidationError):
        _Holder.model_validate({"tenor": 3})


def test_json_schema_is_a_string() -> None:
    schema = _Holder.model_json_schema()["properties"]["tenor"]
    assert schema["type"] == "string"

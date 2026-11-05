import math

import pytest
from pydantic import ValidationError

from mdx_assessment.core.domain import Evidence, canonical_json, compute_evidence_id


def test_same_content_gives_same_id_regardless_of_key_order() -> None:
    a = Evidence.create("point_change", "mdx_data.get_flags", {"curve": "JPY", "tenor": "3M"})
    b = Evidence.create("point_change", "mdx_data.get_flags", {"tenor": "3M", "curve": "JPY"})
    assert a.evidence_id == b.evidence_id


def test_id_is_stable_across_runs() -> None:
    # Pinned value: if this changes, every stored evidence ID changes. Only update
    # deliberately (it breaks citations in recorded sessions).
    ev = Evidence.create("point_change", "mdx_data.get_flags", {"curve": "JPY", "tenor": "3M"})
    assert ev.evidence_id == compute_evidence_id(
        "point_change", "mdx_data.get_flags", {"curve": "JPY", "tenor": "3M"}
    )
    assert ev.evidence_id == "ev-c5b82df2fabe4d39"


@pytest.mark.parametrize(
    ("kind", "source", "data"),
    [
        ("point_change", "mdx_data.get_flags", {"curve": "JPY", "tenor": "6M"}),  # data
        ("history_stats", "mdx_data.get_flags", {"curve": "JPY", "tenor": "3M"}),  # kind
        ("point_change", "mdx_data.get_history", {"curve": "JPY", "tenor": "3M"}),  # source
    ],
)
def test_any_content_change_gives_a_different_id(
    kind: str, source: str, data: dict[str, str]
) -> None:
    base = Evidence.create("point_change", "mdx_data.get_flags", {"curve": "JPY", "tenor": "3M"})
    assert Evidence.create(kind, source, data).evidence_id != base.evidence_id  # type: ignore[arg-type]


def test_id_format() -> None:
    ev = Evidence.create("k", "s")
    assert ev.evidence_id.startswith("ev-")
    assert len(ev.evidence_id) == 3 + 16


def test_id_that_does_not_match_content_is_rejected() -> None:
    ev = Evidence.create("k", "s", {"x": 1})
    with pytest.raises(ValidationError, match="does not match its content"):
        Evidence(evidence_id=ev.evidence_id, kind="k", source="s", data={"x": 2})


def test_evidence_is_immutable() -> None:
    ev = Evidence.create("k", "s", {"x": 1})
    with pytest.raises(ValidationError):
        ev.kind = "other"  # type: ignore[misc]


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json({"x": math.nan})


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "é"]}) == '{"a":[1.5,"é"],"b":1}'

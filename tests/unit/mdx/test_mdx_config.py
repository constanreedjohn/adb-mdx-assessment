"""MDX config schemas, and the repository's real `config/` tree."""

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import ConfigError
from mdx_assessment.core.orchestration.playbooks import PlaybookDocument
from mdx_assessment.settings import PROJECT_ROOT
from mdx_assessment.usecases.mdx.config import (
    DATA_TYPES_PATH,
    EXPECTED_FILES_PATH,
    DataTypeProfiles,
    ExpectedFiles,
)
from mdx_assessment.usecases.mdx.domain import DataType, MarketDataKind, Unit

REPO_CONFIG = PROJECT_ROOT / "config"


def test_repository_config_is_valid() -> None:
    loaded = load_config(REPO_CONFIG)
    rate = loaded.get(DATA_TYPES_PATH, DataTypeProfiles).for_kind(MarketDataKind.RATE)
    assert (rate.quote_unit, rate.measure_unit) == (Unit.PERCENT, Unit.BPS)
    assert rate.provisional, "near-zero floor stays provisional until Q-05 is answered"
    files = loaded.get(EXPECTED_FILES_PATH, ExpectedFiles).for_data_type(DataType.LINEAR)
    assert [f.expected_record_count for f in files] == [80]
    assert loaded.get("playbooks/normal_eod.yaml", PlaybookDocument).id == "normal_eod"
    assert "smoke_test" in loaded.prompts


def test_an_invalid_repository_file_fails_startup(tmp_path: Path) -> None:
    broken = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, broken)
    text = (
        (broken / DATA_TYPES_PATH)
        .read_text()
        .replace("near_zero_floor: 0.10", "near_zero_floor: 0")
    )
    (broken / DATA_TYPES_PATH).write_text(text)
    with pytest.raises(ConfigError) as exc:
        build_registries().config.load_config(broken)
    assert exc.value.problems == [
        "profiles/data_types.yaml: profiles.rate.near_zero_floor: Input should be greater than 0"
    ]


def _profile(**overrides: Any) -> dict[str, Any]:
    return {"quote_unit": "percent", "measure_unit": "bps", "near_zero_floor": 0.1, **overrides}


def test_profile_units_must_fit_the_kind() -> None:
    with pytest.raises(ValidationError, match="rate: measure_unit pips"):
        DataTypeProfiles.model_validate({"profiles": {"rate": _profile(measure_unit="pips")}})


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DataTypeProfiles.model_validate({"profiles": {"equity": _profile()}})


def test_missing_profile_lookup() -> None:
    profiles = DataTypeProfiles.model_validate({"profiles": {"rate": _profile()}})
    with pytest.raises(KeyError, match="fx_vol"):
        profiles.for_kind(MarketDataKind.FX_VOL)


def _files(*entries: dict[str, Any], timezone: str = "Asia/Manila") -> dict[str, Any]:
    return {"timezone": timezone, "files": list(entries)}


def _file(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "mdx_linear_rates",
        "data_type": "linear",
        "name_pattern": "mdx_linear_rates_{yyyymmdd}.csv",
        "expected_record_count": 80,
        **overrides,
    }


def test_file_name_for_business_date() -> None:
    files = ExpectedFiles.model_validate(_files(_file()))
    assert files.files[0].file_name_for(date(2026, 10, 15)) == "mdx_linear_rates_20261015.csv"


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        ("mdx_linear_rates.csv", "exactly once"),
        ("a_{yyyymmdd}_{yyyymmdd}.csv", "exactly once"),
        ("../etc/{yyyymmdd}.csv", "may only contain"),
        ("rates {yyyymmdd}.csv", "may only contain"),
    ],
)
def test_bad_name_patterns(pattern: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ExpectedFiles.model_validate(_files(_file(name_pattern=pattern)))


def test_duplicate_file_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate id"):
        ExpectedFiles.model_validate(_files(_file(), _file(name_pattern="other_{yyyymmdd}.csv")))


def test_unknown_timezone_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown IANA timezone"):
        ExpectedFiles.model_validate(_files(_file(), timezone="Mars/Olympus"))


def test_record_count_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExpectedFiles.model_validate(_files(_file(expected_record_count=0)))

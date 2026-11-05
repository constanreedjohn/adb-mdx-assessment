"""Acceptance (E2-01): the same seed gives an identical day, both planted cases are
present, and about 25 flags are raised. Plus the day's internal consistency."""

import hashlib
import json
import statistics
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mdx_assessment.app import load_config
from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.settings import PROJECT_ROOT
from mdx_assessment.settings import get_settings as settings_cache
from mdx_assessment.usecases.mdx.config import EXPECTED_FILES_PATH, ExpectedFiles
from mdx_assessment.usecases.mdx.domain import MdxFlag, Tenor, relative_change_pct
from mdx_assessment.usecases.mdx.sim import __main__ as sim_cli
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.calendar import business_days_ending
from mdx_assessment.usecases.mdx.sim.config import SIM_LINEAR_DAY_PATH, PlantedCase, SimLinearDay
from mdx_assessment.usecases.mdx.sim.generator import (
    ExpectedClass,
    GroundTruth,
    SimulatedDay,
    generate_linear_day,
)
from mdx_assessment.usecases.mdx.sim.writer import MANIFEST, render_files, write_day

BUSINESS_DATE = date(2026, 10, 15)  # a Thursday
JPY_3M = ("JPY", Tenor.parse("3M"))
GBP_2Y = ("GBP", Tenor.parse("2Y"))


@pytest.fixture(scope="module")
def config() -> LoadedConfig:
    return load_config(PROJECT_ROOT / "config")


@pytest.fixture(scope="module")
def day(config: LoadedConfig) -> SimulatedDay:
    return build_linear_day(config, BUSINESS_DATE)


def history_z(day: SimulatedDay, point: tuple[str, Tenor]) -> float:
    """Today's change against the standard deviation of the 60 previous daily changes."""
    values = day.values[point]
    changes = [(b - a) * 100 for a, b in pairwise(values[:-1])]
    return (values[-1] - values[-2]) * 100 / statistics.pstdev(changes)


def flag(day: SimulatedDay, point: tuple[str, Tenor]) -> MdxFlag:
    return next(f for f in day.flags if (f.curve_id, f.tenor) == point)


# --- acceptance --------------------------------------------------------------------------


def test_same_seed_gives_an_identical_day(config: LoadedConfig, tmp_path: Path) -> None:
    a = render_files(build_linear_day(config, BUSINESS_DATE), config_version=config.version)
    b = render_files(build_linear_day(config, BUSINESS_DATE), config_version=config.version)
    assert a == b  # byte for byte, every file
    write_day(build_linear_day(config, BUSINESS_DATE), tmp_path / "one", config_version="v")
    write_day(build_linear_day(config, BUSINESS_DATE), tmp_path / "two", config_version="v")
    first, second = (tmp_path / d / "sim" / "2026-10-15" for d in ("one", "two"))
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }


def test_other_seeds_and_dates_give_other_days(config: LoadedConfig, day: SimulatedDay) -> None:
    assert build_linear_day(config, BUSINESS_DATE, seed=7).vendor_file != day.vendor_file
    assert build_linear_day(config, date(2026, 10, 16)).vendor_file != day.vendor_file


def test_jpy_3m_threshold_artifact_is_planted(day: SimulatedDay) -> None:
    f = flag(day, JPY_3M)
    assert (f.previous_value, f.current_value) == (0.04, 0.07)
    truth = next(t for t in day.ground_truth.flags if t.flag_id == f.flag_id)
    assert truth.change_bps == 3.0
    assert truth.relative_change_pct == 75.0
    assert truth.near_zero
    assert truth.expected_class is ExpectedClass.THRESHOLD_ARTIFACT
    assert abs(history_z(day, JPY_3M)) < 1.5  # normal against its own history


def test_gbp_2y_genuine_move_is_planted(day: SimulatedDay) -> None:
    f = flag(day, GBP_2Y)
    truth = next(t for t in day.ground_truth.flags if t.flag_id == f.flag_id)
    assert truth.change_bps == 18.0
    assert not truth.near_zero
    assert truth.expected_class is ExpectedClass.GENUINE_LARGE_MOVE
    assert history_z(day, GBP_2Y) > 4  # far outside its own history
    assert day.ground_truth.review_first == (f.flag_id,)


def test_about_25_flags(day: SimulatedDay) -> None:
    assert 20 <= len(day.flags) <= 32
    assert day.ground_truth.flag_count == len(day.flags)


# --- consistency -------------------------------------------------------------------------


def test_planted_moves_are_coherent_with_neighbours(day: SimulatedDay) -> None:
    for (curve, tenor), bps in ((JPY_3M, 2.5), (GBP_2Y, 9.0)):
        for neighbour in ("1M", "6M") if curve == "JPY" else ("1Y", "3Y"):
            values = day.values[(curve, Tenor.parse(neighbour))]
            assert round((values[-1] - values[-2]) * 100, 4) == bps, (curve, neighbour, tenor)


def test_flags_follow_the_mdx_rule_exactly(config: LoadedConfig, day: SimulatedDay) -> None:
    threshold = config.get(SIM_LINEAR_DAY_PATH, SimLinearDay).mdx_rule.relative_change_pct
    flagged = {(f.curve_id, f.tenor) for f in day.flags}
    for point, values in day.values.items():
        rel = relative_change_pct(values[-2], values[-1])
        breaches = values[-1] != values[-2] if rel is None else abs(rel) > threshold
        assert breaches == (point in flagged), point


def test_history_is_60_business_days_plus_today(day: SimulatedDay) -> None:
    assert day.dates == tuple(business_days_ending(BUSINESS_DATE, 61))
    assert all(d.weekday() < 5 for d in day.dates)
    assert len(day.values) == 80
    assert all(len(series) == 61 for series in day.values.values())


def test_vendor_file_and_metadata_agree(day: SimulatedDay) -> None:
    lines = day.vendor_file.decode().splitlines()
    assert lines[0] == "curve_id,tenor,value,unit"
    assert len(lines) - 1 == day.file_metadata.record_count == 80
    assert day.file_metadata.size_bytes == len(day.vendor_file)
    assert day.file_metadata.content_sha256 == hashlib.sha256(day.vendor_file).hexdigest()
    assert day.file_metadata.file_name == "mdx_linear_rates_20261015.csv"
    assert day.file_metadata.arrived_at == datetime(
        2026, 10, 15, 16, 42, tzinfo=ZoneInfo("Asia/Manila")
    )


def test_one_ground_truth_entry_per_flag(day: SimulatedDay) -> None:
    assert [t.flag_id for t in day.ground_truth.flags] == [f.flag_id for f in day.flags]
    assert set(day.ground_truth.planted) == {"jpy_3m_artifact", "gbp_2y_genuine"}


def test_weekend_date_is_rejected(config: LoadedConfig) -> None:
    with pytest.raises(ValueError, match="not a business day"):
        build_linear_day(config, date(2026, 10, 17))


def test_a_planted_case_the_rule_misses_is_an_error(config: LoadedConfig) -> None:
    cfg = config.get(SIM_LINEAR_DAY_PATH, SimLinearDay)
    quiet = PlantedCase.model_validate(
        {
            "id": "too_small",
            "kind": "genuine_large_move",
            "curve": "USD",
            "tenor": "10Y",
            "change_bps": 0.5,
        }
    )
    with pytest.raises(ValueError, match="not flagged by the MDX rule: \\['too_small'\\]"):
        generate_linear_day(
            cfg.model_copy(update={"planted": (quiet,)}),
            BUSINESS_DATE,
            expected_file=config.get(EXPECTED_FILES_PATH, ExpectedFiles).files[0],
            timezone="Asia/Manila",
            near_zero_floor=0.1,
        )


# --- written files and the CLI ----------------------------------------------------------


def test_written_files_match_the_manifest(day: SimulatedDay, tmp_path: Path) -> None:
    folder = write_day(day, tmp_path, config_version="sha256:test")
    manifest = json.loads((folder / MANIFEST).read_text())
    assert manifest["flag_count"] == len(day.flags)
    assert manifest["config_version"] == "sha256:test"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    flags = [MdxFlag.model_validate(f) for f in json.loads((folder / "flags.json").read_text())]
    assert flags == list(day.flags)
    assert (
        GroundTruth.model_validate_json((folder / "ground_truth.json").read_text())
        == day.ground_truth
    )


def test_cli_writes_the_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MDX_DATA_DIR", str(tmp_path))
    settings_cache.cache_clear()
    try:
        assert sim_cli.main(["--date", "2026-10-15"]) == 0
    finally:
        settings_cache.cache_clear()
    out = capsys.readouterr().out
    assert "MDX flags     : 28" in out
    assert "jpy_3m_artifact: 0.04 -> 0.07  (+75.0%, +3.00 bps" in out
    assert (tmp_path / "sim" / "2026-10-15" / "mdx_linear_rates_20261015.csv").is_file()

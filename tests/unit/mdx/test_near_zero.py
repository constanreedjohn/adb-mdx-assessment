"""E8-01: Market Data Assessment, the near-zero check.

Acceptance (unit tests): JPY 3M → artifact (+3 bps); GBP 2Y → not an artifact (+18 bps,
z > 4). Also every flag of the simulated day against the answer key, which only tests read.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import PlanStep, Severity
from mdx_assessment.core.orchestration.state import ExecutionEntry, RunState, TaskStatus
from mdx_assessment.core.orchestration.tasks import TaskContext, TaskInput
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    INPUT_VALIDATION,
    MARKET_DATA_ASSESSMENT,
    NEAR_ZERO_CHECK,
)
from mdx_assessment.usecases.mdx.agents.input_validation import check_files_and_intake
from mdx_assessment.usecases.mdx.agents.market_data import MarketAssessmentResult, near_zero_check
from mdx_assessment.usecases.mdx.checks.near_zero import (
    Classification,
    HistoryStats,
    assess_near_zero,
    history_stats,
)
from mdx_assessment.usecases.mdx.checks.normalize import NormalizedFlag
from mdx_assessment.usecases.mdx.config import (
    DATA_TYPES_PATH,
    NEAR_ZERO_PATH,
    DataTypeProfiles,
    NearZeroSettings,
)
from mdx_assessment.usecases.mdx.domain import (
    ColorCode,
    MarketDataKind,
    Observation,
    Tenor,
    Unit,
)
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
CONFIG = load_config(PROJECT_ROOT / "config")
SETTINGS = CONFIG.get(NEAR_ZERO_PATH, NearZeroSettings)
RATE = CONFIG.get(DATA_TYPES_PATH, DataTypeProfiles).for_kind(MarketDataKind.RATE)
EV = ("ev-" + "a" * 16,)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def flag(curve: str, tenor: str, previous: float, current: float) -> NormalizedFlag:
    change = round((current - previous) * 100, 10)
    return NormalizedFlag(
        flag_id=f"20261015-{curve}-{tenor}",
        curve_id=curve,
        tenor=Tenor.parse(tenor),
        kind=MarketDataKind.RATE,
        color_code=ColorCode.KEY,
        rule="relative_change",
        previous=previous,
        current=current,
        quote_unit=Unit.PERCENT,
        change=change,
        measure_unit=Unit.BPS,
        relative_change_pct=round((current - previous) / abs(previous) * 100, 10),
        evidence_ids=EV,
    )


def stats(stdev: float | None, changes: int = 59) -> HistoryStats:
    return HistoryStats(changes=changes, stdev=stdev, first_date=None, last_date=None)


# --- acceptance: the two planted cases -------------------------------------------------------


def test_jpy_3m_is_a_threshold_artifact() -> None:
    result = assess_near_zero(flag("JPY", "3M", 0.04, 0.07), stats(3.5), RATE, SETTINGS)
    assert result.classification is Classification.THRESHOLD_ARTIFACT
    assert (result.change, result.relative_change_pct) == (3.0, 75.0)
    assert (result.near_zero, result.relative_change_reliable) == (True, False)
    assert result.z == 0.86
    assert result.severity is Severity.LOW


def test_gbp_2y_is_a_genuine_large_move_not_an_artifact() -> None:
    result = assess_near_zero(flag("GBP", "2Y", 4.0227, 4.2027), stats(3.5), RATE, SETTINGS)
    assert result.classification is Classification.GENUINE_LARGE_MOVE
    assert result.change == pytest.approx(18.0)
    assert result.z is not None and result.z > 4
    assert (result.near_zero, result.relative_change_reliable) == (False, True)
    assert result.severity is Severity.HIGH


# --- the rules, case by case -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("f", "stdev", "expected"),
    [
        # A large move on a near-zero rate is still a large move: rule 1 comes first.
        (flag("JPY", "1M", 0.04, 0.20), 3.5, Classification.GENUINE_LARGE_MOVE),
        # Near zero, normal z, but more than 5 bps: not an artifact.
        (flag("JPY", "1M", 0.02, 0.08), 3.5, Classification.ORDINARY),
        # Exactly at the bps cap still counts as small.
        (flag("JPY", "1M", 0.04, 0.09), 3.5, Classification.THRESHOLD_ARTIFACT),
        # Small move, but not near zero: ordinary.
        (flag("EUR", "3M", 2.0156, 2.0752), 3.5, Classification.ORDINARY),
        # Exactly at the large-move cut-off: large.
        (flag("USD", "5Y", 4.00, 4.14), 3.5, Classification.GENUINE_LARGE_MOVE),
    ],
    ids=["near-zero-large", "near-zero-over-cap", "at-cap", "not-near-zero", "at-cut-off"],
)
def test_label_rules(f: NormalizedFlag, stdev: float, expected: Classification) -> None:
    assert assess_near_zero(f, stats(stdev), RATE, SETTINGS).classification is expected


@pytest.mark.parametrize(
    "history",
    [stats(3.5, changes=10), stats(0.0), stats(None, changes=1)],
    ids=["short", "flat", "none"],
)
def test_without_usable_history_z_is_not_guessed(history: HistoryStats) -> None:
    result = assess_near_zero(flag("JPY", "3M", 0.04, 0.07), history, RATE, SETTINGS)
    assert (result.classification, result.z) == (Classification.INSUFFICIENT_HISTORY, None)
    assert "can't be computed" in result.reasons[-1]


def test_each_label_says_why() -> None:
    result = assess_near_zero(flag("JPY", "3M", 0.04, 0.07), stats(3.5), RATE, SETTINGS)
    assert result.reasons == (
        "previous 0.04% is below the 0.1% floor: relative change unreliable",
        "change +3.00 bps is within 5 bps and z +0.86 is normal: the percentage overstates "
        "a small move",
    )


def test_history_excludes_today_and_keeps_the_window() -> None:
    values = [1.0 + 0.01 * (i % 2) for i in range(70)]  # alternating ±1 bp
    days = [date.fromordinal(DAY.toordinal() - 69 + i) for i in range(70)]
    observations = [
        Observation.model_construct(business_date=d, value=v)
        for d, v in zip(days, values, strict=True)
    ]
    observations[-1] = Observation.model_construct(business_date=DAY, value=5.0)  # today: huge
    result = history_stats(observations, DAY, RATE, SETTINGS)
    assert result.changes == SETTINGS.history_window_days - 1
    assert result.stdev == pytest.approx(1.0, abs=0.02)  # today's jump left out
    assert result.last_date < DAY if result.last_date else False


def test_settings_reject_a_window_too_small_for_the_minimum() -> None:
    with pytest.raises(ValidationError, match="can't exceed"):
        NearZeroSettings(
            version="1",
            history_window_days=10,
            min_history_changes=20,
            large_move_z=4,
            artifact_max_change=5,
        )


# --- the task on the simulated day -----------------------------------------------------------


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(CONFIG, DAY), path, config_version=CONFIG.version)
    return path


@dataclass
class Outcome:
    result: MarketAssessmentResult
    intake: dict[str, Any]
    audit: InMemoryAuditTrail


@asynccontextmanager
async def assessed(config: LoadedConfig, data_dir: Path) -> AsyncIterator[Outcome]:
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    audit = InMemoryAuditTrail()
    context = ToolContext(config=config, settings=settings, audit=audit, cases=InMemoryCaseStore())
    matrix = config.get(PERMISSIONS_PATH, PermissionMatrix)
    registries = build_registries()
    budgets = {t.id: t.budget for t in registries.tasks}
    async with ToolHub(registries.tools, matrix, context) as hub:

        def ctx(state: RunState, step_id: str, task: str, agent: str) -> TaskContext:
            return TaskContext(
                inputs=TaskInput(
                    session_id="s1", business_date=DAY, data_type="linear", step_id=step_id
                ),
                state=state,
                step=PlanStep(step_id=step_id, task=task),
                tools_client=hub.client(
                    agent, session_id="s1", task_id=f"s1/{step_id}", budget=budgets[task]
                ),
                gateway=None,
                loaded_config=config,
            )

        state = RunState(session_id="s1", business_date=DAY, data_type="linear")
        intake = await check_files_and_intake(
            ctx(state, "v", CHECK_FILES_AND_INTAKE, INPUT_VALIDATION)
        )
        dumped = intake.model_dump(mode="json")
        now = state.business_date
        entry = ExecutionEntry.model_validate(
            {
                "step_id": "v",
                "task": CHECK_FILES_AND_INTAKE,
                "status": TaskStatus.DONE,
                "started_at": f"{now}T09:00:00Z",
                "finished_at": f"{now}T09:00:01Z",
            }
        )
        state = state.model_copy(update={"execution": [entry], "agent_results": {"v": dumped}})
        result = await near_zero_check(ctx(state, "a", NEAR_ZERO_CHECK, MARKET_DATA_ASSESSMENT))
        yield Outcome(result, dumped, audit)


async def test_every_flag_gets_the_answer_keys_label(data_dir: Path) -> None:
    truth = json.loads((data_dir / "sim" / "2026-10-15" / "ground_truth.json").read_text())
    expected = {f["flag_id"]: f["expected_class"] for f in truth["flags"]}
    async with assessed(CONFIG, data_dir) as outcome:
        got = {
            i.assessment.flag_id: i.assessment.classification.value for i in outcome.result.items
        }
    assert got == expected
    assert outcome.result.counts == {
        Classification.GENUINE_LARGE_MOVE: 1,
        Classification.THRESHOLD_ARTIFACT: 4,
        Classification.ORDINARY: 23,
        Classification.INSUFFICIENT_HISTORY: 0,
    }


async def test_review_order_puts_gbp_first_and_artifacts_last(data_dir: Path) -> None:
    async with assessed(CONFIG, data_dir) as outcome:
        items = [i.assessment for i in outcome.result.items]
    assert items[0].item_key == "GBP 2Y"
    assert items[0].z is not None and items[0].z > 4
    assert {a.item_key for a in items[-4:]} == {"JPY 3M", "JPY 1M", "CHF 1M", "CHF 6M"}
    jpy = next(a for a in items if a.item_key == "JPY 3M")
    assert jpy.z == pytest.approx(0.86, abs=0.25)  # measured from history; ADB's example: 0.8


async def test_each_finding_cites_existing_evidence(data_dir: Path) -> None:
    async with assessed(CONFIG, data_dir) as outcome:
        known = {e["evidence_id"] for e in outcome.intake["evidence"]}
        known |= {e.evidence_id for e in outcome.result.evidence}
        for item in outcome.result.items:
            assert set(item.finding.evidence_ids) <= known
            assert item.finding.kind == item.assessment.classification.value
        gbp = outcome.result.items[0].finding
    assert gbp.summary.startswith("GBP 2Y: +4.5%, +18.00 bps, z +")
    assert {m.name for m in gbp.measures} == {"change", "relative_change", "z", "normal_daily_move"}


async def test_history_is_read_as_market_data_assessment(data_dir: Path) -> None:
    async with assessed(CONFIG, data_dir) as outcome:
        calls = [e for e in outcome.audit.read() if e.payload.get("operation") == "get_history"]
    assert len(calls) == 28
    assert {e.actor for e in calls} == {MARKET_DATA_ASSESSMENT}


async def test_the_check_needs_the_intake_step() -> None:
    ctx = TaskContext(
        inputs=TaskInput(session_id="s1", business_date=DAY, data_type="linear", step_id="a"),
        state=RunState(session_id="s1", business_date=DAY, data_type="linear"),
        step=PlanStep(step_id="a", task=NEAR_ZERO_CHECK),
        tools_client=None,
        gateway=None,
        loaded_config=CONFIG,
    )
    with pytest.raises(ValueError, match="needs a completed intake step"):
        await near_zero_check(ctx)

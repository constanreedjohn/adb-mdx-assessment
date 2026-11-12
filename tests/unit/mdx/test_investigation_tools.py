"""E8-03: the Market Data Assessment Agent's investigation tools.

Acceptance: each tool returns the expected result on the simulated day as an evidence
record; `flag_frequency` names its source; no tool can write.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import JsonValue

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.domain import Evidence, PlanStep
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.orchestration.tasks import TaskContext, TaskInput
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.registry.specs import ToolAccess
from mdx_assessment.core.tools.client import ToolCallError, ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.ids import (
    CHECK_FILES_AND_INTAKE,
    INPUT_VALIDATION,
    MARKET_DATA_ASSESSMENT,
    NEAR_ZERO_CHECK,
)
from mdx_assessment.usecases.mdx.agents.input_validation import (
    IntakeResult,
    check_files_and_intake,
)
from mdx_assessment.usecases.mdx.config import (
    INVESTIGATION_PATH,
    NEAR_ZERO_PATH,
    InvestigationSettings,
    NearZeroSettings,
)
from mdx_assessment.usecases.mdx.domain import DataType
from mdx_assessment.usecases.mdx.investigation.compute import is_flagged_by_rule
from mdx_assessment.usecases.mdx.investigation.reads import MdxReads
from mdx_assessment.usecases.mdx.investigation.toolkit import (
    FLAG_FREQUENCY,
    NEIGHBOUR_MOVES,
    POINT_HISTORY,
    SAME_TENOR_OTHER_CURVES,
    Investigation,
    InvestigationError,
    catalogue,
)
from mdx_assessment.usecases.mdx.investigation.toolkit import (
    TOOLS as TOOL_LIST,
)
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.config import SIM_LINEAR_DAY_PATH, SimLinearDay
from mdx_assessment.usecases.mdx.sim.writer import write_day
from mdx_assessment.usecases.mdx.tools.mdx import MDX_TOOL_SPEC

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
CONFIG = load_config(PROJECT_ROOT / "config")
TOOLS = CONFIG.get(INVESTIGATION_PATH, InvestigationSettings)
WINDOW = CONFIG.get(NEAR_ZERO_PATH, NearZeroSettings).history_window_days
GBP_2Y, JPY_3M = "20261015-GBP-2Y", "20261015-JPY-3M"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_day(build_linear_day(CONFIG, DAY), path, config_version=CONFIG.version)
    return path


@dataclass
class Bench:
    investigation: Investigation
    reads: MdxReads
    intake: IntakeResult
    audit: InMemoryAuditTrail
    crucial_calls: int
    """MDX calls the crucial reads made, before any tool ran."""


@asynccontextmanager
async def bench(data_dir: Path, settings: InvestigationSettings = TOOLS) -> AsyncIterator[Bench]:
    base = get_settings()
    app = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    audit = InMemoryAuditTrail()
    context = ToolContext(config=CONFIG, settings=app, audit=audit, cases=InMemoryCaseStore())
    registries = build_registries()
    budgets = {t.id: t.budget for t in registries.tasks}
    async with ToolHub(
        registries.tools, CONFIG.get(PERMISSIONS_PATH, PermissionMatrix), context
    ) as hub:

        def ctx(step_id: str, task: str, agent: str) -> TaskContext:
            return TaskContext(
                inputs=TaskInput(
                    session_id="s1", business_date=DAY, data_type="linear", step_id=step_id
                ),
                state=RunState(session_id="s1", business_date=DAY, data_type="linear"),
                step=PlanStep(step_id=step_id, task=task),
                tools_client=hub.client(
                    agent, session_id="s1", task_id=f"s1/{step_id}", budget=budgets[task]
                ),
                gateway=None,
                loaded_config=CONFIG,
            )

        intake = await check_files_and_intake(ctx("v", CHECK_FILES_AND_INTAKE, INPUT_VALIDATION))
        reads = MdxReads(ctx("a", NEAR_ZERO_CHECK, MARKET_DATA_ASSESSMENT).tools)
        for f in intake.normalized_flags:  # what the near-zero check reads first
            await reads.history(f.curve_id, f.tenor, DAY, WINDOW + 1)
        yield Bench(
            Investigation(
                reads,
                intake.normalized_flags,
                intake.profiles,
                DAY,
                DataType.LINEAR,
                settings,
                WINDOW + 1,
            ),
            reads,
            intake,
            audit,
            reads.calls,
        )


async def run(b: Bench, tool: str, flag_id: str, **args: JsonValue) -> dict[str, Any]:
    evidence = await b.investigation.run(tool, {"flag_id": flag_id, **args})
    assert evidence.kind == tool and evidence.source == f"investigation.{tool}"
    return evidence.data


# --- acceptance: each tool on the simulated day ----------------------------------------------
# Planted cases (config/sim/linear_day.yaml): GBP 2Y +18 bps with its 1Y and 3Y +9 bps;
# JPY 3M +3 bps with its neighbours +2.5 bps.


async def test_neighbour_moves_shows_gbp_2y_moved_twice_its_neighbours(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        gbp = await run(b, NEIGHBOUR_MOVES, GBP_2Y)
        jpy = await run(b, NEIGHBOUR_MOVES, JPY_3M)
    assert [(n["tenor"], n["change"]) for n in gbp["neighbours"]] == [("1Y", 9.0), ("3Y", 9.0)]
    assert (gbp["change"], gbp["neighbour_mean_change"], gbp["gap_to_neighbour_mean"]) == (
        18.0,
        9.0,
        9.0,
    )
    assert gbp["same_direction"] == 2 and gbp["unit"] == "bps"
    assert gbp["previous_date"] == "2026-10-14"
    assert [(n["tenor"], n["change"]) for n in jpy["neighbours"]] == [("1M", 2.5), ("6M", 2.5)]
    assert jpy["gap_to_neighbour_mean"] == 0.5


async def test_same_tenor_other_curves_covers_every_other_curve(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        gbp = await run(b, SAME_TENOR_OTHER_CURVES, GBP_2Y)
    curves = {c.id for c in CONFIG.get(SIM_LINEAR_DAY_PATH, SimLinearDay).curves} - {"GBP"}
    assert [o["curve_id"] for o in gbp["others"]] == sorted(curves)
    assert all(o["tenor"] == "2Y" for o in gbp["others"])
    # The move is GBP's own: no other curve's 2Y moved even half as much.
    assert max(abs(o["change"]) for o in gbp["others"]) < 9
    changes = [o["change"] for o in gbp["others"]]
    assert gbp["others_mean_change"] == pytest.approx(sum(changes) / len(changes))


async def test_point_history_returns_the_window_ending_today(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        five = await run(b, POINT_HISTORY, JPY_3M, days=5)
        full = await run(b, POINT_HISTORY, JPY_3M, days=TOOLS.point_history_max_days)
        assert b.reads.calls == b.crucial_calls  # both sliced from the crucial read
    assert (five["days"], five["days_requested"]) == (5, 5)
    obs = five["observations"]
    assert obs[-1] == {"business_date": "2026-10-15", "value": 0.07, "change": 3.0}
    assert obs[0]["change"] is None
    assert five["min_value"] == min(o["value"] for o in obs)
    assert five["largest_change"] == max((o["change"] for o in obs[1:]), key=abs)
    assert full["days"] == TOOLS.point_history_max_days
    tail = full["observations"][-5:]
    assert [o["value"] for o in tail] == [o["value"] for o in obs]
    assert tail[1:] == obs[1:]  # the first of the short window has no day before it


async def test_flag_frequency_names_its_source_on_the_simulated_day(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        jpy = await run(b, FLAG_FREQUENCY, JPY_3M)
        gbp = await run(b, FLAG_FREQUENCY, GBP_2Y)
    for result in (jpy, gbp):
        # The simulated MDX holds no past flags, so the configured rule is re-applied.
        assert result["source"] == "configured_rule"
        assert result["rule"] == {"name": "relative_change", "relative_change_pct": 2.0}
        assert "no past flags" in result["source_note"]
        assert result["days_checked"] == TOOLS.flag_frequency_window_days - 1
        assert result["last_date"] == "2026-10-14"  # today is never re-detected
        assert len(result["flagged_dates"]) == result["days_flagged"]
    # A near-zero rate trips the percentage rule almost every day; GBP 2Y rarely does.
    assert jpy["days_flagged"] >= 50
    assert gbp["days_flagged"] <= 5


# --- the configured rule is MDX's rule ----------------------------------------------------------


def test_the_configured_rule_matches_the_simulated_mdx() -> None:
    sim = CONFIG.get(SIM_LINEAR_DAY_PATH, SimLinearDay)
    assert TOOLS.mdx_rule.relative_change_pct == sim.mdx_rule.relative_change_pct


async def test_re_applying_the_rule_to_today_reproduces_mdxs_flags(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        today = await b.reads.curve_values(DAY)
        previous = await b.reads.curve_values(date(2026, 10, 14))
    assert today is not None and previous is not None
    flagged = {
        f"20261015-{c}-{t}"
        for (c, t), value in today.items()
        if is_flagged_by_rule(previous[(c, t)], value, TOOLS.mdx_rule)
    }
    assert flagged == {f.flag_id for f in b.intake.normalized_flags}


@pytest.mark.parametrize(
    ("previous", "current", "flagged"),
    [
        (2.0, 2.04, False),
        (2.0, 2.0401, True),
        (2.0, 1.959, True),
        (0.0, 0.0, False),
        (0.0, 0.01, True),
    ],
    ids=["at-threshold", "above", "down", "zero-unchanged", "from-zero"],
)
def test_rule_edges(previous: float, current: float, flagged: bool) -> None:
    assert is_flagged_by_rule(previous, current, TOOLS.mdx_rule) is flagged


# --- MDX's own past flags, when it has them -------------------------------------------------------


class FakeMdx:
    """Stands in for the MDX tool client: six business days, past flags on two of them."""

    DAYS: ClassVar = (
        date(2026, 10, 8),
        date(2026, 10, 9),
        date(2026, 10, 12),
        date(2026, 10, 13),
        date(2026, 10, 14),
        DAY,
    )
    FLAGGED: ClassVar = {date(2026, 10, 9), date(2026, 10, 13)}

    def __init__(self, has_flags: bool) -> None:
        self.has_flags = has_flags
        self.calls: list[str] = []

    async def call(
        self, tool_id: str, operation: str, arguments: Mapping[str, JsonValue] | None = None
    ) -> dict[str, Any]:
        self.calls.append(operation)
        args = dict(arguments or {})
        if operation == "get_history":
            return {
                "curve_id": "JPY",
                "tenor": "3M",
                "kind": "rate",
                "unit": "percent",
                "observations": [
                    {"business_date": d.isoformat(), "value": 0.04 + i / 100}
                    for i, d in enumerate(self.DAYS)
                ],
            }
        on = date.fromisoformat(str(args["business_date"]))
        if not self.has_flags:
            raise ToolCallError(f"no MDX data for {on}")
        flags = (
            [
                {
                    "flag_id": f"{on:%Y%m%d}-JPY-3M",
                    "business_date": on.isoformat(),
                    "data_type": "linear",
                    "kind": "rate",
                    "curve_id": "JPY",
                    "tenor": "3M",
                    "rule": "relative_change",
                    "unit": "percent",
                    "color_code": "key",
                }
            ]
            if on in self.FLAGGED
            else []
        )
        return {
            "business_date": on.isoformat(),
            "data_type": "linear",
            "count": len(flags),
            "flags": flags,
        }


def fake_investigation(data_dir_intake: IntakeResult, mdx: FakeMdx) -> Investigation:
    settings = TOOLS.model_copy(update={"flag_frequency_window_days": 5})
    jpy = [f for f in data_dir_intake.normalized_flags if f.flag_id == JPY_3M]
    reads = MdxReads(mdx)  # type: ignore[arg-type]
    return Investigation(reads, jpy, data_dir_intake.profiles, DAY, DataType.LINEAR, settings, 6)


async def test_flag_frequency_uses_mdxs_own_flags_when_it_has_them(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        intake = b.intake
    mdx = FakeMdx(has_flags=True)
    data = (await fake_investigation(intake, mdx).run(FLAG_FREQUENCY, {"flag_id": JPY_3M})).data
    assert data["source"] == "mdx_flags"
    assert data["rule"] == {"name": "relative_change"}
    assert (data["days_checked"], data["days_flagged"]) == (4, 2)
    assert data["flagged_dates"] == ["2026-10-09", "2026-10-13"]
    assert mdx.calls.count("get_flags") == 4  # one per day checked, never today's


async def test_flag_frequency_falls_back_when_mdx_has_no_past_flags(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        intake = b.intake
    mdx = FakeMdx(has_flags=False)
    investigation = fake_investigation(intake, mdx)
    first = (await investigation.run(FLAG_FREQUENCY, {"flag_id": JPY_3M})).data
    await investigation.run(FLAG_FREQUENCY, {"flag_id": JPY_3M})
    assert first["source"] == "configured_rule"
    assert first["days_checked"] == 4
    assert mdx.calls.count("get_flags") == 1  # one probe per step, then cached


# --- evidence, reuse and limits -------------------------------------------------------------------


async def test_the_same_facts_give_the_same_evidence_id(data_dir: Path) -> None:
    async with bench(data_dir) as b:
        first = await b.investigation.run(NEIGHBOUR_MOVES, {"flag_id": GBP_2Y})
    async with bench(data_dir) as b:
        again = await b.investigation.run(NEIGHBOUR_MOVES, {"flag_id": GBP_2Y})
    assert first == again
    assert Evidence.model_validate(first.model_dump()) == first


async def test_every_tool_on_every_flag_costs_three_mdx_calls(data_dir: Path) -> None:
    """Today's and yesterday's curve values once, and one past-flags probe. History comes
    from the crucial reads."""
    async with bench(data_dir) as b:
        for f in b.intake.normalized_flags:
            for tool in (NEIGHBOUR_MOVES, SAME_TENOR_OTHER_CURVES, FLAG_FREQUENCY):
                await b.investigation.run(tool, {"flag_id": f.flag_id})
            await b.investigation.run(POINT_HISTORY, {"flag_id": f.flag_id, "days": 20})
        assert b.reads.calls - b.crucial_calls == 3


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("compute_z", {"flag_id": GBP_2Y}, "unknown tool"),
        (NEIGHBOUR_MOVES, {"flag_id": "20261015-XXX-2Y"}, "unknown flag"),
        (NEIGHBOUR_MOVES, {"flag_id": GBP_2Y, "write": True}, "invalid arguments"),
        (POINT_HISTORY, {"flag_id": GBP_2Y}, "invalid arguments"),
        (POINT_HISTORY, {"flag_id": GBP_2Y, "days": 61}, "at most 60 days"),
    ],
    ids=["unknown-tool", "unknown-flag", "extra-argument", "missing-days", "window-too-long"],
)
async def test_bad_requests_are_refused_in_words(
    data_dir: Path, tool: str, args: dict[str, JsonValue], message: str
) -> None:
    async with bench(data_dir) as b:
        with pytest.raises(InvestigationError, match=message):
            await b.investigation.run(tool, args)


# --- no tool can write ------------------------------------------------------------------------------


async def test_the_reader_only_calls_read_operations() -> None:
    reads = MdxReads(FakeMdx(has_flags=True))  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="not a read operation"):
        await reads._read("create_case", {})
    assert all(access is ToolAccess.READ for access in MDX_TOOL_SPEC.operations.values())


def test_the_assessment_agent_may_only_read_mdx() -> None:
    matrix = CONFIG.get(PERMISSIONS_PATH, PermissionMatrix)
    assert matrix.tools["mdx_data"][MARKET_DATA_ASSESSMENT] == (ToolAccess.READ,)


def test_the_catalogue_gives_the_student_ids_descriptions_and_arguments() -> None:
    entries = catalogue(TOOLS)
    assert [e["id"] for e in entries] == [t.id for t in TOOL_LIST]
    assert [e["id"] for e in entries] == [
        NEIGHBOUR_MOVES,
        FLAG_FREQUENCY,
        POINT_HISTORY,
        SAME_TENOR_OTHER_CURVES,
    ]
    for entry in entries:
        schema = entry["arguments"]
        assert isinstance(schema, dict) and isinstance(entry["description"], str)
        assert "flag_id" in schema["required"]  # type: ignore[operator]
        assert schema["additionalProperties"] is False
    history = next(e for e in entries if e["id"] == POINT_HISTORY)["arguments"]
    assert history["properties"]["days"]["maximum"] == TOOLS.point_history_max_days  # type: ignore[index,call-overload]

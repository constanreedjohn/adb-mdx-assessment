"""Acceptance (E3-01): each operation returns the simulated day's data through an MCP client.
Also: read-only tools, readable errors, and the answer key never reachable."""

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from mdx_assessment.app import load_config
from mdx_assessment.core.registry import ToolSpec
from mdx_assessment.settings import PROJECT_ROOT
from mdx_assessment.usecases.mdx.config import EXPECTED_FILES_PATH, ExpectedFiles
from mdx_assessment.usecases.mdx.domain import DataType, FileMetadata, MdxFlag, Series, Tenor
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.generator import SimulatedDay
from mdx_assessment.usecases.mdx.sim.writer import GROUND_TRUTH, write_day
from mdx_assessment.usecases.mdx.tools.mdx import (
    GET_CURVE_VALUES,
    GET_FILE_METADATA,
    GET_FLAGS,
    GET_HISTORY,
    LIST_EXPECTED_FILES,
    MDX_TOOL_SPEC,
    MdxDataError,
    SimulatedMdxBackend,
    create_mdx_server,
)
from mdx_assessment.usecases.mdx.tools.mdx.backend import SERVED_FILES

pytestmark = pytest.mark.anyio
DAY = "2026-10-15"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def sim(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SimulatedDay, ExpectedFiles]:
    config = load_config(PROJECT_ROOT / "config")
    day = build_linear_day(config, date(2026, 10, 15))
    data_dir = tmp_path_factory.mktemp("data")
    write_day(day, data_dir, config_version=config.version)
    return data_dir, day, config.get(EXPECTED_FILES_PATH, ExpectedFiles)


@pytest.fixture
async def client(sim: tuple[Path, SimulatedDay, ExpectedFiles]) -> AsyncIterator[Client]:
    data_dir, _, expected = sim
    async with Client(create_mdx_server(SimulatedMdxBackend(data_dir, expected))) as c:
        yield c


async def call(client: Client, tool: str, **args: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def error_of(client: Client, tool: str, **args: Any) -> str:
    result = await client.call_tool(tool, args)
    assert result.is_error
    return " ".join(c.text for c in result.content if isinstance(c, TextContent))


# --- tool surface ------------------------------------------------------------------------


async def test_five_read_only_tools(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        LIST_EXPECTED_FILES,
        GET_FILE_METADATA,
        GET_FLAGS,
        GET_CURVE_VALUES,
        GET_HISTORY,
    }
    for tool in tools.values():
        assert tool.annotations is not None and tool.annotations.read_only_hint is True
        assert tool.output_schema is not None


async def test_tool_spec() -> None:
    assert isinstance(MDX_TOOL_SPEC, ToolSpec)
    assert (MDX_TOOL_SPEC.id, MDX_TOOL_SPEC.group.value) == ("mdx_data", "adb_system")


# --- acceptance: each operation returns the simulated day's data -------------------------


async def test_list_expected_files(client: Client) -> None:
    result = await call(client, LIST_EXPECTED_FILES, business_date=DAY)
    [expected] = result["files"]
    assert expected["file_name"] == "mdx_linear_rates_20261015.csv"
    assert expected["expected_record_count"] == 80
    assert expected["expected_by"] == "2026-10-15T17:00:00+08:00"
    assert (await call(client, LIST_EXPECTED_FILES, business_date=DAY, data_type="nonlinear"))[
        "files"
    ] == []


async def test_get_file_metadata(
    client: Client, sim: tuple[Path, SimulatedDay, ExpectedFiles]
) -> None:
    result = await call(client, GET_FILE_METADATA, business_date=DAY)
    assert [FileMetadata.model_validate(f) for f in result["files"]] == [sim[1].file_metadata]


async def test_get_flags(client: Client, sim: tuple[Path, SimulatedDay, ExpectedFiles]) -> None:
    result = await call(client, GET_FLAGS, business_date=DAY)
    assert result["count"] == 28
    assert [MdxFlag.model_validate(f) for f in result["flags"]] == list(sim[1].flags)
    jpy = next(f for f in result["flags"] if f["flag_id"] == "20261015-JPY-3M")
    assert (jpy["previous_value"], jpy["current_value"], jpy["unit"]) == (0.04, 0.07, "percent")


async def test_get_curve_values(client: Client) -> None:
    result = await call(client, GET_CURVE_VALUES, business_date=DAY)
    assert result["count"] == 80
    jpy_3m = next(p for p in result["points"] if (p["curve_id"], p["tenor"]) == ("JPY", "3M"))
    assert jpy_3m["value"] == 0.07
    gbp = await call(client, GET_CURVE_VALUES, business_date=DAY, curve_id="GBP")
    assert gbp["count"] == 10
    assert {p["curve_id"] for p in gbp["points"]} == {"GBP"}


async def test_curve_values_for_an_earlier_day_come_from_history(client: Client) -> None:
    result = await call(client, GET_CURVE_VALUES, business_date="2026-10-14", curve_id="JPY")
    assert next(p for p in result["points"] if p["tenor"] == "3M")["value"] == 0.04


async def test_get_history(client: Client, sim: tuple[Path, SimulatedDay, ExpectedFiles]) -> None:
    series = Series.model_validate(
        await call(client, GET_HISTORY, curve_id="JPY", tenor="3M", end_date=DAY, days=3)
    )
    assert [o.value for o in series.observations][-2:] == [0.04, 0.07]
    assert [o.business_date.isoformat() for o in series.observations] == [
        "2026-10-13",
        "2026-10-14",
        DAY,
    ]
    full = Series.model_validate(
        await call(client, GET_HISTORY, curve_id="GBP", tenor="2Y", end_date=DAY)
    )
    assert len(full.observations) == 61  # default: 60 days of history + today
    assert list(full.values) == list(sim[1].values[("GBP", Tenor.parse("2Y"))])


async def test_history_is_capped_by_what_exists(client: Client) -> None:
    series = await call(client, GET_HISTORY, curve_id="USD", tenor="10Y", end_date=DAY, days=250)
    assert len(series["observations"]) == 61


# --- errors ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        (GET_FLAGS, {"business_date": "2026-10-16"}, "no MDX data for 2026-10-16"),
        (GET_FILE_METADATA, {"business_date": "2026-10-16"}, "no MDX data for 2026-10-16"),
        (GET_CURVE_VALUES, {"business_date": DAY, "curve_id": "XYZ"}, "unknown curve 'XYZ'"),
        (
            GET_HISTORY,
            {"curve_id": "JPY", "tenor": "4M", "end_date": DAY},
            "unknown curve point JPY 4M",
        ),
        (GET_HISTORY, {"curve_id": "JPY", "tenor": "3Q", "end_date": DAY}, "tenor"),
        (GET_HISTORY, {"curve_id": "JPY", "tenor": "3M", "end_date": DAY, "days": 251}, "days"),
        (GET_FLAGS, {"business_date": "2026-10-17"}, "weekend"),
    ],
)
async def test_bad_requests_get_readable_errors(
    client: Client, tool: str, args: dict[str, Any], message: str
) -> None:
    assert message in await error_of(client, tool, **args)


# --- the answer key is never reachable ----------------------------------------------------


ANSWER_KEY_FIELDS = ("expected_class", "true_z", "planted", "review_first", "threshold_artifact")


async def test_no_tool_output_contains_answer_key_fields(client: Client) -> None:
    outputs = [
        await client.call_tool(LIST_EXPECTED_FILES, {"business_date": DAY}),
        await client.call_tool(GET_FILE_METADATA, {"business_date": DAY}),
        await client.call_tool(GET_FLAGS, {"business_date": DAY}),
        await client.call_tool(GET_CURVE_VALUES, {"business_date": DAY}),
        await client.call_tool(GET_HISTORY, {"curve_id": "JPY", "tenor": "3M", "end_date": DAY}),
    ]
    text = " ".join(str(r.structured_content) + str(r.content) for r in outputs)
    for field in ANSWER_KEY_FIELDS:
        assert field not in text


async def test_backend_works_without_the_answer_key(
    sim: tuple[Path, SimulatedDay, ExpectedFiles], tmp_path: Path
) -> None:
    import shutil

    data_dir, _, expected = sim
    copy = tmp_path / "data"
    shutil.copytree(data_dir, copy)
    (copy / "sim" / DAY / GROUND_TRUTH).unlink()
    backend = SimulatedMdxBackend(copy, expected)
    assert len(backend.flags(date(2026, 10, 15), DataType.LINEAR)) == 28
    assert len(backend.curve_values(date(2026, 10, 15))) == 80


def test_backend_reader_refuses_files_it_does_not_serve(tmp_path: Path) -> None:
    assert GROUND_TRUTH not in SERVED_FILES
    with pytest.raises(MdxDataError, match="not an MDX file"):
        SimulatedMdxBackend._read(tmp_path, GROUND_TRUTH)

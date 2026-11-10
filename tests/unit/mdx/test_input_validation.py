"""E7-01: Input Validation, file presence and record count.

Acceptance: complete on the simulated day; a test day with a short file is marked incomplete.
"""

import json
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mdx_assessment.app import build_registries, load_config
from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.domain import PlanStep, Severity
from mdx_assessment.core.orchestration.run_context import FactsRequest
from mdx_assessment.core.orchestration.state import RunState
from mdx_assessment.core.orchestration.tasks import TaskContext, TaskInput
from mdx_assessment.core.persistence import InMemoryAuditTrail, InMemoryCaseStore
from mdx_assessment.core.tools.client import ToolHub
from mdx_assessment.core.tools.permissions import PERMISSIONS_PATH, PermissionMatrix
from mdx_assessment.core.tools.registry import ToolContext
from mdx_assessment.settings import PROJECT_ROOT, AppSettings, get_settings
from mdx_assessment.usecases.mdx.agents.input_validation import (
    DATASET_COMPLETE,
    DATASET_INCOMPLETE,
    IntakeResult,
    check_files_and_intake,
)
from mdx_assessment.usecases.mdx.agents.skeleton import CHECK_FILES_AND_INTAKE, INPUT_VALIDATION
from mdx_assessment.usecases.mdx.checks.completeness import FileStatus, assess_completeness
from mdx_assessment.usecases.mdx.facts import build_mdx_facts
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day

pytestmark = pytest.mark.anyio
DAY = date(2026, 10, 15)
FILE = "mdx_linear_rates_20261015.csv"
FILE_KINDS = {"expected_file", "file_metadata"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def config() -> LoadedConfig:
    return load_config(PROJECT_ROOT / "config")


@pytest.fixture(scope="module")
def happy_dir(config: LoadedConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("happy")
    write_day(build_linear_day(config, DAY), path, config_version=config.version)
    return path


def variant(happy_dir: Path, tmp_path: Path, files: list[dict[str, Any]] | None) -> Path:
    """A copy of the happy day with its file metadata replaced (None: no metadata at all)."""
    data_dir = tmp_path / "data"
    shutil.copytree(happy_dir, data_dir)
    metadata = data_dir / "sim" / "2026-10-15" / "file_metadata.json"
    if files is None:
        metadata.unlink()
    else:
        metadata.write_text(json.dumps(files))
    return data_dir


def happy_metadata(happy_dir: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = json.loads(
        (happy_dir / "sim" / "2026-10-15" / "file_metadata.json").read_text()
    )
    return loaded


@dataclass
class Run:
    result: IntakeResult
    audit: InMemoryAuditTrail
    hub: ToolHub


@asynccontextmanager
async def run_step(config: LoadedConfig, data_dir: Path) -> AsyncIterator[Run]:
    base = get_settings()
    settings = base.model_copy(
        update={"app": AppSettings(config_dir=base.app.config_dir, data_dir=data_dir)}
    )
    audit = InMemoryAuditTrail()
    context = ToolContext(config=config, settings=settings, audit=audit, cases=InMemoryCaseStore())
    matrix = config.get(PERMISSIONS_PATH, PermissionMatrix)
    async with ToolHub(build_registries().tools, matrix, context) as hub:
        state = RunState(session_id="s1", business_date=DAY, data_type="linear")
        ctx = TaskContext(
            inputs=TaskInput(
                session_id="s1", business_date=DAY, data_type="linear", step_id="validate"
            ),
            state=state,
            step=PlanStep(step_id="validate", task=CHECK_FILES_AND_INTAKE),
            tools_client=hub.client(INPUT_VALIDATION, session_id="s1", task_id="s1/validate"),
            gateway=None,
            loaded_config=config,
        )
        yield Run(await check_files_and_intake(ctx), audit, hub)


async def intake(config: LoadedConfig, data_dir: Path) -> IntakeResult:
    async with run_step(config, data_dir) as run:
        return run.result


# --- the simulated day -----------------------------------------------------------------------


async def test_complete_on_the_simulated_day(config: LoadedConfig, happy_dir: Path) -> None:
    result = await intake(config, happy_dir)
    assert result.dataset_complete is True
    [check] = result.completeness.files
    assert (check.file_name, check.status, check.received_records) == (FILE, FileStatus.OK, 80)
    finding = result.finding
    assert (finding.kind, finding.severity) == (DATASET_COMPLETE, Severity.INFO)
    assert finding.summary == "Complete: 1 of 1 expected files received, 80 of 80 records."
    assert {m.name: m.value for m in finding.measures} == {
        "files_received": 1,
        "records_received": 80,
    }
    assert result.flags_read == 28


async def test_the_finding_cites_the_recorded_evidence(
    config: LoadedConfig, happy_dir: Path
) -> None:
    result = await intake(config, happy_dir)
    files = [e for e in result.evidence if e.kind in FILE_KINDS]
    assert sorted(e.kind for e in files) == ["expected_file", "file_metadata"]
    assert set(result.finding.evidence_ids) == {e.evidence_id for e in files}
    metadata = next(e for e in result.evidence if e.kind == "file_metadata")
    assert metadata.source == "mdx_data.get_file_metadata"
    assert metadata.data["record_count"] == 80
    again = await intake(config, happy_dir)  # content-derived: the same IDs every run
    assert again.finding.evidence_ids == result.finding.evidence_ids


async def test_the_step_reads_mdx_as_input_validation(
    config: LoadedConfig, happy_dir: Path
) -> None:
    async with run_step(config, happy_dir) as run:
        calls = [e for e in run.audit.read() if e.event_type == "tool_call"]
        assert {e.actor for e in calls} == {INPUT_VALIDATION}
        assert [e.payload["operation"] for e in calls] == [
            "list_expected_files",
            "get_file_metadata",
            "get_flags",
            "get_curve_values",
        ]


# --- incomplete days -------------------------------------------------------------------------


async def test_a_short_file_is_marked_incomplete(
    config: LoadedConfig, happy_dir: Path, tmp_path: Path
) -> None:
    short = [{**happy_metadata(happy_dir)[0], "record_count": 79}]
    result = await intake(config, variant(happy_dir, tmp_path, short))
    assert result.dataset_complete is False
    assert result.completeness.files[0].status is FileStatus.SHORT
    finding = result.finding
    assert (finding.kind, finding.severity) == (DATASET_INCOMPLETE, Severity.HIGH)
    assert finding.summary == (
        "Incomplete: 1 of 1 expected files received, 79 of 80 records. "
        f"{FILE} has 79 of 80 records (short)."
    )


async def test_a_file_with_extra_records_is_marked_incomplete(
    config: LoadedConfig, happy_dir: Path, tmp_path: Path
) -> None:
    over = [{**happy_metadata(happy_dir)[0], "record_count": 81}]
    result = await intake(config, variant(happy_dir, tmp_path, over))
    assert result.completeness.files[0].status is FileStatus.OVER
    assert "(over)" in result.finding.summary


@pytest.mark.parametrize("files", [[], None], ids=["not-delivered", "no-metadata-for-the-date"])
async def test_a_missing_file_is_marked_incomplete(
    config: LoadedConfig, happy_dir: Path, tmp_path: Path, files: list[dict[str, Any]] | None
) -> None:
    result = await intake(config, variant(happy_dir, tmp_path, files))
    assert result.dataset_complete is False
    [check] = result.completeness.files
    assert (check.status, check.received_records) == (FileStatus.MISSING, None)
    assert f"{FILE} was not received" in result.finding.summary
    files = [e.kind for e in result.evidence if e.kind in FILE_KINDS]
    assert files == ["expected_file"]  # nothing received to cite


# --- one check for the facts and the agent ---------------------------------------------------


async def test_the_facts_and_the_agent_agree_on_a_short_day(
    config: LoadedConfig, happy_dir: Path, tmp_path: Path
) -> None:
    short = [{**happy_metadata(happy_dir)[0], "record_count": 79}]
    data_dir = variant(happy_dir, tmp_path, short)
    async with run_step(config, data_dir) as run:
        request = FactsRequest(
            state=RunState(session_id="s1", business_date=DAY, data_type="linear"),
            tools=run.hub.client("master_agent", session_id="s1", task_id="s1/facts"),
            config=config,
        )
        facts = {f.name: f for f in (await build_mdx_facts(request)).facts}
        records = facts["records_received"]
        assert (records.value, records.expected) == (
            run.result.completeness.records_received,
            run.result.completeness.records_expected,
        )


@dataclass(frozen=True)
class Expected:
    file_name: str
    expected_record_count: int


@dataclass(frozen=True)
class Received:
    file_name: str
    record_count: int


def test_unexpected_files_are_listed_but_do_not_make_the_dataset_incomplete() -> None:
    result = assess_completeness(
        [Expected("a.csv", 10)], [Received("a.csv", 10), Received("z.csv", 3)]
    )
    assert result.complete is True
    assert result.unexpected_files == ("z.csv",)
    assert result.records_received == 10  # only expected files count


def test_every_problem_is_listed() -> None:
    result = assess_completeness(
        [Expected("a.csv", 10), Expected("b.csv", 5), Expected("c.csv", 1)],
        [Received("a.csv", 9), Received("c.csv", 1)],
    )
    assert [(f.file_name, f.status) for f in result.problems] == [
        ("a.csv", FileStatus.SHORT),
        ("b.csv", FileStatus.MISSING),
    ]
    assert (result.files_received, result.records_received, result.records_expected) == (2, 10, 16)

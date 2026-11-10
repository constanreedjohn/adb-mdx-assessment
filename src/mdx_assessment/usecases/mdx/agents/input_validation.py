"""Input Validation Agent, task `input_validation.check_files_and_intake` (E7-01, E7-02).

Deterministic core, no LLM.

Part 1 (E7-01), file presence and record counts:

1. Read the expected files for the date and data type (from `config/expected_files.yaml`,
   through the MDX tool) and the metadata of the files MDX received.
2. Compare them with `assess_completeness`, the same check the run-context facts use.
3. Record each file as evidence, and write one dataset finding citing it:
   `dataset_complete` (severity info) or `dataset_incomplete` (severity high), naming
   every file that is missing, short or over.

An incomplete dataset doesn't fail the step: the step reports it, and the review point
(E10-01) decides what happens next. If MDX has no metadata for the date, every expected
file counts as missing rather than the step guessing.

Part 2 (E7-02), intake:

4. Read the day's MDX flags and stored curve values through the MDX tool, and record
   each flag and each flagged point as evidence.
5. Normalize every flag with `normalize_flags`: its change in the native unit of its
   data-type profile (rates: bps), checked against MDX's stored value. Flags that can't
   be normalized are listed with the reason, never dropped.
6. Attach the profiles used, so later steps measure in the same units and floors.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mdx_assessment.core.domain import Evidence, EvidenceId, Finding, Measure, Severity
from mdx_assessment.core.orchestration.tasks import TaskContext
from mdx_assessment.core.tools.client import ToolCallError
from mdx_assessment.usecases.mdx.checks.completeness import (
    Completeness,
    FileCheck,
    FileStatus,
    assess_completeness,
)
from mdx_assessment.usecases.mdx.checks.normalize import (
    FlagIssue,
    NormalizedFlag,
    normalize_flags,
)
from mdx_assessment.usecases.mdx.config import DATA_TYPES_PATH, DataTypeProfile, DataTypeProfiles
from mdx_assessment.usecases.mdx.domain import MarketDataKind, MdxFlag, Point
from mdx_assessment.usecases.mdx.tools.mdx import (
    GET_CURVE_VALUES,
    GET_FILE_METADATA,
    GET_FLAGS,
    LIST_EXPECTED_FILES,
    CurveValuesResult,
    ExpectedFileInfo,
    ExpectedFilesResult,
    FileMetadataResult,
    FlagsResult,
)
from mdx_assessment.usecases.mdx.tools.mdx import TOOL_ID as MDX_DATA

DATASET_COMPLETE = "dataset_complete"
DATASET_INCOMPLETE = "dataset_incomplete"
FINDING_ID = "dataset-completeness"


class IntakeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_complete: bool
    completeness: Completeness
    finding: Finding
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    flags_read: int | None = Field(ge=0)
    """Raw MDX flags for the date. None if MDX had none to give."""
    normalized_flags: tuple[NormalizedFlag, ...] = ()
    flag_issues: tuple[FlagIssue, ...] = ()
    """Flags that couldn't be normalized, with the reason."""
    profiles: dict[MarketDataKind, DataTypeProfile] = Field(default_factory=dict)
    """The data-type profiles the flags were normalized with."""


async def _call_or_none(
    ctx: TaskContext, operation: str, arguments: dict[str, JsonValue]
) -> dict[str, Any] | None:
    try:
        return await ctx.tools.call(MDX_DATA, operation, arguments)
    except ToolCallError:  # MDX has no data for this date
        return None


def _expected_evidence(business_date: str, file: ExpectedFileInfo) -> Evidence:
    return Evidence.create(
        "expected_file",
        f"{MDX_DATA}.{LIST_EXPECTED_FILES}",
        {
            "business_date": business_date,
            "file_name": file.file_name,
            "data_type": file.data_type.value,
            "expected_record_count": file.expected_record_count,
        },
    )


def _flag_evidence(flag: MdxFlag) -> Evidence:
    return Evidence.create("mdx_flag", f"{MDX_DATA}.{GET_FLAGS}", flag.model_dump(mode="json"))


def _point_evidence(point: Point) -> Evidence:
    return Evidence.create(
        "curve_value", f"{MDX_DATA}.{GET_CURVE_VALUES}", point.model_dump(mode="json")
    )


def _describe(check: FileCheck) -> str:
    if check.status is FileStatus.MISSING:
        return f"{check.file_name} was not received"
    return (
        f"{check.file_name} has {check.received_records} of {check.expected_records} "
        f"records ({check.status.value})"
    )


def completeness_finding(
    completeness: Completeness, subject: str, evidence_ids: tuple[str, ...]
) -> Finding:
    files = (
        f"{completeness.files_received} of {completeness.files_expected} expected files received"
    )
    records = f"{completeness.records_received} of {completeness.records_expected} records"
    if completeness.complete:
        kind, severity = DATASET_COMPLETE, Severity.INFO
        summary = f"Complete: {files}, {records}."
    else:
        kind, severity = DATASET_INCOMPLETE, Severity.HIGH
        problems = "; ".join(_describe(c) for c in completeness.problems)
        summary = f"Incomplete: {files}, {records}. {problems}."
    return Finding(
        finding_id=FINDING_ID,
        kind=kind,
        subject=subject,
        summary=summary,
        severity=severity,
        measures=(
            Measure(
                name="files_received",
                value=completeness.files_received,
                unit="files",
                evidence_ids=evidence_ids,
            ),
            Measure(
                name="records_received",
                value=completeness.records_received,
                unit="records",
                evidence_ids=evidence_ids,
            ),
        ),
        evidence_ids=evidence_ids,
    )


async def check_files_and_intake(ctx: TaskContext) -> IntakeResult:
    business_date = ctx.inputs.business_date.isoformat()
    day: dict[str, JsonValue] = {"business_date": business_date}
    typed: dict[str, JsonValue] = {**day, "data_type": ctx.inputs.data_type}

    expected = ExpectedFilesResult.model_validate(
        await ctx.tools.call(MDX_DATA, LIST_EXPECTED_FILES, typed)
    ).files
    if not expected:  # "complete" would hold vacuously, with nothing to cite
        raise ValueError(f"no expected files are configured for {ctx.inputs.data_type!r}")
    metadata = await _call_or_none(ctx, GET_FILE_METADATA, day)
    received = FileMetadataResult.model_validate(metadata).files if metadata else []
    completeness = assess_completeness(expected, received)

    evidence = [_expected_evidence(business_date, f) for f in expected]
    wanted = {f.file_name for f in expected}
    evidence += [
        Evidence.create(
            "file_metadata", f"{MDX_DATA}.{GET_FILE_METADATA}", f.model_dump(mode="json")
        )
        for f in received
        if f.file_name in wanted
    ]
    ids = tuple(e.evidence_id for e in evidence)
    subject = f"{ctx.inputs.data_type} MDX files for {business_date}"

    raw_flags = await _call_or_none(ctx, GET_FLAGS, typed)
    flags = FlagsResult.model_validate(raw_flags).flags if raw_flags is not None else []
    raw_points = await _call_or_none(ctx, GET_CURVE_VALUES, day) if flags else None
    points = CurveValuesResult.model_validate(raw_points).points if raw_points else []

    stored = {(p.curve_id, str(p.tenor)): p for p in points}
    flag_evidence: dict[str, tuple[EvidenceId, ...]] = {}
    for flag in flags:
        records = [_flag_evidence(flag)]
        point = stored.get((flag.curve_id, str(flag.tenor)))
        if point is not None:
            records.append(_point_evidence(point))
        evidence += records
        flag_evidence[flag.flag_id] = tuple(r.evidence_id for r in records)

    profiles = ctx.config.get(DATA_TYPES_PATH, DataTypeProfiles)
    normalization = normalize_flags(flags, points, profiles, flag_evidence)
    used = {f.kind for f in flags} & set(profiles.profiles)
    return IntakeResult(
        dataset_complete=completeness.complete,
        completeness=completeness,
        finding=completeness_finding(completeness, subject, ids),
        evidence=tuple(_unique(evidence)),
        flags_read=len(flags) if raw_flags is not None else None,
        normalized_flags=normalization.flags,
        flag_issues=normalization.issues,
        profiles={kind: profiles.profiles[kind] for kind in sorted(used)},
    )


def _unique(evidence: list[Evidence]) -> list[Evidence]:
    """Two flags on one point would record it twice; the content-derived ID dedupes it."""
    return list({e.evidence_id: e for e in evidence}.values())

"""MDX data MCP server (PDF §2.1 tool layer: ADB system tool, read-only).

| Tool                 | Returns                                                         |
| -------------------- | --------------------------------------------------------------- |
| list_expected_files  | The files MDX expects for a business date and data type          |
| get_file_metadata    | Delivery facts for the files received (time, size, records, hash)|
| get_flags            | MDX exception flags for the date                                 |
| get_curve_values     | Every curve point's value on a date (optionally one curve)       |
| get_history          | One curve point's values for the N business days up to a date    |

All tools are read-only: the solution never changes data in an ADB system (PDF §3.5).
Who may call them (PDF §2.1: Master, Input Validation and Market Data Assessment read;
Escalation and Reporting has no access) is enforced by the client wrapper (E3-03).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mdx_assessment.core.registry.specs import ToolAccess, ToolGroup
from mdx_assessment.usecases.mdx.domain import (
    BusinessDate,
    DataType,
    FileMetadata,
    MdxFlag,
    Point,
    Series,
    Tenor,
)
from mdx_assessment.usecases.mdx.tools.mdx.backend import MdxBackend, MdxDataError

SERVER_NAME = "mdx-data"
TOOL_ID = "mdx_data"
LIST_EXPECTED_FILES = "list_expected_files"
GET_FILE_METADATA = "get_file_metadata"
GET_FLAGS = "get_flags"
GET_CURVE_VALUES = "get_curve_values"
GET_HISTORY = "get_history"
MAX_HISTORY_DAYS = 250

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

DateArg = Annotated[BusinessDate, Field(description="Business date, YYYY-MM-DD.")]
DataTypeArg = Annotated[
    DataType, Field(description="linear (curves, rates, FX) or nonlinear (volatilities).")
]
CurveArg = Annotated[str, Field(min_length=3, max_length=40, description="Curve ID, e.g. 'JPY'.")]


class MdxToolSpec(BaseModel):
    """This tool as the registries see it (satisfies `core.registry.ToolSpec`)."""

    id: str = TOOL_ID
    group: ToolGroup = ToolGroup.ADB_SYSTEM
    description: str = (
        "MDX market data (read-only): expected files, file metadata, exception flags, "
        "curve values and history."
    )
    operations: dict[str, ToolAccess] = {
        name: ToolAccess.READ
        for name in (
            LIST_EXPECTED_FILES,
            GET_FILE_METADATA,
            GET_FLAGS,
            GET_CURVE_VALUES,
            GET_HISTORY,
        )
    }


MDX_TOOL_SPEC = MdxToolSpec()


class ExpectedFileInfo(BaseModel):
    file_id: str
    file_name: str
    data_type: DataType
    expected_record_count: int
    expected_by: datetime | None = Field(description="Latest expected arrival, timezone-aware.")


class ExpectedFilesResult(BaseModel):
    business_date: date
    files: list[ExpectedFileInfo]


class FileMetadataResult(BaseModel):
    business_date: date
    files: list[FileMetadata] = Field(
        description="Files received. An expected file missing here was not delivered."
    )


class FlagsResult(BaseModel):
    business_date: date
    data_type: DataType
    count: int
    flags: list[MdxFlag]


class CurveValuesResult(BaseModel):
    business_date: date
    count: int
    points: list[Point]


def _not_found(exc: MdxDataError) -> ToolError:
    return ToolError(str(exc))


def create_mdx_server(backend: MdxBackend) -> MCPServer:
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        title="MDX data",
        instructions=(
            "Read-only access to MDX: expected files, received file metadata, exception "
            "flags, curve values and history. Values are as MDX holds them (rates in percent)."
        ),
    )

    @server.tool(
        name=LIST_EXPECTED_FILES,
        description="List the input files MDX expects for a business date and data type.",
        annotations=READ_ONLY,
    )
    def list_expected_files(
        business_date: DateArg, data_type: DataTypeArg = DataType.LINEAR
    ) -> ExpectedFilesResult:
        zone = ZoneInfo(backend.timezone)
        return ExpectedFilesResult(
            business_date=business_date,
            files=[
                ExpectedFileInfo(
                    file_id=f.id,
                    file_name=f.file_name_for(business_date),
                    data_type=f.data_type,
                    expected_record_count=f.expected_record_count,
                    expected_by=(
                        datetime.combine(business_date, f.expected_by, zone)
                        if f.expected_by
                        else None
                    ),
                )
                for f in backend.expected_files(data_type)
            ],
        )

    @server.tool(
        name=GET_FILE_METADATA,
        description="Delivery facts (arrival time, size, record count, content hash) for the files received on a business date.",
        annotations=READ_ONLY,
    )
    def get_file_metadata(business_date: DateArg) -> FileMetadataResult:
        try:
            return FileMetadataResult(
                business_date=business_date, files=backend.file_metadata(business_date)
            )
        except MdxDataError as exc:
            raise _not_found(exc) from exc

    @server.tool(
        name=GET_FLAGS,
        description="MDX exception flags raised for a business date and data type.",
        annotations=READ_ONLY,
    )
    def get_flags(business_date: DateArg, data_type: DataTypeArg = DataType.LINEAR) -> FlagsResult:
        try:
            flags = backend.flags(business_date, data_type)
        except MdxDataError as exc:
            raise _not_found(exc) from exc
        return FlagsResult(
            business_date=business_date, data_type=data_type, count=len(flags), flags=flags
        )

    @server.tool(
        name=GET_CURVE_VALUES,
        description="Every curve point's value on a business date, optionally for one curve.",
        annotations=READ_ONLY,
    )
    def get_curve_values(
        business_date: DateArg, curve_id: CurveArg | None = None
    ) -> CurveValuesResult:
        try:
            points = backend.curve_values(business_date, curve_id)
        except MdxDataError as exc:
            raise _not_found(exc) from exc
        return CurveValuesResult(business_date=business_date, count=len(points), points=points)

    @server.tool(
        name=GET_HISTORY,
        description=(
            "One curve point's values for the `days` business days up to and including "
            "`end_date`, oldest first."
        ),
        annotations=READ_ONLY,
    )
    def get_history(
        curve_id: CurveArg,
        tenor: Annotated[
            str, Field(pattern=Tenor.LABEL_PATTERN, description="Tenor, e.g. '3M', '2Y'.")
        ],
        end_date: DateArg,
        days: Annotated[int, Field(ge=1, le=MAX_HISTORY_DAYS)] = 61,
    ) -> Series:
        try:
            return backend.history(curve_id, Tenor.parse(tenor), end_date, days)
        except MdxDataError as exc:
            raise _not_found(exc) from exc

    return server

"""The student's investigation tools: registry, arguments and execution (E8-03).

| Tool                      | Answers                                                        |
| ------------------------- | -------------------------------------------------------------- |
| `neighbour_moves`         | Did the adjacent tenors on the same curve move similarly?      |
| `flag_frequency`          | How often did MDX's rule flag this point in its history?       |
| `point_history`           | The point over a chosen window, today included                 |
| `same_tenor_other_curves` | Did this tenor move in other curves today?                     |

The Market Data Assessment Agent (student, E8-04) chooses which tools to run for a
flag; `catalogue()` is what its prompt gets, as data. Code runs the chosen tool here:
it reads through the MDX tool (`MdxReads`, read operations only), computes in code
(`compute.py`), and returns one `Evidence` record whose ID is derived from its content,
so the same facts give the same ID in every run.

These tools are code, not MCP servers: they compute over data MDX already serves, and
only ADB source systems are MCP tools (adb-mdx-agent ADR-0003).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from mdx_assessment.core.domain import Evidence
from mdx_assessment.core.observability.tracing import Tracer
from mdx_assessment.usecases.mdx.checks.normalize import NormalizedFlag
from mdx_assessment.usecases.mdx.config import DataTypeProfile, InvestigationSettings
from mdx_assessment.usecases.mdx.domain import DataType, MarketDataKind, Series
from mdx_assessment.usecases.mdx.domain.units import conversion_factor
from mdx_assessment.usecases.mdx.investigation import compute
from mdx_assessment.usecases.mdx.investigation.reads import MdxReads

NEIGHBOUR_MOVES = "neighbour_moves"
FLAG_FREQUENCY = "flag_frequency"
POINT_HISTORY = "point_history"
SAME_TENOR_OTHER_CURVES = "same_tenor_other_curves"
SOURCE_PREFIX = "investigation"

FlagIdArg = Annotated[str, Field(min_length=1, description="The flag to investigate.")]


class InvestigationError(ValueError):
    """The tool can't run as asked (unknown tool or flag, bad arguments, no data).
    The message is safe to show the student, which may choose differently."""


class FlagArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag_id: FlagIdArg


class PointHistoryArgs(FlagArgs):
    days: Annotated[
        int,
        Field(ge=2, description="Business days to return, today included."),
    ]


@dataclass(frozen=True)
class InvestigationTool:
    id: str
    description: str
    arguments: type[FlagArgs]


TOOLS: tuple[InvestigationTool, ...] = (
    InvestigationTool(
        NEIGHBOUR_MOVES,
        "Today's move of the tenors next to the flagged one on the same curve, in the "
        "measure unit, with their mean and how many moved the same way.",
        FlagArgs,
    ),
    InvestigationTool(
        FLAG_FREQUENCY,
        "How many of the business days before today MDX's rule flagged this point. "
        "Uses MDX's own past flags when it has them, otherwise re-applies the configured "
        "rule to the point's history; the result names its source.",
        FlagArgs,
    ),
    InvestigationTool(
        POINT_HISTORY,
        "The point's values over the last `days` business days, today included, with "
        "each day's change, the range, the largest change and the normal daily move.",
        PointHistoryArgs,
    ),
    InvestigationTool(
        SAME_TENOR_OTHER_CURVES,
        "Today's move of the same tenor on every other curve, in the measure unit, with "
        "their mean and how many moved the same way.",
        FlagArgs,
    ),
)
_BY_ID = {t.id: t for t in TOOLS}


def catalogue(settings: InvestigationSettings) -> list[dict[str, JsonValue]]:
    """The tools as the student's prompt gets them: ID, description, argument schema."""
    entries: list[dict[str, JsonValue]] = []
    for tool in TOOLS:
        schema = tool.arguments.model_json_schema()
        if tool.id == POINT_HISTORY:
            schema["properties"]["days"]["maximum"] = settings.point_history_max_days
        entries.append({"id": tool.id, "description": tool.description, "arguments": schema})
    return entries


class Investigation:
    """Runs investigation tools for the flags of one step."""

    def __init__(
        self,
        reads: MdxReads,
        flags: Sequence[NormalizedFlag],
        profiles: Mapping[MarketDataKind, DataTypeProfile],
        today: date,
        data_type: DataType,
        settings: InvestigationSettings,
        history_days: int,
        tracer: Tracer | None = None,
    ) -> None:
        """`history_days`: how many business days the step's crucial reads fetch for each
        point (the near-zero window plus today), so tools reuse them. Each run is a span
        in the session's trace when `tracer` is given (E4-04)."""
        self._tracer = tracer or Tracer(None)
        self._reads = reads
        self._flags = {f.flag_id: f for f in flags}
        self._profiles = profiles
        self._today = today
        self._data_type = data_type
        self._settings = settings
        self._history_days = max(history_days, settings.flag_frequency_window_days + 1)

    async def run(self, tool_id: str, arguments: Mapping[str, JsonValue]) -> Evidence:
        with self._tracer.span(
            f"{SOURCE_PREFIX}.{tool_id}", kind="tool", input=dict(arguments)
        ) as span:
            try:
                evidence = await self._run(tool_id, arguments)
            except Exception as exc:
                if span is not None:
                    span.update(error=f"{type(exc).__name__}: {exc}"[:500])
                raise
            if span is not None:
                span.update(output={"evidence_id": evidence.evidence_id, **evidence.data})
            return evidence

    async def _run(self, tool_id: str, arguments: Mapping[str, JsonValue]) -> Evidence:
        tool = _BY_ID.get(tool_id)
        if tool is None:
            raise InvestigationError(f"unknown tool {tool_id!r}; choose one of {sorted(_BY_ID)}")
        try:
            args = tool.arguments.model_validate(dict(arguments))
        except ValidationError as exc:
            raise InvestigationError(f"{tool_id}: invalid arguments: {exc.errors()}") from exc
        flag = self._flags.get(args.flag_id)
        if flag is None:
            raise InvestigationError(f"{tool_id}: unknown flag {args.flag_id!r}")
        if tool_id == NEIGHBOUR_MOVES:
            result: BaseModel = await self._neighbour_moves(flag)
        elif tool_id == SAME_TENOR_OTHER_CURVES:
            result = await self._same_tenor(flag)
        elif tool_id == POINT_HISTORY:
            assert isinstance(args, PointHistoryArgs)
            result = await self._point_history(flag, args.days)
        else:
            result = await self._flag_frequency(flag)
        return Evidence.create(
            tool_id, f"{SOURCE_PREFIX}.{tool_id}", result.model_dump(mode="json")
        )

    # --- shared reads ------------------------------------------------------------------

    def _factor(self, flag: NormalizedFlag) -> float:
        profile = self._profiles[flag.kind]
        return conversion_factor(profile.quote_unit, profile.measure_unit) or 1.0

    async def _series(self, flag: NormalizedFlag) -> Series:
        return await self._reads.history(flag.curve_id, flag.tenor, self._today, self._history_days)

    async def _previous_date(self, flag: NormalizedFlag) -> date:
        before = [
            o for o in (await self._series(flag)).observations if o.business_date < self._today
        ]
        if not before:
            raise InvestigationError(
                f"MDX has no business day before {self._today} for {flag.item_key}"
            )
        return before[-1].business_date

    async def _two_days(
        self, flag: NormalizedFlag
    ) -> tuple[compute.CurveValues, compute.CurveValues, date]:
        previous_date = await self._previous_date(flag)
        today = await self._reads.curve_values(self._today)
        previous = await self._reads.curve_values(previous_date)
        if today is None or previous is None:
            missing = self._today if today is None else previous_date
            raise InvestigationError(f"MDX has no curve values for {missing}")
        return today, previous, previous_date

    # --- tools -------------------------------------------------------------------------

    async def _neighbour_moves(self, flag: NormalizedFlag) -> compute.NeighbourMoves:
        today, previous, previous_date = await self._two_days(flag)
        return compute.neighbour_moves(
            flag,
            today,
            previous,
            previous_date,
            self._factor(flag),
            self._settings.neighbours_each_side,
        )

    async def _same_tenor(self, flag: NormalizedFlag) -> compute.SameTenorOtherCurves:
        today, previous, previous_date = await self._two_days(flag)
        return compute.same_tenor_other_curves(
            flag, today, previous, previous_date, self._factor(flag)
        )

    async def _point_history(self, flag: NormalizedFlag, days: int) -> compute.PointHistory:
        if days > self._settings.point_history_max_days:
            raise InvestigationError(
                f"{POINT_HISTORY}: at most {self._settings.point_history_max_days} days, got {days}"
            )
        series = await self._reads.history(flag.curve_id, flag.tenor, self._today, days)
        if not series.observations:
            raise InvestigationError(f"MDX has no history for {flag.item_key}")
        return compute.point_history(
            flag, series.observations, self._today, days, self._factor(flag)
        )

    async def _flag_frequency(self, flag: NormalizedFlag) -> compute.FlagFrequency:
        window_days = self._settings.flag_frequency_window_days
        series = await self._series(flag)
        window = [o.business_date for o in series.observations if o.business_date < self._today]
        checked = window[-window_days:][1:]  # each day with a day before it, as the rule needs
        why = await self._mdx_has_past_flags(checked)
        if why is None:
            flagged_on: dict[date, bool] = {}
            for on in checked:
                past = await self._reads.flags(on, self._data_type) or []
                flagged_on[on] = any(
                    f.curve_id == flag.curve_id and f.tenor == flag.tenor and f.rule == flag.rule
                    for f in past
                )
            return compute.flag_frequency_from_mdx(flag, flagged_on, window_days)
        return compute.flag_frequency_from_rule(
            flag, series.observations, self._today, window_days, self._settings.mdx_rule, why
        )

    async def _mdx_has_past_flags(self, checked: Sequence[date]) -> str | None:
        """None when MDX holds flags for every day checked; otherwise why it doesn't.
        Decided once per step: it holds for every flag alike."""
        if not checked:
            return "no business days before today to check"
        for on in reversed(checked):
            if await self._reads.flags(on, self._data_type) is None:
                return (
                    f"MDX has no past flags for {on.isoformat()}: the configured rule was "
                    "re-applied to the point's history"
                )
        return None

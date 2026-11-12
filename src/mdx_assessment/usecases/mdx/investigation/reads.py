"""Read-only, cached access to MDX for one Market Data Assessment step (E8-03).

The crucial steps (the near-zero check) and the investigation tools read through the
same `MdxReads`, so a tool reuses what the crucial steps already read: the history
fetched for the z-score serves `point_history` and `flag_frequency` without another
call, and the day's curve values are read once for every flag. Fewer calls keep the
step inside its tool budget and the end-of-day window.

Only operations the MDX tool declares as reads can be called: the tools cannot write.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import JsonValue

from mdx_assessment.core.registry.specs import ToolAccess
from mdx_assessment.core.tools.client import AgentToolClient, ToolCallError
from mdx_assessment.usecases.mdx.domain import DataType, MdxFlag, Series, Tenor
from mdx_assessment.usecases.mdx.investigation.compute import CurveValues
from mdx_assessment.usecases.mdx.tools.mdx import (
    GET_CURVE_VALUES,
    GET_FLAGS,
    GET_HISTORY,
    MDX_TOOL_SPEC,
)
from mdx_assessment.usecases.mdx.tools.mdx import TOOL_ID as MDX_DATA
from mdx_assessment.usecases.mdx.tools.mdx.server import CurveValuesResult, FlagsResult


class MdxReads:
    def __init__(self, tools: AgentToolClient) -> None:
        self._tools = tools
        self._history: dict[tuple[str, Tenor, date], tuple[int, Series]] = {}
        self._curves: dict[date, CurveValues | None] = {}
        self._flags: dict[date, list[MdxFlag] | None] = {}
        self.calls = 0
        """MDX calls made through this reader (cache hits aren't counted)."""

    async def _read(self, operation: str, arguments: dict[str, JsonValue]) -> dict[str, Any]:
        if MDX_TOOL_SPEC.operations.get(operation) is not ToolAccess.READ:
            raise PermissionError(f"{MDX_DATA}.{operation} is not a read operation")
        self.calls += 1
        return await self._tools.call(MDX_DATA, operation, arguments)

    async def history(self, curve_id: str, tenor: Tenor, end_date: date, days: int) -> Series:
        """The point's last `days` business days up to `end_date`. A longer read already
        made for the same point and end date is sliced instead of read again."""
        key = (curve_id, tenor, end_date)
        cached = self._history.get(key)
        if cached is None or cached[0] < days:
            raw = await self._read(
                GET_HISTORY,
                {
                    "curve_id": curve_id,
                    "tenor": str(tenor),
                    "end_date": end_date.isoformat(),
                    "days": days,
                },
            )
            cached = (days, Series.model_validate(raw))
            self._history[key] = cached
        series = cached[1]
        return series.model_copy(update={"observations": series.observations[-days:]})

    async def curve_values(self, on: date) -> CurveValues | None:
        """Every point's value on `on`, or None when MDX has no data for the date."""
        if on not in self._curves:
            try:
                raw = await self._read(GET_CURVE_VALUES, {"business_date": on.isoformat()})
            except ToolCallError:
                self._curves[on] = None
            else:
                points = CurveValuesResult.model_validate(raw).points
                self._curves[on] = {(p.curve_id, p.tenor): p.value for p in points}
        return self._curves[on]

    async def flags(self, on: date, data_type: DataType) -> list[MdxFlag] | None:
        """MDX's flags for `on`, or None when MDX has none to give for the date."""
        if on not in self._flags:
            try:
                raw = await self._read(
                    GET_FLAGS, {"business_date": on.isoformat(), "data_type": data_type.value}
                )
            except ToolCallError:
                self._flags[on] = None
            else:
                self._flags[on] = FlagsResult.model_validate(raw).flags
        return self._flags[on]

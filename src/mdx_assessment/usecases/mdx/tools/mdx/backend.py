"""What the MDX data tool reads from: an interface, and the simulated implementation.

`MdxBackend` is the contract the MCP server depends on. The PoC uses
`SimulatedMdxBackend` (files written by `make sim`); production would add a backend
over ADB's approved MDX interface without changing the server or the agents
(PDF §1: "replaced by approved connections without redesign").

The simulated backend reads only the files that stand in for MDX. It never opens
`ground_truth.json` (the answer key for tests and evaluation).
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from mdx_assessment.usecases.mdx.config import ExpectedFile, ExpectedFiles
from mdx_assessment.usecases.mdx.domain import (
    DataType,
    FileMetadata,
    MarketDataKind,
    MdxFlag,
    Observation,
    Point,
    Series,
    Tenor,
    Unit,
)

# The only files the simulated backend may open (ground_truth.json is deliberately absent).
SERVED_FILES = ("curve_values.csv", "flags.json", "file_metadata.json")


class MdxDataError(LookupError):
    """Requested data doesn't exist (unknown date, curve or tenor). Message is safe to show."""


class MdxBackend(Protocol):
    def expected_files(self, data_type: DataType) -> tuple[ExpectedFile, ...]: ...

    @property
    def timezone(self) -> str: ...

    def file_metadata(self, business_date: date) -> list[FileMetadata]: ...

    def flags(self, business_date: date, data_type: DataType) -> list[MdxFlag]: ...

    def curve_values(self, business_date: date, curve_id: str | None = None) -> list[Point]: ...

    def history(self, curve_id: str, tenor: Tenor, end_date: date, days: int) -> Series: ...


@dataclass(frozen=True)
class _Day:
    metadata: list[FileMetadata]
    flags: list[MdxFlag]
    dates: list[date]  # oldest first; the business date is last
    values: dict[tuple[str, Tenor], list[float]]  # aligned with `dates`


class SimulatedMdxBackend:
    """Reads `<data_dir>/sim/<business date>/`, as written by `make sim`. Caches each day."""

    def __init__(self, data_dir: Path, expected: ExpectedFiles) -> None:
        self._root = data_dir / "sim"
        self._expected = expected
        self._cache: dict[date, _Day] = {}
        self._lock = threading.Lock()

    @property
    def timezone(self) -> str:
        return self._expected.timezone

    def expected_files(self, data_type: DataType) -> tuple[ExpectedFile, ...]:
        return self._expected.for_data_type(data_type)

    def file_metadata(self, business_date: date) -> list[FileMetadata]:
        return list(self._day(business_date).metadata)

    def flags(self, business_date: date, data_type: DataType) -> list[MdxFlag]:
        return [f for f in self._day(business_date).flags if f.data_type is data_type]

    def curve_values(self, business_date: date, curve_id: str | None = None) -> list[Point]:
        day = self._latest_day_covering(business_date)
        i = day.dates.index(business_date)
        points = [
            Point(curve_id=c, tenor=t, business_date=business_date, value=v[i], unit=Unit.PERCENT)
            for (c, t), v in day.values.items()
            if curve_id is None or c == curve_id
        ]
        if curve_id is not None and not points:
            raise MdxDataError(f"unknown curve {curve_id!r}")
        return points

    def history(self, curve_id: str, tenor: Tenor, end_date: date, days: int) -> Series:
        day = self._latest_day_covering(end_date)
        series = day.values.get((curve_id, tenor))
        if series is None:
            raise MdxDataError(f"unknown curve point {curve_id} {tenor}")
        end = day.dates.index(end_date) + 1
        start = max(0, end - days)
        return Series(
            curve_id=curve_id,
            tenor=tenor,
            kind=MarketDataKind.RATE,
            unit=Unit.PERCENT,
            observations=tuple(
                Observation(business_date=d, value=v)
                for d, v in zip(day.dates[start:end], series[start:end], strict=True)
            ),
        )

    # --- loading -------------------------------------------------------------------------

    def _day(self, business_date: date) -> _Day:
        with self._lock:
            if business_date not in self._cache:
                self._cache[business_date] = self._load(business_date)
            return self._cache[business_date]

    def _latest_day_covering(self, on: date) -> _Day:
        """The simulated day whose history includes `on` (itself, or a later day)."""
        if (self._root / on.isoformat()).is_dir():
            return self._day(on)
        for folder in sorted(self._root.iterdir(), reverse=True) if self._root.is_dir() else []:
            try:
                candidate = date.fromisoformat(folder.name)
            except ValueError:
                continue
            if candidate > on and on in self._day(candidate).dates:
                return self._day(candidate)
        raise MdxDataError(f"no MDX data for {on.isoformat()}")

    def _load(self, business_date: date) -> _Day:
        folder = self._root / business_date.isoformat()
        if not folder.is_dir():
            raise MdxDataError(f"no MDX data for {business_date.isoformat()}")
        metadata = [
            FileMetadata.model_validate(m)
            for m in json.loads(self._read(folder, "file_metadata.json"))
        ]
        flags = [MdxFlag.model_validate(f) for f in json.loads(self._read(folder, "flags.json"))]
        dates: list[date] = []
        values: dict[tuple[str, Tenor], list[float]] = {}
        for row in csv.DictReader(self._read(folder, "curve_values.csv").splitlines()):
            on = date.fromisoformat(row["business_date"])
            if not dates or dates[-1] != on:
                dates.append(on)
            values.setdefault((row["curve_id"], Tenor.parse(row["tenor"])), []).append(
                float(row["value"])
            )
        return _Day(metadata=metadata, flags=flags, dates=dates, values=values)

    @staticmethod
    def _read(folder: Path, name: str) -> str:
        if name not in SERVED_FILES:  # defence in depth: never read anything else
            raise MdxDataError(f"{name} is not an MDX file")
        return (folder / name).read_text(encoding="utf-8")

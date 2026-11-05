"""Simulated linear MDX day (plan §2.3, E2-01). Pure: config + date in, a `SimulatedDay` out.

Model, per curve and tenor (values in percent, shocks in bps):

    base(t)  = long + (short - long) * exp(-t / decay)
    r[d]     = r[d-1] + kappa * (base - r[d-1]) + sigma * shock[d] / 100
    shock    = sqrt(1 - iota) * (wL * L[d] + wS(t) * S[d])  +  sqrt(iota) * eps[d]

kappa = `mean_reversion`, iota = `idiosyncratic_share`, sigma = `daily_vol_bps`.
L and S are a curve's level and slope shocks (shared by its tenors, so curves move
coherently), eps is each tenor's own noise, all standard normal. wS(t) =
0.5 * (1 - 2 * exp(-t / decay)) and wL = sqrt(1 - wS^2) keep each shock at unit
variance, so sigma is the true daily volatility of every point. That sigma is what
the ground truth's `true_z` uses.

Planted cases are applied on top, then values are rounded to 4 decimals (0.01 bp),
and the MDX-style relative-change rule raises the flags. Everything is driven by a
`random.Random` seeded with (seed, business date): same inputs, same day.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import random
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from mdx_assessment.core.domain import DomainModel
from mdx_assessment.usecases.mdx.config import ExpectedFile
from mdx_assessment.usecases.mdx.domain import (
    DataType,
    FileMetadata,
    MarketDataKind,
    MdxFlag,
    Percent,
    Tenor,
    Unit,
    change_bps,
    relative_change_pct,
)
from mdx_assessment.usecases.mdx.sim.calendar import business_days_ending
from mdx_assessment.usecases.mdx.sim.config import PlantedCase, PlantedKind, SimCurve, SimLinearDay

GENERATOR_VERSION = "linear-day/1"
DECIMALS = 4
RULE_NAME = "relative_change"

Point = tuple[str, Tenor]


class ExpectedClass(StrEnum):
    THRESHOLD_ARTIFACT = "threshold_artifact"
    GENUINE_LARGE_MOVE = "genuine_large_move"
    ORDINARY = "ordinary"


ARTIFACT_MAX_ABS_Z = 2.0  # near-zero base and a normal-sized move → the % flag is an artifact
LARGE_MOVE_MIN_ABS_Z = 4.0


class GroundTruthFlag(DomainModel):
    """What a correct assessment should conclude about one flag."""

    flag_id: str
    curve_id: str
    tenor: Tenor
    previous: float
    current: float
    change_bps: float
    relative_change_pct: float | None
    true_sigma_bps: float
    true_z: float
    """change_bps / true sigma: how unusual the move is under the generating model."""
    near_zero: bool
    """|previous| below the data-type profile's near-zero floor."""
    expected_class: ExpectedClass
    planted_case: str | None = None


class GroundTruth(DomainModel):
    business_date: date
    seed: int
    generator_version: str
    rule_relative_change_pct: float
    near_zero_floor: float
    flag_count: int
    planted: dict[str, str]
    """Planted case ID → flag ID."""
    review_first: tuple[str, ...]
    """Flag IDs a good run opinion puts first: the genuine large moves."""
    flags: tuple[GroundTruthFlag, ...]


@dataclass(frozen=True)
class SimulatedDay:
    business_date: date
    dates: tuple[date, ...]  # history then the business date, oldest first
    values: dict[Point, tuple[float, ...]]  # aligned with `dates`, percent, 4 dp
    flags: tuple[MdxFlag, ...]
    ground_truth: GroundTruth
    vendor_file_name: str
    vendor_file: bytes
    file_metadata: FileMetadata

    def value(self, curve_id: str, tenor: Tenor, on: date) -> float:
        return self.values[(curve_id, tenor)][self.dates.index(on)]


def base_curve(curve: SimCurve, tenor: Tenor) -> float:
    t = float(tenor.years)
    return curve.long_rate + (curve.short_rate - curve.long_rate) * math.exp(-t / curve.decay_years)


def _slope_weight(curve: SimCurve, tenor: Tenor) -> float:
    return 0.5 * (1 - 2 * math.exp(-float(tenor.years) / curve.decay_years))


def _simulate_paths(cfg: SimLinearDay, n_days: int, rng: random.Random) -> dict[Point, list[float]]:
    paths: dict[Point, list[float]] = {}
    common, idio = math.sqrt(1 - cfg.idiosyncratic_share), math.sqrt(cfg.idiosyncratic_share)
    for curve in cfg.curves:
        weights = {t: _slope_weight(curve, t) for t in cfg.tenors}
        bases = {t: base_curve(curve, t) for t in cfg.tenors}
        for t in cfg.tenors:
            paths[(curve.id, t)] = [bases[t]]
        for _ in range(1, n_days):
            level, slope = rng.gauss(0, 1), rng.gauss(0, 1)
            for t in cfg.tenors:
                w_s = weights[t]
                shock = common * (math.sqrt(1 - w_s**2) * level + w_s * slope) + idio * rng.gauss(
                    0, 1
                )
                prev = paths[(curve.id, t)][-1]
                step = cfg.mean_reversion * (bases[t] - prev) + curve.daily_vol_bps * shock / 100
                paths[(curve.id, t)].append(prev + step)
    return paths


def _neighbours(cfg: SimLinearDay, tenor: Tenor) -> list[Tenor]:
    i = cfg.tenors.index(tenor)
    return [cfg.tenors[j] for j in (i - 1, i + 1) if 0 <= j < len(cfg.tenors)]


def _apply_planted(cfg: SimLinearDay, paths: dict[Point, list[float]]) -> None:
    planted_points = {(c.curve, c.tenor) for c in cfg.planted}
    for case in cfg.planted:
        path = paths[(case.curve, case.tenor)]
        if case.previous is not None:  # shift the whole history so it ends exactly there
            offset = case.previous - path[-2]
            path[:] = [v + offset for v in path]
        path[-1] = _planted_today(case, path[-2])
        for tenor in _neighbours(cfg, case.tenor):
            if (case.curve, tenor) in planted_points:
                continue
            neighbour = paths[(case.curve, tenor)]
            neighbour[-1] = neighbour[-2] + case.neighbour_change_bps / 100


def _planted_today(case: PlantedCase, previous: float) -> float:
    if case.current is not None:
        return case.current
    assert case.change_bps is not None  # guaranteed by PlantedCase validation
    return previous + case.change_bps / 100


def _is_flagged(previous: float, current: float, threshold_pct: float) -> tuple[bool, float | None]:
    rel = relative_change_pct(previous, current)
    if rel is None:
        return current != previous, None
    return abs(rel) > threshold_pct, rel


def _expected_class(near_zero: bool, true_z: float, planted: PlantedCase | None) -> ExpectedClass:
    if planted is not None:
        return ExpectedClass(planted.kind.value)
    if near_zero and abs(true_z) < ARTIFACT_MAX_ABS_Z:
        return ExpectedClass.THRESHOLD_ARTIFACT
    if abs(true_z) > LARGE_MOVE_MIN_ABS_Z:
        return ExpectedClass.GENUINE_LARGE_MOVE
    return ExpectedClass.ORDINARY


def _vendor_file(cfg: SimLinearDay, values: dict[Point, tuple[float, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["curve_id", "tenor", "value", "unit"])
    for curve in cfg.curves:
        for tenor in cfg.tenors:
            value = values[(curve.id, tenor)][-1]
            writer.writerow([curve.id, str(tenor), f"{value:.{DECIMALS}f}", Unit.PERCENT.value])
    return buffer.getvalue().encode("utf-8")


def generate_linear_day(
    cfg: SimLinearDay,
    business_date: date,
    *,
    expected_file: ExpectedFile,
    timezone: str,
    near_zero_floor: float,
    seed: int | None = None,
) -> SimulatedDay:
    """Generate the day. `seed` overrides the config seed (evaluation variants, SP4)."""
    run_seed = cfg.seed if seed is None else seed
    dates = tuple(business_days_ending(business_date, cfg.history_days + 1))
    rng = random.Random(f"{GENERATOR_VERSION}:{run_seed}:{business_date.isoformat()}")  # noqa: S311 (reproducible simulation, not security)

    paths = _simulate_paths(cfg, len(dates), rng)
    _apply_planted(cfg, paths)
    values = {p: tuple(round(v, DECIMALS) for v in path) for p, path in paths.items()}

    planted_by_point = {(c.curve, c.tenor): c for c in cfg.planted}
    flags: list[MdxFlag] = []
    truths: list[GroundTruthFlag] = []
    for curve in cfg.curves:
        for tenor in cfg.tenors:
            previous, current = values[(curve.id, tenor)][-2:]
            flagged, rel = _is_flagged(previous, current, cfg.mdx_rule.relative_change_pct)
            if not flagged:
                continue
            flag_id = f"{business_date:%Y%m%d}-{curve.id}-{tenor}"
            flags.append(
                MdxFlag(
                    flag_id=flag_id,
                    business_date=business_date,
                    data_type=DataType.LINEAR,
                    kind=MarketDataKind.RATE,
                    curve_id=curve.id,
                    tenor=tenor,
                    rule=RULE_NAME,
                    previous_value=previous,
                    current_value=current,
                    unit=Unit.PERCENT,
                    color_code=curve.color_overrides.get(str(tenor), curve.color_code),
                )
            )
            move = change_bps(Percent(previous), Percent(current))
            true_z = round(move / curve.daily_vol_bps, 4)
            near_zero = abs(previous) < near_zero_floor
            planted = planted_by_point.get((curve.id, tenor))
            truths.append(
                GroundTruthFlag(
                    flag_id=flag_id,
                    curve_id=curve.id,
                    tenor=tenor,
                    previous=previous,
                    current=current,
                    change_bps=move,
                    relative_change_pct=None if rel is None else round(rel, 4),
                    true_sigma_bps=curve.daily_vol_bps,
                    true_z=true_z,
                    near_zero=near_zero,
                    expected_class=_expected_class(near_zero, true_z, planted),
                    planted_case=planted.id if planted else None,
                )
            )

    missing = [c.id for c in cfg.planted if c.id not in {t.planted_case for t in truths}]
    if missing:
        raise ValueError(f"planted case(s) not flagged by the MDX rule: {missing}")

    vendor_file = _vendor_file(cfg, values)
    file_name = expected_file.file_name_for(business_date)
    metadata = FileMetadata(
        file_name=file_name,
        business_date=business_date,
        data_type=expected_file.data_type,
        arrived_at=datetime.combine(business_date, cfg.arrival_time, ZoneInfo(timezone)),
        size_bytes=len(vendor_file),
        record_count=len(cfg.curves) * len(cfg.tenors),
        content_sha256=hashlib.sha256(vendor_file).hexdigest(),
    )
    truth = GroundTruth(
        business_date=business_date,
        seed=run_seed,
        generator_version=GENERATOR_VERSION,
        rule_relative_change_pct=cfg.mdx_rule.relative_change_pct,
        near_zero_floor=near_zero_floor,
        flag_count=len(flags),
        planted={t.planted_case: t.flag_id for t in truths if t.planted_case},
        review_first=tuple(
            t.flag_id for t in truths if t.expected_class is ExpectedClass.GENUINE_LARGE_MOVE
        ),
        flags=tuple(truths),
    )
    return SimulatedDay(
        business_date=business_date,
        dates=dates,
        values=values,
        flags=tuple(flags),
        ground_truth=truth,
        vendor_file_name=file_name,
        vendor_file=vendor_file,
        file_metadata=metadata,
    )


__all__ = [
    "GENERATOR_VERSION",
    "ExpectedClass",
    "GroundTruth",
    "GroundTruthFlag",
    "PlantedKind",
    "SimulatedDay",
    "base_curve",
    "generate_linear_day",
]

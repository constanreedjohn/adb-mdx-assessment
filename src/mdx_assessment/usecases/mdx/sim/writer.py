"""Write a simulated day to `<data_dir>/sim/<business date>/`.

Layout (read by the MDX data tool's simulated backend, E3-01):

    mdx_linear_rates_<yyyymmdd>.csv  the vendor file: today's 80 values
    curve_values.csv                 every value, history and today (date, curve, tenor, value)
    flags.json                       MDX exception flags (MdxFlag)
    file_metadata.json               delivery facts per file (FileMetadata)
    ground_truth.json                expected outcome per flag (tests and evaluation only;
                                     never served to agents)
    manifest.json                    seed, generator and config versions, file hashes

Output is byte-for-byte deterministic: no wall-clock times, fixed formatting.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mdx_assessment.usecases.mdx.sim.generator import DECIMALS, GENERATOR_VERSION, SimulatedDay

CURVE_VALUES = "curve_values.csv"
FLAGS = "flags.json"
FILE_METADATA = "file_metadata.json"
GROUND_TRUTH = "ground_truth.json"
MANIFEST = "manifest.json"


def day_dir(data_dir: Path, day: SimulatedDay) -> Path:
    return data_dir / "sim" / day.business_date.isoformat()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _dump(models: BaseModel | list[BaseModel] | tuple[BaseModel, ...]) -> Any:
    if isinstance(models, BaseModel):
        return models.model_dump(mode="json")
    return [m.model_dump(mode="json") for m in models]


def _curve_values(day: SimulatedDay) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["business_date", "curve_id", "tenor", "value"])
    for i, on in enumerate(day.dates):
        for (curve_id, tenor), series in day.values.items():
            writer.writerow([on.isoformat(), curve_id, str(tenor), f"{series[i]:.{DECIMALS}f}"])
    return buffer.getvalue().encode("utf-8")


def render_files(day: SimulatedDay, *, config_version: str) -> dict[str, bytes]:
    """Every output file's name and exact bytes (manifest last, covering the others)."""
    files: dict[str, bytes] = {
        day.vendor_file_name: day.vendor_file,
        CURVE_VALUES: _curve_values(day),
        FLAGS: _json(_dump(day.flags)),
        FILE_METADATA: _json(_dump([day.file_metadata])),
        GROUND_TRUTH: _json(_dump(day.ground_truth)),
    }
    manifest = {
        "business_date": day.business_date.isoformat(),
        "seed": day.ground_truth.seed,
        "generator_version": GENERATOR_VERSION,
        "config_version": config_version,
        "history_days": len(day.dates) - 1,
        "flag_count": len(day.flags),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files[MANIFEST] = _json(manifest)
    return files


def write_day(day: SimulatedDay, data_dir: Path, *, config_version: str) -> Path:
    """Write (or overwrite) the day's folder. Returns its path."""
    target = day_dir(data_dir, day)
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.iterdir():  # a re-run replaces the day completely
        if stale.is_file():
            stale.unlink()
    for name, data in render_files(day, config_version=config_version).items():
        (target / name).write_bytes(data)
    return target

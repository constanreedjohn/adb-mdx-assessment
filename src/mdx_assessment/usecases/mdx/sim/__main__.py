"""`make sim DATE=2026-10-15 [SEED=…]`: write a simulated linear MDX day to data/sim/<date>/."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date

from mdx_assessment.app import load_config  # entry point: may use the composition root
from mdx_assessment.settings import get_settings
from mdx_assessment.usecases.mdx.sim.build import build_linear_day
from mdx_assessment.usecases.mdx.sim.writer import write_day


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make sim", description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat, help="business date")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    args = parser.parse_args(argv)

    try:
        config = load_config()
        day = build_linear_day(config, args.date, seed=args.seed)
    except ValueError as exc:
        print(f"FAILED: {exc}")
        return 1
    target = write_day(day, get_settings().app.data_dir, config_version=config.version)

    truth = day.ground_truth
    classes = Counter(f.expected_class.value for f in truth.flags)
    print(f"Business date : {day.business_date}  (history: {len(day.dates) - 1} business days)")
    print(f"Seed          : {truth.seed}")
    print(f"Vendor file   : {day.vendor_file_name}  ({day.file_metadata.record_count} records)")
    print(
        f"MDX flags     : {len(day.flags)}  "
        + ", ".join(f"{k}={v}" for k, v in sorted(classes.items()))
    )
    for case, flag_id in truth.planted.items():
        flag = next(f for f in truth.flags if f.flag_id == flag_id)
        rel = f"{flag.relative_change_pct:+.1f}%" if flag.relative_change_pct is not None else "n/a"
        print(
            f"Planted       : {case}: {flag.previous} -> {flag.current}  ({rel}, {flag.change_bps:+.2f} bps, z={flag.true_z:+.2f})"
        )
    print(f"Written to    : {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

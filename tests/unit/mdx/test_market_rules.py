from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from mdx_assessment.usecases.mdx.domain import (
    FileMetadata,
    MarketDataKind,
    Observation,
    Point,
    Series,
    Tenor,
    Unit,
)

THU = date(2026, 10, 15)
SAT = date(2026, 10, 17)


def test_business_date_rejects_weekends() -> None:
    with pytest.raises(ValidationError, match="weekend"):
        Point(
            curve_id="JPY",
            tenor=Tenor.parse("3M"),
            business_date=SAT,
            value=0.07,
            unit=Unit.PERCENT,
        )


@pytest.mark.parametrize("curve_id", ["JPY", "USD.SOFR", "EUR-ESTR", "GBP_OIS"])
def test_valid_curve_ids(curve_id: str) -> None:
    Point(
        curve_id=curve_id, tenor=Tenor.parse("3M"), business_date=THU, value=1.0, unit=Unit.PERCENT
    )


@pytest.mark.parametrize("curve_id", ["jpy", "JP", "JPY..X", "", "JPY "])
def test_invalid_curve_ids(curve_id: str) -> None:
    with pytest.raises(ValidationError):
        Point(
            curve_id=curve_id,
            tenor=Tenor.parse("3M"),
            business_date=THU,
            value=1.0,
            unit=Unit.PERCENT,
        )


def test_series_requires_increasing_dates() -> None:
    with pytest.raises(ValidationError, match="increasing"):
        Series(
            curve_id="JPY",
            tenor=Tenor.parse("3M"),
            kind=MarketDataKind.RATE,
            unit=Unit.PERCENT,
            observations=(
                Observation(business_date=date(2026, 10, 15), value=0.04),
                Observation(business_date=date(2026, 10, 14), value=0.05),
            ),
        )


def test_point_rejects_nan() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Point(
            curve_id="JPY",
            tenor=Tenor.parse("3M"),
            business_date=THU,
            value=float("nan"),
            unit=Unit.PERCENT,
        )


def test_file_metadata_checks_hash_and_counts() -> None:
    common = {
        "file_name": "linear.csv",
        "business_date": THU,
        "data_type": "linear",
        "arrived_at": datetime(2026, 10, 15, 17, 0, tzinfo=UTC),
        "size_bytes": 10,
    }
    with pytest.raises(ValidationError):
        FileMetadata.model_validate({**common, "record_count": -1, "content_sha256": "0" * 64})
    with pytest.raises(ValidationError):
        FileMetadata.model_validate({**common, "record_count": 1, "content_sha256": "abc"})

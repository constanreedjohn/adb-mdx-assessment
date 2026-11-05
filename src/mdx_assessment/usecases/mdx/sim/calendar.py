"""Business calendar for the simulator: weekdays only (holidays come later if needed)."""

from __future__ import annotations

from datetime import date, timedelta


def is_business_day(day: date) -> bool:
    return day.weekday() < 5


def previous_business_day(day: date) -> date:
    candidate = day - timedelta(days=1)
    while not is_business_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def business_days_ending(day: date, count: int) -> list[date]:
    """The `count` business days up to and including `day`, oldest first."""
    if not is_business_day(day):
        raise ValueError(f"{day.isoformat()} is not a business day")
    if count < 1:
        raise ValueError("count must be at least 1")
    days = [day]
    while len(days) < count:
        days.append(previous_business_day(days[-1]))
    return days[::-1]

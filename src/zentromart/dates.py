"""Deterministic date arithmetic used by the agent's calculator tools.

Rules from KB-00.3: business days are Monday–Saturday excluding the holidays in
``data/holidays.csv``; Sunday is never a business day. Return windows are calendar days.
"N business days from day D" is counted starting the day *after* D (D itself is day 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]


@dataclass
class Calendar:
    holidays: dict[date, str]
    today: date

    def is_business_day(self, d: date) -> bool:
        return d.weekday() != 6 and d not in self.holidays

    def add_business_days(self, start: date, n: int) -> date:
        if n < 0:
            raise ValueError("n must be >= 0")
        d, counted = start, 0
        while counted < n:
            d += timedelta(days=1)
            if self.is_business_day(d):
                counted += 1
        return d

    def business_days_between(self, start: date, end: date) -> int:
        """Business days in (start, end]; negative if end < start."""
        if end < start:
            return -self.business_days_between(end, start)
        d, count = start, 0
        while d < end:
            d += timedelta(days=1)
            if self.is_business_day(d):
                count += 1
        return count

    def skipped_days(self, start: date, end: date) -> list[str]:
        out, d = [], start
        while d < end:
            d += timedelta(days=1)
            if d.weekday() == 6:
                out.append(f"{d.isoformat()} (Sunday)")
            elif d in self.holidays:
                out.append(f"{d.isoformat()} ({self.holidays[d]})")
        return out

    # ------------------------------------------------------------------ tool functions
    def tool_business_day_window(self, start_date: str, min_days: int, max_days: int) -> dict:
        start = date.fromisoformat(start_date)
        lo, hi = int(min_days), int(max_days)
        if lo > hi:
            lo, hi = hi, lo
        earliest = self.add_business_days(start, lo)
        latest = self.add_business_days(start, hi)
        return {
            "start_date": start.isoformat(),
            "earliest": earliest.isoformat(), "earliest_spoken": spoken_date(earliest),
            "latest": latest.isoformat(), "latest_spoken": spoken_date(latest),
            "skipped_non_business_days": self.skipped_days(start, latest),
            "latest_is_before_today": latest < self.today,
            "today": self.today.isoformat(),
        }

    def tool_add_calendar_days(self, start_date: str, days: int) -> dict:
        start = date.fromisoformat(start_date)
        end = start + timedelta(days=int(days))
        return {
            "start_date": start.isoformat(), "days": int(days),
            "result_date": end.isoformat(), "result_spoken": spoken_date(end),
            "today": self.today.isoformat(),
            "result_is_on_or_after_today": end >= self.today,
            "calendar_days_from_today": (end - self.today).days,
        }

    def tool_business_days_between(self, start_date: str, end_date: str) -> dict:
        s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
        return {"start_date": s.isoformat(), "end_date": e.isoformat(),
                "business_days": self.business_days_between(s, e),
                "skipped_non_business_days": self.skipped_days(min(s, e), max(s, e))}


def spoken_date(d: date) -> str:
    return f"{d.day} {MONTHS_EN[d.month - 1]} {d.year}"

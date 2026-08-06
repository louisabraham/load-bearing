"""Paths, corpus window, and the deterministic GH Archive sampling schedule."""

from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("LBDETECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
DOCS = DATA / "docs"  # cleaned document shards, partitioned by month
SERIES = DATA / "series"  # expression x period frequency tables
ARTIFACTS = DATA / "artifacts"  # models, clusters, atlas
OUT = ROOT / "out"  # human-facing reports and plots

for _p in (DATA, DOCS, SERIES, ARTIFACTS, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- corpus window

START = (2018, 1)
END = (2026, 7)  # inclusive; GH Archive lags ~1 day

# Prose artifact types. Commit messages are a separate register and are kept out
# of the default series so they cannot shift the denominator over time.
PROSE_ARTIFACTS = (
    "issue",
    "issue_comment",
    "pr",
    "pr_comment",
    "pr_review",
    "pr_review_comment",
    "commit_comment",
)
COMMIT_ARTIFACTS = ("commit_msg",)

# GH Archive stopped capturing most non-Push events in 2026: issue comments per
# hour fall ~95% while PushEvent volume holds. We compensate by sampling many
# more hours per month once the density collapses, so the recent tail still gets
# a usable document count. Anything below `min_docs_per_month` in the final
# corpus should be treated as low-power rather than as a real frequency drop.
DENSITY_COLLAPSE = (2026, 1)


@dataclass(frozen=True)
class SamplingPlan:
    """How many hour-files to draw per month, and how much of each one to read.

    `max_bytes` truncates the download with an HTTP Range request. Because the
    archive stores events in time order inside an hour, reading the first N bytes
    is a contiguous slice of that hour rather than a random sample -- fine for
    frequency estimation, and the cheapest available knob on a slow connection.
    Set to 0 for the whole file.

    Yield is roughly 90 eligible documents per megabyte downloaded, so the byte
    budget sets the statistical power directly: rare expressions need volume.
    """

    baseline_hours: int = 4  # 2018-01 .. 2021-12 (pre-LLM baseline)
    era_hours: int = 5  # 2022-01 .. 2025-12 (the interesting window)
    sparse_hours: int = 40  # 2026+ (degraded archive)
    max_bytes: int = 0

    def hours_for(self, year: int, month: int) -> int:
        if (year, month) >= DENSITY_COLLAPSE:
            return self.sparse_hours
        if year >= 2022:
            return self.era_hours
        return self.baseline_hours


DEFAULT_PLAN = SamplingPlan()

# A deliberately small run for validating the pipeline end to end. It trades
# sensitivity for bytes: expect to recover expressions around 1e-3 document
# frequency, not the 1e-5 ones the full plan reaches.
PILOT_PLAN = SamplingPlan(baseline_hours=1, era_hours=2, sparse_hours=8,
                          max_bytes=16 << 20)


def months(start: tuple[int, int] = START, end: tuple[int, int] = END) -> list[tuple[int, int]]:
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def sample_hours(year: int, month: int, n: int) -> list[tuple[int, int, int, int]]:
    """Deterministically pick `n` (y, m, d, h) slots inside a month.

    Days are spread across the month and hours across the clock on a diagonal,
    so a month's sample covers both the weekly and the diurnal cycle instead of
    repeatedly hitting the same time of day. The pattern is identical every
    month, which keeps months comparable; day-of-week drifts naturally because
    month lengths differ.
    """
    dim = calendar.monthrange(year, month)[1]
    n = max(1, min(n, dim * 24))
    slots = []
    for i in range(n):
        # spread days over 1..dim, wrapping if n > dim
        day = 1 + int(round(i * (dim - 1) / max(1, n - 1))) if n > 1 else 15
        day = 1 + ((day - 1 + (i // dim)) % dim)
        hour = int((i * 24) / n + 12 / n) % 24
        slots.append((year, month, day, hour))
    # de-duplicate while keeping order (possible when n > dim)
    seen, uniq = set(), []
    for s in slots:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def alternates(year: int, month: int, day: int, hour: int) -> list[tuple[int, int, int, int]]:
    """Fallback slots when an hour-file is missing or corrupt (the archive has
    genuine gaps). Nearby hours first, then nearby days."""
    dim = calendar.monthrange(year, month)[1]
    out = []
    for dh in (1, -1, 2, -2, 3, 4):
        out.append((year, month, day, (hour + dh) % 24))
    for dd in (1, -1, 2, -2):
        d = day + dd
        if 1 <= d <= dim:
            out.append((year, month, d, hour))
    return out


def period_key(ts: str, freq: str = "M") -> str:
    """`ts` is an ISO-8601 GH Archive timestamp. Returns 'YYYY-MM' or 'YYYY-Www'."""
    if freq == "M":
        return ts[:7]
    import datetime as _dt

    d = _dt.date(int(ts[0:4]), int(ts[5:7]), int(ts[8:10]))
    iso = d.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"

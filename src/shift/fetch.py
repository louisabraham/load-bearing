"""A uniform random sample of GitHub comments, 2024-01 onward.

One event type -- ``IssueCommentEvent``, the only prose-bearing stream GH Archive
still emits in every month of the window (``PullRequestEvent`` bodies stop in
2025-11, ``PushEvent`` commit arrays in 2025-10).

Every week contributes the same number of hours, drawn uniformly at random from its
168 without replacement. "Uniformly" is across the window -- the same sampling
effort in 2024-01 as in 2026-07, so a difference between two periods cannot come
from having looked harder at one of them. "At random" is inside the week -- no fixed
hour-of-day, so the sample represents the whole week rather than a chosen slice of
the clock.

The week is the sampling unit, not the comparison unit: the detector pools two weeks
on each side of every weekly boundary. Sampling weekly is what makes those two-week
windows slide one week at a time.

Nothing is authored away: bot accounts are kept, and the ``author`` column is
stored so any filtering is a decision made at analysis time rather than baked into
the data.

A missing hour is never substituted from a neighbour -- that would buy volume by
breaking the draw. It just leaves that week thinner, which the per-week
normalisation already handles.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import orjson
import polars as pl

UA = "lbdetect/0.2 (research; longitudinal language change)"
BASE = "https://data.gharchive.org"
RAW = Path("data/uniform")

ANCHOR = date(2024, 1, 1)          # a Monday
WEEK_DAYS = 7
WEEK_HOURS = WEEK_DAYS * 24
PROBE = b'"type":"IssueCommentEvent"'

HOURS_PER_WEEK = 5     # hours sampled from each week
CAP_MB = 3             # bytes read from each hour-file
SEED = 0

SCHEMA = {
    "ts": pl.Utf8,
    "repo": pl.Utf8,
    "author": pl.Utf8,
    "is_pr": pl.Boolean,
    "body": pl.Utf8,
}


def n_weeks(today: date | None = None) -> int:
    """Number of complete weeks since the anchor."""
    today = today or date.today()
    return (today - ANCHOR).days // WEEK_DAYS


def week_start(k: int) -> date:
    return ANCHOR + timedelta(days=WEEK_DAYS * int(k))


def draws(k: int, hours_per_week: int = HOURS_PER_WEEK, seed: int = SEED
          ) -> list[tuple[date, int]]:
    """The hours sampled from week `k`.

    A permutation of the week's 168 hours truncated to length K, rather than K
    independent draws: raising `hours_per_week` then extends the sample instead of
    replacing it, so a deeper run keeps every hour already fetched. Seeding on the
    week index makes the draw reproducible and independent between weeks.
    """
    perm = np.random.default_rng([seed, k]).permutation(WEEK_HOURS)
    start = week_start(k)
    return [(start + timedelta(days=int(h) // 24), int(h) % 24)
            for h in perm[:hours_per_week]]


def sample(hours_per_week: int = HOURS_PER_WEEK, seed: int = SEED,
           today: date | None = None) -> list[tuple[date, int]]:
    """The whole sample, draw-index major.

    Every week gets its first hour before any gets its second, so an interrupted
    fetch is thin everywhere rather than absent from the recent end -- the one
    arrangement that would make partial data useless here.
    """
    per_week = [draws(k, hours_per_week, seed) for k in range(n_weeks(today))]
    return [d[i] for i in range(hours_per_week) for d in per_week]


def path(day: date, hour: int) -> Path:
    return RAW / f"{day.isoformat()}-{hour:02d}.parquet"


def _lines(blob: bytes):
    """Decompress incrementally, tolerating the truncation a byte cap creates."""
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf, pos, step = b"", 0, 1 << 22
    while pos < len(blob):
        try:
            chunk = dec.decompress(blob[pos : pos + step])
        except zlib.error:
            break
        pos += step
        if chunk:
            buf += chunk
            if b"\n" in buf:
                *out, buf = buf.split(b"\n")
                yield from out


def fetch_slot(day: date, hour: int, cap_mb: int = CAP_MB, timeout: int = 180) -> dict:
    """Download one hour-file and write its comment shard. Never raises.

    The cap takes the opening slice of the hour rather than a random offset inside
    it, because a gzip stream cannot be decoded from an arbitrary byte. The same
    slice is taken from every hour, so it is a constant, not a drift.
    """
    out = path(day, hour)
    tag = out.stem
    if out.exists():
        return {"slot": tag, "ok": True, "cached": True}
    RAW.mkdir(parents=True, exist_ok=True)

    url = f"{BASE}/{day.year:04d}-{day.month:02d}-{day.day:02d}-{hour}.json.gz"
    headers = {"User-Agent": UA}
    if cap_mb:
        headers["Range"] = f"bytes=0-{(cap_mb << 20) - 1}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as r:
            blob = r.read()
    except urllib.error.HTTPError as ex:
        return {"slot": tag, "ok": False, "error": f"http {ex.code}"}
    except Exception as ex:
        return {"slot": tag, "ok": False, "error": type(ex).__name__}

    rows = []
    for line in _lines(blob):
        if PROBE not in line[:120]:
            continue
        try:
            e = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        p = e.get("payload") or {}
        c = p.get("comment") or {}
        body = c.get("body")
        if not body:
            continue
        rows.append(
            {
                "ts": e.get("created_at") or "",
                "repo": (e.get("repo") or {}).get("name") or "",
                "author": (c.get("user") or {}).get("login") or "",
                "is_pr": bool((p.get("issue") or {}).get("pull_request")),
                "body": body[:8000],
            }
        )

    df = pl.DataFrame(rows, schema=SCHEMA)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=7)
    tmp.replace(out)
    return {"slot": tag, "ok": True, "docs": len(df), "bytes": len(blob)}


def _log(msg: str) -> None:
    print(msg, flush=True)  # a background run must show progress as it happens


def run(hours_per_week: int = HOURS_PER_WEEK, cap_mb: int = CAP_MB, seed: int = SEED,
        workers: int = 12, log=_log) -> dict:
    """Fetch the sample. Resumable: an hour with a shard on disk is skipped."""
    todo = sample(hours_per_week, seed)
    log(f"fetch: {len(todo)} hours ({hours_per_week} per week) x {cap_mb}MB "
        f"= {len(todo) * cap_mb / 1024:.2f}GB ceiling, {workers} workers")
    done = docs = nbytes = failed = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_slot, d, h, cap_mb) for d, h in todo]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            docs += r.get("docs", 0)
            nbytes += r.get("bytes", 0)
            failed += 0 if r["ok"] else 1
            if done % 25 == 0 or done == len(todo):
                log(f"  {done}/{len(todo)}  docs={docs:,}  "
                    f"fetched={nbytes / 1e9:.2f}GB  failed={failed}")
    return {"hours": len(todo), "docs": docs, "bytes": nbytes, "failed": failed}

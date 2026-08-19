"""A uniform sample of GitHub issues and pull requests, taken from GitHub itself.

GH Archive is not a usable source any more, and the cause is upstream of it. Its own
tracker carries "Drastic Drop Off in Events After 2025-05-23" (issue #310, open since
July 2025, no maintainer reply) and "WatchEvent capture rate has degraded significantly
since June 2025" (#320). On GitHub's community forum the same loss was traced to "a
GitHub Event API outage propagated downstream, not an OpenDigger parsing issue", and the
same gaps appear in OSSInsight, which reads the Events API directly rather than through
the archive. GitHub has published no fix and no alternative. Measured from the files:
the archive carried 3,000-10,000 issue comments an hour through 2025-10 and 77 an hour
by 2026-07.

Nothing built on the Events API can be repaired, so this module does not use it. It uses
the search API, which has three properties that together make a clean sample possible:

* `created:` accepts timestamps, not just dates, so a window can be minutes wide;
* a window that narrow holds few enough items to be enumerated rather than sampled;
* the response carries the full body, so one request yields a hundred documents.

The budget is wall-clock, and it is latency-bound: a search query takes about five
seconds, so one hour buys roughly 700 windows and, at ~97 documents each, about 68,000
documents. Five windows a week across 137 weeks is 685 windows -- an hour's work. The
draw is a truncated permutation, so a second hour adds five more windows to every week
and refetches nothing.

Each week gets the same number of randomly placed windows, so the sample is uniform
across the window and random inside each week -- and it is authoritative, because it is
GitHub answering about its own present state rather than a replay of a broken feed.

A window wider than one page is truncated to its earliest hundred items. That is not a
bias in time: the window's placement is uniformly random, so what is sampled is still
"everything created in a uniformly random interval", with the interval's effective width
varying as GitHub's volume does.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import orjson
import polars as pl
from concurrent.futures import ThreadPoolExecutor

from .fetch import ANCHOR, SCHEMA, WEEK_DAYS, n_weeks, week_start

BASE = Path("data")     # patched in tests; the corpora live under it
UA = "lbdetect/0.2 (research; longitudinal language change)"
WINDOW_S = 300          # seconds per window; ~100 items in 2024, ~1.2 min worth in 2026
WINDOWS_PER_WEEK = 5    # ~12 windows/min serially, so 137 weeks x 5 fits in an hour

# Two filters can be pushed into the query, and together they take a hundred-item page
# from 43 usable documents to 97.
#
# Apps: `-author:app/NAME` works one slug at a time; `-author:app/*` is a 422, so there
# is no way to say "no apps". On pull requests these four are 90% of App-authored bodies
# and cut a page from 29 bot documents to 1. Adding five more changed the total by
# nothing, so the list stops here.
EXCLUDE_APPS = ("pull", "dependabot", "renovate", "github-actions")

# Empty bodies: there is no emptiness qualifier -- `-body:""` is a 422, `has:body` and
# `-in:body` are silently ignored, and `body:*` is a text match on the asterisk that cuts
# 94% rather than the 22% that are empty. But requiring any one of a few function words
# *in the body* does the job exactly: `(the in:body OR a in:body OR ...)` returns zero
# empty bodies in every era tested, from 2024-02 to 2026-07.
#
# Note `in:body` does not distribute over an OR group -- `(the OR a) in:body` matches
# titles too and lets empty bodies back in -- so the qualifier is repeated per term.
#
# It also happens to *reduce* the drift in the sampling frame rather than adding to it.
# The share of pull requests with a usable body climbs 23% to 58% across the window
# because empty descriptions are disappearing; with this filter the admitted share climbs
# only 50% to 76%, because the thing that was moving is the thing it removes. What it
# misses are short bulleted lists and CJK bodies; adding Romance-language function words
# gained 0.5%, so the set stays English and small.
PROSE_TERMS = ("the", "a", "to", "of", "and", "in", "is", "for", "that", "with")
PER_PAGE = 100
RATE = 26.0             # requests a minute; the documented search cap is 30
# Serial by default. A search query takes about five seconds, so one thread reaches
# ~12 requests a minute and never approaches the 30/min cap -- the cap is not the
# binding constraint, latency is. Raising this is tempting and was tried: GitHub's REST
# guidance asks for serial requests per credential, and five threads stalled behind
# secondary rate limiting rather than going five times faster.
WORKERS = 1
SLOTS = WEEK_DAYS * 24 * 60 * 60 // WINDOW_S


def raw_dir(exclude_apps: bool = False, require_prose: bool = False) -> Path:
    """Where a corpus lives. Each filter combination is a different population, so each
    gets its own directory rather than sharing one and being neither."""
    return BASE / ("api" + ("-noapps" if exclude_apps else "")
                   + ("-prose" if require_prose else ""))


def windows(k: int, per_week: int = WINDOWS_PER_WEEK, seed: int = 0,
            window_s: int = WINDOW_S) -> list[datetime]:
    """Start times of the windows sampled from week `k`.

    A truncated permutation of the week's slots, so raising `per_week` extends the
    sample rather than replacing it and a deeper run refetches nothing.
    """
    perm = np.random.default_rng([seed, k, window_s]).permutation(
        WEEK_DAYS * 24 * 60 * 60 // window_s)
    start = datetime.combine(week_start(k), datetime.min.time(), timezone.utc)
    return [start + timedelta(seconds=int(s) * window_s) for s in perm[:per_week]]


def path(start: datetime, kind: str, exclude_apps: bool = False,
         require_prose: bool = False) -> Path:
    return raw_dir(exclude_apps, require_prose) / f"{start:%Y-%m-%d-%H%M%S}-{kind}.parquet"


def _search(query: str, token: str) -> dict | None:
    url = ("https://api.github.com/search/issues?advanced_search=true&sort=created"
           f"&order=asc&per_page={PER_PAGE}&q={urllib.parse.quote(query)}")
    headers = {"User-Agent": UA, "Accept": "application/vnd.github+json",
               "Authorization": f"Bearer {token}"}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as r:
                return orjson.loads(r.read())
        except urllib.error.HTTPError as ex:
            if ex.code not in (403, 429):
                return None
            time.sleep(20 * (attempt + 1))     # secondary rate limit; back off
        except Exception:
            time.sleep(10)
    return None


def fetch_window(start: datetime, kind: str = "issue", token: str | None = None,
                 window_s: int = WINDOW_S, exclude_apps: tuple[str, ...] = (),
                 prose_terms: tuple[str, ...] = ()) -> dict:
    """Fetch one window into a shard. Never raises."""
    out = path(start, kind, bool(exclude_apps), bool(prose_terms))
    if out.exists():
        return {"window": out.stem, "ok": True, "cached": True}
    token = token or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        return {"window": out.stem, "ok": False, "error": "no GITHUB_TOKEN"}
    out.parent.mkdir(parents=True, exist_ok=True)

    hi = start + timedelta(seconds=window_s)
    q = (f"created:{start:%Y-%m-%dT%H:%M:%SZ}..{hi:%Y-%m-%dT%H:%M:%SZ} is:{kind}"
         + "".join(f" -author:app/{a}" for a in exclude_apps))
    if prose_terms:
        # repeated per term on purpose: in:body does not distribute over an OR group
        q += " (" + " OR ".join(f"{t} in:body" for t in prose_terms) + ")"
    data = _search(q, token)
    if data is None:
        return {"window": out.stem, "ok": False, "error": "search failed"}

    rows = [
        {
            "ts": it.get("created_at") or "",
            # the search result names the repo only by API url; the last two
            # segments are owner and name
            "repo": "/".join((it.get("repository_url") or "").split("/")[-2:]),
            "author": ((it.get("user") or {}).get("login") or "").lower(),
            "is_pr": kind == "pr",
            "body": (it.get("body") or "")[:8000],
        }
        for it in data.get("items", [])
    ]
    df = pl.DataFrame(rows, schema=SCHEMA)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=7)
    tmp.replace(out)
    return {"window": out.stem, "ok": True, "docs": len(df),
            "available": int(data.get("total_count", 0)),
            "truncated": int(data.get("total_count", 0)) > PER_PAGE}


def _log(msg: str) -> None:
    print(msg, flush=True)  # a background run must show progress as it happens


class _Pace:
    """Hand out request slots at a fixed rate, whatever the latency.

    The limit that matters is requests per minute across the whole run, so the slots
    are issued from one clock rather than by sleeping after each reply. Workers then
    overlap GitHub's several-second search latency instead of adding to it.
    """

    def __init__(self, per_minute: float):
        self.gap = 60.0 / per_minute
        self.next = 0.0
        self.lock = __import__("threading").Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next)
            self.next = at + self.gap
        if at > now:
            time.sleep(at - now)


def run(per_week: int = WINDOWS_PER_WEEK, kind: str = "issue", seed: int = 0,
        window_s: int = WINDOW_S, token: str | None = None, rate: float = RATE,
        workers: int = WORKERS, exclude_apps: tuple[str, ...] = (),
        prose_terms: tuple[str, ...] = (), log=_log) -> dict:
    """Fetch the sample, window-index major so a partial run covers every week."""
    weeks = range(n_weeks())
    todo = [w for i in range(per_week)
            for k in weeks for w in [windows(k, per_week, seed, window_s)[i]]]
    log(f"api sample: {len(todo)} windows of {window_s}s ({per_week} per week) over "
        f"{len(weeks)} weeks, kind={kind}, {rate:.0f}/min in {workers} threads, "
        f"~{len(todo) / rate:.0f} min")
    pace = _Pace(rate)
    done = docs = trunc = failed = 0

    def one(w):
        if not path(w, kind, bool(exclude_apps), bool(prose_terms)).exists():
            pace.wait()
        return fetch_window(w, kind, token, window_s, exclude_apps, prose_terms)

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            done += 1
            docs += r.get("docs", 0)
            trunc += bool(r.get("truncated"))
            failed += 0 if r["ok"] else 1
            if done % 50 == 0 or done == len(todo):
                log(f"  {done}/{len(todo)}  docs={docs:,}  truncated={trunc}  "
                    f"failed={failed}")
    return {"windows": len(todo), "docs": docs, "truncated": trunc, "failed": failed}


def week_of(shard: Path) -> int:
    from datetime import date
    return (date.fromisoformat(shard.stem[:10]) - ANCHOR).days // WEEK_DAYS


def groups(kind: str = "issue", exclude_apps: bool = False,
           require_prose: bool = False) -> dict[int, list[Path]]:
    """Shards of one kind, by week.

    The kind is part of the sample definition, not a detail of the filename: issue
    bodies and pull request bodies are different populations written by different people
    for different reasons, and a corpus silently holding both would be neither.
    """
    out: dict[int, list[Path]] = {}
    for f in sorted(raw_dir(exclude_apps, require_prose).glob(f"*-{kind}.parquet")):
        k = week_of(f)
        if 0 <= k < n_weeks():
            out.setdefault(k, []).append(f)
    return dict(sorted(out.items()))

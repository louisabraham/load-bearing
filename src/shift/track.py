"""Count one expression per month, straight from GitHub, on any writing surface.

GH Archive stopped being usable for this. Measured from the files themselves -- comment
count in a fixed byte window, scaled by the hour-file's true size -- the feed carried
between three and ten thousand issue comments an hour from 2024-01 through 2025-10, and
then decayed: about 3,900/hour that winter, 1,590 in 2026-03, 1,076 in 2026-04, 866 in
2026-06 and 77 in 2026-07. A complete hour of 2024-08-12 holds 13,555 IssueCommentEvents
against 86 for the same hour of 2026-08-10. Issues, pull requests and reviews fell with
them; pushes survive at full volume but carry no text, GitHub having removed the commit
array from the payload in October 2025. So the archive holds no usable prose for 2026 on
any surface, and every bulk mirror is derived from the same feed.

What does work is GitHub's own search, which reports how many items match instead of
returning them, and so answers a frequency question in one request. The surfaces below
are different populations and worth reading against each other: comments are heavily
bot-written, commit messages are short and mostly human, issue and pull request bodies
sit in between. An expression that rises on all of them is a change in how people
write; one that rises only in comments is probably a tool.

Two limits. On the issue surfaces a match is an *item*, not an occurrence, so a busy
thread counts once. And the date is the item's creation date, not the moment the words
were written, so a comment added in 2026 to an issue opened in 2024 lands in 2024 --
which blunts a rising signal rather than inventing one.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import orjson
import polars as pl

UA = "lbdetect/0.2 (research; longitudinal language change)"
DENOM_CACHE = Path("data/denominators.json")
# Unauthenticated search allows 10 requests a minute, a token 30. Staying under the
# limit by construction is cheaper than handling the 403 that follows exceeding it.
PAUSE_ANON = 7.0
PAUSE_TOKEN = 2.5


class Surface(NamedTuple):
    endpoint: str      # which search endpoint
    date: str          # the date qualifier that endpoint understands
    scope: str         # qualifiers that pick the surface out
    denom: str         # qualifiers for "everything on this surface"
    unit: str          # what one match is, for reading the rate


SURFACES: dict[str, Surface] = {
    "comment": Surface("issues", "created", "in:comments", "is:issue", "issues"),
    "issue": Surface("issues", "created", "in:body is:issue", "is:issue", "issues"),
    "pr": Surface("issues", "created", "in:body is:pr", "is:pr", "PRs"),
    "title": Surface("issues", "created", "in:title", "is:issue", "issues"),
    "commit": Surface("commits", "author-date", "", "", "commits"),
}


def _get(endpoint: str, query: str, token: str | None) -> int | None:
    url = (f"https://api.github.com/search/{endpoint}"
           f"?per_page=1&advanced_search=true&q={urllib.parse.quote(query)}")
    headers = {"User-Agent": UA, "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as r:
                return int(orjson.loads(r.read()).get("total_count", 0))
        except urllib.error.HTTPError as ex:
            if ex.code not in (403, 429):
                return None
            time.sleep(20 * (attempt + 1))     # secondary rate limit; back off
        except Exception:
            time.sleep(10)
    return None


def months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    out, (y, m) = [], start
    while (y, m) <= end:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _span(s: Surface, y: int, m: int) -> str:
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{s.date}:{y}-{m:02d}-01..{ny}-{nm:02d}-01"


def _denominators(surface: str, todo: list[tuple[int, int]], token: str | None,
                  pause: float, log) -> dict[str, int]:
    """How much of this surface exists per month, cached across runs.

    The denominator does not depend on the expression, so caching it means a second
    expression costs half as many requests -- which matters when the budget is ten
    requests a minute.
    """
    cache: dict = {}
    if DENOM_CACHE.exists():
        cache = json.loads(DENOM_CACHE.read_text())
    got = cache.setdefault(surface, {})
    s = SURFACES[surface]
    for y, m in todo:
        key = f"{y}-{m:02d}"
        if key in got:
            continue
        n = _get(s.endpoint, f"{s.denom} {_span(s, y, m)}".strip(), token)
        time.sleep(pause)
        if n is not None:
            got[key] = n
            log(f"  denominator {surface} {key}: {n:,}")
    DENOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DENOM_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return got


def monthly(term: str, start: tuple[int, int], end: tuple[int, int],
            surface: str = "comment", token: str | None = None, log=print
            ) -> pl.DataFrame:
    """Monthly matches for `term` on one surface, and the rate per 10k of it."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; have {sorted(SURFACES)}")
    token = token or os.environ.get("GITHUB_TOKEN") or None
    pause = PAUSE_TOKEN if token else PAUSE_ANON
    s = SURFACES[surface]
    todo = months(start, end)
    denom = _denominators(surface, todo, token, pause, log)

    rows = []
    for y, m in todo:
        key = f"{y}-{m:02d}"
        hits = _get(s.endpoint, f'"{term}" {s.scope} {_span(s, y, m)}'.replace("  ", " "),
                    token)
        time.sleep(pause)
        rows.append({"month": key, "surface": surface, "hits": hits,
                     "total": denom.get(key)})
        log(f"  {surface:8s} {key}  {hits!s:>8} of {denom.get(key)!s:>12}")
    return pl.DataFrame(rows).with_columns(
        (pl.col("hits") / pl.col("total") * 10_000).alias("per_10k")
    )

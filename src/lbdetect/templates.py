"""Data-driven removal of repeated boilerplate *inside* otherwise-unique documents.

Whole-document de-duplication cannot catch the most damaging kind of template:
a banner or checklist that a platform or tool inserts into text a human really
wrote. The surrounding document is unique, so nothing collides, yet the inserted
line appears in tens of thousands of documents and looks exactly like a fast,
broad, persistent rise in language.

The rule is deliberately mechanical rather than a list of known banners, because
the set of banners changes faster than any list: a line is boilerplate if the
same long line appears verbatim in many documents across many repositories. Short
lines are exempt, so genuine formulaic human utterances ("lgtm", "thanks!")
survive as language.
"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from . import config as C

LINES = C.SERIES / "template_lines.parquet"

_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")

MIN_TOKENS = 5  # shorter lines are idiom, not boilerplate
MIN_REPOS = 5
MIN_FRAC = 3e-4  # of a period's documents
MIN_ABS = 4


def norm_line(line: str) -> str:
    """Normalise a line for template matching.

    Digit runs collapse to a placeholder: the most common templates interpolate a
    count ("reviewed 67 of 69 files", "ready in 3 minutes"), so matching on
    literal text would treat every instance as a different line and mine none of
    them.
    """
    s = _PUNCT.sub("", _WS.sub(" ", line).strip().lower())
    return _DIGITS.sub("#", s)


def _mine_period(period: str, min_tokens: int, min_repos: int,
                 min_frac: float, min_abs: int) -> list[str]:
    from .series import load_period

    df = load_period(period, apply_templates=False)
    if df.height == 0:
        return []
    counts: dict[str, int] = defaultdict(int)
    repos: dict[str, set[int]] = defaultdict(set)
    for text, repo_id in zip(df["text"].to_list(), df["repo_id"].to_list()):
        seen = set()
        for raw in text.split("\n"):
            ln = norm_line(raw)
            if ln.count(" ") + 1 < min_tokens or ln in seen:
                continue
            seen.add(ln)
            counts[ln] += 1
            repos[ln].add(repo_id)
    floor = max(min_abs, int(min_frac * df.height))
    return [ln for ln, c in counts.items() if c >= floor and len(repos[ln]) >= min_repos]


def mine(periods: list[str] | None = None, workers: int = 12, min_tokens: int = MIN_TOKENS,
         min_repos: int = MIN_REPOS, min_frac: float = MIN_FRAC, min_abs: int = MIN_ABS,
         log=print) -> frozenset[str]:
    """Mine template lines per period and keep the union.

    Union rather than per-period application: a line that is boilerplate in 2025
    is boilerplate whenever it appears, and applying period-specific cleaning
    would make the text pipeline itself a function of time -- which is precisely
    the confound the study is trying to avoid.
    """
    from .series import available_periods

    periods = periods or available_periods()
    out: set[str] = set()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_mine_period, p, min_tokens, min_repos, min_frac, min_abs)
                for p in periods]
        for i, fut in enumerate(as_completed(futs), 1):
            out.update(fut.result())
            if i % 20 == 0 or i == len(periods):
                log(f"  {i}/{len(periods)} periods, {len(out):,} template lines")
    pl.DataFrame({"line": sorted(out)}).write_parquet(LINES)
    log(f"templates: {len(out):,} boilerplate lines")
    return frozenset(out)


_CACHE: dict[str, frozenset[str]] = {}


def load() -> frozenset[str]:
    if "t" not in _CACHE:
        if LINES.exists():
            _CACHE["t"] = frozenset(pl.read_parquet(LINES)["line"].to_list())
        else:
            _CACHE["t"] = frozenset()
    return _CACHE["t"]


def strip(text: str, templates: frozenset[str] | None = None) -> str:
    t = load() if templates is None else templates
    if not t:
        return text
    keep = [ln for ln in text.split("\n") if norm_line(ln) not in t]
    return "\n".join(keep)

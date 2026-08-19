"""Fortnightly document-frequency matrix over whitespace words.

A word is a run of non-space characters, lowercased, with surrounding punctuation
removed -- that is the only normalisation. No stemming, no n-grams, no stopword
list. Each word is counted at most once per document, so an entry is a number of
documents and a single loquacious comment cannot move a rate however often it
repeats itself.

Every author counts, bots included. Machine-written comments are a large and
growing part of what GitHub prose is, and excluding them would remove the clearest
carrier of the change being looked for. The `author` column is kept in the shards,
so filtering stays a decision made here rather than one baked into the data.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from .fetch import (ANCHOR, HOURS_PER_WEEK, RAW, SEED, WEEK_DAYS, draws,
                    n_weeks, path)

MATRIX = Path("data/matrix.npz")
MIN_WORDS = 5          # a document needs this many distinct words to be prose
MIN_DF = 100           # a word needs this many documents across the window
DOCS_PER_WEEK = 800    # every week contributes this many documents, or all it has
STRIP = "\"'`*_~<>|#.,;:!?()[]{}"


def words(body: str) -> set[str]:
    """The distinct words of one document.

    Purely numeric tokens are dropped. They are dates, versions, counts and line
    numbers rather than vocabulary, and they are an active nuisance here: the
    calendar advances at every boundary, so a bare `10` or `2026` arrives and departs
    on a schedule of its own. Month abbreviations do the same thing and are not
    filtered, because they are words -- `apr` jumping from 0.05% to 9% of documents
    at the end of March is a real artifact to read past, not a real change.
    """
    return {w for w in (t.strip(STRIP) for t in body.lower().split())
            if w and not w.isdigit()}


def week_of(shard: Path) -> int:
    """Which week a shard belongs to.

    Taken from the filename, not the timestamps: a shard is a single hour of a
    single day, so every document in it falls in the same week.
    """
    return (date.fromisoformat(shard.stem[:10]) - ANCHOR).days // WEEK_DAYS


def groups(hours: int = HOURS_PER_WEEK, seed: int = SEED, source: str = "archive"
           ) -> dict[int, list[Path]]:
    """The sampled shards, by week.

    For the archive this is driven by the draw definition rather than by whatever
    happens to be in the directory: a stray file, or one fetched under a different byte
    cap, is not part of that sample and must not silently join it. For the API sampler
    the directory *is* the sample, because each shard is one enumerated window.
    """
    if source.startswith("api"):
        from . import apifetch
        # "api" is issue bodies, "api:pr" pull request bodies; tags on the prefix pick
        # the filtered corpora, e.g. "api-noapps-prose:pr"
        base, _, kind = source.partition(":")
        return apifetch.groups(kind or "issue", "noapps" in base, "prose" in base)
    out: dict[int, list[Path]] = {}
    for k in range(n_weeks()):
        got = [p for p in (path(d, h) for d, h in draws(k, hours, seed))
               if p.exists()]
        if got:
            out[k] = got
    return out


def documents(shard: Path, drop_bots: bool = False):
    """(repo, is_bot, word set) for the eligible documents of one shard."""
    df = pl.read_parquet(shard, columns=["author", "repo", "body"])
    bot = pl.col("author").str.ends_with("[bot]")   # GitHub's own label for Apps
    if drop_bots:
        df = df.filter(~bot)
    df = df.with_columns(bot.alias("is_bot"))
    for repo, is_bot, body in zip(df["repo"].to_list(), df["is_bot"].to_list(),
                                  df["body"].to_list()):
        w = words(body)
        if len(w) >= MIN_WORDS:
            yield repo, is_bot, w


def week(group: list[Path], drop_bots: bool = False):
    """The documents of one week, verbatim repeats collapsed.

    Two documents with the identical set of words count once. This is about text, not
    authorship: one ordinary human account ran a mass-close script and posted 147
    copies of the same sentence inside a fortnight -- 16% of that fortnight's
    documents -- and every word of its template moved with it. Exact-set equality
    costs a hash lookup, so no near-duplicate machinery is involved. And because it
    applies inside each week rather than across the window, a template that runs for
    months contributes one document to every week alike, which is a level rather
    than a change.
    """
    seen: set[frozenset[str]] = set()
    for shard in group:
        for repo, is_bot, w in documents(shard, drop_bots):
            key = frozenset(w)
            if key not in seen:
                seen.add(key)
                yield repo, is_bot, w


def week_docs(group: list[Path], k: int, cap: int = DOCS_PER_WEEK,
              seed: int = SEED, drop_bots: bool = False):
    """One week's documents, repeats collapsed and then thinned to `cap`.

    Sampling the same number of *hours* from every week does not give the same number
    of *documents*: the archive's own comment volume swings by a factor of two across
    the window, and mid-2026 it collapses by a factor of twenty. That matters because
    real text is overdispersed -- words cluster inside repositories and threads -- so
    a z-score computed on more documents is inflated rather than merely more precise,
    and a boundary score would rank the busy stretches of the archive above the busy
    stretches of language. Thinning every week to a common size is what makes two
    boundaries in different years comparable at all.

    The subsample is seeded on the week, and the document order is deterministic, so
    both passes of the build see the same documents.
    """
    docs = list(week(group, drop_bots))
    if cap and len(docs) > cap:
        keep = np.random.default_rng([seed, k]).choice(len(docs), cap, replace=False)
        docs = [docs[i] for i in np.sort(keep)]
    return docs


def build(min_df: int = MIN_DF, hours: int = HOURS_PER_WEEK, seed: int = SEED,
          cap: int = DOCS_PER_WEEK, drop_bots: bool = False, source: str = "archive",
          log=print) -> dict:
    """Two passes over the sample: choose the vocabulary, then fill the matrix."""
    T = n_weeks()
    by_week = groups(hours, seed, source)
    if not by_week:
        raise SystemExit(f"no sampled shards in {RAW} -- run `fetch-data` first")
    have = np.array([len(by_week.get(k, ())) for k in range(T)])
    unit = "windows" if source == "api" else "hours"
    log(f"source={source}: {have.sum()} sampled {unit} present over {T} weeks; "
        f"per week min {have.min()}, median {int(np.median(have))}")

    df_total: Counter[str] = Counter()
    n = np.zeros(T, dtype=np.int64)
    for k, group in by_week.items():
        for _, _, w in week_docs(group, k, cap, seed, drop_bots):
            n[k] += 1
            df_total.update(w)
    short = int(((n < cap) & (n > 0)).sum()) if cap else 0
    log(f"pass 1: {n.sum():,} documents, {len(df_total):,} distinct words"
        + (f"; {short} of {T} weeks hold fewer than the {cap} asked for" if cap else ""))

    vocab = sorted(w for w, c in df_total.items() if c >= min_df)
    index = {w: j for j, w in enumerate(vocab)}
    V = len(vocab)
    log(f"vocabulary: {V:,} words with df >= {min_df}")

    X = np.zeros((T, V), dtype=np.int64)
    for k, group in by_week.items():
        idx = [index[w] for _, _, ws in week_docs(group, k, cap, seed, drop_bots)
               for w in ws if w in index]
        if idx:
            X[k] += np.bincount(np.asarray(idx), minlength=V)
    log("pass 2: matrix filled")

    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    # min_df travels with the matrix: it is a property of this matrix, and a report
    # that printed the module default instead would misdescribe its own input
    np.savez_compressed(MATRIX, X=X.astype(np.int32), n=n,
                        vocab=np.array(vocab, dtype=object), min_df=min_df)
    return {"weeks": T, "hours": int(have.sum()), "docs": int(n.sum()),
            "vocab": V, "docs_per_week": int(np.median(n)),
            "short_weeks": short, "empty_weeks": int((n == 0).sum())}


class Matrix(NamedTuple):
    X: np.ndarray          # (weeks, words) document counts
    n: np.ndarray          # (weeks,) documents
    vocab: list[str]
    min_df: int


def load() -> Matrix:
    d = np.load(MATRIX, allow_pickle=True)
    return Matrix(d["X"], d["n"], list(d["vocab"]), int(d["min_df"]))


class Support(NamedTuple):
    repos: int             # distinct repositories the word appeared in
    bot_share: float       # fraction of its documents written by an App account


def support(weeks: range, target: list[str], hours: int = HOURS_PER_WEEK,
            seed: int = SEED, cap: int = DOCS_PER_WEEK, source: str = "archive"
            ) -> dict[str, Support]:
    """Who and where carried each target word over a span of weeks.

    Two questions the matrix cannot answer, because one document is one document.
    A rate can rise because many people started writing a word or because one busy
    repository did, and -- since every author counts here -- because a tool was
    deployed rather than because people changed how they write. Both are checked
    afterwards, on the handful of words actually reported, where a second look
    costs nothing.
    """
    want = set(target)
    repos: dict[str, set[str]] = {w: set() for w in want}
    docs: Counter[str] = Counter()
    bots: Counter[str] = Counter()
    by_week = groups(hours, seed, source)
    for k in weeks:
        # the same thinned documents the matrix counted, or the shares would
        # describe a different corpus from the one that produced the z-scores
        for repo, is_bot, ws in week_docs(by_week.get(k, []), k, cap, seed):
            for w in ws & want:
                repos[w].add(repo)
                docs[w] += 1
                bots[w] += is_bot
    return {w: Support(len(v), bots[w] / docs[w] if docs[w] else 0.0)
            for w, v in repos.items()}

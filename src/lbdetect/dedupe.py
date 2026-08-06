"""MinHash near-duplicate and template detection.

Two distinct problems, one mechanism:

* **Duplicate documents** - the same text posted many times (cross-posted reports,
  copied answers). Counted once, they would look like a real frequency rise.
* **Templates** - many documents sharing most of their text with each other. A
  new issue template or PR checklist rolling out across a repo produces exactly
  the "sudden broad rise" signature we are hunting for.

Both are found by banded LSH over MinHash signatures of 5-gram shingles. The
output is a cluster id per document plus a `is_template` flag for members of
large cross-repository clusters.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

MERSENNE = (1 << 61) - 1


def _shingles(text: str, k: int = 5) -> list[int]:
    toks = text.lower().split()
    if len(toks) < k:
        return [hash(" ".join(toks)) & 0xFFFFFFFFFFFFFFF] if toks else []
    return [
        hash(" ".join(toks[i : i + k])) & 0xFFFFFFFFFFFFFFF
        for i in range(len(toks) - k + 1)
    ]


class MinHasher:
    def __init__(self, n_perm: int = 64, seed: int = 17):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, size=n_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, size=n_perm, dtype=np.uint64)
        self.n_perm = n_perm

    def signature(self, text: str) -> np.ndarray | None:
        sh = _shingles(text)
        if not sh:
            return None
        x = np.array(sh, dtype=np.uint64)
        # (a*x + b) mod (2^61 - 1), vectorised over permutations
        h = (np.outer(self.a, x) + self.b[:, None]) % np.uint64(MERSENNE)
        return h.min(axis=1)


def cluster(
    texts: list[str],
    n_perm: int = 64,
    bands: int = 16,
    seed: int = 17,
) -> np.ndarray:
    """Union-find over LSH band collisions. Returns a cluster id per document.

    `bands` controls the similarity threshold: with 64 permutations and 16 bands
    (4 rows each) documents sharing ~0.6+ Jaccard collide with high probability.
    """
    n = len(texts)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    mh = MinHasher(n_perm, seed)
    sigs: dict[int, np.ndarray] = {}
    for i, t in enumerate(texts):
        s = mh.signature(t)
        if s is not None:
            sigs[i] = s

    rows = n_perm // bands
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        lo, hi = band * rows, (band + 1) * rows
        for i, s in sigs.items():
            buckets[s[lo:hi].tobytes()].append(i)
        for members in buckets.values():
            if len(members) > 1:
                first = members[0]
                for other in members[1:]:
                    union(first, other)

    return np.array([find(i) for i in range(n)])


def _prefix_key(text: str, n_tokens: int = 6) -> str:
    return " ".join(text.lower().split()[:n_tokens])


def templated_authors(
    df: pl.DataFrame,
    min_docs: int = 8,
    min_repos: int = 3,
    min_prefix_share: float = 0.5,
) -> set[str]:
    """Authors whose documents share a structural opening across repositories.

    Near-duplicate detection misses tools whose output is templated in *shape* but
    unique in content -- an automated reviewer describing a different pull request
    each time collides with nothing, yet every one of its comments opens
    identically. Comparing modal opening phrases finds these without a name list,
    which matters because the set of AI review tools changes faster than any
    hand-maintained list.

    A prolific human contributor is not caught: they would have to open half their
    comments with the same six words across three or more repositories.
    """
    if df.height == 0 or "author" not in df.columns:
        return set()
    keys = [_prefix_key(t) for t in df["text"].to_list()]
    tmp = df.select("author", "repo_id").with_columns(prefix=pl.Series(keys))
    stats = (
        tmp.group_by("author")
        .agg(
            pl.len().alias("n_docs"),
            pl.col("repo_id").n_unique().alias("n_repos"),
            pl.col("prefix").mode().first().alias("modal"),
        )
        .filter((pl.col("n_docs") >= min_docs) & (pl.col("n_repos") >= min_repos))
    )
    if stats.height == 0:
        return set()
    share = (
        tmp.join(stats.select("author", "modal"), on="author", how="inner")
        .group_by("author")
        .agg((pl.col("prefix") == pl.col("modal")).mean().alias("prefix_share"))
        .filter(pl.col("prefix_share") >= min_prefix_share)
    )
    return set(share["author"].to_list())


def annotate(df: pl.DataFrame, min_template_size: int = 8,
             min_template_repos: int = 3) -> pl.DataFrame:
    """Add `dup_cluster`, `is_dup` (non-canonical copy) and `is_template`.

    A cluster confined to one repository is a repeated local template; a cluster
    spanning several repositories is a platform-wide or tool-generated template.
    Both are excluded from counting, but they are flagged separately so the
    confounder report can say which kind it was.
    """
    if df.height == 0:
        return df.with_columns(
            dup_cluster=pl.lit(0, pl.Int64),
            is_dup=pl.lit(False),
            is_template=pl.lit(False),
            is_templated_author=pl.lit(False),
        )
    bots = templated_authors(df)
    df = df.with_columns(is_templated_author=pl.col("author").is_in(list(bots)))
    cid = cluster(df["text"].to_list())
    df = df.with_columns(dup_cluster=pl.Series(cid, dtype=pl.Int64))
    stats = df.group_by("dup_cluster").agg(
        pl.len().alias("_size"), pl.col("repo_id").n_unique().alias("_repos")
    )
    df = (
        df.join(stats, on="dup_cluster", how="left")
        .with_columns(
            is_template=(pl.col("_size") >= min_template_size)
            & (pl.col("_repos") >= min_template_repos),
            _rank=pl.int_range(pl.len()).over("dup_cluster"),
        )
        .with_columns(is_dup=pl.col("_rank") > 0)
        .drop("_size", "_repos", "_rank")
    )
    return df

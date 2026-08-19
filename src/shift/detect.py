"""Find the weeks where the whole word distribution moves at once.

The test is applied at every week boundary, and each test pools two weeks on each
side of it: weeks k-2 and k-1 against weeks k and k+1. Two weeks a side buys enough
documents for a word-level comparison; testing every week means the pair of windows
slides one week at a time, so a change is located to the week rather than to
whichever fortnight it happened to fall inside. Neighbouring tests therefore share
three quarters of their data and their scores are correlated: a real change shows up
as a short run of elevated boundaries, and the peak of the run is the estimate.

At each boundary every word in the vocabulary gets a z-score for "documents
containing it before" against "after". That turns the boundary into a single vector
of V z-scores, and the question "did a group of words change abruptly here?" becomes
a question about the length of that vector: under no change the z are roughly
standard normal, so the sum of squares is chi-square and

    S = (sum z^2 - (V-1)) / sqrt(2 (V-1))

is the same quantity in standard-deviation units for any vocabulary size.

The z is a difference of log-odds rather than of rates, with the median difference
across the vocabulary removed, and that detail carries most of the weight. Document
frequency depends on document length: if comments simply get longer, every common
word appears in more of them, and a raw rate comparison then reports `to`, `of` and
`is` all doubling -- a change in verbosity wearing the costume of a change in
vocabulary. Longer documents shift every word's log-odds by roughly the same
additive amount (a word appears if any of L tokens is it, so log-odds ~ log q +
log L), so subtracting the median difference removes exactly the part of the move
that hit the whole vocabulary equally and leaves what is specific to each word.
The median is the right summary here because it is unmoved by the minority of words
that genuinely changed. The cost is a real limit: if a majority of the vocabulary
moved the same way at once, the median would absorb that too -- but a change that
broad is not distinguishable from a length change using document counts alone, so
the limit is in the data, not in the choice of estimator.

Real text is overdispersed -- words cluster inside repositories and threads -- so
S has a heavier null than theory says. Rather than model that, S is calibrated
against its own spread across the other boundaries in the window (median and MAD),
which is honest about the scale actually observed instead of the one assumed.

The z-vector then answers the second question for free: the words that moved are
the large components, and how many are large says whether a boundary is one word
or a whole register shifting.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from .fetch import WEEK_DAYS, week_start

# Model and coding-assistant releases inside the window. Annotations only: the
# list is here so a detected boundary can be read next to what shipped near it,
# not so alignment can be scored. It is deliberately confined to well-documented
# public launches, and it stops in 2025-10 -- boundaries in 2026 have no candidate
# here, which means "unknown", not "nothing shipped".
RELEASES: list[tuple[str, str]] = [
    ("2024-02-15", "Gemini 1.5 Pro"),
    ("2024-03-04", "Claude 3 (Opus/Sonnet)"),
    ("2024-05-13", "GPT-4o"),
    ("2024-06-20", "Claude 3.5 Sonnet"),
    ("2024-09-12", "OpenAI o1-preview"),
    ("2024-10-22", "Claude 3.5 Sonnet (new)"),
    ("2024-12-05", "OpenAI o1"),
    ("2024-12-11", "Gemini 2.0 Flash"),
    ("2025-01-20", "DeepSeek-R1"),
    ("2025-02-24", "Claude 3.7 Sonnet + Claude Code preview"),
    ("2025-03-25", "Gemini 2.5 Pro"),
    ("2025-04-14", "GPT-4.1"),
    ("2025-04-16", "OpenAI o3 / Codex CLI"),
    ("2025-05-16", "OpenAI Codex agent"),
    ("2025-05-22", "Claude 4 (Opus/Sonnet)"),
    ("2025-08-07", "GPT-5"),
    ("2025-09-29", "Claude Sonnet 4.5"),
    ("2025-10-15", "Claude Haiku 4.5"),
]

HALF = 2            # weeks pooled on each side of a boundary
Z_GATE = 4.0        # |z| above which a word counts as part of the moving group
MIN_DOCS = 1000     # a boundary needs this many documents on both sides


def pool(X: np.ndarray, n: np.ndarray, half: int = HALF):
    """Rolling `half`-week sums either side of every week boundary.

    Returns (before, after, docs before, docs after, cut week) with one row per
    testable boundary. Cumulative sums do the pooling, so widening the window costs
    nothing and the two sides are always exactly `half` weeks each.
    """
    T = len(n)
    cx = np.vstack([np.zeros((1, X.shape[1]), dtype=np.int64),
                    np.cumsum(X.astype(np.int64), axis=0)])
    cn = np.concatenate([[0], np.cumsum(n.astype(np.int64))])
    lo = np.arange(max(T - 2 * half + 1, 0))
    mid, hi = lo + half, lo + 2 * half
    return (cx[mid] - cx[lo], cx[hi] - cx[mid],
            cn[mid] - cn[lo], cn[hi] - cn[mid], mid)


def _logodds(a: np.ndarray, b: np.ndarray, na: np.ndarray, nb: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Per-word log-odds change at every boundary, and its standard error."""
    # half a document on each side of each cell: the standard continuity correction
    # for log-odds, and it keeps a word absent from one window finite
    a, b = a + 0.5, b + 0.5
    na, nb = na[:, None] + 1.0, nb[:, None] + 1.0
    a_, b_ = na - a, nb - b
    d = np.log(b / b_) - np.log(a / a_)
    se = np.sqrt(1.0 / a + 1.0 / a_ + 1.0 / b + 1.0 / b_)
    return d, se


def common_shift(X: np.ndarray, n: np.ndarray, half: int = HALF) -> np.ndarray:
    """The odds ratio the median word moved by -- how much the boundary shifted
    the whole vocabulary at once. Mostly document length, and reported next to
    every boundary so it is visible rather than silently subtracted."""
    a, b, na, nb, _ = pool(X, n, half)
    d, _ = _logodds(a, b, na, nb)
    return np.exp(np.median(d, axis=1))


def zscores(X: np.ndarray, n: np.ndarray, half: int = HALF) -> np.ndarray:
    """One row of per-word z per boundary, net of the common shift."""
    a, b, na, nb, _ = pool(X, n, half)
    d, se = _logodds(a, b, na, nb)
    d = d - np.median(d, axis=1, keepdims=True)
    return np.nan_to_num(d / se, nan=0.0, posinf=0.0, neginf=0.0)


def strength(z: np.ndarray) -> np.ndarray:
    dof = z.shape[1] - 1                          # one lost to the median
    return (np.square(z).sum(axis=1) - dof) / np.sqrt(2.0 * dof)


def _robust(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Standardise against the boundaries we trust, not against all of them."""
    ref = s[mask]
    if ref.size < 4:
        return np.zeros_like(s)
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (s - med) / (mad if mad > 0 else 1.0)


NEAR_DAYS = 60      # how far a release may sit from a cut and still be a candidate


def nearest_release(cut: date, within: int = NEAR_DAYS) -> tuple[str, int]:
    """Closest release and its signed distance in days (negative = before cut).

    Nothing within `within` days returns blank rather than the least-distant entry
    on the calendar: a release eight months away is not an explanation, and printing
    it next to a boundary invites reading one anyway. The window is generous because
    adoption lags a release by an unknown amount.
    """
    best, best_d = ("", 10**6)
    for iso, name in RELEASES:
        d = (date.fromisoformat(iso) - cut).days
        if abs(d) < abs(best_d):
            best, best_d = name, d
    return (best, best_d) if abs(best_d) <= within else ("", 0)


def scan(X: np.ndarray, n: np.ndarray, half: int = HALF, min_docs: int = MIN_DOCS,
         z_gate: float = Z_GATE) -> tuple[pl.DataFrame, np.ndarray]:
    """One row per week boundary, ordered in time. Also returns the z matrix."""
    a, b, na, nb, mid = pool(X, n, half)
    z = zscores(X, n, half)
    s = strength(z)
    usable = (na >= min_docs) & (nb >= min_docs)
    cuts = [week_start(k) for k in mid]
    near = [nearest_release(c) for c in cuts]
    frame = pl.DataFrame(
        {
            "i": np.arange(len(s)),
            "cut": cuts,
            "docs_before": na,
            "docs_after": nb,
            "S": s,
            "shift": _robust(s, usable),
            "common": common_shift(X, n, half),
            "n_up": (z > z_gate).sum(axis=1),
            "n_down": (z < -z_gate).sum(axis=1),
            "usable": usable,
            "release": [name for name, _ in near],
            "release_days": [days for _, days in near],
        }
    )
    return frame, z


def movers(z: np.ndarray, X: np.ndarray, n: np.ndarray, vocab: list[str], i: int,
           k: int = 25, rising: bool = True, half: int = HALF) -> pl.DataFrame:
    """The words that moved most at boundary `i`."""
    A, B, NA, NB, _ = pool(X, n, half)
    col = z[i]
    order = np.argsort(-col if rising else col)[:k]
    before, after = A[i][order], B[i][order]
    nb, na = max(int(NA[i]), 1), max(int(NB[i]), 1)
    # the fold change adds half a document to each side: a word that was absent
    # before would otherwise divide by zero and report an infinity, which says
    # less than "at least this much" does
    ratio = ((after + 0.5) / na) / ((before + 0.5) / nb)
    return pl.DataFrame(
        {
            "word": [vocab[j] for j in order],
            "z": col[order],
            "pct_before": 100.0 * before / nb,
            "pct_after": 100.0 * after / na,
            "docs_before": before,
            "docs_after": after,
            "ratio": ratio,
        }
    )


def window(cut: date, half: int = HALF) -> tuple[date, date, date]:
    """The two spans a boundary compares: [start, cut) against [cut, end)."""
    span = timedelta(days=WEEK_DAYS * half)
    return cut - span, cut, cut + span


def weeks_after(i: int, half: int = HALF) -> range:
    """The weeks on the after side of boundary `i`, for a provenance check."""
    return range(i + half, i + 2 * half)

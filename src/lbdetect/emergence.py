"""Per-expression emergence detection.

For every candidate expression we locate the single best level shift in its
frequency series and describe it: how large, how fast, how significant, and
whether it held. Two statistics do most of the work and answer different
questions:

* a **binomial z**, which asks whether the post-shift document count is
  compatible with the pre-shift rate. This is the right test for rare events,
  where a rate can triple on a handful of documents.
* a **robust z** against a rolling median with MAD scale, which asks whether the
  jump is unusual *for this expression*, given how noisy it normally is.

An expression that only the first statistic likes is probably rare and lucky; one
that only the second likes is probably noisy. Both are kept and reported.

A later decline never reduces the emergence score -- an expression that rose in
2023 and faded in 2025 still emerged in 2023 -- but the decline is recorded,
because a rise-then-fall profile is what pins down a narrow date range.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .series import Series

MIN_PRE_PERIODS = 8
MIN_POST_PERIODS = 4
EPS = 1e-9


def _regularized(p: np.ndarray, n: np.ndarray | float) -> np.ndarray:
    """Floor a rate at half an observation, so a zero pre-period yields a large
    but finite growth ratio instead of infinity."""
    return np.maximum(p, 0.5 / np.maximum(np.asarray(n, dtype=float), 1.0))


def find_changepoints(s: Series, min_docs: int = 3000) -> dict[str, np.ndarray]:
    """Best single level shift per expression, found by maximising a binomial z
    over every admissible split. Fully vectorised over expressions.

    Periods with too few documents are dropped rather than treated as zero-usage,
    otherwise a thin month reads as a collapse in every expression at once.
    """
    keep = s.usable_mask(min_docs)
    counts = np.asarray(s.counts, dtype=np.float64)[:, keep]
    docs = s.docs[keep]
    periods = [p for p, k in zip(s.periods, keep) if k]
    T = len(periods)
    if T < MIN_PRE_PERIODS + MIN_POST_PERIODS:
        raise ValueError(f"only {T} usable periods; need "
                         f"{MIN_PRE_PERIODS + MIN_POST_PERIODS}")

    cum_c = np.cumsum(counts, axis=1)
    cum_d = np.cumsum(docs)
    tot_c = cum_c[:, -1:]
    tot_d = cum_d[-1]

    lo, hi = MIN_PRE_PERIODS, T - MIN_POST_PERIODS
    splits = np.arange(lo, hi)  # split s: pre = [0, s), post = [s, T)

    n_pre = cum_d[splits - 1]
    n_post = tot_d - n_pre
    k_pre = cum_c[:, splits - 1]
    k_post = tot_c - k_pre

    p_pre = _regularized(k_pre / n_pre, n_pre)
    var = n_post * p_pre * (1.0 - p_pre)
    z = (k_post - n_post * p_pre) / np.sqrt(np.maximum(var, EPS))

    best = np.argmax(z, axis=1)
    rows = np.arange(counts.shape[0])
    cp = splits[best]

    p_pre_b = p_pre[rows, best]
    k_post_b = k_post[rows, best]
    n_post_b = n_post[best]
    p_post_b = _regularized(k_post_b / n_post_b, n_post_b)

    return {
        "cp": cp,
        "cp_period": np.array([periods[i] for i in cp], dtype=object),
        "binom_z": z[rows, best],
        "pre_rate": k_pre[rows, best] / n_pre[best],
        "post_rate": k_post_b / n_post_b,
        "log_growth": np.log(p_post_b / p_pre_b),
        "pre_docs": n_pre[best],
        "post_count": k_post_b,
        "periods": np.array(periods, dtype=object),
        "keep": keep,
        "rates": counts / docs,
    }


def _robust_z(rates: np.ndarray, cp: np.ndarray, window: int = 12) -> np.ndarray:
    """z of the value at the changepoint against a trailing median/MAD baseline."""
    n, T = rates.shape
    out = np.zeros(n)
    for i in range(n):
        c = int(cp[i])
        lo = max(0, c - window)
        hist = rates[i, lo:c]
        if hist.size < 3:
            continue
        med = np.median(hist)
        mad = np.median(np.abs(hist - med)) * 1.4826
        # floor the scale at the sampling noise of a rate this small
        floor = np.sqrt(max(med, EPS) / max(1.0, hist.size)) * 1e-2
        scale = max(mad, floor, EPS)
        post = rates[i, c : min(T, c + 6)]
        out[i] = (np.median(post) - med) / scale if post.size else 0.0
    return out


def _shape_stats(rates: np.ndarray, cp: np.ndarray, pre: np.ndarray,
                 post: np.ndarray) -> dict[str, np.ndarray]:
    """Persistence, rise speed, peak and subsequent decline."""
    n, T = rates.shape
    persistence = np.zeros(n)
    rise = np.full(n, np.nan)
    peak_i = np.zeros(n, dtype=int)
    peak_v = np.zeros(n)
    decline = np.zeros(n)

    for i in range(n):
        c = int(cp[i])
        r = rates[i]
        lo, hi = pre[i], post[i]
        span = hi - lo
        tail = r[c:]
        if tail.size:
            persistence[i] = float(np.mean(tail >= 0.5 * hi))
        if span > 0:
            hi_thr, lo_thr = lo + 0.8 * span, lo + 0.2 * span
            up = np.nonzero(r[c:] >= hi_thr)[0]
            if up.size:
                first_up = c + up[0]
                below = np.nonzero(r[: first_up + 1] <= lo_thr)[0]
                last_low = below[-1] if below.size else 0
                rise[i] = first_up - last_low
        pk = int(np.argmax(r))
        peak_i[i] = pk
        peak_v[i] = r[pk]
        after = r[pk + 1 :]
        if after.size >= 3 and peak_v[i] > 0:
            decline[i] = 1.0 - float(np.median(after[-3:]) / peak_v[i])
    return {
        "persistence": persistence,
        "rise_periods": rise,
        "peak_idx": peak_i,
        "peak_rate": peak_v,
        "decline_from_peak": decline,
    }


def core_score() -> pl.Expr:
    """Series-only emergence strength, before breadth is known.

    Deliberately a bounded sum of saturating terms rather than a product: any one
    component being extreme should not by itself carry an expression to the top of
    the ranking, since each component has its own failure mode.
    """
    growth = (pl.col("log_growth") / np.log(10)).clip(0, 3) / 3          # up to 1000x
    sig = (pl.col("binom_z") / 40.0).clip(0, 1)
    robust = (pl.col("robust_z") / 10.0).clip(0, 1)
    persist = pl.col("persistence").clip(0, 1)
    # NaN is not null in polars: an expression whose level never spans a
    # measurable rise has no rise time, and must be sent through fill_nan first
    # or the whole score silently becomes NaN
    speed = (1.0 - (pl.col("rise_periods").fill_nan(None).fill_null(12) / 12.0)).clip(0, 1)
    rarity = (-pl.col("pre_rate").log10().clip(-7, -3) - 3) / 4          # rarer before = better
    raw = (
        0.28 * growth
        + 0.22 * sig
        + 0.12 * robust
        + 0.18 * persist
        + 0.10 * speed
        + 0.10 * rarity
    )
    # The best split is a maximum even for an expression that only ever declined,
    # so the score has to be gated on the shift actually being upward. Declines are
    # kept in the table -- they are useful for dating -- but they are not
    # emergences and must not rank as such.
    is_rise = (pl.col("log_growth") > 0) & (pl.col("binom_z") > 0)
    return pl.when(is_rise).then(raw).otherwise(0.0).fill_nan(0.0).alias("core_score")


def analyze(s: Series, min_docs: int = 3000, log=print) -> pl.DataFrame:
    cps = find_changepoints(s, min_docs)
    rates = cps["rates"]
    log(f"emergence: {rates.shape[0]:,} expressions over {rates.shape[1]} usable periods")
    rz = _robust_z(rates, cps["cp"])
    shape = _shape_stats(rates, cps["cp"], cps["pre_rate"], cps["post_rate"])
    periods = cps["periods"]

    from .ngrams import family_key, n_words

    df = pl.DataFrame(
        {
            "term": s.terms,
            "family": [family_key(t) for t in s.terms],
            "n_words": [n_words(t) for t in s.terms],
            "cp_period": [str(x) for x in cps["cp_period"]],
            "pre_rate": cps["pre_rate"],
            "post_rate": cps["post_rate"],
            "log_growth": cps["log_growth"],
            "binom_z": cps["binom_z"],
            "robust_z": rz,
            "persistence": shape["persistence"],
            "rise_periods": shape["rise_periods"],
            "peak_period": [str(periods[i]) for i in shape["peak_idx"]],
            "peak_rate": shape["peak_rate"],
            "decline_from_peak": shape["decline_from_peak"],
            "post_count": cps["post_count"],
        }
    )
    return df.with_columns(core_score()).sort("core_score", descending=True)


def growth_matrix(s: Series, min_docs: int = 3000, eps: float | None = None
                  ) -> tuple[np.ndarray, list[str]]:
    """Per-period growth series used for clustering: g = d log(rate + eps).

    `eps` defaults to the smallest rate the corpus can resolve, which keeps the
    log from amplifying the difference between "zero" and "one document".
    """
    keep = s.usable_mask(min_docs)
    counts = np.asarray(s.counts, dtype=float)[:, keep]
    docs = s.docs[keep]
    rates = counts / docs
    if eps is None:
        eps = 1.0 / float(np.median(docs))
    lg = np.log(rates + eps)
    g = np.diff(lg, axis=1)
    periods = [p for p, k in zip(s.periods, keep) if k]
    return g, periods[1:]

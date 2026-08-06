"""Relate expression clusters to the release timeline.

Release dates are annotations, so this module is built to be able to say "no
alignment". The observed alignment statistic is always reported next to a null
distribution built by shifting the release calendar at random: if real releases
explain cluster timing no better than fake ones, the number says so.

Adoption lags release, and by an unknown amount, so alignment is scored over a
window rather than at a point, and the best-fitting lag is reported as a finding
rather than fixed in advance.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .clustering import Cluster
from .releases import RELEASES, Release
from .util import month_index, month_label

LAG_GRID = np.arange(0.0, 7.0, 0.5)  # months between release and visible adoption


def _release_times(weighted: bool) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([month_index(r.date[:7]) for r in RELEASES], dtype=float)
    w = np.array([r.weight for r in RELEASES]) if weighted else np.ones(len(RELEASES))
    return t, w


def distance_to_nearest(cp_months: np.ndarray, lag: float, weighted: bool = True,
                        release_t: np.ndarray | None = None) -> np.ndarray:
    """Months from each changepoint to the nearest lag-shifted release.

    Plain months, not weight-discounted: dividing by a release weight produced a
    quantity in no real units, which made the placebo comparison impossible to
    read. Weights belong on the clusters when averaging, not inside the distance.
    """
    t = _release_times(weighted)[0] if release_t is None else release_t
    d = np.abs(cm := cp_months[:, None] - (t + lag)[None, :])
    del cm
    return d.min(axis=1)


def align_clusters(clusters: list[Cluster], weighted: bool = True) -> pl.DataFrame:
    if not clusters:
        return pl.DataFrame(schema={"cid": pl.Int64})
    cp = np.array([month_index(c.cp_period) for c in clusters], dtype=float)
    strength = np.array([len(c.terms) * max(c.coherence, 0.0) for c in clusters])

    rows = []
    for c, m in zip(clusters, cp):
        best_r, best_d, best_lag = None, 1e9, 0.0
        for lag in LAG_GRID:
            for r in RELEASES:
                d = m - (month_index(r.date[:7]) + lag)
                if abs(d) < abs(best_d):
                    best_r, best_d, best_lag = r, d, lag
        assert best_r is not None
        rows.append(
            {
                "cid": c.cid,
                "cluster_cp": c.cp_period,
                "size": len(c.terms),
                "n_families": c.n_families,
                "coherence": round(c.coherence, 3),
                "nearest_release": best_r.name,
                "release_date": best_r.date,
                "generation": best_r.generation,
                "months_after_release": round(month_index(c.cp_period)
                                              - month_index(best_r.date[:7]), 1),
                "release_weight": best_r.weight,
            }
        )
    return pl.DataFrame(rows).sort("cluster_cp")


def best_lag(clusters: list[Cluster], weighted: bool = True) -> tuple[float, float]:
    """The adoption lag that best explains cluster timing, and its mean distance."""
    if not clusters:
        return 0.0, float("nan")
    cp = np.array([month_index(c.cp_period) for c in clusters], dtype=float)
    w = np.array([len(c.terms) * max(c.coherence, 1e-3) for c in clusters])
    best = (0.0, np.inf)
    for lag in LAG_GRID:
        d = distance_to_nearest(cp, lag, weighted)
        stat = float((d * w).sum() / w.sum())
        if stat < best[1]:
            best = (float(lag), stat)
    return best


def placebo_test(clusters: list[Cluster], n_draws: int = 2000, seed: int = 0,
                 weighted: bool = True, window: tuple[str, str] | None = None,
                 log=print) -> dict:
    """Compare real releases against two nulls, and report the test's own power.

    Null A **circularly** shifts the release calendar inside the observed window.
    A plain shift would push the whole calendar out of the data range and lose by
    construction, making any real calendar look significant; wrapping keeps the
    same number of releases in the window, so only their *positions* are random.

    Null B keeps the real releases and scatters the changepoints instead, which
    asks the complementary question: are the observed dates special, or would any
    dates have been close to something?

    The reported power check matters more than either p-value. With twenty
    releases in a sixty-month window the mean gap is three months, so *every*
    date is within about 1.5 months of some release. When that is the case the
    test cannot discriminate and says so rather than returning a small p.
    """
    if len(clusters) < 2:
        return {"n_clusters": len(clusters), "verdict": "too few clusters to test"}

    cp = np.array([month_index(c.cp_period) for c in clusters], dtype=float)
    w = np.array([len(c.terms) * max(c.coherence, 1e-3) for c in clusters])
    lo, hi = ((month_index(window[0]), month_index(window[1])) if window
              else (float(cp.min()) - 6, float(cp.max()) + 6))
    span = max(hi - lo, 12.0)

    t_all, _ = _release_times(weighted)
    inside = (t_all >= lo) & (t_all <= hi)
    t = t_all[inside]
    if t.size == 0:
        return {"n_clusters": len(clusters),
                "verdict": "no releases inside the observed window"}

    def stat(cps: np.ndarray, rel: np.ndarray, lag: float) -> float:
        d = np.abs(cps[:, None] - (rel + lag)[None, :]).min(axis=1)
        return float((d * w).sum() / w.sum())

    lag = min(LAG_GRID, key=lambda L: stat(cp, t, L))
    observed = stat(cp, t, lag)

    rng = np.random.default_rng(seed)
    null_a = np.empty(n_draws)
    null_b = np.empty(n_draws)
    for i in range(n_draws):
        shifted = lo + ((t - lo + rng.uniform(0, span)) % span)
        null_a[i] = stat(cp, shifted, lag)
        null_b[i] = stat(rng.uniform(lo, hi, size=cp.size), t, lag)

    p_a = float((null_a <= observed).mean())
    p_b = float((null_b <= observed).mean())
    mean_gap = span / t.size
    underpowered = observed >= mean_gap / 3

    out = {
        "n_clusters": len(clusters),
        "n_releases_in_window": int(t.size),
        "window": [month_label(lo), month_label(hi)],
        "best_lag_months": float(lag),
        "observed_mean_distance": round(observed, 3),
        "null_a_shifted_calendar": {
            "mean": round(float(null_a.mean()), 3),
            "p05": round(float(np.quantile(null_a, 0.05)), 3),
            "p_value": p_a,
        },
        "null_b_shuffled_changepoints": {
            "mean": round(float(null_b.mean()), 3),
            "p05": round(float(np.quantile(null_b, 0.05)), 3),
            "p_value": p_b,
        },
        "mean_release_gap_months": round(mean_gap, 2),
        "underpowered": bool(underpowered),
    }
    if underpowered:
        out["verdict"] = (
            f"inconclusive: {t.size} releases in a {span:.0f}-month window leave a "
            f"{mean_gap:.1f}-month mean gap, so any date lands close to some release. "
            "Alignment cannot be distinguished from coincidence at this release density."
        )
    elif p_a < 0.05 and p_b < 0.05:
        out["verdict"] = "clusters align with releases better than chance under both nulls"
    elif p_a < 0.05 or p_b < 0.05:
        out["verdict"] = "alignment survives one null but not the other; treat as weak"
    else:
        out["verdict"] = "no better than chance"
    log(f"placebo: observed {observed:.2f} months | null A {null_a.mean():.2f} "
        f"(p={p_a:.3f}) | null B {null_b.mean():.2f} (p={p_b:.3f}) | "
        f"{'UNDERPOWERED' if underpowered else 'powered'}")
    return out


def generation_table(align: pl.DataFrame) -> pl.DataFrame:
    """Clusters grouped by release generation, since releases weeks apart are not
    separable by this method and should not be reported as if they were."""
    if align.height == 0:
        return align
    return (
        align.group_by("generation")
        .agg(
            pl.len().alias("n_clusters"),
            pl.col("size").sum().alias("n_expressions"),
            pl.col("cluster_cp").min().alias("first_cp"),
            pl.col("cluster_cp").max().alias("last_cp"),
            pl.col("coherence").mean().round(3).alias("mean_coherence"),
        )
        .sort("first_cp")
    )

"""Find expressions whose frequency changes happen together.

The project's central hypothesis is that a model generation leaves a *bundle* of
expressions rather than one catchphrase. So the unit of evidence is a group of
expressions that move at the same time, and the test of an expression is whether
it belongs to such a group.

Similarity between two expressions combines three views, because any one of them
is easy to fool:

* correlation of growth series (shape of the whole trajectory)
* agreement of changepoint dates (when the shift happened)
* small-lag cross-correlation (one expression may follow another by a month)

Clusters then get a latent curve L_k(t) and per-expression loadings a_e, fitted
as a rank-one factor per cluster: g_e(t) ~ a_e * L_k(t). The loading says how
strongly an expression responds to the shared shock, which is what the text
scorer needs -- an expression with a weak loading contributes little even if its
own series looks dramatic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .util import month_index, month_label

MAX_LAG = 2  # periods; adoption of related phrasing need not be simultaneous


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-12)


def lagged_correlation(g: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Max over small lags of the cross-correlation between growth series.

    Returns a dense (n, n) matrix, so callers must keep n in the low thousands.
    """
    z = _zscore_rows(g)
    T = z.shape[1]
    best = np.full((z.shape[0], z.shape[0]), -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = z[:, lag:], z[:, : T - lag] if lag else z
        else:
            a, b = z[:, : T + lag], z[:, -lag:]
        c = (a @ b.T) / a.shape[1]
        best = np.maximum(best, np.maximum(c, c.T))
    np.fill_diagonal(best, 1.0)
    return np.clip(best, -1.0, 1.0)


def changepoint_affinity(cp_periods: list[str], tol_months: float = 3.0) -> np.ndarray:
    """1 when two expressions share a changepoint date, decaying with distance."""
    t = np.array([month_index(p) for p in cp_periods], dtype=float)
    d = np.abs(t[:, None] - t[None, :])
    return np.exp(-((d / tol_months) ** 2))


def similarity(g: np.ndarray, cp_periods: list[str], w_corr: float = 0.6,
               w_cp: float = 0.4, max_lag: int = MAX_LAG) -> np.ndarray:
    corr = lagged_correlation(g, max_lag)
    cp = changepoint_affinity(cp_periods)
    return w_corr * np.clip(corr, 0, 1) + w_cp * cp


@dataclass
class Cluster:
    cid: int
    terms: list[str]
    loadings: np.ndarray
    latent: np.ndarray  # L_k(t) over growth periods
    periods: list[str]
    coherence: float  # variance of member growth explained by the rank-one factor
    cp_period: str  # shared emergence estimate
    cp_spread: float  # months of disagreement among members
    n_families: int
    members: pl.DataFrame = field(default_factory=pl.DataFrame)

    def summary(self) -> dict:
        return {
            "cid": self.cid,
            "size": len(self.terms),
            "n_families": self.n_families,
            "cp_period": self.cp_period,
            "cp_spread": round(self.cp_spread, 2),
            "coherence": round(self.coherence, 3),
            "peak_growth_period": self.periods[int(np.argmax(self.latent))]
            if len(self.latent) else "",
            "terms": self.terms[:25],
        }


def _rank_one(gz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Leading factor of a member growth block, sign-fixed so the shock is a rise.

    Uses the SVD rather than a plain mean so that members contribute in
    proportion to how well they track the common signal.
    """
    u, s, vt = np.linalg.svd(gz, full_matrices=False)
    latent = vt[0]
    load = u[:, 0] * s[0]
    if load.sum() < 0:  # a shared *rise* is the interpretable orientation
        latent, load = -latent, -load
    total = float((gz**2).sum())
    coherence = float(s[0] ** 2 / total) if total > 0 else 0.0
    return latent, load, coherence


def build(
    g: np.ndarray,
    terms: list[str],
    families: list[str],
    cp_periods: list[str],
    growth_periods: list[str],
    threshold: float = 0.55,
    min_size: int = 3,
    min_families: int = 3,
    log=print,
) -> list[Cluster]:
    """Average-linkage clustering on the combined similarity, then a rank-one fit.

    `min_families` is the load-bearing constraint: three inflections of one phrase
    are one habit, not three independent witnesses. Clusters are required to span
    distinct expression families before they count as corroborating evidence.
    """
    n = len(terms)
    if n < min_size:
        return []
    sim = similarity(g, cp_periods)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    log(f"clustering: {n} expressions -> {labels.max()} raw groups")

    gz = _zscore_rows(g)
    out: list[Cluster] = []
    for lab in np.unique(labels):
        idx = np.nonzero(labels == lab)[0]
        if idx.size < min_size:
            continue
        fams = {families[i] for i in idx}
        if len(fams) < min_families:
            continue
        latent, load, coh = _rank_one(gz[idx])
        cps = np.array([month_index(cp_periods[i]) for i in idx], dtype=float)
        w = np.abs(load)
        centre = float((cps * w).sum() / w.sum()) if w.sum() > 0 else float(cps.mean())
        out.append(
            Cluster(
                cid=int(lab),
                terms=[terms[i] for i in idx],
                loadings=load,
                latent=latent,
                periods=growth_periods,
                coherence=coh,
                cp_period=month_label(centre),
                cp_spread=float(np.std(cps)),
                n_families=len(fams),
            )
        )
    out.sort(key=lambda c: (-len(c.terms) * c.coherence))
    for i, c in enumerate(out):
        c.cid = i
    log(f"  {len(out)} clusters with >={min_size} members and >={min_families} families")
    return out


def to_frame(clusters: list[Cluster]) -> pl.DataFrame:
    rows = []
    for c in clusters:
        for term, load in zip(c.terms, c.loadings):
            rows.append(
                {
                    "term": term,
                    "cid": c.cid,
                    "loading": float(load),
                    "cluster_cp": c.cp_period,
                    "cluster_coherence": c.coherence,
                    "cluster_size": len(c.terms),
                    "cluster_families": c.n_families,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={"term": pl.Utf8, "cid": pl.Int64, "loading": pl.Float64,
                    "cluster_cp": pl.Utf8, "cluster_coherence": pl.Float64,
                    "cluster_size": pl.Int64, "cluster_families": pl.Int64}
        )
    return pl.DataFrame(rows)


def latent_frame(clusters: list[Cluster]) -> pl.DataFrame:
    rows = []
    for c in clusters:
        cum = np.cumsum(c.latent)  # growth -> level, for plotting the shared curve
        for p, gval, lvl in zip(c.periods, c.latent, cum):
            rows.append({"cid": c.cid, "period": p, "latent_growth": float(gval),
                         "latent_level": float(lvl)})
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"cid": pl.Int64, "period": pl.Utf8, "latent_growth": pl.Float64,
                "latent_level": pl.Float64}
    )


def stability(g: np.ndarray, terms: list[str], families: list[str],
              cp_periods: list[str], growth_periods: list[str],
              thresholds=(0.5, 0.55, 0.6), log=print) -> pl.DataFrame:
    """How often each pair of expressions lands in the same cluster across
    settings. Pairs that only co-cluster at one threshold are not evidence."""
    from collections import Counter

    pair_hits: Counter[tuple[str, str]] = Counter()
    runs = 0
    for th in thresholds:
        cs = build(g, terms, families, cp_periods, growth_periods,
                   threshold=th, log=lambda *_: None)
        runs += 1
        for c in cs:
            ts = sorted(c.terms)
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    pair_hits[(ts[i], ts[j])] += 1
    rows = [{"term_a": a, "term_b": b, "co_cluster_frac": n / runs}
            for (a, b), n in pair_hits.items()]
    log(f"stability: {len(rows)} co-clustered pairs over {runs} settings")
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"term_a": pl.Utf8, "term_b": pl.Utf8, "co_cluster_frac": pl.Float64})

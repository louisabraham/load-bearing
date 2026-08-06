"""Estimate when an undated text was written from its expressions.

P(t | d) ∝ P(t) · Π_e P(e | t)

Two details matter more than the formula.

First, each present expression contributes the *ratio* of its rate in a period to
its rate averaged over periods, not its raw probability. Absence looks like
evidence -- a document lacking every 2025 expression is weak evidence against
2025 -- but over a fixed feature set it is swamped by document length: a
two-sentence comment lacks almost every expression whenever it was written. Using
absence dated recent documents years early, with error growing the more recent the
document was. The ratio form is invariant to how many features are absent. Absence
remains available via ``use_absence=True`` and is only sound on documents of
comparable length.

Second, features from one cluster are not independent -- that is the point of
clusters -- so treating twelve members of the same cluster as twelve observations
produces absurdly sharp posteriors. The cluster-level estimator collapses each
cluster to one observation and is the one to trust when the document is short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .ngrams import features
from .scoring import Lexicon
from .series import Series
from .util import month_index, month_label

SMOOTH = 0.5  # pseudo-counts, in documents
SHRINK_DOCS = 250.0
LOG_RATIO_CAP = 1.0  # max evidence in nats from one expression in one period  # strength of shrinkage toward an expression's global rate


def informative_terms(series: Series, k: int = 300, min_docs: int = 1500,
                      candidates: list[str] | None = None,
                      min_total: int = 25) -> list[str]:
    """Pick the expressions that carry the most information about *when*.

    Emergence rank is the wrong criterion for dating: it rewards the size of a
    single jump, while dating needs presence to vary informatively across the
    whole timeline. This selects by mutual information between "expression is
    present" and "which period", computed on shrunk rates so that a thin month
    cannot buy its way in with sampling noise.
    """
    keep = series.usable_mask(min_docs)
    docs = series.docs[keep]
    idx = (np.arange(len(series.terms)) if candidates is None
           else np.array([series.index[t] for t in candidates if t in series.index]))
    if idx.size == 0:
        return []
    counts = series.counts[idx, :][:, keep].astype(float)
    enough = counts.sum(axis=1) >= min_total
    idx, counts = idx[enough], counts[enough]
    if idx.size == 0:
        return []

    global_rate = (counts.sum(axis=1) + SMOOTH) / (docs.sum() + 2 * SMOOTH)
    p = np.clip((counts + SHRINK_DOCS * global_rate[:, None]) / (docs + SHRINK_DOCS),
                1e-7, 1 - 1e-7)
    pbar = p.mean(axis=1, keepdims=True)
    mi = (p * np.log(p / pbar) + (1 - p) * np.log((1 - p) / (1 - pbar))).mean(axis=1)
    order = np.argsort(-mi)[:k]
    return [series.terms[int(idx[i])] for i in order]


@dataclass
class DateEstimate:
    best_period: str
    interval: tuple[str, str]  # highest-posterior-density range
    posterior: np.ndarray
    periods: list[str]
    alternatives: list[tuple[str, float]]
    n_features: int
    method: str

    def explain(self, top: int = 5) -> str:
        lines = [
            f"most likely period: {self.best_period}",
            f"credible range: {self.interval[0]} .. {self.interval[1]}",
            f"features used: {self.n_features} ({self.method})",
            "alternatives:",
        ]
        for p, prob in self.alternatives[:top]:
            lines.append(f"  {p}  {prob:.3f}")
        return "\n".join(lines)


class Dater:
    """Naive-Bayes-over-periods with presence/absence likelihoods."""

    def __init__(self, series: Series, terms: list[str], min_docs: int = 1500,
                 cluster_of: dict[str, int] | None = None):
        keep = series.usable_mask(min_docs)
        self.periods = [p for p, k in zip(series.periods, keep) if k]
        idx = [series.index[t] for t in terms if t in series.index]
        self.terms = [t for t in terms if t in series.index]
        counts = series.counts[np.array(idx), :][:, keep].astype(float)
        docs = series.docs[keep]
        # Shrink each period's rate toward the expression's own global rate with a
        # fixed strength in document-equivalents. Add-half smoothing would put a
        # floor of 0.5/docs on every cell, which is an order of magnitude higher in
        # a thin month than a fat one -- so absence evidence would penalise thin
        # months and drag every estimate toward whichever years were sampled most
        # heavily, which is a property of the sampling plan, not of the language.
        global_rate = (counts.sum(axis=1) + SMOOTH) / (docs.sum() + 2 * SMOOTH)
        k = SHRINK_DOCS
        self.p = (counts + k * global_rate[:, None]) / (docs[None, :] + k)
        self.p = np.clip(self.p, 1e-7, 1 - 1e-7)
        self.log_p = np.log(self.p)
        self.log_q = np.log1p(-self.p)
        # Rate in this period relative to the expression's period-averaged rate,
        # clipped. A term seen twice in a thin month yields an enormous ratio that
        # spikes the posterior onto that month; capping the evidence any single
        # expression can contribute keeps a few noisy cells from deciding the date.
        self.log_ratio = np.clip(
            self.log_p - np.log(self.p.mean(axis=1, keepdims=True)),
            -LOG_RATIO_CAP, LOG_RATIO_CAP)
        self.cluster_of = cluster_of or {}
        # uniform prior over periods with data: a corpus-frequency prior would
        # just re-impose the sampling schedule on the answer
        self.log_prior = np.full(len(self.periods), -np.log(len(self.periods)))

    def _posterior(self, present_mask: np.ndarray, use_absence: bool = False
                   ) -> np.ndarray:
        """Posterior over periods.

        By default only *present* expressions contribute, and each contributes the
        log ratio of its rate in that period to its rate averaged over periods.

        Including absence is tempting -- it is genuinely informative -- but with a
        fixed feature set it is dominated by document length: a two-sentence
        comment lacks 299 of 300 expressions whenever it was written, so the
        absence product simply favours whichever periods have the lowest overall
        rates. That put every recent document decades early. The likelihood-ratio
        form is invariant to how many features are absent, which is what makes it
        usable across documents of wildly different length.
        """
        if use_absence:
            ll = (self.log_p[present_mask].sum(axis=0)
                  + self.log_q[~present_mask].sum(axis=0))
        else:
            ll = self.log_ratio[present_mask].sum(axis=0)
        ll = ll + self.log_prior
        ll -= ll.max()
        post = np.exp(ll)
        return post / post.sum()

    def estimate(self, text: str, hdi: float = 0.8,
                 use_absence: bool = False) -> DateEstimate:
        present = features(text)
        mask = np.array([t in present for t in self.terms])
        if not mask.any():
            return self._package(np.exp(self.log_prior), 0, "expression-level (no features)")
        post = self._posterior(mask, use_absence)
        return self._package(post, int(mask.sum()), "expression-level")

    def estimate_by_cluster(self, text: str, hdi: float = 0.8) -> DateEstimate:
        """Collapse each cluster to a presence indicator before multiplying.

        Members of a cluster rose together, so they carry nearly the same
        information; counting them once keeps the posterior honest.
        """
        present = features(text)
        by_cluster: dict[int, list[int]] = {}
        for i, t in enumerate(self.terms):
            by_cluster.setdefault(self.cluster_of.get(t, -1 - i), []).append(i)

        ll = self.log_prior.copy()
        used = 0
        for cid, members in by_cluster.items():
            m = np.array(members)
            if not any(self.terms[i] in present for i in m):
                continue  # absence excluded for the same length-bias reason
            # cluster-level rate: the mean member rate, so one cluster contributes
            # one observation regardless of how many members it has
            p = np.clip(self.p[m].mean(axis=0), 1e-7, 1 - 1e-7)
            ll += np.log(p) - np.log(p.mean())
            used += 1
        if used == 0:
            return self._package(np.exp(self.log_prior), 0, "cluster-level (no features)")
        ll -= ll.max()
        post = np.exp(ll)
        post /= post.sum()
        return self._package(post, used, "cluster-level")

    def _package(self, post: np.ndarray, n_features: int, method: str) -> DateEstimate:
        order = np.argsort(-post)
        if n_features == 0:
            # A uniform posterior has no maximum; argmax would silently return the
            # first period, which reads as a confident answer of "the earliest
            # month in the corpus". Say the middle and mark it as evidence-free.
            mid = len(self.periods) // 2
            return DateEstimate(
                best_period=self.periods[mid],
                interval=(self.periods[0], self.periods[-1]),
                posterior=post, periods=self.periods,
                alternatives=[(self.periods[mid], float(post[mid]))],
                n_features=0, method=method + " [no evidence]",
            )
        best = self.periods[int(order[0])]
        # highest-density interval by accumulating the most probable periods
        cum, chosen = 0.0, []
        for i in order:
            chosen.append(int(i))
            cum += post[i]
            if cum >= 0.8:
                break
        lo, hi = min(chosen), max(chosen)
        return DateEstimate(
            best_period=best,
            interval=(self.periods[lo], self.periods[hi]),
            posterior=post,
            periods=self.periods,
            alternatives=[(self.periods[int(i)], float(post[i])) for i in order[:8]],
            n_features=n_features,
            method=method,
        )


def evaluate(dater: Dater, docs: pl.DataFrame, use_clusters: bool = False,
             sample: int = 2000, seed: int = 0, log=print) -> dict:
    """Hide known timestamps and measure recovery error in months.

    Sampling is stratified by period. Sampling documents proportionally would let
    the most heavily ingested months dominate the score, so the number would
    describe the sampling plan rather than the model -- and it would flatter any
    model that simply guesses those months.
    """
    if docs.height == 0:
        return {"n": 0}
    usable = set(dater.periods)
    d = docs.filter(pl.col("period").is_in(list(usable)))
    if d.height == 0:
        return {"n": 0}
    n_periods = d["period"].n_unique()
    per_period = max(5, sample // max(1, n_periods))
    d = (d.with_columns(_r=pl.int_range(pl.len()).shuffle(seed=seed).over("period"))
         .filter(pl.col("_r") < per_period).drop("_r"))

    errs, naive, by_era = [], [], {}
    n_no_evidence = 0
    mid = month_index(dater.periods[len(dater.periods) // 2])
    for text, period in zip(d["text"].to_list(), d["period"].to_list()):
        est = (dater.estimate_by_cluster(text) if use_clusters
               else dater.estimate(text))
        if est.n_features == 0:
            # scoring these as predictions would measure the tie-break, not the model
            n_no_evidence += 1
            continue
        err = abs(month_index(est.best_period) - month_index(period))
        errs.append(err)
        naive.append(abs(mid - month_index(period)))
        by_era.setdefault(period[:4], []).append(err)
    if not errs:
        return {"n": 0}
    out = {
        "n": len(errs),
        "n_no_evidence": n_no_evidence,
        "coverage": round(len(errs) / (len(errs) + n_no_evidence), 3),
        "mae_months": round(float(np.mean(errs)), 2),
        "median_ae_months": round(float(np.median(errs)), 2),
        "baseline_mae_months": round(float(np.mean(naive)), 2),
        "within_6_months": round(float(np.mean(np.array(errs) <= 6)), 3),
        "mae_by_year": {y: round(float(np.mean(v)), 1)
                        for y, v in sorted(by_era.items())},
        "method": "cluster-level" if use_clusters else "expression-level",
    }
    log(f"date prediction ({out['method']}): MAE {out['mae_months']} months "
        f"vs {out['baseline_mae_months']} for always guessing the midpoint")
    return out

"""Score a text for LLM-era expressions, with and without a date.

Two guards shape the arithmetic, both aimed at the same failure: one striking
phrase carrying a whole verdict.

* **Family collapse.** Only the strongest expression from a family counts.
  "load-bearing", "load bearing" and "load-bearing assumption" are one habit.
* **Cluster saturation.** Expressions from the same temporal cluster are pooled
  with a concave function, so three members of a 2025 cluster count for more than
  one but much less than three times one. Independent clusters add linearly,
  which is what makes a document with several unrelated era markers score higher
  than a document leaning on one phrase eight times.

The output is a continuous score, not a human-versus-model verdict. A high score
says the text uses expressions that spread during a period -- which humans who
read LLM output also do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .ngrams import family_key, features
from .series import Series
from .util import month_index

CLUSTER_SATURATION = 0.6  # exponent on within-cluster evidence: 1 = additive, 0.5 = sqrt


@dataclass
class Lexicon:
    """Expression weights and temporal profiles, ready for scoring."""

    weights: dict[str, float]
    cluster_of: dict[str, int]
    cluster_cp: dict[int, str]
    family_of: dict[str, str]
    cp_period: dict[str, str]
    rates: dict[str, np.ndarray] = field(default_factory=dict)  # term -> rate per period
    periods: list[str] = field(default_factory=list)

    @classmethod
    def from_atlas(cls, atlas: pl.DataFrame, series: Series | None = None,
                   min_weight: float = 0.02) -> "Lexicon":
        a = atlas.filter(pl.col("weight") >= min_weight)
        weights = dict(zip(a["term"], a["weight"]))
        cluster_of = {t: (int(c) if c is not None else -1)
                      for t, c in zip(a["term"], a["cid"])} if "cid" in a.columns else {}
        cluster_cp = {}
        if "cluster_cp" in a.columns:
            for c, p in zip(a["cid"], a["cluster_cp"]):
                if c is not None and p is not None:
                    cluster_cp[int(c)] = str(p)
        rates: dict[str, np.ndarray] = {}
        periods: list[str] = []
        if series is not None:
            periods = series.periods
            r = series.rates
            for t in weights:
                i = series.index.get(t)
                if i is not None:
                    rates[t] = r[i]
        return cls(
            weights=weights,
            cluster_of=cluster_of,
            cluster_cp=cluster_cp,
            family_of={t: family_key(t) for t in weights},
            cp_period=dict(zip(a["term"], a["cp_period"])) if "cp_period" in a.columns else {},
            rates=rates,
            periods=periods,
        )

    def save(self, path) -> None:
        pl.DataFrame(
            {
                "term": list(self.weights),
                "weight": [self.weights[t] for t in self.weights],
                "cid": [self.cluster_of.get(t, -1) for t in self.weights],
                "cluster_cp": [self.cluster_cp.get(self.cluster_of.get(t, -1), "")
                               for t in self.weights],
                "cp_period": [self.cp_period.get(t, "") for t in self.weights],
            }
        ).write_parquet(path)


def build_weights(atlas: pl.DataFrame) -> pl.DataFrame:
    """Turn atlas columns into a single expression weight.

    Cluster membership is a multiplier rather than an additive bonus: an
    expression that rose alone is not evidence of a shared shock however dramatic
    its own curve, and one that rose with twenty others is worth more than its own
    series suggests.
    """
    has_cluster = "cid" in atlas.columns
    cluster_factor = (
        (
            1.0
            + 0.8 * (pl.col("cluster_size").fill_null(1).log1p() / math.log(20))
            * pl.col("cluster_coherence").fill_null(0.0)
            + 0.4 * pl.col("loading").fill_null(0.0).abs().clip(0, 2) / 2
        )
        if has_cluster
        else pl.lit(1.0)
    )
    breadth_factor = (
        0.5 + 0.5 * (pl.col("repo_spread").fill_null(0) / 200).clip(0, 1)
    )
    return atlas.with_columns(
        weight=(pl.col("adj_score") * cluster_factor * breadth_factor).clip(0, 5)
    )


@dataclass
class DocScore:
    score: float
    n_expressions: int
    contributions: list[tuple[str, float]]
    clusters: list[tuple[int, str, int, float]]  # cid, cp, n_members, contribution
    families_used: int

    def explain(self, top: int = 12) -> str:
        lines = [f"LLM-era expression score: {self.score:.2f} "
                 f"({self.n_expressions} expressions, {self.families_used} families)"]
        for t, w in self.contributions[:top]:
            lines.append(f"  {w:6.3f}  {t}")
        if self.clusters:
            lines.append("  clusters:")
            for cid, cp, n, contrib in sorted(self.clusters, key=lambda x: -x[3]):
                lines.append(f"    cluster {cid} (emerged {cp}): {n} members, "
                             f"contributes {contrib:.2f}")
        return "\n".join(lines)


def score_text(text: str, lex: Lexicon) -> DocScore:
    """Undated score: sum of expression weights, collapsed by family and cluster."""
    present = features(text) & lex.weights.keys()

    # strongest member per family only
    by_family: dict[str, tuple[str, float]] = {}
    for t in present:
        w = lex.weights[t]
        fam = lex.family_of.get(t, t)
        if fam not in by_family or w > by_family[fam][1]:
            by_family[fam] = (t, w)

    per_cluster: dict[int, list[tuple[str, float]]] = {}
    for t, w in by_family.values():
        per_cluster.setdefault(lex.cluster_of.get(t, -1), []).append((t, w))

    total = 0.0
    clusters: list[tuple[int, str, int, float]] = []
    contributions: list[tuple[str, float]] = []
    for cid, members in per_cluster.items():
        members.sort(key=lambda x: -x[1])
        raw = sum(w for _, w in members)
        if cid < 0:
            # unclustered expressions have no shared shock to saturate against,
            # so they stay additive -- their weights are already the smaller ones
            contrib = raw
        else:
            contrib = math.pow(raw, CLUSTER_SATURATION) if raw > 0 else 0.0
            clusters.append((cid, lex.cluster_cp.get(cid, "?"), len(members), contrib))
        total += contrib
        contributions.extend(members)

    contributions.sort(key=lambda x: -x[1])
    return DocScore(
        score=total,
        n_expressions=len(present),
        contributions=contributions,
        clusters=clusters,
        families_used=len(by_family),
    )


def score_text_dated(text: str, date: str, lex: Lexicon,
                     reference: str = "pre_llm") -> DocScore:
    """Date-aware score: log-likelihood ratio of the observed expressions between
    the document's own period and a reference period.

    Positive means the text uses expressions that were more common then than in
    the reference window. Requires the rate series, so it is only available for
    expressions that survived to the matrix.
    """
    if not lex.rates:
        raise ValueError("dated scoring needs a Lexicon built with a Series")
    from .releases import ERAS

    idx = {p: i for i, p in enumerate(lex.periods)}
    t_i = idx.get(date[:7])
    if t_i is None:
        raise ValueError(f"period {date[:7]} not in the corpus")

    if reference in ERAS:
        lo, hi = ERAS[reference]
        ref_idx = [i for p, i in idx.items() if lo <= p <= hi]
    elif reference == "previous_generation":
        ref_idx = [i for i in range(max(0, t_i - 18), max(1, t_i - 6))]
    else:  # all history before the document
        ref_idx = list(range(0, t_i))
    if not ref_idx:
        raise ValueError(f"reference window '{reference}' is empty")

    present = features(text) & lex.weights.keys()
    contributions: list[tuple[str, float]] = []
    total = 0.0
    for t in present:
        r = lex.rates.get(t)
        if r is None:
            continue
        p_t = r[t_i]
        p_ref = float(np.nanmean(r[ref_idx]))
        if not np.isfinite(p_t) or not np.isfinite(p_ref):
            continue
        floor = 1e-6
        llr = math.log(max(p_t, floor) / max(p_ref, floor))
        contributions.append((t, llr))
        total += llr
    contributions.sort(key=lambda x: -abs(x[1]))
    return DocScore(
        score=total,
        n_expressions=len(present),
        contributions=contributions,
        clusters=[],
        families_used=len({lex.family_of.get(t, t) for t in present}),
    )


def score_frame(texts: list[str], lex: Lexicon) -> pl.DataFrame:
    rows = []
    for i, t in enumerate(texts):
        s = score_text(t, lex)
        rows.append({"i": i, "score": s.score, "n_expressions": s.n_expressions,
                     "n_clusters": len(s.clusters), "families": s.families_used})
    return pl.DataFrame(rows)

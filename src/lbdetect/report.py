"""Human-facing outputs: the expression atlas, cluster report and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from .clustering import Cluster
from .releases import RELEASES
from .series import Series
from .util import month_index

ATLAS = C.ARTIFACTS / "atlas.parquet"


def build_atlas(em: pl.DataFrame, clusters_frame: pl.DataFrame,
                breadth: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per expression with everything known about it."""
    df = em
    if breadth is not None and breadth.height:
        cols = [c for c in breadth.columns if c != "term" and c not in df.columns]
        df = df.join(breadth.select(["term", *cols]), on="term", how="left")
    if clusters_frame.height:
        df = df.join(clusters_frame, on="term", how="left")
    else:
        df = df.with_columns(
            cid=pl.lit(None, pl.Int64), loading=pl.lit(None, pl.Float64),
            cluster_cp=pl.lit(None, pl.Utf8), cluster_coherence=pl.lit(None, pl.Float64),
            cluster_size=pl.lit(None, pl.Int64), cluster_families=pl.lit(None, pl.Int64),
        )
    if "adj_score" not in df.columns:
        df = df.with_columns(adj_score=pl.col("core_score"),
                             confounder_penalty=pl.lit(1.0))
    from .scoring import build_weights

    return build_weights(df).sort("weight", descending=True)


def atlas_markdown(atlas: pl.DataFrame, top: int = 60) -> str:
    """The table a human reads first."""
    cols = ["term", "cp_period", "pre_rate", "post_rate", "log_growth", "binom_z",
            "persistence", "repo_spread", "top_repo_share", "cid", "weight"]
    have = [c for c in cols if c in atlas.columns]
    d = atlas.head(top).select(have)
    lines = ["| " + " | ".join(have) + " |", "|" + "---|" * len(have)]
    for row in d.iter_rows():
        cells = []
        for c, v in zip(have, row):
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.2e}" if abs(v) < 1e-2 and v != 0 else f"{v:.3g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cluster_markdown(clusters: list[Cluster], align: pl.DataFrame,
                     top_terms: int = 14) -> str:
    if not clusters:
        return "_No clusters met the size and family thresholds._"
    amap = {r["cid"]: r for r in align.to_dicts()} if align.height else {}
    out = []
    for c in clusters:
        a = amap.get(c.cid, {})
        out.append(
            f"### Cluster {c.cid} — emerged {c.cp_period} "
            f"({len(c.terms)} expressions, {c.n_families} families, "
            f"coherence {c.coherence:.2f})\n"
        )
        if a:
            out.append(
                f"Nearest release: **{a.get('nearest_release')}** "
                f"({a.get('release_date')}, generation `{a.get('generation')}`), "
                f"{a.get('months_after_release')} months before/after the cluster shift.\n"
            )
        order = np.argsort(-np.abs(c.loadings))
        terms = [f"`{c.terms[i]}` ({c.loadings[i]:+.2f})" for i in order[:top_terms]]
        out.append("Members by loading: " + ", ".join(terms) + "\n")
    return "\n".join(out)


# ------------------------------------------------------------------------- plots

def plot_expression(s: Series, terms: list[str], path: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = [month_index(p) for p in s.periods]
    for t in terms:
        if t not in s.index:
            continue
        ax.plot(x, s.rate_of(t) * 1e4, marker=".", lw=1.4, label=t)
    _release_lines(ax)
    ax.set_ylabel("documents per 10,000")
    ax.set_title(title or "expression frequency")
    _month_axis(ax, s.periods)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cluster(c: Cluster, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = [month_index(p) for p in c.periods]
    ax.plot(x, np.cumsum(c.latent), color="#c0392b", lw=2.2,
            label=f"latent level L_{c.cid}(t)")
    ax.axvline(month_index(c.cp_period), color="#c0392b", ls="--", lw=1,
               label=f"shared changepoint {c.cp_period}")
    _release_lines(ax)
    ax.set_title(f"Cluster {c.cid}: {len(c.terms)} expressions, "
                 f"coherence {c.coherence:.2f}")
    ax.set_ylabel("cumulative latent growth")
    _month_axis(ax, c.periods)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _release_lines(ax) -> None:
    for r in RELEASES:
        if r.weight < 0.4:
            continue
        ax.axvline(month_index(r.date[:7]), color="#7f8c8d", lw=0.7, alpha=0.5)
        ax.text(month_index(r.date[:7]), ax.get_ylim()[1], r.name.split(" (")[0][:18],
                rotation=90, fontsize=6, va="top", ha="right", color="#7f8c8d")


def _month_axis(ax, periods: list[str]) -> None:
    ticks = [month_index(p) for p in periods if p.endswith(("-01", "-07"))]
    labels = [p for p in periods if p.endswith(("-01", "-07"))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)


def coverage_plot(meta: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.2))
    x = [month_index(p) for p in meta["period"]]
    ax.bar(x, meta["docs"], width=0.8, color="#2c7fb8")
    ax.set_ylabel("eligible documents")
    ax.set_title("corpus coverage per month (denominator of every rate)")
    _month_axis(ax, meta["period"].to_list())
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

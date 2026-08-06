"""Figures for the summary report.

Print-targeted, light mode only. Colours are the validated categorical slots in
fixed order; every multi-series chart carries a legend *and* direct end-labels,
which is also what the contrast relief rule requires for the aqua and yellow
slots. No dual axes anywhere: where two quantities share a panel they share units.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from lbdetect import config as C
from lbdetect.series import Series
from lbdetect.util import month_index

OUT = C.ROOT / "out" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8981"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRID = "#e3e2dc"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
})

REGIME = "2025-11"  # archive stops carrying PR descriptions


def _months(ax, periods, step=6):
    ticks = [i for i, p in enumerate(periods) if p.endswith(("-01", "-07"))]
    ax.set_xticks(ticks)
    ax.set_xticklabels([periods[i] for i in ticks], rotation=55, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)


def _regime_band(ax, periods):
    """Mark the window where the archive's artifact mix changes."""
    if REGIME not in periods:
        return
    i = periods.index(REGIME)
    ax.axvspan(i - 0.5, len(periods) - 0.5, color="#f0efe8", zorder=0)
    ax.axvline(i - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(i - 1.5, ax.get_ylim()[1] * 0.85, "archive coverage\nchanges",
            fontsize=6.5, color=INK2, va="top", ha="right")


# ---------------------------------------------------------------- 1. coverage

def fig_coverage() -> None:
    m = pl.read_parquet(C.SERIES / "periods.parquet").sort("period")
    periods = m["period"].to_list()
    docs = m["docs"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.bar(range(len(periods)), docs / 1000, width=0.78, color=SERIES[0],
           linewidth=0)
    ax.set_ylabel("eligible documents (thousands)")
    ax.set_title("Corpus after cleaning: 2.69M eligible documents over 103 months")
    _months(ax, periods)
    _regime_band(ax, periods)
    fig.tight_layout()
    fig.savefig(OUT / "coverage.pdf")
    plt.close(fig)


# ------------------------------------------------------- 2. machine-authored

def fig_machine_share() -> None:
    from lbdetect import ingest
    from lbdetect.textclean import bot_expr

    d = (ingest.corpus()
         .group_by("period")
         .agg(pl.len().alias("n"), bot_expr("author").sum().alias("bot"))
         .sort("period").collect())
    periods = d["period"].to_list()
    share = (d["bot"] / d["n"] * 100).to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.plot(range(len(periods)), share, color=SERIES[1], lw=1.8)
    ax.set_ylabel("% of documents")
    ax.set_ylim(0, max(60, share.max() * 1.15))
    ax.set_title("Machine-authored share of GitHub prose (excluded from the corpus)")
    _months(ax, periods)
    for i, lab in ((0, None), (len(periods) - 1, f"{share[-1]:.0f}%")):
        if lab:
            ax.annotate(lab, (i, share[i]), textcoords="offset points",
                        xytext=(-4, 6), fontsize=8, color=INK, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "machine_share.pdf")
    plt.close(fig)


# ------------------------------------------------- 3. declared AI assistance

def fig_provenance() -> None:
    d = pl.read_parquet(C.ARTIFACTS / "provenance.parquet").sort("period")
    d = d.filter(pl.col("n_commit") > 0)
    periods = d["period"].to_list()
    per10k = (d["commit_claude_code"] / d["n_commit"] * 1e4).to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.plot(range(len(periods)), per10k, color=SERIES[0], lw=1.8, marker="o",
            markersize=3.2, markerfacecolor=SURFACE, markeredgewidth=1.2)
    ax.set_ylabel("commits per 10,000")
    ax.set_title("Commits declaring Claude Code assistance (Co-Authored-By trailer)")
    ax.set_xticks(range(0, len(periods), 3))
    ax.set_xticklabels([periods[i] for i in range(0, len(periods), 3)],
                       rotation=55, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.annotate(f"{per10k[-1]:.1f} per 10k\n({periods[-1]})",
                (len(periods) - 1, per10k[-1]), textcoords="offset points",
                xytext=(-8, -2), fontsize=7.5, color=INK, ha="right", va="top")
    ax.annotate("first appearance\n2025-04", (list(per10k > 0).index(True), 2),
                textcoords="offset points", xytext=(6, 14), fontsize=7,
                color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "provenance.pdf")
    plt.close(fig)


# ------------------------------------------------- 4. expression trajectories

SHOW = [
    ("TYPO:em_dash", "em dash (—)"),
    ("HYPH:follow-up", "follow-up"),
    ("CONSTR:canonical", "canonical"),
    ("load bearing", "load-bearing"),
]


def fig_expressions() -> None:
    s = Series.load().standardize()
    periods = s.periods
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for slot, (term, label) in enumerate(SHOW):
        i = s.index.get(term)
        if i is None:
            continue
        r = np.asarray(s.counts)[i] / np.where(s.docs > 0, s.docs, np.nan) * 1e4
        ax.plot(range(len(periods)), r, color=SERIES[slot], lw=1.8, label=label)
        last = np.nanargmax(np.where(np.isfinite(r), np.arange(len(r)), -1))
        ax.annotate(label, (last, r[last]), textcoords="offset points",
                    xytext=(4, 0), fontsize=7, color=INK, va="center")
    ax.set_yscale("log")
    ax.set_ylabel("documents per 10,000 (log)")
    ax.set_title("Standardised frequency of four expressions")
    _months(ax, periods)
    # the steepest part of every curve falls inside the window where the archive
    # changed what it carries; not marking that would overstate the finding
    _regime_band(ax, periods)
    ax.set_xlim(-1, len(periods) + 9)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "expressions.pdf")
    plt.close(fig)


# ------------------------------------------------------- 5. held-out generator

def fig_dating() -> None:
    j = json.loads((C.ARTIFACTS / "llm_date_model.json").read_text())
    rows = [("all generators\n(repo holdout)", j["expression_level"]["mae_months"],
             j["expression_level"]["baseline_mae_months"])]
    for tool, v in (j.get("leave_one_tool_out") or {}).items():
        if v.get("mae_months"):
            rows.append((f"held out:\n{tool}", v["mae_months"],
                         v["baseline_mae_months"]))
    labels = [r[0] for r in rows]
    mae = np.array([r[1] for r in rows])
    base = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    w = 0.34
    off = w / 2 + 0.025  # keep a visible surface gap between the paired fills
    ax.bar(x - off, mae, w, color=SERIES[0], label="model", linewidth=0)
    ax.bar(x + off, base, w, color=MUTED, label="baseline (guess the midpoint)",
           linewidth=0)
    for xi, (m, b) in enumerate(zip(mae, base)):
        ax.text(xi - off, m + 0.25, f"{m:.1f}", ha="center", fontsize=7, color=INK)
        ax.text(xi + off, b + 0.25, f"{b:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean absolute error (months)")
    ax.set_ylim(0, max(base.max(), mae.max()) * 1.22)
    ax.set_title("Dating LLM-written text: mean absolute error, lower is better")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "dating.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_coverage()
    fig_machine_share()
    fig_provenance()
    fig_expressions()
    fig_dating()
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.pdf"))))

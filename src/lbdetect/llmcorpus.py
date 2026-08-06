"""Dating LLM-generated text.

The rest of the project measures how *human* GitHub prose changed, and excludes
machine authors to do it. This module uses that excluded material as its subject:
review bots and coding agents write dated, unambiguously LLM-generated prose, so
they supply both the likelihood and a labelled test set for estimating when a
piece of generated text was produced.

The distinction matters for the likelihood. P(expression | period) estimated on
general GitHub prose describes how developers write; estimated here it describes
how *generated* text reads in that month. Dating generated text needs the second,
because the populations differ -- a phrase can be ubiquitous in model output while
staying rare in human comments.

Tool boilerplate is stripped like everywhere else. A tool's fixed banner would let
the model identify the tool and recover the date from the tool's popularity curve
rather than from the language, which would not transfer to a pasted paragraph of
model prose.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from . import config as C
from . import dedupe, templates
from .series import Series

# Authors that are LLM writing tools, as opposed to CI, linters or dependency
# bots. Only these produce model-generated *prose*.
LLM_AUTHOR = re.compile(
    r"^(?:copilot|copilot-swe-agent|copilot-pull-request-reviewer|coderabbitai|"
    r"claude|claude-bot|devin-ai-integration|cursor|cursoragent|codex|"
    r"chatgpt-codex-connector|gemini-code-assist|sweep-ai|korbit-ai|qodo-merge-pro|"
    r"pr-agent|ellipsis-dev|greptile-apps(?:-staging)?|cubic-dev-ai|sourcery-ai|"
    r"codeant-ai|bito-ai|entelligence-ai|macroscope-so|charliehelps)"
    r"(?:\[bot\])?$",
    re.I,
)

MIN_DOCS_PER_PERIOD = 25


def tool_of(author: str | None) -> str:
    if not author:
        return ""
    a = author.lower().removesuffix("[bot]")
    return re.sub(r"-(?:staging|preview)$", "", a)


def load_period(period: str, strip_templates: bool = True) -> pl.DataFrame:
    """LLM-authored documents of one period, cleaned the same way as the rest."""
    files = sorted((C.DOCS / period).glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    from .series import _concat_shards

    df = _concat_shards(files)
    df = df.filter(pl.col("artifact").is_in(C.PROSE_ARTIFACTS))
    if df.height == 0:
        return df
    df = df.unique(subset=["doc_id"], keep="first")
    df = df.with_columns(
        tool=pl.col("author").map_elements(tool_of, return_dtype=pl.Utf8)
    ).filter(
        pl.col("author").map_elements(
            lambda a: bool(LLM_AUTHOR.match((a or "").strip())), return_dtype=pl.Boolean
        )
    )
    if df.height == 0:
        return df
    if strip_templates:
        tmpl = templates.load()
        if tmpl:
            df = df.with_columns(
                text=pl.col("text").map_elements(
                    lambda t: templates.strip(t, tmpl), return_dtype=pl.Utf8)
            ).filter(pl.col("text").str.len_chars() > 0)
    if df.height == 0:
        return df
    # exact duplicates only: a tool repeating itself should count once, but its
    # house style is the signal and must not be filtered away
    return df.unique(subset=["text"], keep="first")


def corpus(periods: list[str], strip_templates: bool = True) -> pl.DataFrame:
    frames = [d for p in periods if (d := load_period(p, strip_templates)).height]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def build_series(docs: pl.DataFrame, terms: list[str], periods: list[str],
                 max_tokens: int = 400) -> Series:
    """Count the discovered expressions over the LLM-authored corpus."""
    from .ngrams import features

    wanted = set(terms)
    pos = {p: i for i, p in enumerate(periods)}
    idx = {t: i for i, t in enumerate(terms)}
    counts = np.zeros((len(terms), len(periods)), dtype=np.float64)
    ndocs = np.zeros(len(periods), dtype=float)
    for text, period in zip(docs["text"].to_list(), docs["period"].to_list()):
        j = pos.get(period)
        if j is None:
            continue
        ndocs[j] += 1
        for f in features(text, max_tokens=max_tokens) & wanted:
            counts[idx[f], j] += 1
    return Series(counts, list(terms), list(periods), ndocs)


def split_by_repo(docs: pl.DataFrame, holdout: float = 0.35, seed: int = 0
                  ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split on repository, so a project's reviews never straddle the split.

    A random document split would let near-identical reviews of the same pull
    request land on both sides and flatter the model.
    """
    h = (pl.col("repo_id") * 2654435761 + seed) % 1000
    thr = int(holdout * 1000)
    return docs.filter(h >= thr), docs.filter(h < thr)


def evaluate(train: pl.DataFrame, test: pl.DataFrame, terms: list[str],
             periods: list[str], use_clusters: bool = False,
             cluster_of: dict[str, int] | None = None,
             per_period_cap: int = 60, seed: int = 0,
             min_docs: int = MIN_DOCS_PER_PERIOD, log=print) -> dict:
    """Fit P(expression | period) on `train`, recover dates for `test`."""
    from .dating import Dater
    from .util import month_index

    s = build_series(train, terms, periods)
    usable = [p for p, n in zip(periods, s.docs) if n >= min_docs]
    if len(usable) < 6:
        return {"error": f"only {len(usable)} periods have "
                         f">={min_docs} LLM-authored documents"}
    dater = Dater(s, terms, min_docs=min_docs, cluster_of=cluster_of)

    t = test.filter(pl.col("period").is_in(dater.periods))
    if t.height == 0:
        return {"error": "no test documents in the fitted period range"}
    t = (t.with_columns(_r=pl.int_range(pl.len()).shuffle(seed=seed).over("period"))
         .filter(pl.col("_r") < per_period_cap).drop("_r"))

    mid = month_index(dater.periods[len(dater.periods) // 2])
    errs, naive, no_ev, by_year, by_tool, signed = [], [], 0, {}, {}, []
    for text, period, tool in zip(t["text"].to_list(), t["period"].to_list(),
                                  t["tool"].to_list()):
        est = (dater.estimate_by_cluster(text) if use_clusters
               else dater.estimate(text))
        if est.n_features == 0:
            no_ev += 1
            continue
        err = month_index(est.best_period) - month_index(period)
        errs.append(abs(err))
        signed.append(err)
        naive.append(abs(mid - month_index(period)))
        by_year.setdefault(period[:4], []).append(abs(err))
        by_tool.setdefault(tool, []).append(abs(err))
    if not errs:
        return {"error": "no test document contained a known expression"}

    out = {
        "n_train_docs": train.height,
        "n_test_scored": len(errs),
        "n_test_no_evidence": no_ev,
        "coverage": round(len(errs) / (len(errs) + no_ev), 3),
        "periods_fitted": [dater.periods[0], dater.periods[-1]],
        "n_periods": len(dater.periods),
        "mae_months": round(float(np.mean(errs)), 2),
        "median_ae_months": round(float(np.median(errs)), 2),
        "bias_months": round(float(np.mean(signed)), 2),
        "baseline_mae_months": round(float(np.mean(naive)), 2),
        "within_3_months": round(float(np.mean(np.array(errs) <= 3)), 3),
        "within_6_months": round(float(np.mean(np.array(errs) <= 6)), 3),
        "mae_by_year": {y: round(float(np.mean(v)), 1) for y, v in sorted(by_year.items())},
        "mae_by_tool": {k: round(float(np.mean(v)), 1)
                        for k, v in sorted(by_tool.items(), key=lambda x: -len(x[1]))[:6]},
        "method": "cluster-level" if use_clusters else "expression-level",
    }
    out["skill_vs_baseline"] = round(out["baseline_mae_months"] / out["mae_months"], 2)
    log(f"LLM-text dating ({out['method']}): MAE {out['mae_months']} months vs "
        f"{out['baseline_mae_months']} baseline "
        f"({out['skill_vs_baseline']}x), coverage {out['coverage']:.0%}")
    return out


def leave_one_tool_out(docs: pl.DataFrame, terms: list[str], periods: list[str],
                       min_docs: int = MIN_DOCS_PER_PERIOD, log=print) -> dict:
    """Fit without one tool, then date that tool's text.

    Tools have house styles and their own popularity curves, so a model fitted on
    all of them can recover a date by recognising *which* tool wrote the text. This
    asks the harder question the use case actually implies: does the language carry
    the date for a generator the model never saw?
    """
    counts = docs.group_by("tool").agg(pl.len().alias("n")).sort("n", descending=True)
    out = {}
    for tool in counts.filter(pl.col("n") >= 200)["tool"].to_list()[:3]:
        tr = docs.filter(pl.col("tool") != tool)
        te = docs.filter(pl.col("tool") == tool)
        res = evaluate(tr, te, terms, periods, min_docs=min_docs, log=lambda *_: None)
        out[tool] = {k: res.get(k) for k in
                     ("mae_months", "baseline_mae_months", "n_test_scored",
                      "coverage", "skill_vs_baseline", "error")}
        log(f"  held-out tool {tool}: MAE {res.get('mae_months')} vs "
            f"baseline {res.get('baseline_mae_months')}")
    return out

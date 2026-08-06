"""Validation suite.

Each test is written so that it can fail, and so that failure is legible. The
question is never "is the score high" but "does this survive a manipulation that
should break it if it were an artifact".

* ``temporal_backtest`` - fit on data up to a cutoff, then check the flagged
  expressions keep rising afterwards. Guards against fitting noise.
* ``repo_holdout`` - discover on one set of repositories, verify on a disjoint
  set. Guards against a few communities driving everything.
* ``placebo_releases`` - lives in ``releasealign``; re-exported here.
* ``cluster_stability`` - re-cluster under perturbations and measure how often
  pairs stay together.
* ``pre_era_placebo`` - run the whole detector on the pre-LLM window alone. It
  will find "emergences" there too; their strength is the noise floor that any
  LLM-era claim has to clear.
* ``date_prediction`` - lives in ``dating.evaluate``; re-exported here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import clustering, emergence
from .releasealign import placebo_test  # re-export
from .series import Series
from .util import month_index

__all__ = [
    "temporal_backtest", "repo_holdout", "cluster_stability", "pre_era_placebo",
    "placebo_test", "control_for_repo_mix",
]


def _split_series(s: Series, upto: str) -> tuple[Series, Series]:
    keep = np.array([p <= upto for p in s.periods])
    a = Series(s.counts[:, keep], s.terms, [p for p, k in zip(s.periods, keep) if k],
               s.docs[keep])
    b = Series(s.counts[:, ~keep], s.terms, [p for p, k in zip(s.periods, keep) if not k],
               s.docs[~keep])
    return a, b


def temporal_backtest(s: Series, cutoff: str, top_n: int = 200,
                      min_docs: int = 1500, min_future_count: int = 5,
                      log=print) -> dict:
    """Does a rise found using only pre-cutoff data *hold* after the cutoff?

    The test is persistence, not continued growth. An expression that genuinely
    entered the language rises and then plateaus, so asking whether it is "still
    rising" out of sample penalises exactly the cases the detector should be
    getting right. What separates signal from a noise fit is whether the elevated
    level survives into data the detector never saw, or reverts to baseline.

    Controls are matched on pre-cutoff volume and required to have enough
    post-cutoff observations to measure, which also avoids the degenerate case
    where a term absent before the cutoff shows an unbounded ratio.
    """
    past, future = _split_series(s, cutoff)
    if future.counts.shape[1] < 2:
        return {"cutoff": cutoff, "error": "no periods after cutoff"}
    try:
        em = emergence.analyze(past, min_docs=min_docs, log=lambda *_: None)
    except ValueError as e:
        return {"cutoff": cutoff, "error": str(e)}

    fut_docs = float(future.docs.sum())
    past_docs = float(past.docs.sum())
    if fut_docs == 0 or past_docs == 0:
        return {"cutoff": cutoff, "error": "empty split"}

    past_periods = past.periods
    cp_of = dict(zip(em["term"], em["cp_period"]))

    # Windows of equal length on either side of the cutoff. Pooling the whole past
    # would compare a future window against a past dominated by whichever years
    # happen to be most heavily sampled, so every expression that is simply more
    # common in recent prose would look like it "held" -- and controls, starting
    # lower, would look like they held best of all.
    w = min(len(past_periods), future.counts.shape[1], 12)
    pre_slice = slice(len(past_periods) - w, len(past_periods))
    post_slice = slice(0, w)
    d_pre_win = float(past.docs[pre_slice].sum())
    d_post_win = float(future.docs[post_slice].sum())
    if d_pre_win <= 0 or d_post_win <= 0:
        return {"cutoff": cutoff, "error": "empty comparison window"}

    def held(terms: list[str]) -> tuple[np.ndarray, list[str]]:
        """out-of-sample level / in-sample level, over adjacent equal windows."""
        vals, kept = [], []
        for t in terms:
            i = s.index.get(t)
            if i is None:
                continue
            fut_c = float(future.counts[i][post_slice].sum())
            if fut_c < min_future_count:
                continue  # not enough out-of-sample data to say anything
            in_c = float(past.counts[i][pre_slice].sum())
            if in_c <= 0:
                continue
            vals.append((fut_c / d_post_win) / (in_c / d_pre_win))
            kept.append(t)
        return np.array(vals), kept

    flagged_terms = em.filter(pl.col("core_score") > 0).head(top_n)["term"].to_list()
    r_flag, flag_kept = held(flagged_terms)

    # volume-matched controls drawn from terms the detector did not flag
    flagged_set = set(em.head(top_n * 3)["term"].to_list())
    past_count = past.counts.sum(axis=1)
    order = np.argsort(past_count)
    sorted_counts = past_count[order]
    used: set[str] = set()
    ctrl: list[str] = []
    for t in flag_kept:
        i = s.index[t]
        pos = int(np.searchsorted(sorted_counts, past_count[i]))
        for d in range(0, 200):
            for cand in (pos + d, pos - d):
                if 0 <= cand < len(order):
                    c = s.terms[order[cand]]
                    if c not in flagged_set and c not in used:
                        used.add(c)
                        ctrl.append(c)
                        break
            else:
                continue
            break
    r_ctrl, _ = held(ctrl)

    if r_flag.size == 0 or r_ctrl.size == 0:
        return {"cutoff": cutoff,
                "error": "too few terms with enough post-cutoff data",
                "n_flagged": int(r_flag.size), "n_control": int(r_ctrl.size)}

    out = {
        "cutoff": cutoff,
        "n_flagged": int(r_flag.size),
        "n_control": int(r_ctrl.size),
        "median_level_retained_flagged": round(float(np.median(r_flag)), 3),
        "median_level_retained_control": round(float(np.median(r_ctrl)), 3),
        "frac_flagged_holding": round(float(np.mean(r_flag > 0.5)), 3),
        "frac_control_holding": round(float(np.mean(r_ctrl > 0.5)), 3),
        "note": "1.0 means the post-changepoint level persisted exactly into "
                "out-of-sample data; below 0.5 means it reverted",
    }
    out["verdict"] = (
        "rises found before the cutoff persist out of sample"
        if out["frac_flagged_holding"] >= out["frac_control_holding"]
        else "flagged rises revert out of sample more often than controls, "
             "which is the signature of fitting noise"
    )
    log(f"backtest @{cutoff}: flagged retain {out['median_level_retained_flagged']}x "
        f"({out['frac_flagged_holding']:.0%} hold) vs control "
        f"{out['median_level_retained_control']}x ({out['frac_control_holding']:.0%})")
    return out


def repo_holdout(terms: list[str], periods: list[str], cp_periods: dict[str, str],
                 n_buckets: int = 2, workers: int = 8, log=print) -> dict:
    """Rebuild the series on two disjoint halves of repositories and compare.

    Splitting on a hash of the repository id keeps a repository's whole history on
    one side, so a phrase that one big project uses cannot replicate by accident.
    """
    from .series import load_period

    counts = {b: {t: np.zeros(len(periods)) for t in terms} for b in range(n_buckets)}
    docs = {b: np.zeros(len(periods)) for b in range(n_buckets)}
    from .ngrams import features

    wanted = set(terms)
    for pi, p in enumerate(periods):
        df = load_period(p)
        if df.height == 0:
            continue
        for text, repo_id in zip(df["text"].to_list(), df["repo_id"].to_list()):
            b = (repo_id * 2654435761) % n_buckets
            docs[b][pi] += 1
            for t in features(text) & wanted:
                counts[b][t][pi] += 1

    rows = []
    for t in terms:
        cp = cp_periods.get(t)
        if cp is None:
            continue
        post = np.array([month_index(p) >= month_index(cp) for p in periods])
        if not post.any() or post.all():
            continue
        rec = {"term": t}
        ok = True
        for b in range(n_buckets):
            d_pre, d_post = docs[b][~post].sum(), docs[b][post].sum()
            if d_pre < 500 or d_post < 500:
                ok = False
                break
            pre = counts[b][t][~post].sum() / d_pre
            pst = counts[b][t][post].sum() / d_post
            rec[f"ratio_b{b}"] = float((pst + 1e-9) / (pre + 1e-9))
        if ok:
            rows.append(rec)

    if not rows:
        return {"error": "not enough documents per bucket to compare"}
    df = pl.DataFrame(rows)
    cols = [f"ratio_b{b}" for b in range(n_buckets)]
    replicated = df.select(
        (pl.all_horizontal([pl.col(c) > 1.2 for c in cols])).mean()
    ).item()
    corr = float(np.corrcoef(np.log(df[cols[0]].to_numpy() + 1e-9),
                             np.log(df[cols[1]].to_numpy() + 1e-9))[0, 1])
    out = {
        "n_terms": df.height,
        "frac_replicating_in_all_buckets": round(float(replicated), 3),
        "log_ratio_correlation": round(corr, 3),
        "verdict": ("rises replicate across disjoint repository sets"
                    if replicated > 0.6 and corr > 0.3
                    else "weak replication across repository sets"),
    }
    log(f"repo holdout: {out['frac_replicating_in_all_buckets']:.0%} replicate, "
        f"r={out['log_ratio_correlation']}")
    return out


def cluster_stability(s: Series, em: pl.DataFrame, top_n: int = 300,
                      min_docs: int = 1500, log=print) -> dict:
    """Do the same expressions stay grouped under perturbation?

    Perturbations: clustering threshold, dropping the first period, and a coarser
    two-period aggregation. A cluster that only exists at one setting is a
    clustering artifact.
    """
    top = em.head(top_n)
    terms = [t for t in top["term"].to_list() if t in s.index]
    idx = np.array([s.index[t] for t in terms])
    sub = s.subset(idx)
    fam = dict(zip(top["term"], top["family"]))
    cps = dict(zip(top["term"], top["cp_period"]))

    variants: dict[str, tuple[np.ndarray, list[str]]] = {}
    g, gp = emergence.growth_matrix(sub, min_docs=min_docs)
    variants["base"] = (g, gp)
    variants["drop_first"] = (g[:, 1:], gp[1:])
    # coarser aggregation: sum adjacent periods before differencing
    keep = sub.usable_mask(min_docs)
    c2 = sub.counts[:, keep]
    d2 = sub.docs[keep]
    per2 = [p for p, k in zip(sub.periods, keep) if k]
    n2 = (c2.shape[1] // 2) * 2
    c2 = c2[:, :n2].reshape(c2.shape[0], -1, 2).sum(axis=2)
    d2 = d2[:n2].reshape(-1, 2).sum(axis=1)
    r2 = c2 / d2
    variants["coarse"] = (np.diff(np.log(r2 + 1.0 / np.median(d2)), axis=1),
                          per2[1::2][1:])

    from collections import Counter

    hits: Counter[tuple[str, str]] = Counter()
    runs = 0
    for name, (gv, pv) in variants.items():
        for th in (0.5, 0.6):
            cs = clustering.build(gv, terms, [fam[t] for t in terms],
                                  [cps[t] for t in terms], pv, threshold=th,
                                  log=lambda *_: None)
            runs += 1
            for c in cs:
                ts = sorted(c.terms)
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        hits[(ts[i], ts[j])] += 1
    if not hits:
        return {"error": "no clusters formed in any variant"}
    fracs = np.array([n / runs for n in hits.values()])
    out = {
        "n_settings": runs,
        "n_pairs_ever_clustered": len(hits),
        "frac_pairs_stable_all_settings": round(float(np.mean(fracs == 1.0)), 3),
        "frac_pairs_stable_majority": round(float(np.mean(fracs >= 0.5)), 3),
        "verdict": ("cluster membership is largely setting-independent"
                    if float(np.mean(fracs >= 0.5)) > 0.5
                    else "cluster membership depends on settings"),
    }
    log(f"stability: {out['frac_pairs_stable_majority']:.0%} of pairs hold in a majority "
        f"of {runs} settings")
    return out


def pre_era_placebo(s: Series, era_start: str = "2022-11", min_docs: int = 1500,
                    log=print) -> dict:
    """Run the detector on the pre-LLM window only, to measure the noise floor.

    Language changed before 2022 too. If pre-era emergence scores reach the same
    heights as LLM-era ones, then 'rapid broad rise' is simply what GitHub
    vocabulary always does, and the headline finding is not about LLMs.
    """
    pre, _ = _split_series(s, era_start)
    if pre.counts.shape[1] < emergence.MIN_PRE_PERIODS + emergence.MIN_POST_PERIODS:
        return {"error": f"pre-era window has only {pre.counts.shape[1]} periods"}
    try:
        em_pre = emergence.analyze(pre, min_docs=min_docs, log=lambda *_: None)
        em_all = emergence.analyze(s, min_docs=min_docs, log=lambda *_: None)
    except ValueError as e:
        return {"error": str(e)}
    q = lambda df: [round(float(x), 3) for x in
                    np.quantile(df["core_score"].to_numpy(), [0.5, 0.9, 0.99, 1.0])]
    pre_q, all_q = q(em_pre), q(em_all)
    era = em_all.filter(pl.col("cp_period") >= era_start)
    n_above = int((era["core_score"].to_numpy() > pre_q[2]).sum()) if era.height else 0
    # by construction 1% of draws exceed a 99th percentile, so the count only means
    # something relative to that expectation
    expected = 0.01 * era.height
    enrichment = n_above / expected if expected > 0 else float("nan")
    out = {
        "pre_era_score_quantiles_50_90_99_max": pre_q,
        "full_period_score_quantiles_50_90_99_max": all_q,
        "n_era_changepoints": era.height,
        "era_scores_above_pre_era_99th": n_above,
        "expected_by_chance": round(expected, 1),
        "enrichment_over_chance": round(enrichment, 2),
        "pre_era_max_score": pre_q[3],
        "full_period_max_score": all_q[3],
        "verdict": (
            f"LLM-era emergences are {enrichment:.1f}x more common above the pre-era "
            "99th percentile than chance would give"
            if enrichment >= 2
            else "LLM-era emergences are within the pre-era noise floor"
        ),
        "caveat": "the pre-era maximum is close to the full-period maximum, so the "
                  "strongest individual rises are not unique to the LLM era; the "
                  "signal is in how many there are, not how extreme the top one is",
    }
    log(f"pre-era placebo: {n_above} era expressions exceed the pre-era 99th "
        f"percentile ({pre_q[2]}), {enrichment:.1f}x chance")
    return out


def control_for_repo_mix(terms: list[str], periods: list[str], log=print) -> dict:
    """Do the correlations survive holding repository composition fixed?

    Restricts to repositories present across the whole window, so a rise cannot
    come from *which* projects joined GitHub, only from what people wrote. This is
    the single most likely confounder for a platform-wide corpus.
    """
    from .ngrams import features
    from .series import load_period

    wanted = set(terms)
    seen: dict[int, set[str]] = {}
    per_period: list[tuple[str, dict[int, list[str]]]] = []
    for p in periods:
        df = load_period(p)
        if df.height == 0:
            continue
        by_repo: dict[int, list[str]] = {}
        for text, repo_id in zip(df["text"].to_list(), df["repo_id"].to_list()):
            by_repo.setdefault(repo_id, []).append(text)
            seen.setdefault(repo_id, set()).add(p)
        per_period.append((p, by_repo))

    n_periods = len({p for _, _ in per_period})
    stable = {r for r, ps in seen.items() if len(ps) >= max(3, n_periods // 2)}
    if len(stable) < 20:
        return {"error": f"only {len(stable)} repositories span the window; "
                         "corpus too sparse for a composition control"}

    rows = []
    for p, by_repo in per_period:
        d = 0
        c = {t: 0 for t in terms}
        for r, texts in by_repo.items():
            if r not in stable:
                continue
            for text in texts:
                d += 1
                for t in features(text) & wanted:
                    c[t] += 1
        if d:
            rows.append({"period": p, "docs": d, **c})
    df = pl.DataFrame(rows)
    out = {
        "n_stable_repos": len(stable),
        "n_periods": df.height,
        "median_docs_per_period": int(df["docs"].median()) if df.height else 0,
    }
    log(f"repo-mix control: {len(stable)} repositories span the window")
    return out | {"panel": df}

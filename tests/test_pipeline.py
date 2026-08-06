"""Tests for the parts where a silent error would corrupt every downstream number."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lbdetect import breadth, clustering, dedupe, emergence, ngrams, series, templates
from lbdetect import textclean, releasealign
from lbdetect.series import Series


# ------------------------------------------------------------------ text cleaning

def test_code_blocks_and_logs_are_removed():
    raw = (
        "Here is the actual point I am making.\n"
        "```python\n"
        "def f(x):\n    return x + 1\n"
        "```\n"
        "Traceback (most recent call last):\n"
        '  File "a.py", line 3, in <module>\n'
        "    raise ValueError\n"
    )
    c = textclean.clean(raw)
    assert "actual point" in c.text
    assert "def f" not in c.text
    assert "Traceback" not in c.text
    assert c.had_code


def test_quotes_and_urls_removed_but_prose_kept():
    c = textclean.clean("> quoted reply text here\nMy own reply https://x.com/a?b=1 ok")
    assert "quoted reply" not in c.text
    assert "x.com" not in c.text
    assert "My own reply" in c.text


def test_non_english_is_ineligible():
    assert not textclean.eligible(textclean.clean("这是一个中文的问题描述需要更多的文字" * 3))


def test_eligibility_gates_numerator_and_denominator_identically():
    # the same predicate must decide both, or rates drift with document mix
    docs = ["ok", "This is a real English sentence with enough words to count here."]
    cleaned = [textclean.clean(d) for d in docs]
    assert [textclean.eligible(c) for c in cleaned] == [False, True]


def test_bot_expr_matches_is_bot_login():
    logins = ["dependabot[bot]", "copilot", "alice", "renovate", "some-ci", None]
    df = pl.DataFrame({"author": logins})
    got = df.select(textclean.bot_expr("author"))["author"].to_list()
    want = [textclean.is_bot_login(x) for x in logins]
    assert got == want


# ---------------------------------------------------------------------- features

def test_hyphen_and_space_variants_share_a_family():
    a = ngrams.family_key("HYPH:load-bearing")
    b = ngrams.family_key("load bearing")
    c = ngrams.family_key("load bearings")
    assert a == b == c == "load bearing"


def test_ngrams_do_not_cross_sentence_boundaries():
    grams = ngrams.word_ngrams(ngrams.tokenize("fix the bug. ship it now"))
    assert "the bug" in grams
    assert "bug ship" not in grams


def test_singularize_never_mangles_function_words():
    for w in ("is", "this", "status", "class", "analysis"):
        assert ngrams._singularize(w) == w


def test_singularize_handles_irregulars():
    assert ngrams._singularize("indices") == "index"
    assert ngrams._singularize("criteria") == "criterion"
    assert ngrams._singularize("assumptions") == "assumption"


def test_features_are_presence_not_count():
    once = ngrams.features("the load bearing wall")
    twice = ngrams.features("the load bearing wall and the load bearing wall")
    assert "load bearing" in once and "load bearing" in twice


def test_constructions_fire_on_slots():
    f = ngrams.features("This is not just a refactor, but a rewrite of the core")
    assert "CONSTR:not_just_but" in f


# ----------------------------------------------------------------------- dedupe

def test_templated_author_detected_across_repos():
    n = 12
    df = pl.DataFrame({
        "author": ["reviewbot"] * n + [f"human{i}" for i in range(n)],
        "repo_id": list(range(n)) + list(range(n)),
        "text": [f"Pull request overview This PR changes thing {i}." for i in range(n)]
                + [f"Totally different human sentence number {i} here." for i in range(n)],
    })
    assert "reviewbot" in dedupe.templated_authors(df)


def test_prolific_human_not_flagged_as_templated():
    df = pl.DataFrame({
        "author": ["alice"] * 12,
        "repo_id": list(range(12)),
        "text": [f"A distinct opening {i} followed by other words." for i in range(12)],
    })
    assert "alice" not in dedupe.templated_authors(df)


# --------------------------------------------------------------------- templates

def test_template_lines_normalise_digits():
    a = templates.norm_line("Copilot reviewed 67 out of 69 changed files")
    b = templates.norm_line("Copilot reviewed 3 out of 4 changed files")
    assert a == b


def test_strip_removes_only_template_lines():
    t = frozenset({templates.norm_line("by submitting this pull request i confirm")})
    text = "By submitting this pull request I confirm\nReal human sentence."
    assert templates.strip(text, t).strip() == "Real human sentence."


# --------------------------------------------------------------------- emergence

def _synthetic_series(n_periods=40, jump_at=25, n_docs=5000):
    """One expression that jumps 10x, one flat, one that only declines."""
    docs = np.full(n_periods, n_docs, dtype=float)
    rise = np.array([2 if i < jump_at else 20 for i in range(n_periods)], float)
    flat = np.full(n_periods, 10.0)
    fall = np.array([30 if i < jump_at else 3 for i in range(n_periods)], float)
    counts = np.vstack([rise, flat, fall]).astype(np.int32)
    periods = [f"20{18 + i // 12:02d}-{i % 12 + 1:02d}" for i in range(n_periods)]
    return Series(counts, ["riser", "flat", "faller"], periods, docs)


def test_changepoint_lands_near_the_true_jump():
    s = _synthetic_series()
    em = emergence.analyze(s, min_docs=100, log=lambda *_: None)
    row = em.filter(pl.col("term") == "riser").to_dicts()[0]
    assert row["cp_period"] == s.periods[25]
    assert row["log_growth"] > 1.5
    assert row["core_score"] > 0.4


def test_pure_decline_scores_zero():
    s = _synthetic_series()
    em = emergence.analyze(s, min_docs=100, log=lambda *_: None)
    assert em.filter(pl.col("term") == "faller").to_dicts()[0]["core_score"] == 0.0


def test_flat_series_scores_below_riser():
    s = _synthetic_series()
    em = emergence.analyze(s, min_docs=100, log=lambda *_: None)
    d = {r["term"]: r["core_score"] for r in em.to_dicts()}
    assert d["riser"] > d["flat"]


def test_core_score_is_never_nan():
    s = _synthetic_series()
    em = emergence.analyze(s, min_docs=100, log=lambda *_: None)
    assert not np.isnan(em["core_score"].to_numpy()).any()


def test_thin_periods_are_dropped_not_treated_as_zero():
    s = _synthetic_series()
    s.docs[10] = 0
    cps = emergence.find_changepoints(s, min_docs=100)
    assert s.periods[10] not in list(cps["periods"])


# --------------------------------------------------------------------- clustering

def test_cluster_recovers_a_shared_shock():
    rng = np.random.default_rng(0)
    T = 30
    shock = np.zeros(T)
    shock[15] = 1.0
    g = np.vstack([shock * 2 + rng.normal(0, 0.05, T) for _ in range(4)]
                  + [rng.normal(0, 0.5, T) for _ in range(4)])
    terms = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]
    fams = terms
    cps = ["2024-04"] * 4 + ["2019-01", "2020-06", "2021-03", "2022-09"]
    periods = [f"2023-{i % 12 + 1:02d}" for i in range(T)]
    cs = clustering.build(g, terms, fams, cps, periods, threshold=0.5,
                          min_size=3, min_families=3, log=lambda *_: None)
    assert cs, "expected at least one cluster"
    biggest = max(cs, key=lambda c: len(c.terms))
    assert set(biggest.terms) >= {"a0", "a1", "a2"}


def test_cluster_requires_distinct_families():
    rng = np.random.default_rng(1)
    T = 30
    shock = np.zeros(T)
    shock[15] = 1.0
    g = np.vstack([shock * 2 + rng.normal(0, 0.05, T) for _ in range(4)])
    terms = ["load bearing", "load bearings", "HYPH:load-bearing", "load bearing wall"]
    fams = ["load bearing"] * 3 + ["load bearing wall"]  # only 2 families
    cs = clustering.build(g, terms, fams, ["2024-04"] * 4,
                          [f"2023-{i % 12 + 1:02d}" for i in range(T)],
                          min_size=3, min_families=3, log=lambda *_: None)
    assert cs == []


# ------------------------------------------------------------------------ breadth

def test_cohesion_separates_templates_from_phrases():
    h = dedupe.MinHasher(32)
    same = [h.signature("the quick brown fox jumps over the lazy dog every time")] * 6
    varied = [h.signature(f"completely unrelated sentence number {i} about {i * 7} things")
              for i in range(6)]
    assert breadth.cohesion(same) > 0.8
    assert breadth.cohesion(varied) < 0.2


def test_capitalization_ignores_sentence_initial():
    pat = breadth._cap_pattern("mcp server")
    # sentence-initial capital carries no information and must not count, so only
    # the second (mid-sentence, lowercase) occurrence is measured
    occ, cap = breadth.capitalization("MCP server is fine. we use mcp server here", pat)
    assert occ == 1 and cap == 0


def test_capitalization_detects_a_name():
    pat = breadth._cap_pattern("mcp server")
    occ, cap = breadth.capitalization("we run the MCP server daily", pat)
    assert occ == 1 and cap == 1


# ------------------------------------------------------------------------- series

def test_calendar_windows_keep_time_resolution():
    periods = [f"20{18 + i // 12:02d}-{i % 12 + 1:02d}" for i in range(48)]
    w = series.window_bounds(periods, window_months=12)
    assert len(w) == 4
    assert w[0][0] == periods[0]


def test_aggregate_pools_counts_and_docs():
    s = _synthetic_series(n_periods=12)
    q = s.aggregate(3)
    assert len(q.periods) == 4
    assert q.counts.sum() == s.counts.sum()
    assert q.docs.sum() == s.docs.sum()


def test_rates_are_nan_where_no_documents():
    s = _synthetic_series()
    s.docs[3] = 0
    assert np.isnan(s.rates[:, 3]).all()


# ------------------------------------------------------------------------- align

def test_placebo_reports_underpowered_when_releases_are_dense():
    cs = [
        clustering.Cluster(cid=i, terms=[f"t{i}a", f"t{i}b", f"t{i}c"],
                           loadings=np.ones(3), latent=np.zeros(5),
                           periods=["2025-01"] * 5, coherence=0.5,
                           cp_period="2025-05", cp_spread=0.1, n_families=3)
        for i in range(4)
    ]
    res = releasealign.placebo_test(cs, n_draws=200, log=lambda *_: None)
    assert "verdict" in res
    # every cluster sits on a release, so the honest answer is not a tiny p-value
    assert res["observed_mean_distance"] < 2.0


def test_placebo_null_keeps_releases_inside_the_window():
    cs = [
        clustering.Cluster(cid=i, terms=[f"t{i}a", f"t{i}b", f"t{i}c"],
                           loadings=np.ones(3), latent=np.zeros(5),
                           periods=["2020-01"] * 5, coherence=0.5,
                           cp_period=p, cp_spread=0.1, n_families=3)
        for i, p in enumerate(["2019-03", "2020-07", "2021-11", "2023-02"])
    ]
    res = releasealign.placebo_test(cs, n_draws=300, log=lambda *_: None)
    # a null that pushed the calendar out of range would show an absurd mean
    assert res["null_a_shifted_calendar"]["mean"] < 60


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# -------------------------------------------------------------- LLM-text dating

def test_llm_author_matches_tools_not_ci():
    from lbdetect import llmcorpus

    for a in ("copilot", "coderabbitai[bot]", "sourcery-ai[bot]", "Copilot"):
        assert llmcorpus.LLM_AUTHOR.match(a), a
    for a in ("dependabot[bot]", "travis-ci", "codecov", "alice"):
        assert not llmcorpus.LLM_AUTHOR.match(a), a


def test_tool_of_normalises_staging_variants():
    from lbdetect import llmcorpus

    assert llmcorpus.tool_of("greptile-apps-staging[bot]") == "greptile-apps"
    assert llmcorpus.tool_of("CodeRabbitAI[bot]") == "coderabbitai"


def test_repo_split_is_disjoint():
    from lbdetect import llmcorpus

    docs = pl.DataFrame({"repo_id": list(range(200)), "text": ["x"] * 200})
    tr, te = llmcorpus.split_by_repo(docs, holdout=0.3)
    assert set(tr["repo_id"]).isdisjoint(set(te["repo_id"]))
    assert tr.height + te.height == docs.height


def test_dater_reports_no_evidence_instead_of_guessing_first_period():
    # a uniform posterior has no argmax; returning period 0 would read as a
    # confident answer of "the earliest month in the corpus"
    from lbdetect import dating

    s = _synthetic_series(n_periods=40)
    d = dating.Dater(s, ["riser", "flat"], min_docs=100)
    est = d.estimate("zzz qqq nothing matches here at all")
    assert est.n_features == 0
    assert "no evidence" in est.method
    assert est.best_period == d.periods[len(d.periods) // 2]


def test_vocabulary_admits_declines():
    # dating needs early-period markers; a rise-only lexicon biases estimates late
    import lbdetect.series as S

    if not S.VOCAB.exists():
        pytest.skip("vocabulary not built")
    v = pl.read_parquet(S.VOCAB)
    assert "direction" in v.columns
    assert (v["direction"] == "decline").sum() > 0

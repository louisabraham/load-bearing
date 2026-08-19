"""Tests for the simple pipeline: uniform grid, word counting, shift detection."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from shift import counts, detect, factor, fetch


# -------------------------------------------------------------------- sampling

def test_deepening_the_sample_keeps_what_was_already_fetched():
    """The draw is a truncated permutation, so a bigger K extends it."""
    assert fetch.draws(3, 4) == fetch.draws(3, 40)[:4]


def test_hours_within_a_week_are_distinct():
    d = fetch.draws(7, 60)
    assert len(set(d)) == 60


def test_draws_stay_inside_their_own_week():
    for k in (0, 17, 40):
        lo, hi = fetch.week_start(k), fetch.week_start(k + 1)
        assert all(lo <= day < hi for day, _ in fetch.draws(k, 30))


def test_every_week_gets_the_same_number_of_hours():
    """Uniform across the window: no week is looked at harder than another."""
    today = date(2024, 6, 1)
    per_week = {k: len(fetch.draws(k, 9)) for k in range(fetch.n_weeks(today))}
    assert set(per_week.values()) == {9}


def test_weeks_get_different_hours():
    """Independent draws, not one schedule repeated."""
    offsets = [{(day - fetch.week_start(k)).days * 24 + h
                for day, h in fetch.draws(k, 10)} for k in range(12)]
    assert len({frozenset(o) for o in offsets}) == 12


def test_the_draw_is_reproducible_and_seed_dependent():
    assert fetch.draws(5, 10) == fetch.draws(5, 10)
    assert fetch.draws(5, 10, seed=1) != fetch.draws(5, 10, seed=0)


def test_the_sample_spreads_over_the_whole_clock():
    """Random hours rather than a fixed grid: all 24 hours turn up."""
    s = fetch.sample(10, today=date(2026, 8, 18))
    assert len({h for _, h in s}) == 24


def test_the_sample_is_draw_index_major():
    """A truncated fetch must be thin everywhere, not absent from the recent end."""
    today = date(2024, 6, 1)
    n = fetch.n_weeks(today)
    first = fetch.sample(4, today=today)[:n]
    assert len({(d - fetch.ANCHOR).days // fetch.WEEK_DAYS for d, _ in first}) == n


def test_the_anchor_is_a_monday():
    """Weeks that start mid-week would straddle two weekends and mix the mix."""
    assert fetch.ANCHOR.weekday() == 0


# --------------------------------------------------------------- word extraction

def test_words_are_whitespace_separated_and_lowercased():
    assert counts.words("The Load-Bearing wall") == {"the", "load-bearing", "wall"}


def test_surrounding_punctuation_is_dropped_so_a_word_is_one_word():
    assert counts.words("load-bearing, **load-bearing** (load-bearing)") == {
        "load-bearing"
    }


def test_bare_numbers_are_not_words():
    """Dates, versions and counts arrive on the calendar's schedule, not language's."""
    assert counts.words("bumped to 3 from 2 on 2026") == {"bumped", "to", "from", "on"}
    assert "v2" in counts.words("released v2 of the thing")


def test_a_word_counts_once_per_document():
    assert counts.words("very very very robust") == {"very", "robust"}


def test_internal_punctuation_survives():
    assert "don't" in counts.words("i don't think so")
    assert "src/main.py" in counts.words("see src/main.py")


# ------------------------------------------------------------- document loading

def _write(tmp_path, name, rows):
    pl.DataFrame(rows, schema=fetch.SCHEMA).write_parquet(tmp_path / name)
    return tmp_path / name


def _row(author, body, repo="a/b"):
    return {"ts": "2024-01-03T05:00:00Z", "repo": repo, "author": author,
            "is_pr": False, "body": body}


def test_bot_authors_are_kept_and_stubs_are_not(tmp_path):
    """Machine-written comments are part of the language, so they stay in. Only
    documents too short to carry any are dropped."""
    f = _write(tmp_path, "2024-01-03-05.parquet", [
        _row("alice", "this comment has plenty of distinct words in it"),
        _row("coderabbitai[bot]", "walkthrough of the changes in this pull request"),
        _row("bob", "lgtm thanks"),
    ])
    assert len(list(counts.documents(f))) == 2
    assert len(list(counts.documents(f, drop_bots=True))) == 1
    assert [b for _, b, _ in counts.documents(f)] == [False, True]


def test_a_shard_belongs_to_the_week_its_name_says():
    assert counts.week_of(Path("2024-01-01-00.parquet")) == 0
    assert counts.week_of(Path("2024-01-07-23.parquet")) == 0
    assert counts.week_of(Path("2024-01-08-00.parquet")) == 1
    assert counts.week_of(Path("2026-08-10-13.parquet")) == 136


def test_repeated_documents_collapse_inside_a_week(tmp_path, monkeypatch):
    """One account posting the same sentence 50 times is one document, not 50."""
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    flood = "this issue was automatically closed as it was created before 2024"
    f = _write(tmp_path, "2024-01-03-05.parquet",
               [_row("masscloser", flood, f"r/{i}") for i in range(50)]
               + [_row("alice", "a genuine comment with enough distinct words here")])
    assert len(list(counts.documents(f))) == 51
    assert len(list(counts.week([f]))) == 2


def test_duplicates_collapse_across_shards_of_the_same_week(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    flood = "please upgrade to the latest release before reporting this again"
    a = _write(tmp_path, "2024-01-01-00.parquet", [_row("bot-ish", flood)])
    b = _write(tmp_path, "2024-01-06-07.parquet", [_row("bot-ish", flood)])
    assert counts.week_of(a) == counts.week_of(b) == 0
    assert len(list(counts.week([a, b]))) == 1


def test_a_long_running_template_is_a_level_not_a_change(tmp_path, monkeypatch):
    """Collapsing inside the week, not across the window, is what keeps a
    months-long template from looking like it started or stopped."""
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    flood = "closing this as stale after ninety days without any activity"
    for k in range(3):
        day, hour = fetch.draws(k, 1)[0]
        _write(tmp_path, f"{day.isoformat()}-{hour:02d}.parquet",
               [_row("stalebot-user", flood)] * 30)
    got = counts.groups(hours=1)
    assert sorted(got) == [0, 1, 2]
    assert [len(list(counts.week(g))) for g in got.values()] == [1, 1, 1]


def test_only_sampled_hours_join_the_matrix(tmp_path, monkeypatch):
    """A stray file, or one fetched under a different byte cap, is not the sample."""
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    day, hour = fetch.draws(0, 1)[0]
    _write(tmp_path, f"{day.isoformat()}-{hour:02d}.parquet",
           [_row("alice", "a comment with quite enough distinct words to count")])
    _write(tmp_path, "2024-01-07-13.parquet",
           [_row("bob", "another comment with quite enough distinct words here")])
    assert counts.groups(hours=1) == {0: [fetch.path(day, hour)]}


def test_support_counts_repositories_and_bot_share_not_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    day, hour = fetch.draws(0, 1)[0]
    _write(tmp_path, f"{day.isoformat()}-{hour:02d}.parquet", [
        _row("alice", "the wall here is load-bearing and quite solid", "one/repo"),
        _row("bob", "this assumption is load-bearing for the proof", "one/repo"),
        _row("carol", "a load-bearing invariant lives in this module", "two/repo"),
        _row("reviewbot[bot]", "this load-bearing check is automated", "four/repo"),
    ])
    got = counts.support(range(1), ["load-bearing", "wall"], hours=1)
    assert got["load-bearing"].repos == 3
    assert got["load-bearing"].bot_share == 0.25   # one of four documents
    assert got["wall"] == (1, 0.0)


def test_thinning_gives_every_week_the_same_document_count(tmp_path, monkeypatch):
    """The archive's own volume swings by 2x across the window, and a z computed on
    more documents is inflated rather than merely more precise."""
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    day, hour = fetch.draws(0, 1)[0]
    f = _write(tmp_path, f"{day.isoformat()}-{hour:02d}.parquet",
               [_row("u", f"comment tag{i} with enough distinct words to count")
                for i in range(50)])
    assert len(counts.week_docs([f], 0, cap=20)) == 20
    assert len(counts.week_docs([f], 0, cap=0)) == 50
    assert len(counts.week_docs([f], 0, cap=200)) == 50   # a short week keeps all


def test_the_thinned_subsample_is_the_same_every_time(tmp_path, monkeypatch):
    """Both passes of the build must see the same documents, or the vocabulary and
    the counts describe different corpora."""
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    day, hour = fetch.draws(0, 1)[0]
    f = _write(tmp_path, f"{day.isoformat()}-{hour:02d}.parquet",
               [_row("u", f"comment tag{i} with enough distinct words to count")
                for i in range(50)])
    a = [sorted(w) for _, _, w in counts.week_docs([f], 0, cap=20)]
    b = [sorted(w) for _, _, w in counts.week_docs([f], 0, cap=20)]
    assert a == b


def test_different_weeks_get_different_subsamples(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    rows = [_row("u", f"comment tag{i} with enough distinct words to count")
            for i in range(50)]
    f = _write(tmp_path, "2024-01-01-00.parquet", rows)
    a = [sorted(w) for _, _, w in counts.week_docs([f], 0, cap=20)]
    b = [sorted(w) for _, _, w in counts.week_docs([f], 7, cap=20)]
    assert a != b


# ------------------------------------------------------------------- statistics

def _corpus(T=20, V=400, n_per=4000, seed=0):
    """A stationary corpus: every word has a fixed rate in every week."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.002, 0.25, size=V)
    n = np.full(T, n_per, dtype=np.int64)
    X = rng.binomial(n[:, None], p[None, :]).astype(np.int32)
    return X, n, [f"w{j}" for j in range(V)], p


def test_no_change_gives_z_near_standard_normal():
    X, n, _, _ = _corpus()
    z = detect.zscores(X, n)
    assert abs(z.mean()) < 0.05
    assert 0.9 < z.std() < 1.1


def test_no_change_gives_shift_near_zero():
    X, n, _, _ = _corpus()
    s = detect.strength(detect.zscores(X, n))
    assert abs(np.median(s)) < 2.0


def test_a_group_moving_at_one_boundary_is_the_top_boundary():
    X, n, vocab, p = _corpus()
    cut, group = 12, slice(0, 15)
    rng = np.random.default_rng(7)
    lifted = np.minimum(p[group] * 2.5, 0.9)
    X[cut:, group] = rng.binomial(n[cut:, None], lifted[None, :])

    frame, z = detect.scan(X, n, min_docs=100)
    top = frame.sort("shift", descending=True).row(0, named=True)
    assert top["i"] == cut - detect.HALF          # the cut lands on week `cut`
    assert top["cut"] == fetch.week_start(cut)
    assert top["n_up"] >= 10                      # a group, not a single word


def test_the_moved_words_are_the_ones_reported():
    X, n, vocab, p = _corpus()
    cut, size = 12, 15
    rng = np.random.default_rng(7)
    X[cut:, :size] = rng.binomial(n[cut:, None], np.minimum(p[:size] * 2.5, 0.9))

    frame, z = detect.scan(X, n, min_docs=100)
    m = detect.movers(z, X, n, vocab, cut - detect.HALF, k=size + 5)
    moved = {f"w{j}" for j in range(size)}
    # not all fifteen: a 2.5x lift on a word only two documents in a thousand is
    # genuinely less significant than noise on a common one, and the ranking is
    # by significance. What must hold is that the list is dominated by the group.
    assert len(moved & set(m["word"])) >= size - 3
    assert (m["pct_after"] > m["pct_before"]).all()
    assert set(m["word"][:8]) <= moved


def _shift_odds(p, factor):
    o = p / (1.0 - p) * factor
    return o / (1.0 + o)


def test_a_verbosity_shift_is_not_a_vocabulary_shift():
    """Longer comments raise every common word's document frequency at once.

    That is a change in how much people wrote, not in which words they chose, and
    it must not register -- otherwise the detector finds length changes forever.
    """
    X, n, vocab, p = _corpus(T=24)
    cut = 12
    rng = np.random.default_rng(11)
    X[cut:] = rng.binomial(n[cut:, None], _shift_odds(p, 1.8)[None, :])

    frame, _ = detect.scan(X, n, min_docs=100)
    verbosity = frame.filter(pl.col("i") == cut - detect.HALF)["shift"][0]
    assert verbosity < 4.0, f"verbosity shift leaked through as {verbosity:.1f}"


def test_a_real_group_still_registers_after_the_correction():
    """The same machinery, with a genuine minority of words moving instead."""
    X, n, vocab, p = _corpus(T=24)
    cut, size = 12, 15
    rng = np.random.default_rng(11)
    X[cut:, :size] = rng.binomial(n[cut:, None], _shift_odds(p[:size], 4.0)[None, :])

    frame, _ = detect.scan(X, n, min_docs=100)
    assert frame.sort("shift", descending=True).row(0, named=True)["i"] == cut - detect.HALF
    assert frame.filter(pl.col("i") == cut - detect.HALF)["shift"][0] > 8.0


def test_a_group_moving_under_a_verbosity_shift_is_still_found():
    """The two effects together: the group survives, the offset is removed."""
    X, n, vocab, p = _corpus(T=24)
    cut, size = 12, 15
    rng = np.random.default_rng(5)
    lifted = _shift_odds(p, 1.8)
    lifted[:size] = _shift_odds(p[:size], 1.8 * 4.0)
    X[cut:] = rng.binomial(n[cut:, None], lifted[None, :])

    frame, z = detect.scan(X, n, min_docs=100)
    assert frame.sort("shift", descending=True).row(0, named=True)["i"] == cut - detect.HALF
    m = detect.movers(z, X, n, vocab, cut - detect.HALF, k=size)
    assert len(set(m["word"]) & {f"w{j}" for j in range(size)}) >= size - 3


def test_one_word_moving_is_not_reported_as_a_group():
    X, n, vocab, p = _corpus()
    cut = 12
    rng = np.random.default_rng(3)
    X[cut:, 0] = rng.binomial(n[cut:], min(p[0] * 6, 0.9))
    frame, _ = detect.scan(X, n, min_docs=100)
    assert frame.filter(pl.col("i") == cut - detect.HALF)["n_up"][0] <= 2


def test_the_test_runs_at_every_week_boundary():
    X, n, _, _ = _corpus(T=20)
    frame, _ = detect.scan(X, n, min_docs=100)
    assert len(frame) == 20 - 2 * detect.HALF + 1
    cuts = frame["cut"].to_list()
    assert all((b - a).days == fetch.WEEK_DAYS for a, b in zip(cuts, cuts[1:]))


def test_each_boundary_compares_two_weeks_against_two_weeks():
    X, n, _, _ = _corpus(T=20, n_per=1000)
    frame, _ = detect.scan(X, n, min_docs=100)
    assert frame["docs_before"].to_list() == [2000] * len(frame)
    assert frame["docs_after"].to_list() == [2000] * len(frame)


def test_one_thin_week_does_not_disqualify_a_boundary():
    """Pooling two weeks a side is what makes a single sparse week survivable."""
    X, n, _, _ = _corpus()
    n[5] = 50
    X[5] = np.minimum(X[5], 50)
    frame, _ = detect.scan(X, n, min_docs=1000)
    assert frame.filter(pl.col("i") == 4)["usable"][0]


def test_a_run_of_thin_weeks_is_marked_unusable_not_scored():
    X, n, _, _ = _corpus()
    for w in (5, 6):
        n[w] = 100
        X[w] = np.minimum(X[w], 100)
    frame, _ = detect.scan(X, n, min_docs=1000)
    assert not frame.filter(pl.col("i") == 5)["usable"][0]   # weeks 5,6 are "before"
    assert not frame.filter(pl.col("i") == 3)["usable"][0]   # weeks 5,6 are "after"
    assert frame.filter(pl.col("i") == 10)["usable"][0]


def test_shift_is_scaled_by_the_boundaries_we_trust():
    """A thin bin's noise must not set the yardstick for the rest."""
    X, n, _, _ = _corpus(T=30)
    frame, _ = detect.scan(X, n, min_docs=1000)
    assert abs(np.median(frame.filter("usable")["shift"].to_numpy())) < 0.5


def test_empty_bins_do_not_produce_infinities():
    X, n, _, _ = _corpus()
    n[3] = 0
    X[3] = 0
    frame, z = detect.scan(X, n, min_docs=1000)
    assert np.isfinite(z).all()
    assert frame["S"].is_finite().all()


# -------------------------------------------------------------------- releases

def test_nearest_release_is_signed_relative_to_the_cut():
    name, days = detect.nearest_release(date(2025, 5, 25))
    assert name == "Claude 4 (Opus/Sonnet)"
    assert days == -3                              # shipped three days before


def test_a_distant_release_is_not_offered_as_the_nearest():
    """The 2026 boundaries have no candidate on this calendar, and should say so."""
    assert detect.nearest_release(date(2026, 6, 1)) == ("", 0)


def test_window_reports_the_two_fortnights_compared():
    before, at, after = detect.window(date(2025, 6, 1))
    assert (at - before).days == 14 and (after - at).days == 14


# ----------------------------------------------------------------- factorisation

def _synthetic(T=60, V=300, on=30, size=12, seed=0):
    """A background register, plus one that switches on at week `on`."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.01, 0.30, size=V)
    P = np.tile(base, (T, 1))
    P[on:, :size] *= 4.0                       # the arriving bundle
    n = np.full(T, 20_000, dtype=np.int64)     # large, so sampling noise is small
    X = rng.binomial(n[:, None], np.clip(P, 0, 1)).astype(np.int64)
    return X, n, [f"w{j}" for j in range(V)], size, on


def test_relative_scaling_puts_every_word_on_the_same_footing():
    X, n, vocab, _, _ = _synthetic()
    A, base, kept = factor.relative(X, n, vocab)
    assert np.allclose(A.mean(axis=0), 1.0, atol=1e-9)
    assert len(kept) == len(base) == A.shape[1]


def test_a_word_never_seen_is_dropped_rather_than_dividing_by_zero():
    X, n, vocab, _, _ = _synthetic(V=50)
    X[:, 7] = 0
    A, base, kept = factor.relative(X, n, vocab)
    assert "w7" not in kept and np.isfinite(A).all()


def test_mass_is_a_share():
    X, n, vocab, _, _ = _synthetic()
    f = factor.fit(X, n, vocab, k=4)
    assert abs(factor.mass(f).sum() - 1.0) < 1e-9


def _component_of(f, words, size):
    """Whichever component is actually made of `words`."""
    best, score = 0, -1
    for k in range(f.weight.shape[1]):
        overlap = len(words & set(factor.characteristic(f, k, top=size)["word"]))
        if overlap > score:
            best, score = k, overlap
    return best, score


def test_an_arriving_bundle_becomes_a_component_that_switches_on():
    """The whole premise: a bundle of words arriving at once is a rank-one piece."""
    X, n, vocab, size, on = _synthetic()
    f = factor.fit(X, n, vocab, k=3)
    best, score = _component_of(f, {f"w{j}" for j in range(size)}, size)
    assert score >= size - 2, "no component is made of the arriving words"
    w = f.weight[:, best]
    assert w[:on].mean() < 0.25 * w[on:].mean(), "its weight does not switch on"


def test_the_switch_lands_on_the_right_week():
    """Dated by the arriving component, not by whichever component jumped most: a
    background register can out-jump a real arrival on one noisy week."""
    X, n, vocab, size, on = _synthetic()
    f = factor.fit(X, n, vocab, k=3)
    k, _ = _component_of(f, {f"w{j}" for j in range(size)}, size)
    row = factor.shapes(f).filter(pl.col("k") == k).row(0, named=True)
    assert row["jump_week"] == fetch.week_start(on)


def test_a_background_register_is_on_throughout():
    X, n, vocab, _, _ = _synthetic()
    f = factor.fit(X, n, vocab, k=3)
    sh = factor.shapes(f).sort("mass", descending=True)
    assert sh["off"][0] < 0.1, "the largest component should be the background"


def test_the_curve_has_one_line_per_week():
    X, n, vocab, _, _ = _synthetic(T=40)
    f = factor.fit(X, n, vocab, k=3)
    assert len(factor.curve(f, 0)) == 40


# ------------------------------------------------------- the API sampler as a source

def test_api_windows_are_a_stable_prefix_inside_their_week():
    from shift import apifetch
    from datetime import datetime, timezone
    assert apifetch.windows(5, 3) == apifetch.windows(5, 20)[:3]
    lo = datetime.combine(fetch.week_start(5), datetime.min.time(), timezone.utc)
    hi = datetime.combine(fetch.week_start(6), datetime.min.time(), timezone.utc)
    assert all(lo <= w < hi for w in apifetch.windows(5, 40))


def test_every_week_gets_the_same_number_of_api_windows():
    from shift import apifetch
    assert {len(apifetch.windows(k, 7)) for k in range(20)} == {7}


def test_api_shards_are_grouped_by_the_week_in_their_name(tmp_path, monkeypatch):
    from shift import apifetch
    monkeypatch.setattr(apifetch, "BASE", tmp_path)
    apifetch.raw_dir().mkdir()
    for name in ("2024-01-01-060000-issue", "2024-01-07-231500-issue",
                 "2024-01-08-000000-issue"):
        pl.DataFrame([_row("u", "a body with quite enough distinct words to count")],
                     schema=fetch.SCHEMA).write_parquet(
                         apifetch.raw_dir() / f"{name}.parquet")
    got = apifetch.groups()
    assert sorted(got) == [0, 1]
    assert len(got[0]) == 2 and len(got[1]) == 1


def test_the_api_source_feeds_the_same_counting_path(tmp_path, monkeypatch):
    """Its shards must be readable by exactly the code the archive shards use."""
    from shift import apifetch
    monkeypatch.setattr(apifetch, "BASE", tmp_path)
    apifetch.raw_dir().mkdir()
    pl.DataFrame(
        [_row("alice", "the wall in question here is entirely load-bearing", "a/b"),
         _row("helper[bot]", "an automated note with enough distinct words", "c/d")],
        schema=fetch.SCHEMA,
    ).write_parquet(apifetch.raw_dir() / "2024-01-01-060000-issue.parquet")
    by_week = counts.groups(source="api")
    assert list(by_week) == [0]
    assert len(list(counts.week(by_week[0]))) == 2
    got = counts.support(range(1), ["load-bearing"], source="api")
    assert got["load-bearing"].repos == 1


def test_excluded_and_unexcluded_windows_land_in_different_directories():
    """They enumerate different populations; a directory holding both is neither."""
    from datetime import datetime, timezone
    from shift import apifetch
    t = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    a, b = apifetch.path(t, "issue", False), apifetch.path(t, "issue", True)
    assert a.name == b.name and a.parent != b.parent


def test_the_app_exclusion_list_is_slugs_not_logins():
    """The query qualifier is -author:app/NAME, so the [bot] suffix must not be there."""
    from shift import apifetch
    assert all("[bot]" not in a and "/" not in a for a in apifetch.EXCLUDE_APPS)


def test_issue_and_pr_shards_do_not_share_a_corpus(tmp_path, monkeypatch):
    """Different populations: a directory holding both would be neither."""
    from shift import apifetch
    monkeypatch.setattr(apifetch, "BASE", tmp_path)
    apifetch.raw_dir().mkdir()
    for name in ("2024-01-01-060000-issue", "2024-01-01-070000-pr"):
        pl.DataFrame([_row("u", "a body with quite enough distinct words to count")],
                     schema=fetch.SCHEMA).write_parquet(
                         apifetch.raw_dir() / f"{name}.parquet")
    assert len(apifetch.groups("issue")[0]) == 1
    assert len(apifetch.groups("pr")[0]) == 1
    assert counts.groups(source="api")[0][0].stem.endswith("issue")
    assert counts.groups(source="api:pr")[0][0].stem.endswith("pr")


def test_the_source_string_selects_kind_and_exclusion(tmp_path, monkeypatch):
    from shift import apifetch
    monkeypatch.setattr(apifetch, "BASE", tmp_path)
    row = pl.DataFrame([_row("u", "a body with quite enough distinct words to count")],
                       schema=fetch.SCHEMA)
    for exc, prose, name in ((False, False, "2024-01-01-060000-issue"),
                             (False, False, "2024-01-01-070000-pr"),
                             (True, False, "2024-01-01-080000-pr"),
                             (True, True, "2024-01-01-090000-pr")):
        d = apifetch.raw_dir(exc, prose); d.mkdir(parents=True, exist_ok=True)
        row.write_parquet(d / f"{name}.parquet")
    assert counts.groups(source="api")[0][0].stem.endswith("issue")
    assert counts.groups(source="api:pr")[0][0].parent == apifetch.raw_dir()
    assert counts.groups(source="api-noapps:pr")[0][0].parent == apifetch.raw_dir(True)
    assert counts.groups(source="api-noapps-prose:pr")[0][0].parent == \
        apifetch.raw_dir(True, True)


def test_both_models_find_an_arriving_bundle_on_normalised_input():
    """The normalisation is what makes a rare bundle separable, not the choice of model.

    Only the positive half is asserted. On the real corpus, LDA fitted on raw counts puts
    16% of its mass on the twenty commonest words and never separates the register, while
    the same model on normalised input finds it with 14% of mass -- but a synthetic corpus
    cannot stand in for that, because an injected 4x lift is far stronger than the real
    signal and LDA recovers it from the counts too. Asserting the failure here would be
    asserting something this fixture does not show.
    """
    X, n, vocab, size, on = _synthetic(T=60, V=300, on=30, size=12)
    rng = np.random.default_rng(1)
    X[:, -20:] = rng.binomial(n[:, None], 0.75, size=(len(n), 20))   # function words
    moved = {f"w{j}" for j in range(size)}

    def finds(f):
        return any(len(moved & set(factor.characteristic(f, k, top=size + 6)["word"])) >= 4
                   for k in range(f.weight.shape[1]))

    assert finds(factor.fit(X, n, vocab, k=4)), "NMF on normalised input missed it"
    assert finds(factor.fit_lda(X, n, vocab, k=4, scale="relative")), \
        "LDA on normalised input missed it"


def test_lda_scales_are_both_wired_and_differ():
    X, n, vocab, _, _ = _synthetic(T=40, V=120)
    a = factor.fit_lda(X, n, vocab, k=3, scale="relative")
    b = factor.fit_lda(X, n, vocab, k=3, scale="counts")
    assert a.weight.shape == b.weight.shape == (40, 3)
    assert not np.allclose(a.profile, b.profile)


def test_the_transforms_all_preserve_non_negativity_and_shape():
    X, n, vocab, _, _ = _synthetic(T=30, V=80)
    shapes = set()
    for tr in ("mean", "sqrt", "log"):
        A, base, kept = factor.relative(X, n, vocab, tr)
        assert (A >= 0).all(), tr
        shapes.add(A.shape)
    assert len(shapes) == 1


def test_mean_transform_leaves_every_word_averaging_one():
    X, n, vocab, _, _ = _synthetic(T=30, V=80)
    A, _, _ = factor.relative(X, n, vocab, "mean")
    assert np.allclose(A.mean(axis=0), 1.0, atol=1e-9)


def test_compressing_transforms_shrink_the_dynamic_range():
    """That is the mechanism by which they narrow the register to its rarest words."""
    X, n, vocab, _, _ = _synthetic(T=30, V=80)
    spread = {tr: factor.relative(X, n, vocab, tr)[0].max()
              for tr in ("mean", "sqrt", "log")}
    assert spread["log"] < spread["sqrt"] < spread["mean"]


def test_an_unknown_transform_is_refused():
    X, n, vocab, _, _ = _synthetic(T=20, V=40)
    with pytest.raises(ValueError):
        factor.relative(X, n, vocab, "tfidf")

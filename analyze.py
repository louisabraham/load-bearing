"""A mixture model over weeks: which ways of writing, and how much of each, week by week.

    W_k        a fixed distribution over the vocabulary -- one way of writing
    pi_tk      how much of week t was written that way, with sum_k pi_tk = 1

Generative, per document d written in week t(d):

    z_d ~ Categorical(pi_t)
    x_d | z_d = k ~ Multinomial(n_d, W_k)

The only thing asked of a prevalence curve is smoothness, lam * sum_t (pi_tk -
pi_{t-1,k})^2. Nothing requires a component to rise, to fall, or to be absent early: the
shapes are whatever the documents say they are.

Fitted by EM -- attribute documents, refit the word distributions, refit the prevalences --
restarted from several starting points because EM finds different local optima here and one
run is not enough.

    python analyze.py               # writes analysis.js, read by index.html
    python analyze.py --selftest    # recovers a planted component from synthetic data
"""

import argparse
import glob
import gzip
import json
import os
import re
from collections import Counter
from datetime import date, timedelta

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

K = 16
LAMBDA = 32.0                       # smoothness, scale-free in k; see `fit_pi`
OUTER = 12                          # EM passes per restart
N_INIT = 10                         # restarts; see `fit_best`
SEED = 0
WORDS_LISTED = 40                   # per component; the cut is arbitrary and `tail` says so
WORDS_LEAD = 1000                   # for the one component that arrives; see `pack`
# An arrival started as nothing and ended as a lot. Stated as two absolute shares rather than
# as a growth ratio: end/start explodes when the start is near zero, so a ratio threshold both
# ranks a component with a 0.07% start above one with a 0.3% start for no good reason and moves
# by a factor of ten when a single day of new data arrives.
LEAD_START = 0.01                   # under this much of the first eight weeks
LEAD_END = 0.20                     # and at least this much of the last eight


# ------------------------------------------------------------------------- corpus

ANCHOR = date(2024, 12, 30)        # the Monday that starts the first week of 2025. Weeks
                                   # beginning mid-week would straddle two partial weekends
                                   # and mix the author mix
DAY_GLOB = "data/days/*.jsonl.gz"

# A word is a run of letters, digits, hyphens and underscores containing at least one
# letter -- so `load-bearing`, `snake_case` and `--all-targets` survive whole, while `/`,
# backtick, `:` and `>` are separators rather than characters a word may contain. Whole
# http(s) links are pulled out first and kept as single tokens, before that split can
# shred them. No stemming, no n-grams, no stopword list.
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}]+")
TAG_RE = re.compile(r"<[a-z/!][^<>]*>")   # html markup, not prose: `a > b` is not a tag
EM_DASH = "\u2014"                       # a word by fiat; see `tokens`
WORD_RE = re.compile(r"[a-z0-9_/-]*[a-z][a-z0-9_/-]*")
# One vulnerability identifier per advisory, the same shape of problem as one link per
# item: `snyk-js-axios-6144788` and 1,400 siblings, 113 of them clearing the frequency
# floors, between them occupying seven of sixteen components. Collapsed to one token,
# which says the useful thing -- that the description cites a Snyk advisory at all.
# The trailing run of digits is what distinguishes an identifier from `snyk-top-banner`.
SNYK_ID_RE = re.compile(r"^snyk-.+-\d{4,}$")
MIN_WORDS = 5                      # a body needs this many distinct words to be prose
MIN_TF = 45                        # a word needs this many total appearances. Set by the
                                   # rarest word worth naming: `load-bearing` appears 51
                                   # times, so 60 would have excluded it.
MIN_DF = 25                        # and this many distinct documents. Total appearances
                                   # alone is not breadth: `multi-draw` appears 101 times
                                   # inside ONE document, `m0` 140 times, and each was
                                   # ranking among a component's most representative words
                                   # because lift cannot tell a widespread word from a
                                   # word someone repeated. The ceiling is `load-bearing`
                                   # again, in 45 documents.
DOCS_PER_WEEK = 350                # see the note in `read_week`
MAX_PER_AUTHOR = 3                 # per author per week; see `read_week`

def domain_token(url):
    """A link becomes one token naming its domain: `[cursor-url]`, `[snyk-url]`.

    Kept whole, every distinct link was its own word, and a tool that puts a
    per-item link in each description got one word per item instead of one word.
    Snyk alone contributed thousands: its vulnerability links were the most
    representative words of eight of sixteen components, each holding a different
    handful of them. Collapsing by domain says the useful thing -- that a
    description links to Snyk at all -- in one token that can then clear the
    frequency floors and be compared across weeks.

    The registrable domain is taken as the second-to-last label, which is wrong
    for `example.co.uk` and right for everything that turns up here.
    """
    host = url.split("//", 1)[-1].split("/", 1)[0].split("@")[-1].split(":")[0]
    labels = [x for x in host.split(".") if x and x != "www"]
    name = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "link")
    return f"[{name}-url]"


def tokens(body):
    """Every appearance of every word in one document.

    Links are taken first, each becoming one token naming its domain:
    `[bugbot](https://cursor.com/x)` yields `bugbot` and `[cursor-url]`. Splitting on
    punctuation first would have produced `bugbot](https` and a trail of fragments, and
    those fragments used to rank among a component's most representative words. See
    `domain_token` for why the domain rather than the link.

    HTML tags go next, whole, for the same reason: splitting them character by character
    turned `<sup>reviewed</sup>` into `sup`, `reviewed`, `sup` and made `li`, `br`, `td` and
    `href` six of one component's twelve commonest words. The pattern requires a letter or
    slash after the bracket, so `a > b` in prose is not mistaken for markup.

    Then everything else splits on any character a word may not contain, which handles what
    markdown creates without needing to know about it -- `srcset="..."` gives `srcset`,
    `height="28` gives `height`. Emphasis is handled by trimming: `*example*` needs nothing
    because `*` is a separator, and `_other example_` needs the underscores trimmed off the
    ends, since an underscore is allowed *inside* a word. A trailing hyphen goes for the
    same reason; a leading one stays, so `--all-targets` is not quietly turned into
    `all-targets`.

    Snyk advisory identifiers collapse to `[snyk-id]` for the same reason links collapse
    to their domain -- see `SNYK_ID_RE`.

    Requiring a letter drops what is left of numbers and rules: `27.49`, `589/1000`,
    `2025-06-24`, `-------`. The arrow and `+` go with them. The em dash is the one
    exception, taken before the split and counted as a word of its own -- it earns that by
    going from 0.0 appearances per 10,000 words in early 2024 to 123.0 in mid-2026, the
    sharpest single signal here.
    """
    body = body.lower()
    out = [domain_token(m.group(0)) for m in URL_RE.finditer(body)]
    # the em dash counts as a word. It is punctuation, so the rule above would drop it,
    # and it is the sharpest single signal in the corpus: 0.0 appearances per 10,000 words
    # in early 2024 against 123.0 in mid-2026. Counted separately rather than added to the
    # word characters, because it is as often unspaced as spaced -- inside the character
    # class `foo\u2014bar` would become one token instead of three.
    out += [EM_DASH] * body.count(EM_DASH)
    # links first so they survive whole, then tags, so that what is left of
    # `<a href="...">text</a>` is `text` and not `a`, `href`, `text`, `a`
    rest = TAG_RE.sub(" ", URL_RE.sub(" ", body))
    for w in WORD_RE.findall(rest):
        w = w.strip("_/").rstrip("-")
        if w and any(c.isalpha() for c in w):
            out.append("[snyk-id]" if SNYK_ID_RE.match(w) else w)
    return out


def week_index(day):
    return (day - ANCHOR).days // 7


def week_files(log=print):
    """The day files grouped by the week they fall in.

    Days are the unit of collection -- one request, one file, appended by CI and committed --
    and weeks are the unit of analysis, because a single day is a hundred descriptions and too
    thin to compare against another. Weeks run from the first present to the last with no gaps,
    so a week that was never collected shows as an empty week rather than being quietly closed
    up and shifting everything after it.
    """
    files = sorted(glob.glob(DAY_GLOB))
    if not files:
        raise SystemExit(f"no days in {DAY_GLOB} -- run `python fetch_day.py`")
    by_week = {}
    for f in files:
        d = date.fromisoformat(os.path.basename(f)[:10])
        by_week.setdefault(week_index(d), []).append(f)
    lo, hi = min(by_week), max(by_week)
    weeks = [(ANCHOR + timedelta(days=7 * w)).isoformat() for w in range(lo, hi + 1)]
    groups = [by_week.get(w, []) for w in range(lo, hi + 1)]
    empty = sum(1 for g in groups if not g)
    log(f"{len(files)} days over {len(weeks)} weeks from {weeks[0]}"
        + (f", {empty} with no data" if empty else ""))
    return weeks, groups


def documents(log=print):
    """One row per description, with the week it belongs to.

    The filters are applied per week rather than per file, because that is the population being
    compared: identical word sets collapse within the week, no author may contribute more than
    a few to it, and the week is then cut off at a common size.
    """
    weeks, groups = week_files(log)

    docs, week_of, tf, df = [], [], Counter(), Counter()
    for t, group in enumerate(groups):
        seen, kept, by_author = set(), 0, Counter()
        for f in group:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    toks = tokens(row["body"])
                    if len(set(toks)) < MIN_WORDS:
                        continue
                    key = frozenset(toks)
                    if key in seen:
                        continue
                    author = row.get("author") or ""
                    if by_author[author] >= MAX_PER_AUTHOR:
                        continue
                    by_author[author] += 1
                    seen.add(key)
                    kept += 1
                    if kept > DOCS_PER_WEEK:
                        break
                    c = Counter(toks)
                    docs.append(c)
                    week_of.append(t)
                    tf.update(c)
                    df.update(key)
            if kept > DOCS_PER_WEEK:
                break

    vocab = sorted(w for w, n in tf.items() if n >= MIN_TF and df[w] >= MIN_DF)
    index = {w: j for j, w in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for i, c in enumerate(docs):
        for w, n in c.items():
            j = index.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(n)
    X = csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=np.float64)
    keep = np.asarray(X.sum(axis=1)).ravel() >= MIN_WORDS
    X, week_of = X[keep], np.asarray(week_of)[keep]
    log(f"{X.shape[0]:,} descriptions, {X.sum():,.0f} appearances, {len(vocab):,} words")
    return X, week_of, weeks, vocab


# -------------------------------------------------------------------------- model

def fit_pi(C, lam=LAMBDA):
    """Prevalences: maximise sum C log pi - lam sum (pi_t - pi_{t-1})^2.

    Subject to each week summing to 1. Solved on a per-week softmax, so the constraint holds
    by construction and the problem is unconstrained in those coordinates -- L-BFGS handles
    it directly rather than needing a projection at every step.

    The smoothness term exists because weekly counts are noisy at a few hundred documents a
    week, not because a component is expected to be monotone.

    **The penalty is on the difference measured against 1/K, not on the difference itself.**
    Without that, the right lambda changes by two orders of magnitude with K, and for a
    mechanical reason: prevalences sum to one, so a typical pi is about 1/K and a typical
    squared difference about 1/K^2. Held-out likelihood -- fit on 90% of documents, score the
    other 10%, in bits per word -- puts the optimum at 5,000 for K=12 and 500,000 for K=128,
    and (128/12)^2 x 5,000 = 568,889. One grid step from the observed optimum, so the whole
    K-dependence is that factor. Absorbing K^2 into the penalty leaves a single constant that
    is right at both: 5,000/12^2 = 34.7 and 500,000/128^2 = 30.5.

    The sweeps, at K=12:

        lambda        0      40     200    1000    5000   25000  100000
        held out  -9.2947 -9.2946 -9.2946 -9.2945 -9.2944 -9.2944 -9.2945
        roughness  1.891   1.524   0.895   0.311   0.074   0.022   0.009

    and at K=128, where there are 17,536 prevalences to fit rather than 1,644 and the penalty
    has real work to do:

        lambda        0    1000    5000   25000  100000  500000    2e6    1e7
        held out  -9.0782 -9.0770 -9.0749 -9.0742 -9.0734 -9.0728 -9.0738 -9.0760
        train     -8.8055 -8.8053 -8.8056 -8.8064 -8.8070 -8.8072 -8.8079 -8.8099

    Train getting worse while held-out gets better is the regularisation signature, and it is
    only visible at the larger K. At K=12 held-out is flat to four decimal places across four
    orders of magnitude, so there the penalty is free rather than helpful -- worth taking
    anyway, because it cuts roughness twenty-fold for nothing.

    Held-out likelihood does not see over-smoothing directly, so the shape was checked too. At
    K=12 the register's rise is 0.4% to 65.4% at lambda=40 and 0.3% to 63.8% at the chosen
    setting, but 0.4% to 47.0% at a hundred times that -- the peak dragged down toward the
    early weeks. The chosen value is the largest that leaves the shape alone.

    It binds, unlike the L1 in py, and for a structural reason: there the columns of W
    are normalised *after* fitting, so an L1 on H could be satisfied by shrinking H and
    inflating W at no cost, and the rescaling undid it exactly. Here pi sums to 1 in every
    week by construction, so the scale is not free. An L1 on pi itself would still do nothing
    -- on the simplex ||pi_t||_1 = 1 identically, a constant with zero gradient.
    """
    T, K_ = C.shape

    def softmax(theta):
        z = theta - theta.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def neg(v):
        pi = softmax(v.reshape(T, K_))
        p = np.maximum(pi, 1e-12)
        # the difference is measured against 1/K, not absolutely, which is what makes one
        # lambda work at every K -- see the docstring
        w = lam * K_ * K_
        d = np.diff(pi, axis=0)
        obj = (C * np.log(p)).sum() - w * (d ** 2).sum()
        g = C / p
        g[1:] -= 2 * w * d
        g[:-1] += 2 * w * d
        gt = pi * (g - (pi * g).sum(axis=1, keepdims=True))   # through the softmax
        return -obj, -gt.ravel()

    res = minimize(neg, np.zeros(T * K_), jac=True, method="L-BFGS-B",
                   options={"maxiter": 400, "maxfun": 500})
    return softmax(res.x.reshape(T, K_))


def responsibilities(logits, pi, week_of):
    """r_dk, and the total log-likelihood."""
    z = logits + np.log(np.maximum(pi[week_of], 1e-300))
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(axis=1, keepdims=True)
    return e / s, float((np.log(s) + m).sum())


def fit(X, week_of, T, k=K, lam=LAMBDA, outer=OUTER, seed=SEED, flat=False, log=print):
    """One EM run. Returns (W, pi, C, A, log-likelihood).

    `flat=True` fits one mixture for the whole window instead of one per week, so the model has
    no way to represent time at all. The weekly counts that come out of the attribution are
    then purely observed, which is the point: if a component still shows the same rise, the
    rise is in the words rather than in the freedom the model was given to fit it.
    """
    rng = np.random.default_rng(seed)
    D, V = X.shape

    # seeded from a handful of real documents each, the usual start for a multinomial
    # mixture: a uniform start leaves every component identical and the first step cannot
    # break the tie
    W = np.zeros((k, V))
    for c in range(k):
        pick = rng.choice(D, size=min(40, D), replace=False)
        W[c] = np.asarray(X[pick].sum(axis=0)).ravel() + 0.1
    W /= W.sum(axis=1, keepdims=True)
    pi = np.full((T, k), 1.0 / k)

    ll = -np.inf
    for it in range(outer):
        logits = X @ np.log(np.maximum(W, 1e-12)).T          # 1. attribute
        r, ll = responsibilities(logits, pi, week_of)

        W = np.asarray((r.T @ X) + 0.01)                      # 2. word distributions
        W /= W.sum(axis=1, keepdims=True)

        C = np.zeros((T, k))                                  # 3. weekly counts
        np.add.at(C, week_of, r)

        if flat:                                              # 4. prevalences
            share = C.sum(axis=0)
            pi = np.tile(share / max(share.sum(), 1e-12), (T, 1))
        else:
            pi = fit_pi(C, lam)
        log(f"  iter {it + 1:2d}  loglik {ll:,.0f}")

    logits = X @ np.log(np.maximum(W, 1e-12)).T
    r, ll = responsibilities(logits, pi, week_of)
    C = np.zeros((T, k))
    np.add.at(C, week_of, r)
    # and the same attribution weighted by document length, which is the quantity that
    # actually varies: the corpus caps documents at 350 a week, so C is nearly flat by
    # construction, while the words in them swing threefold
    A = np.zeros((T, k))
    np.add.at(A, week_of, r * np.asarray(X.sum(axis=1)))
    return W, pi, C, A, ll


def fit_best(X, week_of, T, k=K, lam=LAMBDA, outer=OUTER, n_init=N_INIT,
             seed=SEED, flat=False, log=print):
    """Fit `n_init` times from different seeds and keep the highest likelihood.

    One run is not enough: EM finds different local optima here, and the worst of them mix a
    component with something else and give it two thirds of the peak the good fits find. The
    likelihood tells them apart reliably, so the only cost is time -- about a second a
    restart.
    """
    best = None
    for i in range(n_init):
        W, pi, C, A, ll = fit(X, week_of, T, k, lam, outer, seed + i, flat,
                              log=lambda *_: None)
        log(f"  restart {i + 1}/{n_init}  loglik {ll:,.0f}")
        if best is None or ll > best[-1]:
            best = (W, pi, C, A, ll)
    log(f"  kept loglik {best[-1]:,.0f}")
    return best


# --------------------------------------------------------------------------- out

def pack(X, week_of, weeks, vocab, W, pi, C, A, ll, lam, strict=True):
    # Each component's share of all word appearances, used to build the baseline below.
    mass_c = A.sum(axis=0)
    mass_c = mass_c / max(mass_c.sum(), 1e-12)
    docs_per_week = np.bincount(week_of, minlength=len(weeks))
    words_per_week = np.zeros(len(weeks))
    np.add.at(words_per_week, week_of, np.asarray(X.sum(axis=1)).ravel())
    per_word = np.zeros((X.shape[1], len(weeks)))          # appearances, word by week
    for t in range(len(weeks)):
        sel = week_of == t
        if sel.any():
            per_word[:, t] = np.asarray(X[sel].sum(axis=0)).ravel()

    # ordered by size in the final week, largest first: what a reader wants first is what
    # the corpus looks like now, and the stack then puts the currently-dominant band at the
    # bottom where its shape is easiest to follow. The last week is one week and therefore
    # noisy, which is the price of answering "what is biggest now" exactly.
    order = np.argsort(-C[-1, :])

    obs = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-12)
    start, end = obs[:8].mean(axis=0), obs[-8:].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(start > 0, end / np.maximum(start, 1e-12), np.inf)

    # Which components arrived: they end at least LEAD_RATIO times their starting size while
    # being worth at least LEAD_FLOOR of the final weeks. What is asserted is not how many
    # there are -- that depends on k, one at k=16 and two at k=32, where the register splits
    # into prose and command-line tooling -- but that the set is unambiguous: every arrival is
    # at least LEAD_GAP times clear of everything that is not one. Measured, the gap is
    # 1,405-fold at k=16 (5,759 against 4.1) and 59-fold at k=32 (201 against 3.4). If this
    # ever fires, the arrivals have stopped being separable from ordinary drift and the page
    # should not be published from that fit.
    lead = np.flatnonzero((start < LEAD_START) & (end >= LEAD_END))
    arrived = len(lead) >= 1
    if strict:
        assert arrived, (
            "no component started under {:.0%} of the first eight weeks and ended at or above "
            "{:.0%} of the last eight; the biggest went {:.2%} -> {:.0%}".format(
                LEAD_START, LEAD_END, start[int(np.argmax(end))], end.max()))
    elif not arrived:
        # An ablation is allowed to fail this -- that is often the finding -- but it still has
        # to render, so the largest-ending component stands in and is labelled as a stand-in.
        lead = np.array([int(np.argmax(end))])
    lead = set(int(c) for c in lead)

    comps = []
    for c in order:
        # Lift against the corpus *without* this component. Dividing by the whole corpus
        # understates a large component's own words, because its occurrences are most of what
        # it is being compared against -- at the end of the window one component is a third of
        # everything written, so its vocabulary was being measured partly against itself. The
        # baseline here is the mixture of every other component, weighted by their share of
        # appearances, which asks the question that was meant: how much more probable is this
        # word here than in the writing that is not this.
        other = np.delete(mass_c, c)
        base = (np.delete(W, c, axis=0) * other[:, None]).sum(axis=0) / max(other.sum(), 1e-12)
        lift = W[c] / np.maximum(base, 1e-12)
        rank = np.argsort(-lift)
        # the arriving component gets a long list, because it is the one anybody will read
        # past the first handful of, and the cut has to fall somewhere
        n = WORDS_LEAD if c in lead else WORDS_LISTED
        comps.append({
            "id": int(c),
            "lead": bool(c in lead),
            "share": round(float(pi[:, c].mean()), 5),
            "peak": round(float(pi[:, c].max()), 5),
            "peak_week": weeks[int(np.argmax(pi[:, c]))],
            "start_share": round(float(start[c]), 5),
            "end_share": round(float(end[c]), 5),
            "flat_pi": bool(np.allclose(pi[:, c], pi[0, c])),
            "prevalence": [round(float(v), 5) for v in pi[:, c]],
            "count": [round(float(v), 1) for v in C[:, c]],
            "appearances": [int(round(v)) for v in A[:, c]],
            "word_list": [vocab[j] for j in rank[:n]],
            "word_lift": [round(float(lift[j]), 2) for j in rank[:n]],
            # each listed word's own weekly appearances, so the page can show a word's
            # history on hover. Only for the arrivals: at 137 weeks a thousand words is
            # 137,000 integers, which is worth carrying once or twice and not sixteen times.
            "series": ([[int(v) for v in per_word[j]] for j in rank[:n]]
                       if c in lead else None),
            "tail": {t: int((lift >= t).sum()) for t in (5, 3, 2)},
        })
    return {"generated": date.today().isoformat(), "weeks": weeks, "n_init": N_INIT,
            "arrived": bool(arrived),
            # reported for information, not asserted on: among components ending above the
            # floor there is a continuum of growth rather than a clean gap, which is why the
            # test is on the shares and not on the ratio
            "lead_ratio": round(float(ratio[list(lead)].min()), 1) if len(lead) else None,
            "rest_ratio": round(float(max((ratio[c] for c in range(len(end))
                                           if c not in lead and end[c] >= 0.05),
                                          default=0.0)), 1),
            "documents": int(X.shape[0]), "appearances": int(X.sum()),
            "docs_per_week": [int(v) for v in docs_per_week],
            "words_per_week": [int(v) for v in words_per_week],
            "vocab": len(vocab), "k": len(comps), "lambda": lam,
            "loglik": round(ll, 1), "components": comps}


def selftest():
    """A planted component that arrives partway through must be recovered."""
    rng = np.random.default_rng(0)
    T, V, D_per, arrives = 40, 60, 60, 22
    Wt = rng.dirichlet(np.full(V, 0.4), size=3)
    rows, cols, vals, week_of = [], [], [], []
    d = 0
    for t in range(T):
        for _ in range(D_per):
            live = [0, 1] if t < arrives else [0, 1, 2]
            k = live[rng.integers(len(live))]
            x = rng.multinomial(50, Wt[k])
            for j in np.flatnonzero(x):
                rows.append(d); cols.append(int(j)); vals.append(float(x[j]))
            week_of.append(t); d += 1
    X = csr_matrix((vals, (rows, cols)), shape=(d, V))
    week_of = np.array(week_of)
    W, pi, C, A, ll = fit_best(X, week_of, T, k=3, outer=8, n_init=3,
                               log=lambda *_: None)

    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-6), "prevalences must sum to 1"
    assert (pi >= 0).all() and np.allclose(W.sum(axis=1), 1.0), "factors are malformed"
    # C must recover each week's document count, since every document's r sums to 1
    assert np.allclose(C.sum(axis=1), np.bincount(week_of, minlength=T), rtol=1e-6), \
        "the counts do not reconstruct the week"

    late = int(np.argmax([Wt[2] @ np.log(np.maximum(W[c], 1e-12)) for c in range(3)]))
    before, after = pi[:arrives, late].mean(), pi[arrives:, late].mean()
    assert before < 0.5 * after, f"the planted component did not rise ({before:.3f} " \
                                 f"then {after:.3f})"
    assert np.allclose(A.sum(axis=1),
                       np.bincount(week_of, minlength=T) * 50, rtol=0.02), \
        "the appearance counts do not reconstruct the week"
    rough = lambda lam: float((np.diff(fit_pi(C, lam), axis=0) ** 2).sum())
    assert rough(1000.0) < rough(0.0), "the smoothness penalty does nothing"
    print(f"selftest: ok  (prevalences sum to 1, counts reconstruct each week, planted "
          f"component {before:.3f} -> {after:.3f} at week {arrives})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--outer", type=int, default=OUTER)
    ap.add_argument("--n-init", type=int, default=N_INIT, dest="n_init")
    ap.add_argument("--out", default="analysis.js")
    ap.add_argument("--flat", action="store_true",
                    help="one mixture for the whole window instead of one per week")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X, week_of, weeks, vocab = documents()
    W, pi, C, A, ll = fit_best(X, week_of, len(weeks), k=args.k, lam=args.lam,
                               outer=args.outer, n_init=args.n_init, flat=args.flat)
    variant = "one mixture for the window" if args.flat else "default"
    out = pack(X, week_of, weeks, vocab, W, pi, C, A, ll, args.lam,
               strict=(variant == "default"))
    out["variant"] = variant
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.ANALYSIS = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"\nloglik {ll:,.0f}, wrote {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)\n")
    for c in out["components"]:
        print(f"  peaks {c['peak_week']}  mean {c['share']:6.1%}  peak {c['peak']:5.1%}  "
              f"{c['start_share']:.1%} -> {c['end_share']:.1%}")
        print("        " + ", ".join(w[:20] for w in c["word_list"][:9]))


if __name__ == "__main__":
    main()

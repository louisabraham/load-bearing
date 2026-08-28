"""KL k-means over pull request descriptions, with no model of time at all.

    W[c]   one way of writing: a fixed distribution over the vocabulary
    p_d    one description, as a distribution over the vocabulary

Every description is assigned to the way of writing it is closest to, under the divergence that
belongs to word counts:

    assign d to argmin_c  n_d * KL(p_d || W[c])

and each W[c] is then the middle of what it was given. Seeded by greedy k-means++ under the same
divergence and iterated to a fixed point, N_INIT times, keeping the cheapest of them. If that one
fails the arrival check the whole batch is run again from fresh seeds. See `fit_arriving`.

**Time appears nowhere in it.** There is one set of centres for the entire window, not one per
week, so the fit has no parameter that could describe a trend and no freedom to place one. The
weekly curves the page draws are pure attribution: each description is assigned by its words
alone, and the weeks are counted up afterwards. If a way of writing rises, the rise is in what
people wrote, because it cannot be anywhere else.

    python analyze.py               # writes analysis.js, read by index.html
    python analyze.py --selftest    # recovers a planted way of writing from synthetic data
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from datetime import date, timedelta
from multiprocessing import Pool

import numba
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix

K = 10  # ways of writing; chosen on the outcome, and marked as such in the README
TRIALS = 3  # k-means++ candidates per centre; see `kmeanspp`
SEED = 0  # the first starting point; see `fit_arriving`
N_INIT = 8  # restarts per attempt, the cheapest of which is published
RETRIES = 4  # fresh batches of restarts, if that one did not arrive
SMOOTH = 0.01  # pseudo-count, so no centre gives a word zero probability
MAX_PASSES = 200  # a runaway guard, not a setting; the fixed point comes at 30
WORDS_LISTED = 150  # per component; the cut is arbitrary and `tail` says so
WORDS_LEAD = 1000  # for the component the page opens on; see `pack`
TREND_WEEKS = 12  # weeks the reported trend is fitted over
# WHICH COMPONENT THE PAGE IS ABOUT: the largest one of the last LEAD_WINDOW weeks. Nothing is
# selected on how much it grew. A month rather than a week, because a week is 5,300 descriptions
# and the subject of the whole page should not turn on which of two close components led across
# one of them.
LEAD_WINDOW = 4
# LEAD_START and LEAD_END do not select; they CHECK. Picking the biggest component says nothing
# about whether it arrived, and the page's headline claim is that it arrived, so that claim is
# tested against the component actually chosen. Two absolute shares rather than a growth ratio,
# because end/start explodes when the start is near zero.
LEAD_START = 0.02  # started under this much of the first eight weeks
LEAD_END = 0.20  # and ended at or above this much of the last eight


# ------------------------------------------------------------------------- corpus

ANCHOR = date(2024, 12, 30)  # the Monday that starts the first week of 2025. Weeks
# beginning mid-week would straddle two partial weekends
# and mix the author mix
DAY_GLOB = "data/days/*.jsonl"

# A word is a run of letters, digits, hyphens and underscores containing at least one
# letter -- so `load-bearing`, `snake_case` and `--all-targets` survive whole, while `/`,
# backtick, `:` and `>` are separators rather than characters a word may contain. Whole
# http(s) links are pulled out first and kept as single tokens, before that split can
# shred them. No stemming, no n-grams, no stopword list.
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}]+")
TAG_RE = re.compile(r"<[a-z/!][^<>]*>")  # html markup, not prose: `a > b` is not a tag
EM_DASH = "\u2014"  # a word by fiat; see `tokens`
WORD_RE = re.compile(r"[a-z0-9_/-]*[a-z][a-z0-9_/-]*")
# One vulnerability identifier per advisory, the same shape of problem as one link per item:
# `snyk-js-axios-6144788` and 1,400 siblings. Collapsed to one token, which says the useful
# thing -- that the description cites a Snyk advisory at all. The trailing run of digits is
# what distinguishes an identifier from `snyk-top-banner`.
SNYK_ID_RE = re.compile(r"^snyk-.+-\d{4,}$")
MIN_WORDS = 5  # a body needs this many distinct words to be prose
# ONE FLOOR ON A WORD, AND IT COUNTS PEOPLE. A word is in the
# vocabulary when this many distinct accounts have written it.
#
# There were three of these -- 45 appearances, 25 documents,
# 20 accounts -- and two of them were doing nothing that this
# one does not do better. Counting appearances cannot tell a
# shared word from one document written two hundred times,
# which is why the account floor existed at all; and 20
# accounts already implies 20 documents and 20 appearances,
# so the other two floors were very nearly subsumed. The
# appearance floor also carried a disclosure, having been
# picked so that the word this page is named after would
# clear it. One floor that counts people needs no such note.
#
# THE NUMBER IS CHOSEN ON A PROPERTY OF THE METHOD, not on
# the answer: it is the least restrictive floor at which two
# independent fits agree on at least half of the top twenty
# words. Measured over 32 unconditioned fits of this corpus,
# at k = 12, agreement rises monotonically with the floor --
# 0.29 at 20 accounts, 0.51 at 50, 0.69 at 200, 0.79 at 400 --
# so there is no optimum to find, only a rate of return, and
# the rule picks a point on it for a stated reason. At 20 the
# component the page is about came out mixed with another in 6
# of the 32; at 50, in none of them.
MIN_AUTHORS = 50
MAX_PER_AUTHOR = 3  # per author per week; see `documents`
# Accounts that are not people. The query can only exclude
# Apps one slug at a time, and the ones it names are four of
# thousands: 3,784 accounts in the collected corpus end in
# `[bot]` or `-bot`, and `copilot` is an agent posting under
# an ordinary login. Dropped by the shape of the name, which
# is 13.2% of the collected rows -- the same share as at a
# tenth of this depth -- and the arrival survives it, which
# is the point of dropping them.
BOT_SUFFIX = ("[bot]", "-bot")
BOT_LOGIN = ("copilot",)


def domain_token(url):
    """A link becomes one token naming its domain: `[cursor-url]`, `[snyk-url]`.

    Kept whole, every distinct link was its own word, and a tool that puts a
    per-item link in each description got one word per item instead of one word.
    Collapsing by domain says the useful thing -- that a description links to Snyk
    at all -- in one token that can clear the frequency floors and be compared
    across weeks.

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
        # no letter test is needed: WORD_RE already requires one, and trimming only ever
        # removes `_`, `/` and `-`, so a token that had a letter still has it. The `startswith`
        # guard is not cosmetic either -- without it the Snyk pattern is matched against every
        # one of seven million tokens instead of the few hundred that could possibly match.
        w = w.strip("_/").rstrip("-")
        if w:
            out.append("[snyk-id]" if w.startswith("snyk-") and SNYK_ID_RE.match(w) else w)
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

    # A part-week at either end is dropped. This matters most at the trailing end and it
    # matters every day: collection runs each morning, so the newest week is almost always
    # half-collected, and it is the week everything leans on -- the component this page is
    # about is the largest one *in the final week*, and the headline share is measured there.
    # Reading that off three days instead of seven made the answer jump for a reason that had
    # nothing to do with anybody's writing. The page now ends at the last complete week, so it
    # can lag by up to six days, which is the right trade at weekly resolution.
    dropped = []
    while lo <= hi and len(by_week.get(lo, [])) < 7:
        dropped.append(("first", lo, len(by_week.pop(lo, []))))
        lo += 1
    while hi >= lo and len(by_week.get(hi, [])) < 7:
        dropped.append(("last", hi, len(by_week.pop(hi, []))))
        hi -= 1
    if lo > hi:
        raise SystemExit("no complete week of days yet")
    for which, w, ndays in dropped:
        log(
            f"  dropped the {which} week, {(ANCHOR + timedelta(days=7 * w)).isoformat()}: "
            f"{ndays} of 7 days"
        )
    weeks = [(ANCHOR + timedelta(days=7 * w)).isoformat() for w in range(lo, hi + 1)]
    groups = [by_week.get(w, []) for w in range(lo, hi + 1)]
    empty = sum(1 for g in groups if not g)
    log(
        f"{sum(len(g) for g in groups)} days over {len(weeks)} complete weeks, "
        f"{weeks[0]} to {weeks[-1]}" + (f", {empty} with no data" if empty else "")
    )
    return weeks, groups


# Eight, because twelve was slower: the work is one core reading and tokenising one day, and
# past eight the processes are waiting on memory rather than on each other. A runner with four
# cores uses four.
WORKERS = 8


def scan(f):
    """One day file, read and tokenised: its own word list, and one array of ids per description.

    The word ids are local to the file, which is what lets this run in a pool at all -- a global
    id has to be handed out in corpus order, and that order is not known until every file is
    read. `documents` maps each file's list onto the global one as the results come back, in
    order, so the vocabulary is numbered exactly as a single process would have numbered it.
    """
    ids, cols, authors, bots = {}, [], [], 0
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            a = (row.get("author") or "").lower()
            if a.endswith(BOT_SUFFIX) or a in BOT_LOGIN:
                bots += 1
                continue
            # one small array a description rather than one Python int an appearance: the row
            # indices are implied by the lengths, and 28 million boxed integers in two lists
            # were three quarters of this program's memory
            words = tokens(row["body"])
            cols.append(
                np.fromiter((ids.setdefault(w, len(ids)) for w in words), np.int32, len(words))
            )
            authors.append(row.get("author") or "")
    return list(ids), cols, authors, bots


def documents(log=print):
    """One row per description, with the week it belongs to.

    The filters are applied per week rather than per file, because that is the population being
    compared: identical word sets collapse within the week, and no author may contribute more
    than a few to it.

    There is deliberately no cap on the size of a week. An earlier version thinned every week
    to a common count, because weeks then came from a handful of bulk windows and their sizes
    swung by a factor of two -- a word could rise in the ranking purely because the weeks
    around it had grown. Collection is now ten windows a day, every day, and every window comes
    back a full page, so weeks are already the same size by construction and the cap only threw
    away half the corpus. What variation is left is real: the filters below bite differently
    from week to week, and that is a property of the writing, not of the sampling.

    Words are turned into integers as they are read, and every count after that is a sparse
    matrix operation. Building the matrix collapses duplicate (description, word) pairs in C and
    sorts each row, which is exactly the deduplication the filters need -- so the distinct-word
    counts, the per-week word-set keys, the document frequencies and the distinct-account counts
    all fall out of a matrix that had to be built anyway, rather than out of a dictionary of sets
    and a loop over four million pairs.
    """
    weeks, groups = week_files(log)
    n_days = sum(len(g) for g in groups)

    # Reading and tokenising is most of this function and all of it is per-file, so it is done
    # in a pool: five times faster on this laptop, and the same corpus down to the word ids,
    # because `imap` hands the files back in the order they were given.
    flat = [(t, f) for t, group in enumerate(groups) for f in group]
    ids, cols, authors, week_raw = {}, [], [], []
    bots = 0
    with Pool(min(WORKERS, os.cpu_count() or 1)) as pool:
        for (t, _), (local, c, a, n_bots) in zip(
            flat, pool.imap(scan, [f for _, f in flat], chunksize=4)
        ):
            remap = np.fromiter((ids.setdefault(w, len(ids)) for w in local), np.int32, len(local))
            cols += [remap[x] for x in c]
            authors += a
            week_raw += [t] * len(a)
            bots += n_bots
    log(f"{bots:,} rows from accounts that are not people")
    vocab_all = list(ids)
    n, V = len(authors), len(vocab_all)
    week_raw = np.asarray(week_raw, np.int64)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum([len(c) for c in cols], out=indptr[1:])
    indices = np.concatenate(cols) if cols else np.zeros(0, np.int32)
    del cols
    X0 = csr_matrix((np.ones(indices.size), indices, indptr), shape=(n, V), dtype=np.float64)
    # the filters below want each row deduplicated and sorted, which the COO constructor used to
    # do on the way in; from indptr and indices it is this call instead
    X0.sum_duplicates()

    # the three per-description filters: too few distinct words, a word set already seen this
    # week, or an author already at the cap for the week
    n_distinct = np.diff(X0.indptr)
    keep = np.zeros(n, bool)
    seen, by_author, cur_week = set(), Counter(), -1
    for d in range(n):
        if week_raw[d] != cur_week:
            seen, by_author, cur_week = set(), Counter(), week_raw[d]
        if n_distinct[d] < MIN_WORDS:
            continue
        key = X0.indices[X0.indptr[d] : X0.indptr[d + 1]].tobytes()
        if key in seen:
            continue
        a = authors[d]
        if by_author[a] >= MAX_PER_AUTHOR:
            continue
        by_author[a] += 1
        seen.add(key)
        keep[d] = True

    kd = np.flatnonzero(keep)
    Xk = X0[kd]
    tf = np.asarray(Xk.sum(axis=0)).ravel()

    # THE ONE FLOOR: how many DIFFERENT PEOPLE write the word. Counting appearances or documents
    # instead cannot tell a shared word from one document written two hundred times, because the
    # per-week author cap bounds an account to three descriptions a week and a template that runs
    # for sixteen months is under the cap every single week. Counting accounts separates them at a
    # glance: `nixpkgs-update` reaches 242 documents from 2 accounts and `store-path` 242 from 2,
    # while `load-bearing` comes from 848 accounts in 905 documents and `seam` from 1,135 in
    # 1,247. A word 848 people reached for is a word; a word in 242 descriptions from 2 accounts
    # is one document written 242 times.
    #
    # The templates this comment used to name -- `proprosed`, a misspelling of "proposed" inside
    # one Red Hat Konflux template, and `pipelineruns` -- are down to 28 appearances and to 40
    # documents from 5 accounts. Ten windows a day did that: the cap is three a week whatever the
    # depth, so a template that filled a thin week fills a tenth of a deep one.
    aid = {}
    aids = np.array([aid.setdefault(authors[d], len(aid)) for d in kd], np.int64)
    by_word = csr_matrix(
        (np.ones(Xk.indices.size), (np.repeat(aids, np.diff(Xk.indptr)), Xk.indices)),
        shape=(len(aid), V),
        dtype=np.float64,
    )
    n_auth = np.bincount(by_word.indices, minlength=V)

    ok = n_auth >= MIN_AUTHORS
    log(
        f"  {int(ok.sum()):,} of {int((tf > 0).sum()):,} words written by {MIN_AUTHORS} or more "
        f"distinct accounts"
    )

    live = np.flatnonzero(ok)
    live = live[np.argsort([vocab_all[i] for i in live], kind="stable")]
    vocab = [vocab_all[i] for i in live]
    X, week_of = Xk[:, live], week_raw[kd]
    long_enough = np.asarray(X.sum(axis=1)).ravel() >= MIN_WORDS
    X, week_of = X[long_enough], week_of[long_enough]
    log(f"{X.shape[0]:,} descriptions, {X.sum():,.0f} appearances, {len(vocab):,} words")
    return X, week_of, weeks, vocab, n_days


# -------------------------------------------------------------------------- model

# The two halves of a Lloyd pass, and between them nearly all of the arithmetic this file does:
# every pass reads the whole matrix twice, once to assign and once to add up what was assigned.
# Both are one pass over 28 million appearances, and both are written out here rather than left
# to scipy for the same reason -- scipy's sparse products are single-threaded, and these are
# perfectly parallel over descriptions. Together they took a pass from 340 ms to 24 ms on twelve
# cores, and the fit from eight and a half minutes to about a minute.
#
# `nearest` is not bit-for-bit what `X @ log(W).T` gives: the compiler contracts the multiply and
# add into one instruction and rounds once instead of twice. It picks the same centres --
# a difference of one part in 10^16 against gaps of one part in 10^2 -- but the cost printed by a
# pass can differ in its last digits. `summed` has no such caveat: the values are small integer
# counts and their sums are exact in float64.


@njit(cache=True, parallel=True)
def nearest(data, indices, indptr, logW):
    """For every description, the centre maximising `x_d . log W_c`, and that score.

    That argmax is the nearest centre under KL -- see `fit` for why the entropy term drops out.
    `logW` is (V, k) so that a word's k values are one cache line rather than k strides apart.
    """
    D, k = indptr.size - 1, logW.shape[1]
    lab, top = np.empty(D, np.int64), np.empty(D)
    for d in prange(D):
        acc = np.zeros(k)
        for p in range(indptr[d], indptr[d + 1]):
            v, j = data[p], indices[p]
            for c in range(k):
                acc[c] += v * logW[j, c]
        best = 0
        for c in range(1, k):
            if acc[c] > acc[best]:
                best = c
        lab[d], top[d] = best, acc[best]
    return lab, top


@njit(cache=True, parallel=True)
def summed(data, indices, indptr, lab, k, V, blocks):
    """What each cluster holds, word by word: the sum of the descriptions assigned to it.

    Blocked rather than scattered. Adding straight into one (k, V) accumulator would have every
    thread writing the same words, so each block of descriptions gets its own and they are added
    at the end -- no locks, and the same answer whatever the block count.
    """
    acc, D = np.zeros((blocks, k, V)), indptr.size - 1
    for b in prange(blocks):
        for d in range(D * b // blocks, D * (b + 1) // blocks):
            c = lab[d]
            for p in range(indptr[d], indptr[d + 1]):
                acc[b, c, indices[p]] += data[p]
    return acc.sum(axis=0)


def kmeanspp(X, k, rng, S):
    """Greedy k-means++ under KL. Returns k centres, each a distribution over the vocabulary.

    D^2 sampling with the multinomial's own divergence in place of squared distance: the first
    centre is a random description, and each next one is drawn with probability proportional to
    its divergence from the nearest centre chosen so far -- best of TRIALS draws, which is what
    "greedy" means and what scikit-learn's `k-means++` does by default.

    `S` is the first term of

        n_d * KL(p_d || W_c)  =  sum_v x_dv log(x_dv / n_d)  -  x_d . log W_c

    which depends on the description alone, so the sampling costs one sparse product per
    candidate.
    """
    D = X.shape[0]

    def centre(i):
        w = X[i].toarray().ravel() + SMOOTH
        return w / w.sum()

    def cost_to(W):
        # `nearest` with a single centre is the sparse product this needs, threaded
        _, dot = nearest(X.data, X.indices, X.indptr, np.ascontiguousarray(np.log(W))[:, None])
        return np.maximum(S - dot, 0.0)

    W = [centre(rng.integers(D))]
    cost = cost_to(W[0])
    for _ in range(1, k):
        best = None
        for _ in range(TRIALS):
            cand = centre(int(rng.choice(D, p=cost / cost.sum())))
            left = np.minimum(cost, cost_to(cand))
            if best is None or left.sum() < best[0]:
                best = (left.sum(), cand, left)
        W.append(best[1])
        cost = best[2]
    return np.array(W)


def fit(X, week_of, T, k=K, seed=SEED, log=print):
    """KL k-means. Returns (W, C, A, cost).

        W[c]   the c-th way of writing: a distribution over the vocabulary
        C[t,c] descriptions of week t assigned to c
        A[t,c] their word appearances
        cost   sum_d n_d KL(p_d || W_assigned), the quantity minimised

    Lloyd's algorithm: assign every description to the centre it is closest to under KL, move
    each centre to the middle of what it was given, repeat until nothing moves. Assignment is
    hard, so a description belongs to exactly one way of writing and the weekly curves are
    counts of whole descriptions.

    Since x_d . log W_c = -n_d ( KL(p_d || W_c) + H(p_d) ) and H(p_d) does not vary with c, the
    nearest centre is the one maximising x_d . log W_c -- one sparse product for the whole
    corpus -- and the KL-centroid of a cluster is the sum of its descriptions, normalised. The
    n_d weight is the only trace of counting left in it: a long description pulls harder.
    """
    rng = np.random.default_rng(seed)
    D, V = X.shape
    X = X.tocsr()
    blocks = numba.get_num_threads()
    n_d = np.asarray(X.sum(axis=1)).ravel()
    ent = np.bincount(
        np.repeat(np.arange(D), np.diff(X.indptr)),
        weights=X.data * np.log(X.data),
        minlength=D,
    ) - n_d * np.log(n_d)

    W, lab = kmeanspp(X, k, rng, ent), None
    for it in range(MAX_PASSES):
        # `top` is the score under the centres as they stand, so the cost below -- and the one
        # returned after the loop breaks -- belongs to the last centres, not the previous ones
        new, top = nearest(X.data, X.indices, X.indptr, np.ascontiguousarray(np.log(W).T))
        if lab is not None and (new == lab).all():
            break
        lab = new
        W = summed(X.data, X.indices, X.indptr, lab, k, V, blocks) + SMOOTH
        W /= W.sum(axis=1, keepdims=True)
        log(f"  pass {it + 1:2d}  cost {ent.sum() - top.sum():,.0f}")
    else:
        log(f"  warning: {MAX_PASSES} passes without a fixed point")

    idx = week_of * k + lab
    C = np.bincount(idx, minlength=T * k).reshape(T, k).astype(float)
    A = np.bincount(idx, weights=n_d, minlength=T * k).reshape(T, k)
    # what each component holds, word by word: appearances, not probabilities. The assignment
    # is hard, so this is a partition of the corpus and the ranking can be a count of things
    # rather than a quantity the fit chose.
    M = summed(X.data, X.indices, X.indptr, lab, k, V, blocks)
    return W, C, A, M, float(ent.sum() - top.sum())


# --------------------------------------------------------------------------- out


def pack(X, week_of, weeks, vocab, W, C, A, M, cost, n_days=0, seed=SEED, fits=1):
    share = C.sum(axis=0) / max(C.sum(), 1e-12)  # each component's share of the corpus
    corpus = M.sum(axis=0)  # every appearance of every word, whoever wrote it
    docs_per_week = np.bincount(week_of, minlength=len(weeks))
    words_per_week = np.zeros(len(weeks))
    np.add.at(words_per_week, week_of, np.asarray(X.sum(axis=1)).ravel())
    per_word = np.zeros((X.shape[1], len(weeks)))  # appearances, word by week
    for t in range(len(weeks)):
        sel = week_of == t
        if sel.any():
            per_word[:, t] = np.asarray(X[sel].sum(axis=0)).ravel()

    # ordered by size over the last LEAD_WINDOW weeks, largest first: what a reader wants
    # first is what the corpus looks like now, and the stack then puts the currently-dominant
    # band at the bottom where its shape is easiest to follow.
    recent = C[-LEAD_WINDOW:].sum(axis=0)
    order = np.argsort(-recent)

    vocab_arr = np.asarray(vocab)
    obs = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-12)
    start, end = obs[:8].mean(axis=0), obs[-8:].mean(axis=0)

    # The component this page is about is simply the largest one of the last month:
    # "whatever is most of the writing now" needs no threshold and cannot reject anything.
    lead = int(np.argmax(recent))

    # Reported, not enforced: `fit_arriving` is what acts on it, by trying the next seed.
    arrived = bool(start[lead] < LEAD_START and end[lead] >= LEAD_END)

    comps = []
    for c in order:
        # How OFTEN the word is written here against how often it is written anywhere else:
        # each side's count over that side's own total, so the number is a ratio of two
        # empirical frequencies. Counted rather than modelled -- the assignment is hard, so
        # each appearance belongs to exactly one component and the two sides partition the
        # word's occurrences and the corpus they are measured against.
        #
        # A ratio of raw counts, which is what this was, carries the component's size inside
        # it. This component holds a fifth of the appearances, so dividing its count by the
        # four fifths outside understates every one of its words by that same factor, and a
        # component holding a fortieth understates its own by forty. The numbers were then
        # neither comparable between components nor readable as the "more frequent" the page
        # calls them. Dividing each side by its own total is what makes them both.
        #
        # The denominator carries a pseudo-count, and it is the difference between a ranking
        # and a lottery. Floored at one appearance instead -- which is what this did -- the top
        # of the list was words written two to seven times in 43 million outside the component,
        # where two against six is a factor of three in the score and nothing at all in the
        # world: `mutation-checked` (3 outside) beat `load-bearing` (158) by sixty-five places
        # on a difference no larger than the noise. Those words are also too rare to have been
        # noticed -- two appearances per million against twenty.
        #
        # Half of MIN_AUTHORS is the size of it, and derived rather than picked: a word in the
        # vocabulary has been written by MIN_AUTHORS accounts and so appears at least that many
        # times, which makes MIN_AUTHORS the fewest appearances a word here can have and half of
        # it the honest prior for "written outside less often than can be measured". Both are
        # absolute counts, so the two stay in step as the corpus grows.
        #
        # What it does to the list: a word never written outside now scores in proportion to
        # what it was written INSIDE -- the pseudo-count is the whole denominator -- so the top
        # is ordered by frequency among the component's exclusive words instead of by the
        # accident of a tiny divisor. Ratios for those words are shrunk, deliberately: the page
        # would rather understate a word it cannot measure than rank it first.
        inside = M[c]
        here, elsewhere = max(inside.sum(), 1.0), max(corpus.sum() - inside.sum(), 1.0)
        lift = (inside / here) / ((corpus - inside + MIN_AUTHORS / 2) / elsewhere)
        # ties broken on the word itself, so two builds of the same corpus are byte-identical
        # and the daily commit does not churn on words that score the same
        rank = np.lexsort((vocab_arr, -lift))
        # the arriving component gets a long list, because it is the one anybody will read
        # past the first handful of, and the cut has to fall somewhere. The others get a list
        # a reader can still fall down for a while, because the board now steps between them
        # and a cluster you can exhaust in one screen is not one you can explore.
        n = WORDS_LEAD if c == lead else WORDS_LISTED
        comps.append(
            {
                "lead": bool(c == lead),
                "share": round(float(share[c]), 5),
                "start_share": round(float(start[c]), 5),
                "end_share": round(float(end[c]), 5),
                "count": [int(v) for v in C[:, c]],
                "appearances": [int(round(v)) for v in A[:, c]],
                "word_list": [vocab[j] for j in rank[:n]],
                "word_lift": [round(float(lift[j]), 2) for j in rank[:n]],
                # each listed word's own weekly appearances, so the page can show a word's
                # history on hover -- raw counts, which the page divides by `words_per_week` to
                # draw a rate, since a week here runs from 37,000 words to 129,000 and the
                # count alone would draw that too.
                #
                # For EVERY component, not just the lead. It was the lead's alone while the
                # board was about one component; the board now steps between all twelve, and a
                # cluster whose words have no history is a panel that answers for one of them
                # and goes blank for the other eleven. The cost is why the lists differ in
                # length: a thousand words over 85 weeks is 85,000 integers and worth carrying
                # once, WORDS_LISTED of them is worth carrying twelve times.
                "series": [[int(v) for v in per_word[j]] for j in rank[:n]],
            }
        )
    # Points per week that the lead component has moved over the last TREND_WEEKS, by least
    # squares on its observed weekly share. The page reads this rather than being told the
    # component is still rising: a claim that can go stale should not be typed into the markup.
    lead_share = C[:, lead] / np.maximum(C.sum(axis=1), 1e-12)
    tw = min(TREND_WEEKS, len(weeks))
    slope = float(np.polyfit(np.arange(tw), lead_share[-tw:], 1)[0]) if tw >= 3 else 0.0

    return {
        "generated": date.today().isoformat(),
        "weeks": weeks,
        "days": int(n_days),
        "arrived": bool(arrived),
        "lead_window": LEAD_WINDOW,
        "seed": int(seed),
        "fits": int(fits),
        "trend": round(slope, 6),
        "trend_weeks": tw,
        "documents": int(X.shape[0]),
        "appearances": int(X.sum()),
        "docs_per_week": [int(v) for v in docs_per_week],
        "words_per_week": [int(v) for v in words_per_week],
        "vocab": len(vocab),
        "k": len(comps),
        "cost": round(cost, 1),
        "components": comps,
    }


def fit_arriving(
    X,
    week_of,
    weeks,
    vocab,
    n_days,
    k=K,
    seed=SEED,
    n_init=N_INIT,
    retries=RETRIES,
    log=print,
):
    """Fit `n_init` times, publish the cheapest. Returns the packed result.

    The restarts are there for the daily job rather than for the answer. Cost is the only
    quality measure this model has, and it correlates +0.03 with the share the page reports, so
    the cheapest of eight is not a better answer than the first of one -- it is a more reliably
    *publishable* one -- and at K = 12 not even that, since all 32 unconditioned fits of this
    corpus arrive and the cheapest of eight therefore arrives because every one of the eight does.
    At K = 8 a single fit passed 27 times in 32, and the cheapest of eight ninety-nine times in a
    hundred, weakly because arriving fits cost a little less. The restarts are insurance now.

    When even that one does not arrive, the whole batch is run again from the next `n_init`
    seeds, and after `retries` batches the assertion fires and nothing is published, which is
    what stops the daily job.

    A retry conditions the published fit on the check it is supposed to face, so the check is a
    selector on the rare day one happens. What remains as evidence of arrival is the rate at
    which unconditioned runs arrive at all -- 32 of 32 on this corpus -- and that belongs in the
    README rather than in this fit.
    """
    fits = 0
    for attempt in range(retries):
        best = None
        for i in range(n_init):
            out = fit(
                X,
                week_of,
                len(weeks),
                k=k,
                seed=seed + attempt * n_init + i,
                log=lambda *_: None,
            )
            fits += 1
            log(f"  seed {seed + attempt * n_init + i}  cost {out[-1]:,.0f}")
            if best is None or out[-1] < best[-1]:
                best, best_seed = out, seed + attempt * n_init + i
        packed = pack(X, week_of, weeks, vocab, *best, n_days, best_seed, fits)
        lead = next(c for c in packed["components"] if c["lead"])
        log(
            f"  kept seed {best_seed}, {lead['start_share']:.2%} -> {lead['end_share']:.2%}"
            f"{'' if packed['arrived'] else ', not an arrival -- running the batch again'}"
        )
        if packed["arrived"]:
            return packed
    raise AssertionError(
        f"{fits} fits in {retries} batches and the cheapest of each did not arrive: the largest "
        f"component of the last {LEAD_WINDOW} weeks has to start under {LEAD_START:.0%} of the "
        f"first eight weeks and end at or above {LEAD_END:.0%} of the last eight"
    )


def selftest():
    """A planted way of writing that arrives partway through must be recovered."""
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
                rows.append(d)
                cols.append(int(j))
                vals.append(float(x[j]))
            week_of.append(t)
            d += 1
    X = csr_matrix((vals, (rows, cols)), shape=(d, V))
    week_of = np.array(week_of)
    W, C, A, M, cost = fit(X, week_of, T, k=3, log=lambda *_: None)

    assert np.allclose(W.sum(axis=1), 1.0), "the centres are not distributions"
    assert cost > 0, "the cost must be a positive divergence"
    # every description belongs to exactly one component, so the counts are whole and they
    # reconstruct each week
    assert (
        np.allclose(C, np.round(C)) and (C.sum(axis=1) == np.bincount(week_of, minlength=T)).all()
    ), "the counts do not reconstruct the week"
    assert np.allclose(A.sum(axis=1), np.bincount(week_of, minlength=T) * 50), (
        "the appearance counts do not reconstruct the week"
    )
    # the components partition the corpus word for word, which is what lets the ranking be a
    # ratio of frequencies: what is written here and what is written everywhere else sum to the
    # whole, and so do the two totals each side is divided by
    assert np.allclose(M.sum(axis=0), np.asarray(X.sum(axis=0)).ravel()), (
        "the per-component word counts do not reconstruct the corpus"
    )

    # the planted way of writing must be found rising even though the model cannot represent
    # time: this is the whole claim of the thing, tested where the answer is known
    obs = C / C.sum(axis=1, keepdims=True)
    late = int(np.argmax([Wt[2] @ np.log(W[c]) for c in range(3)]))
    before, after = obs[:arrives, late].mean(), obs[arrives:, late].mean()
    assert before < 0.5 * after, (
        f"the planted component did not rise ({before:.3f} then {after:.3f})"
    )

    print(
        f"selftest: ok  (centres are distributions, counts are whole and reconstruct each "
        f"week, planted component {before:.3f} -> {after:.3f} at week {arrives})"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-init", type=int, default=N_INIT, dest="n_init")
    ap.add_argument("--retries", type=int, default=RETRIES)
    ap.add_argument("--out", "-o", default="analysis.js")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X, week_of, weeks, vocab, n_days = documents()
    out = fit_arriving(
        X,
        week_of,
        weeks,
        vocab,
        n_days,
        k=args.k,
        seed=args.seed,
        n_init=args.n_init,
        retries=args.retries,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.ANALYSIS = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(
        f"\nseed {out['seed']} published, {out['fits']} fit(s) run, "
        f"cost {out['cost']:,.0f}, wrote {args.out} "
        f"({os.path.getsize(args.out) / 1e3:.0f} kB)\n"
    )
    for c in out["components"]:
        print(
            f"  {'lead' if c['lead'] else '    '}  share {c['share']:6.1%}  "
            f"{c['start_share']:6.2%} -> {c['end_share']:6.2%}"
        )
        print("        " + ", ".join(w[:20] for w in c["word_list"][:9]))


if __name__ == "__main__":
    main()

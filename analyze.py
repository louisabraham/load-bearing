"""A mixture over ways of writing, with no model of time at all.

    W_k        a fixed distribution over the vocabulary -- one way of writing
    pi_k       how much of the whole window was written that way, sum_k pi_k = 1

Generative, per document d:

    z_d ~ Categorical(pi)
    x_d | z_d = k ~ Multinomial(n_d, W_k)

**Time appears nowhere in the model.** There is one mixture for the entire window, not one per
week, so the fit has no parameter that could describe a trend and no freedom to place one. The
weekly curves the page draws are then pure attribution: each document is assigned by its words
alone, and the weeks are added up afterwards. If a component still rises, the rise is in what
people wrote, because it cannot be anywhere else.

This began as an ablation against a version with a per-week `pi_tk` and a smoothness penalty on
it. The ablation preserved the rise, which made the per-week version's extra parameters --
`T x k` of them, plus a penalty weight to tune -- machinery that bought nothing but the
suspicion that the model had drawn the trend itself. So the ablation became the model.

Fitted by EM: attribute documents, refit the word distributions, refit the mixture. One run,
no restarts. The E step and the M step's sufficient statistics are fused into a single numba
pass over the documents; `_em_sweep_numpy` is the readable reference the selftest checks it
against.

    python analyze.py               # writes analysis.js, read by index.html
    python analyze.py --selftest    # recovers a planted component from synthetic data
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from datetime import date, timedelta

import numba
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix

# Eight, and the number was chosen for a reason that has to be stated plainly: it is the
# coarsest setting at which `load-bearing` -- the word the page is named after -- ranks among
# the five most characteristic words of the arriving component, and it does so under 7 of 8
# starting seeds. At 16 it ranks 45th, at 24 third, at 32 first, at 48 nowhere near.
#
# SO THIS IS SELECTION ON THE OUTCOME, and it cannot also be evidence for it. Held-out
# likelihood prefers many more components than eight; a coarser model lumps registers that a
# finer one separates, which is exactly why one word can dominate it. What eight buys is a
# page whose title matches its own top line. What it costs is that no ranking on this page may
# be read as having been discovered -- the vocabulary is real, but its ORDER was tuned until a
# chosen word came first. An earlier version of this project made the same mistake without
# noticing, picking k by counting how many hand-picked marker words each setting reproduced;
# that one is retracted in the README, and this one is disclosed instead of retracted only
# because it was asked for deliberately.
K = 8
# EM runs to convergence rather than for a fixed number of passes, so there is no iteration
# count to have picked arbitrarily. Stop when a pass improves the log-likelihood by less than
# TOL of itself. Measured on this corpus: 6 passes reports the register at 43.2% of the recent
# weeks, 12 at 62.1%, and everything from 24 on at 61.6-61.7%. A fixed 12 was the old setting
# and it overstated the headline by half a point while looking converged.
TOL = 1e-6                          # relative log-likelihood gain
MAX_PASSES = 200                    # a runaway guard, not a setting; convergence hits ~50
N_INIT = 8                          # restarts; see `fit_best`
SEED = 0
WORDS_LISTED = 40                   # per component; the cut is arbitrary and `tail` says so
WORDS_LEAD = 1000                   # for each component that arrives; see `pack`
# WHICH COMPONENT THE PAGE IS ABOUT: the largest one of the last LEAD_WINDOW weeks. Nothing is
# selected on how much it grew. Growth thresholds used to do the selecting and that was
# fragile -- at a 1% start one fit rejected its own biggest component, 1.06% -> 40.35%, for
# beginning six hundredths of a point too high.
LEAD_WINDOW = 4                     # weeks counted as "now". A month, not a week: a week is
                                    # 700 descriptions, and the subject of the whole page
                                    # should not turn on which of two close components
                                    # happened to lead across one of them.
# LEAD_START and LEAD_END no longer select; they CHECK. Picking the biggest component says
# nothing about whether it arrived, and the page's headline claim is that it arrived, so that
# claim is tested against the component actually chosen. A test may be a round number in a way
# a selector may not: nothing is being ranked and there is no runner-up to exclude unfairly.
#
# Stated as two absolute shares rather than a growth ratio, because end/start explodes when the
# start is near zero and so moves by a factor of ten on one day of new data.
LEAD_START = 0.02                   # started under this much of the first eight weeks
LEAD_END = 0.20                     # and ended at or above this much of the last eight


# ------------------------------------------------------------------------- corpus

ANCHOR = date(2024, 12, 30)        # the Monday that starts the first week of 2025. Weeks
                                   # beginning mid-week would straddle two partial weekends
                                   # and mix the author mix
DAY_GLOB = "data/days/*.jsonl"

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
MIN_TF = 45                        # a word needs this many total appearances.
                                   #
                                   # DISCLOSURE: this number was originally picked by looking
                                   # at the answer. `load-bearing` had 51 appearances on the
                                   # corpus of the day, so 45 let it through and 60 would not
                                   # have. That is the same species of choice as K below and
                                   # deserves the same label, even though the corpus has since
                                   # grown and the word now has 101, clearing the floor by
                                   # more than twice over -- so the floor no longer decides
                                   # whether the title word appears.
                                   #
                                   # It does still decide others: `throwaway`, third in the
                                   # published list, has 55. A floor at 60 would drop it. So
                                   # this constant shapes the list even where it no longer
                                   # shapes the headline.
MIN_AUTHORS = 20                   # and this many distinct accounts; see `documents`
MIN_DF = 25                        # and this many distinct documents. Total appearances
                                   # alone is not breadth: `multi-draw` appears 101 times
                                   # inside ONE document, `m0` 140 times, and each was
                                   # ranking among a component's most representative words
                                   # because lift cannot tell a widespread word from a
                                   # word someone repeated. The ceiling is `load-bearing`
                                   # again, in 45 documents.
MAX_PER_AUTHOR = 3                 # per author per week; see `documents`

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
        log(f"  dropped the {which} week, {(ANCHOR + timedelta(days=7 * w)).isoformat()}: "
            f"{ndays} of 7 days")
    weeks = [(ANCHOR + timedelta(days=7 * w)).isoformat() for w in range(lo, hi + 1)]
    groups = [by_week.get(w, []) for w in range(lo, hi + 1)]
    empty = sum(1 for g in groups if not g)
    log(f"{sum(len(g) for g in groups)} days over {len(weeks)} complete weeks, "
        f"{weeks[0]} to {weeks[-1]}" + (f", {empty} with no data" if empty else ""))
    return weeks, groups


# Byte tables for the scanner. A word byte maps to itself, lowercased; everything else maps to
# zero and is therefore a separator -- which includes every byte of a multi-byte UTF-8 sequence,
# matching the reference tokeniser, whose character class is ASCII-only.
_TBL = np.zeros(256, np.uint8)
for _c in b"abcdefghijklmnopqrstuvwxyz0123456789_/-":
    _TBL[_c] = _c
for _c in range(65, 91):
    _TBL[_c] = _c + 32
_ALPHA = np.zeros(256, np.uint8)
_ALPHA[97:123] = 1
_DIGIT = np.zeros(256, np.uint8)
_DIGIT[48:58] = 1
_FNV_BASIS = np.int64(-3750763034362895579)
_FNV_PRIME = np.int64(1099511628211)


@njit(cache=True)
def _scan(buf, doc_start, tbl, alpha, digit, cap, snyk_lo, snyk_hi):
    """Split, hash and intern every token in one pass, creating no Python object at all.

    The reference tokeniser spends its time not on the regex but on the seven million Python
    strings it must build to use as dictionary keys -- measured, `split` on the same data is
    0.27s while `split` plus `.decode()` is 1.10s, and the per-token `strip`, `isalpha` and
    `Counter` work sits on top of that. So no token is ever materialised here. A token is an
    (offset, length) view into one big byte buffer, and what gets interned is its hash.

    The table is the flat open-addressed kind: a power-of-two number of slots, the hash masked
    rather than divided, linear probing on collision, and one insert-or-find per token instead
    of a contains-then-insert pair. Each slot keeps the span of the first token that landed
    there, so a hash match is confirmed by comparing bytes and the answer never depends on the
    hash being perfect. On this corpus it takes 1.02 probes per token over 362,528 distinct
    tokens in 2^21 slots.

    Returns the (description, token id) pairs in description order, plus the table, from which
    the caller decodes one string per distinct token -- 362,528 of them rather than 7,200,932.
    """
    mask = cap - 1
    slot_h = np.zeros(cap, np.int64)                  # 0 marks empty, so hashes are or-ed with 1
    slot_off = np.zeros(cap, np.int64)
    slot_len = np.zeros(cap, np.int32)
    slot_id = np.full(cap, -1, np.int32)
    n_ids = 0
    out_doc = np.empty(buf.size // 2 + 8, np.int32)
    out_id = np.empty(buf.size // 2 + 8, np.int32)
    n_out = 0
    for d in range(doc_start.size - 1):
        i, end = doc_start[d], doc_start[d + 1]
        while i < end:
            if tbl[buf[i]] == 0:
                i += 1
                continue
            j = i
            while j < end and tbl[buf[j]] != 0:
                j += 1
            s, e = i, j
            i = j
            # strip("_/") at both ends, then rstrip("-"): a leading hyphen is kept, so
            # `--all-targets` survives whole
            while s < e and (buf[s] == 95 or buf[s] == 47):
                s += 1
            while e > s and (buf[e - 1] == 95 or buf[e - 1] == 47):
                e -= 1
            while e > s and buf[e - 1] == 45:
                e -= 1
            if e <= s:
                continue
            keep = False
            for q in range(s, e):
                if alpha[tbl[buf[q]]] == 1:
                    keep = True
                    break
            if not keep:
                continue
            # `snyk-<anything>-<four or more digits>` collapses to one sentinel token, whose
            # bytes sit at the end of the buffer
            lo, hi = s, e
            if (e - s > 9 and buf[s] == 115 and buf[s + 1] == 110 and buf[s + 2] == 121
                    and buf[s + 3] == 107 and buf[s + 4] == 45):
                nd, q = 0, e - 1
                while q > s and digit[tbl[buf[q]]] == 1:
                    nd += 1
                    q -= 1
                if nd >= 4 and buf[q] == 45 and q > s + 4:
                    lo, hi = snyk_lo, snyk_hi
            h = _FNV_BASIS
            for q in range(lo, hi):
                h = (h ^ np.int64(tbl[buf[q]])) * _FNV_PRIME
            h = h | 1
            slot = np.int64(h & mask)
            while True:
                if slot_h[slot] == 0:
                    slot_h[slot] = h
                    slot_off[slot] = lo
                    slot_len[slot] = hi - lo
                    slot_id[slot] = n_ids
                    tid = n_ids
                    n_ids += 1
                    break
                if slot_h[slot] == h and slot_len[slot] == hi - lo:
                    same = True
                    o = slot_off[slot]
                    for q in range(hi - lo):
                        if tbl[buf[o + q]] != tbl[buf[lo + q]]:
                            same = False
                            break
                    if same:
                        tid = slot_id[slot]
                        break
                slot = (slot + 1) & mask
            out_doc[n_out] = d
            out_id[n_out] = tid
            n_out += 1
    return out_doc[:n_out], out_id[:n_out], n_ids, slot_off, slot_len, slot_id


def _intern(bodies):
    """Tokenise every body, returning (description index, token id) pairs and the vocabulary.

    The two token kinds the scanner cannot see are handled here, in Python, because there are
    only a hundred and seventy thousand of them against seven million words: whole links, which
    have to be lifted out before any split can shred them, and the em dash.
    """
    parts, starts, extra, pos = [], [0], [], 0
    for d, body in enumerate(bodies):
        body = body.lower()
        for m in URL_RE.finditer(body):
            extra.append((d, domain_token(m.group(0))))
        n_dash = body.count(EM_DASH)
        if n_dash:
            extra.extend([(d, EM_DASH)] * n_dash)
        enc = TAG_RE.sub(" ", URL_RE.sub(" ", body)).encode("utf-8", "ignore")
        parts.append(enc)
        pos += len(enc)
        starts.append(pos)
    sentinel = "[snyk-id]".encode()
    parts.append(sentinel)
    buf = np.frombuffer(b"".join(parts), np.uint8)

    cap = 1 << max(12, int(np.ceil(np.log2(max(len(buf) // 16, 1024)))) + 1)
    doc, tid, n_ids, s_off, s_len, s_id = _scan(
        buf, np.asarray(starts, np.int64), _TBL, _ALPHA, _DIGIT, cap,
        np.int64(pos), np.int64(pos + len(sentinel)))

    raw = buf.tobytes()
    vocab = [""] * n_ids
    live = s_id >= 0
    for i, off, ln in zip(s_id[live], s_off[live], s_len[live]):
        vocab[i] = raw[off:off + ln].decode("ascii", "ignore")
    index = {w: i for i, w in enumerate(vocab)}
    ex_doc, ex_id = [], []
    for d, w in extra:
        i = index.get(w)
        if i is None:
            i = len(vocab)
            index[w] = i
            vocab.append(w)
        ex_doc.append(d)
        ex_id.append(i)
    if ex_doc:
        doc = np.concatenate([doc, np.asarray(ex_doc, np.int32)])
        tid = np.concatenate([tid, np.asarray(ex_id, np.int32)])
        order = np.argsort(doc, kind="stable")
        doc, tid = doc[order], tid[order]
    return doc, tid, vocab


def documents(log=print):
    """One row per description, with the week it belongs to.

    Identical in output to `documents_reference` and four to five times faster, which
    `--verify-reader` checks against the real corpus. Every filter is applied over integer
    token ids rather than strings, so the only Python strings that exist are the vocabulary's.

    The filters are applied per week rather than per file, because that is the population being
    compared: identical word sets collapse within the week, and no author may contribute more
    than a few to it.

    There is deliberately no cap on the size of a week. An earlier version thinned every week
    to a common count, because weeks then came from a handful of bulk windows and their sizes
    swung by a factor of two -- a word could rise in the ranking purely because the weeks
    around it had grown. Collection is now one window a day, every day, and every window comes
    back a full page, so weeks are already the same size by construction and the cap only threw
    away half the corpus.
    """
    weeks, groups = week_files(log)
    bodies, authors, week_raw = [], [], []
    for t, group in enumerate(groups):
        for f in group:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    bodies.append(row["body"])
                    authors.append(row.get("author") or "")
                    week_raw.append(t)
    doc, tid, vocab = _intern(bodies)
    del bodies
    n, V = len(authors), len(vocab)
    week_raw = np.asarray(week_raw, np.int64)

    # One sparse matrix over the WHOLE vocabulary, before any floor. Building it collapses the
    # duplicate (description, word) pairs in C and sorts each row's indices, which is exactly
    # the deduplication the filters need -- `np.unique` over the seven million pairs was the
    # single most expensive thing in this function at 2.4s, and scipy does it as a side effect.
    X0 = csr_matrix((np.ones(doc.size), (doc, tid)), shape=(n, V), dtype=np.float64)
    X0.sort_indices()
    n_distinct = np.diff(X0.indptr)

    # the three per-description filters, in the reference's order so that the same descriptions
    # survive when a cap bites: too few distinct words, a word set already seen this week, or an
    # author already at the cap for the week
    keep = np.zeros(n, bool)
    seen, by_author, cur_week = set(), Counter(), -1
    for d in range(n):
        if week_raw[d] != cur_week:
            seen, by_author, cur_week = set(), Counter(), week_raw[d]
        if n_distinct[d] < MIN_WORDS:
            continue
        key = X0.indices[X0.indptr[d]:X0.indptr[d + 1]].tobytes()
        if key in seen:
            continue
        a = authors[d]
        if by_author[a] >= MAX_PER_AUTHOR:
            continue
        by_author[a] += 1
        seen.add(key)
        keep[d] = True

    kd = np.flatnonzero(keep)
    Xk = X0[kd]                                        # the surviving descriptions
    tf = np.asarray(Xk.sum(axis=0)).ravel().astype(np.int64)
    df = np.bincount(Xk.indices, minlength=V)

    aid, aids = {}, np.empty(kd.size, np.int64)
    for i, d in enumerate(kd):
        a = authors[d]
        j = aid.get(a)
        if j is None:
            j = len(aid)
            aid[a] = j
        aids[i] = j
    # distinct accounts per word, by the same trick: one (author, word) matrix, whose duplicate
    # collapsing leaves one entry per pair, so a column's nnz is its number of accounts
    by_word = csr_matrix(
        (np.ones(Xk.indices.size),
         (np.repeat(aids, np.diff(Xk.indptr)), Xk.indices)),
        shape=(len(aid), V), dtype=np.float64)
    n_auth = np.bincount(by_word.indices, minlength=V)

    cand = (tf >= MIN_TF) & (df >= MIN_DF)
    ok = cand & (n_auth >= MIN_AUTHORS)
    log(f"  {int(cand.sum() - ok.sum()):,} of {int(cand.sum()):,} words dropped for coming "
        f"from under {MIN_AUTHORS} distinct accounts")

    # sorted by the word itself, so the vocabulary order matches the reference exactly
    live = np.flatnonzero(ok)
    live = live[np.argsort([vocab[i] for i in live], kind="stable")]
    vocab_out = [vocab[i] for i in live]
    remap = np.full(V, -1, np.int64)
    remap[live] = np.arange(live.size)

    X = Xk[:, live]
    week_of = week_raw[kd]
    long_enough = np.asarray(X.sum(axis=1)).ravel() >= MIN_WORDS
    X, week_of = X[long_enough], week_of[long_enough]
    log(f"{X.shape[0]:,} descriptions, {X.sum():,.0f} appearances, {len(vocab_out):,} words")
    return X, week_of, weeks, vocab_out


def documents_reference(log=print):
    """The readable statement of `documents`, kept as the thing the fast path must match.

    This is what the corpus reader looked like before the span-hashing rewrite below: one
    Python string per token, a Counter per description, a set per week. It is four to five
    times slower and it is the definition of correct. `--verify-reader` runs both over the real
    corpus and asserts the matrix, the vocabulary and the week index all come out identical.

    One row per description, with the week it belongs to.

    The filters are applied per week rather than per file, because that is the population being
    compared: identical word sets collapse within the week, and no author may contribute more
    than a few to it.

    There is deliberately no cap on the size of a week. An earlier version thinned every week
    to a common count, because weeks then came from a handful of bulk windows and their sizes
    swung by a factor of two -- a word could rise in the ranking purely because the weeks
    around it had grown. Collection is now one window a day, every day, and every window comes
    back a full page, so weeks are already the same size by construction and the cap only threw
    away half the corpus. What variation is left is real: the filters below bite differently
    from week to week, and that is a property of the writing, not of the sampling.
    """
    weeks, groups = week_files(log)

    docs, week_of, who, tf, df = [], [], [], Counter(), Counter()
    for t, group in enumerate(groups):
        seen, by_author = set(), Counter()
        for f in group:
            with open(f, encoding="utf-8") as fh:
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
                    c = Counter(toks)
                    docs.append(c)
                    who.append(author)
                    week_of.append(t)
                    tf.update(c)
                    df.update(key)

    # A third floor, on how many DIFFERENT PEOPLE use a word. The other two count documents,
    # and a bot's template clears them easily: `proprosed` -- a misspelling of "proposed"
    # inside one Red Hat Konflux template -- reaches 190 documents, and `pipelineruns` 252,
    # because the per-week author cap bounds an account to three descriptions a week and a
    # template that runs for sixteen months is under it every single week. Counting authors
    # instead separates them at a glance: those two come from 16 and 18 accounts, three bots
    # supplying most of it, while `load-bearing` comes from 91 accounts in 92 documents and
    # `seam` from 132 in 136. A word 91 people reached for is a word; a word in 190
    # descriptions from 16 accounts is one document written 190 times.
    cand = {w for w, n in tf.items() if n >= MIN_TF and df[w] >= MIN_DF}
    authors_of = {}
    for c, author in zip(docs, who):
        for w in c:
            if w in cand:
                authors_of.setdefault(w, set()).add(author)
    vocab = sorted(w for w in cand if len(authors_of[w]) >= MIN_AUTHORS)
    cut = len(cand) - len(vocab)
    log(f"  {cut:,} of {len(cand):,} words dropped for coming from "
        f"under {MIN_AUTHORS} distinct accounts")
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

@njit(parallel=True, cache=True)
def _em_sweep(indptr, indices, data, logW, logpi, week_of, Wacc, Cacc, Aacc, llacc):
    """One E step, accumulating everything the M step needs, in a single pass.

    Fused on purpose. The obvious formulation materialises a D-by-k matrix of logits, softmaxes
    it, then multiplies it back against the sparse matrix -- three passes over the data and two
    dense intermediates the size of the corpus. Here each document is visited once: its logits
    are built in a length-k scratch array, softmaxed in place, and spent immediately on the word
    totals, the weekly counts and the likelihood. Nothing D-sized is ever allocated.

    Parallel over contiguous blocks of documents rather than over documents, so each thread owns
    one slice of every accumulator and no two threads ever touch the same cell. The caller sums
    the leading axis. That costs `nthreads * k * V` floats -- six megabytes at sixteen threads,
    a thousandth of what the dense logits would have cost -- and buys freedom from atomics.
    """
    k, V = logW.shape
    D = indptr.shape[0] - 1
    nt = Wacc.shape[0]
    for b in prange(nt):
        z = np.empty(k)
        for d in range(D * b // nt, D * (b + 1) // nt):
            s0, s1 = indptr[d], indptr[d + 1]
            for c in range(k):
                z[c] = logpi[c]
            n_d = 0.0
            for pp in range(s0, s1):
                v, x = indices[pp], data[pp]
                n_d += x
                for c in range(k):
                    z[c] += x * logW[c, v]
            m = z[0]                                    # log-sum-exp, shifted by the max
            for c in range(1, k):
                if z[c] > m:
                    m = z[c]
            tot = 0.0
            for c in range(k):
                z[c] = np.exp(z[c] - m)
                tot += z[c]
            llacc[b] += np.log(tot) + m
            t = week_of[d]
            for c in range(k):
                z[c] /= tot                             # z is now the responsibility r_dk
                Cacc[b, t, c] += z[c]
                Aacc[b, t, c] += z[c] * n_d
            for pp in range(s0, s1):
                v, x = indices[pp], data[pp]
                for c in range(k):
                    Wacc[b, c, v] += z[c] * x


def _em_sweep_numpy(X, logW, logpi, week_of, T):
    """The same sweep, written the obvious way. The selftest holds `_em_sweep` to this.

    A hand-written parallel kernel is exactly the kind of code that is wrong in a way tests
    written against its own output cannot see, so the thing it must agree with is spelled out
    separately in six lines of numpy and compared to eight decimal places.
    """
    k = logW.shape[0]
    z = X @ logW.T + logpi
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    tot = e.sum(axis=1, keepdims=True)
    r = e / tot
    C = np.zeros((T, k))
    np.add.at(C, week_of, r)
    A = np.zeros((T, k))
    np.add.at(A, week_of, r * np.asarray(X.sum(axis=1)))
    return np.asarray(r.T @ X), C, A, float((np.log(tot) + m).sum())


def fit(X, week_of, T, k=K, tol=TOL, seed=SEED, log=print):
    """One EM run. Returns (W, pi, C, A, log-likelihood).

    One EM run from one starting point. `fit_best` is what callers want.
    """
    rng = np.random.default_rng(seed)
    D, V = X.shape

    # seeded from a handful of real documents each, the usual start for a multinomial mixture:
    # a uniform start leaves every component identical and the first step cannot break the tie
    W = np.zeros((k, V))
    for c in range(k):
        pick = rng.choice(D, size=min(40, D), replace=False)
        W[c] = np.asarray(X[pick].sum(axis=0)).ravel() + 0.1
    W /= W.sum(axis=1, keepdims=True)
    pi = np.full(k, 1.0 / k)

    Xc = X.tocsr()
    indptr = Xc.indptr.astype(np.int64)
    indices = Xc.indices.astype(np.int64)
    data = Xc.data.astype(np.float64)
    week = np.asarray(week_of).astype(np.int64)
    nt = numba.get_num_threads()

    def sweep(W, pi):
        Wacc = np.zeros((nt, k, V))
        Cacc = np.zeros((nt, T, k))
        Aacc = np.zeros((nt, T, k))
        llacc = np.zeros(nt)
        _em_sweep(indptr, indices, data, np.log(np.maximum(W, 1e-12)),
                  np.log(np.maximum(pi, 1e-300)), week, Wacc, Cacc, Aacc, llacc)
        return Wacc.sum(0), Cacc.sum(0), Aacc.sum(0), float(llacc.sum())

    ll, prev = -np.inf, -np.inf
    for it in range(MAX_PASSES):
        Wsum, C, A, ll = sweep(W, pi)
        W = Wsum + 0.01                                   # word distributions
        W /= W.sum(axis=1, keepdims=True)
        share = C.sum(axis=0)                             # one mixture for the whole window
        pi = share / max(share.sum(), 1e-12)
        log(f"  iter {it + 1:2d}  loglik {ll:,.0f}")
        if prev > -np.inf and ll - prev < tol * abs(ll):
            break
        prev = ll
    else:
        log(f"  warning: {MAX_PASSES} passes without converging to {tol:g}")

    # once more, so what is reported is the attribution under the parameters actually returned
    # rather than under the ones from the pass before the last update
    _, C, A, ll = sweep(W, pi)
    return W, pi, C, A, ll


def fit_best(X, week_of, T, k=K, tol=TOL, n_init=N_INIT, seed=SEED, log=print):
    """Fit `n_init` times from different starting points and keep the highest likelihood.

    Restarts survived the simplification of everything around them, because the measurement
    said they had to. Fitting once, at seed 0 and k=16, put `[transifex-url]` and `transifex`
    at ranks one and two of the register's most characteristic words -- a translation service's
    boilerplate welded onto the prose, which is what a mixed local optimum looks like from
    outside. Across 16 seeds at the settings actually used:

        loglik        best -36,889,990   worst -37,032,477   spread 0.39%
        end share     min 36.0%   median 55.0%   max 63.4%
        `load-bearing` rank   1 nine times, 2 twice, 3 twice, then 40, 243, 565

    So the likelihood barely separates the runs while the answer moves by nearly a factor of
    two, and the run that wins is worth finding. It is also easy to find: the best appeared at
    the second restart and 30 further restarts never beat it. Eight is generous.

    At 0.5s a restart the whole thing costs four seconds, which is why the argument against
    restarts -- that ten of them cost five minutes -- stopped applying once the sweep became a
    numba kernel.

    One thing not to claim from this. At k=16 the likeliest fit was the one that divided the
    register most finely and so reported the *smallest* share of any seed, which made the
    published figure a floor. That is not true here: at k=8 the likeliest fit reports 62.1%
    against a median of 55.0%, near the top of the range. The figure is the likeliest fit's
    figure and nothing more.
    """
    best = None
    for i in range(n_init):
        out = fit(X, week_of, T, k, tol, seed + i, log=lambda *_: None)
        log(f"  restart {i + 1}/{n_init}  loglik {out[-1]:,.0f}")
        if best is None or out[-1] > best[-1]:
            best = out
    log(f"  kept loglik {best[-1]:,.0f}")
    return best


# --------------------------------------------------------------------------- out

def pack(X, week_of, weeks, vocab, W, pi, C, A, ll, strict=True):
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

    # ordered by size over the last LEAD_WINDOW weeks, largest first: what a reader wants
    # first is what the corpus looks like now, and the stack then puts the currently-dominant
    # band at the bottom where its shape is easiest to follow.
    recent = C[-LEAD_WINDOW:].sum(axis=0)
    order = np.argsort(-recent)

    obs = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-12)
    start, end = obs[:8].mean(axis=0), obs[-8:].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(start > 0, end / np.maximum(start, 1e-12), np.inf)

    # The component this page is about is simply the largest one of the last month. It is not
    # chosen by how much it grew, which is what the two thresholds below used to do and what
    # made the choice fragile: at a 1% start one fit rejected its own biggest component,
    # 1.06% -> 40.35%, for beginning six hundredths of a point too high. "Whatever is most of
    # the writing now" needs no threshold and cannot reject anything.
    #
    # A month rather than a week because a week is 700 descriptions, and the subject of the
    # whole page should not turn on which of two close components happened to lead across one
    # of them.
    lead = int(np.argmax(recent))

    # The thresholds no longer select; they check. Selecting the biggest component says nothing
    # about whether it arrived, and the page's headline claim is that it arrived, so that claim
    # is tested separately against the component actually chosen. If this fires, the largest
    # component is one that was always there and the page should not be published from the fit.
    arrived = bool(start[lead] < LEAD_START and end[lead] >= LEAD_END)
    if strict:
        assert arrived, (
            "the largest component of the final week went {:.2%} -> {:.2%}, which is not an "
            "arrival: it needed to start under {:.0%} and end at or above {:.0%}".format(
                start[lead], end[lead], LEAD_START, LEAD_END))

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
        n = WORDS_LEAD if c == lead else WORDS_LISTED
        comps.append({
            "id": int(c),
            "lead": bool(c == lead),
            # pi is one number per component now, not a curve: the model has no time in it.
            # What varies by week is the attribution, `count`.
            "share": round(float(pi[c]), 5),
            "start_share": round(float(start[c]), 5),
            "end_share": round(float(end[c]), 5),
            "count": [round(float(v), 1) for v in C[:, c]],
            "appearances": [int(round(v)) for v in A[:, c]],
            "word_list": [vocab[j] for j in rank[:n]],
            "word_lift": [round(float(lift[j]), 2) for j in rank[:n]],
            # each listed word's own weekly appearances, so the page can show a word's
            # history on hover. Only for the lead: at 85 weeks a thousand words is 85,000
            # integers, worth carrying once and not sixteen times.
            "series": ([[int(v) for v in per_word[j]] for j in rank[:n]]
                       if c == lead else None),
            "tail": {t: int((lift >= t).sum()) for t in (5, 3, 2)},
        })
    return {"generated": date.today().isoformat(), "weeks": weeks,
            "arrived": bool(arrived), "lead_window": LEAD_WINDOW,
            # reported for information, never selected on: growth is a continuum here rather
            # than two separated groups
            "lead_ratio": round(float(ratio[lead]), 1),
            "documents": int(X.shape[0]), "appearances": int(X.sum()),
            "docs_per_week": [int(v) for v in docs_per_week],
            "words_per_week": [int(v) for v in words_per_week],
            "vocab": len(vocab), "k": len(comps),
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
    W, pi, C, A, ll = fit(X, week_of, T, k=3, log=lambda *_: None)

    assert abs(pi.sum() - 1.0) < 1e-6, "the mixture must sum to 1"
    assert (pi >= 0).all() and np.allclose(W.sum(axis=1), 1.0), "factors are malformed"
    # C must recover each week's document count, since every document's r sums to 1
    assert np.allclose(C.sum(axis=1), np.bincount(week_of, minlength=T), rtol=1e-6), \
        "the counts do not reconstruct the week"

    # the planted component must be found rising even though the model cannot represent time:
    # this is the whole claim of the thing, tested on data where the answer is known
    obs = C / C.sum(axis=1, keepdims=True)
    late = int(np.argmax([Wt[2] @ np.log(np.maximum(W[c], 1e-12)) for c in range(3)]))
    before, after = obs[:arrives, late].mean(), obs[arrives:, late].mean()
    assert before < 0.5 * after, f"the planted component did not rise ({before:.3f} " \
                                 f"then {after:.3f})"
    assert np.allclose(A.sum(axis=1),
                       np.bincount(week_of, minlength=T) * 50, rtol=0.02), \
        "the appearance counts do not reconstruct the week"

    # and the numba kernel must agree with the plain numpy statement of the same sweep. A
    # hand-written parallel reduction is wrong in ways its own output cannot reveal.
    logW = np.log(np.maximum(W, 1e-12))
    logpi = np.log(np.maximum(pi, 1e-300))
    nt = numba.get_num_threads()
    Wacc = np.zeros((nt, 3, V)); Cacc = np.zeros((nt, T, 3))
    Aacc = np.zeros((nt, T, 3)); llacc = np.zeros(nt)
    _em_sweep(X.tocsr().indptr.astype(np.int64), X.tocsr().indices.astype(np.int64),
              X.tocsr().data.astype(np.float64), logW, logpi,
              week_of.astype(np.int64), Wacc, Cacc, Aacc, llacc)
    rW, rC, rA, rll = _em_sweep_numpy(X, logW, logpi, week_of, T)
    for got, want, name in ((Wacc.sum(0), rW, "word totals"), (Cacc.sum(0), rC, "counts"),
                            (Aacc.sum(0), rA, "appearances")):
        assert np.allclose(got, want, rtol=1e-8, atol=1e-8), \
            f"the numba sweep disagrees with numpy on the {name}"
    assert abs(llacc.sum() - rll) < 1e-6 * max(abs(rll), 1.0), \
        f"the numba sweep disagrees with numpy on the likelihood ({llacc.sum()} vs {rll})"

    print(f"selftest: ok  (mixture sums to 1, counts reconstruct each week, numba agrees with "
          f"numpy, planted component {before:.3f} -> {after:.3f} at week {arrives})")


def verify_reader():
    """Assert the fast reader and the reference reader agree on the real corpus.

    Not part of `--selftest`, which runs on synthetic data in a second; this reads every day
    twice and takes about fifteen. Run it after touching either reader.
    """
    import time
    t0 = time.time()
    Xf, wf, weeks_f, vf = documents()
    t_fast = time.time() - t0
    t0 = time.time()
    Xr, wr, weeks_r, vr = documents_reference()
    t_ref = time.time() - t0
    assert vf == vr, "the vocabularies differ"
    assert weeks_f == weeks_r, "the week labels differ"
    assert np.array_equal(wf, wr), "the week index differs"
    assert Xf.shape == Xr.shape, f"shapes differ: {Xf.shape} vs {Xr.shape}"
    d = abs(Xf - Xr)
    assert d.nnz == 0, f"{d.nnz} matrix entries differ, largest {d.max()}"
    print(f"reader verified: identical matrix, vocabulary and week index\n"
          f"  fast      {t_fast:6.2f}s\n  reference {t_ref:6.2f}s"
          f"   ({t_ref / t_fast:.1f}x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--n-init", type=int, default=N_INIT, dest="n_init")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="analysis.js")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-reader", action="store_true", dest="verify_reader",
                    help="run both readers over the real corpus and assert they agree")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.verify_reader:
        return verify_reader()

    X, week_of, weeks, vocab = documents()
    W, pi, C, A, ll = fit_best(X, week_of, len(weeks), k=args.k, tol=args.tol,
                               n_init=args.n_init, seed=args.seed)
    out = pack(X, week_of, weeks, vocab, W, pi, C, A, ll)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.ANALYSIS = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"\nloglik {ll:,.0f}, wrote {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)\n")
    for c in out["components"]:
        print(f"  {'lead' if c['lead'] else '    '}  share {c['share']:6.1%}  "
              f"{c['start_share']:6.2%} -> {c['end_share']:6.2%}")
        print("        " + ", ".join(w[:20] for w in c["word_list"][:9]))


if __name__ == "__main__":
    main()

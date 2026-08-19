"""Find groups of words that arrived on GitHub at the same time.

The model, and the whole of it:

    rate(word w, week t) = sum over components c of  weight(c, t) * profile(c, w)

A component is a way of writing. `profile[c]` says which words it uses, fixed across the
whole window; `weight[c]` says how much of it was in the air each week, shared by every
word in it. Nothing here knows what a model release is. The claim is only that if an
assistant brings a bundle of habits, that bundle is a rank-one piece of the matrix, and
the factorisation has to spend a component on it.

Both factors are non-negative, because a rate and a share of a rate both are. That is what
makes the components additive pieces of the observed rate rather than arbitrary
directions, and so readable at all. A word that falls is still representable: a component
whose weight is high early and low later.

Run `python analyze.py` to write analysis.js. Run `python analyze.py --selftest` to check
the invariants that would silently invalidate the result.
"""

import argparse
import glob
import gzip
import json
import os
from datetime import date, timedelta

import numpy as np
from sklearn.decomposition import NMF

# --------------------------------------------------------------------------- corpus

ANCHOR = date(2024, 1, 1)          # a Monday; weeks starting mid-week would straddle
                                   # two partial weekends and mix the author mix
WEEK_GLOB = "data/weeks/*.jsonl.gz"

# The only normalisation applied to text. No stemming, no n-grams, no stopword list.
STRIP = "\"'`*_~<>|#.,;:!?()[]{}"
MIN_WORDS = 5                      # a body needs this many distinct words to be prose
MIN_DF = 60                        # a word needs this many documents across the window
DOCS_PER_WEEK = 350               # see the note in `read_week`

# ----------------------------------------------------------------------------- model

K = 8                              # two register components plus six vendor templates,
                                   # the templates acting as a control
ALPHA_W = 5e-3                     # L1 on the weekly weights; see `fit`
MAX_ITER = 800
SEED = 0
WORDS_CHARTED = 16                 # sparklines per component
WORDS_LISTED = 40                  # words named per component


def words(body):
    """The distinct words of one document.

    Purely numeric tokens are dropped. They are dates, versions, counts and line numbers
    rather than vocabulary, and they are an active nuisance: the calendar advances every
    week, so a bare `10` or `2026` arrives and departs on a schedule of its own. `apr`
    went from 0.05% to 9% of documents at the end of one March. Month abbreviations are
    words and are not filtered, so that one is an artifact to read past.
    """
    return {w for w in (t.strip(STRIP) for t in body.lower().split())
            if w and not w.isdigit()}


def read_week(path, index):
    """One week's documents as word sets, repeats collapsed and then thinned.

    Two documents with the identical set of words count once. This is about text, not
    authorship: one ordinary account once posted 147 copies of the same sentence inside a
    fortnight, 16% of that fortnight, and every word of its template moved with it. It
    applies inside the week and not across the window on purpose -- collapsing globally
    would make a template that runs for months look as though it started or stopped.

    Then the week is thinned to a common size. Sampling the same number of hours from
    every week does not give the same number of documents; volume swings by a factor of
    two. That matters because real text is overdispersed -- words cluster inside
    repositories -- so a rate computed on more documents comes out inflated rather than
    merely more precise, and busy weeks would outrank busy language. The subsample is
    seeded on the week and the file order is fixed, so both passes see the same documents.
    """
    seen, docs = set(), []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            w = words(json.loads(line)["body"])
            if len(w) < MIN_WORDS:
                continue
            key = frozenset(w)
            if key in seen:
                continue
            seen.add(key)
            docs.append(w)
    if len(docs) > DOCS_PER_WEEK:
        keep = np.random.default_rng([SEED, index]).choice(
            len(docs), DOCS_PER_WEEK, replace=False)
        docs = [docs[i] for i in np.sort(keep)]
    return docs


def build(log=print):
    """The week x word matrix of document counts, in two passes over the text."""
    files = sorted(glob.glob(WEEK_GLOB))
    if not files:
        raise SystemExit(f"no weeks in {WEEK_GLOB} -- run `python fetch_week.py --all`")
    weeks = [os.path.basename(f)[:10] for f in files]

    total, n = {}, np.zeros(len(files), dtype=np.int64)
    for i, f in enumerate(files):
        for w in read_week(f, i):
            n[i] += 1
            for word in w:
                total[word] = total.get(word, 0) + 1
    vocab = sorted(w for w, c in total.items() if c >= MIN_DF)
    index = {w: j for j, w in enumerate(vocab)}
    log(f"{len(files)} weeks, {n.sum():,} documents, {len(total):,} distinct words, "
        f"{len(vocab):,} kept at df >= {MIN_DF}")

    X = np.zeros((len(files), len(vocab)), dtype=np.int32)
    for i, f in enumerate(files):
        for w in read_week(f, i):
            for word in w:
                j = index.get(word)
                if j is not None:
                    X[i, j] += 1
    return X, n, weeks, vocab


def normalise(X, n):
    """Rates, each word divided by its own average over the window.

    This one line is what makes the result visible, so the measurements are worth keeping
    next to it. Scored by whether the fit separates the prose register at all, and how
    much mass lands on the twenty commonest words:

        NMF, these normalised rates      0.4%   found
        NMF, raw rates                  14.4%   not found
        NMF, counts with a KL loss      14.5%   not found
        LDA, counts                     15.6%   not found
        LDA, these normalised rates      0.4%   found

    So it is the input and not the model. Pruning frequent words is not a substitute: only
    twenty words here exceed 25% document frequency, and removing them promotes the next
    tier -- that fit spends 68% of its mass on `it, if, not, as, new, you, change, have`.
    There is no threshold between function words and content words, because the problem is
    the scale of the counts at every level.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.nan_to_num(X / n[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    base = rates.mean(axis=0)
    assert (base > 0).all(), "a word in the vocabulary never appeared"
    return rates / base, base


def fit(X, n, k=K, alpha_w=ALPHA_W, seed=SEED):
    """Factorise. Returns (weight, profile, base).

    `solver="cd"` is the one that reaches exact zeros; `init="nndsvda"` because plain
    `nndsvd` seeds zeros a solver cannot leave.

    An assistant contributes nothing before it exists, so the honest shape for a weight
    curve is zeros and then something, and L1 is the penalty that gives exact zeros rather
    than small ones. Swept on this corpus:

        alpha_W   error   weeks at exactly 0   largest component
        0          892           37%                 36%
        1e-3       892           39%                 37%
        5e-3       892           48%                 39%      <- chosen, free
        1e-2       893           57%                 43%
        3e-2       915           67%                 52%      <- fit degrading

    scikit-learn multiplies `alpha_W` by the number of words, so this value is calibrated
    to a vocabulary of roughly 4,800 and does not transfer to a much larger one. Hence the
    warning below: one component swallowing the corpus is what over-penalising looks like.
    """
    A, base = normalise(X, n)
    model = NMF(n_components=k, init="nndsvda", solver="cd",
                alpha_W=alpha_w, l1_ratio=1.0,
                max_iter=MAX_ITER, random_state=seed)
    weight = model.fit_transform(A)
    return weight, model.components_, base, float(model.reconstruction_err_)


def pack(X, n, weeks, vocab, weight, profile):
    """Shape the fit into the structure the page reads."""
    contrib = weight.mean(axis=0)[:, None] * profile
    total = np.maximum(contrib.sum(axis=0), 1e-12)
    mass = contrib.sum(axis=1) / max(contrib.sum(), 1e-12)
    pct = 100.0 * X / np.maximum(n, 1)[:, None]

    def r(a, nd):
        return [round(float(v), nd) for v in a]

    components = []
    for c in np.argsort(-mass):
        share = contrib[c] / total
        # Ranking by the profile alone would put `the` atop every component, because every
        # component has to reproduce `the`. Contribution times share asks the useful
        # question instead: of everything that made people write this word, how much came
        # from this component?
        order = np.argsort(-(contrib[c] * share))
        w = weight[:, c]
        peak = max(w.max(), 1e-12)
        components.append({
            "id": int(c),
            "mass": round(float(mass[c]), 4),
            "peak_week": weeks[int(np.argmax(w))],
            "zero_weeks": round(float((w == 0).mean()), 3),
            "weight": r(w / peak, 4),
            "words": [{"word": vocab[j],
                       "share": round(float(share[j]), 3),
                       "lift": round(float(profile[c][j]), 3),
                       "pct": r(pct[:, j], 3)}
                      for j in order[:WORDS_CHARTED]],
            "word_list": [vocab[j] for j in order[:WORDS_LISTED]],
        })
    return {
        "generated": date.today().isoformat(),
        "source": "GitHub pull request descriptions, sampled from the search API",
        "weeks": weeks,
        "documents": int(n.sum()),
        "docs_per_week": [int(v) for v in n],
        "vocab": len(vocab),
        "k": len(components),
        "components": components,
    }


def selftest():
    """The invariants whose silent failure would invalidate the result."""
    rng = np.random.default_rng(0)
    T, V, on, size = 60, 300, 30, 12
    p = rng.uniform(0.01, 0.30, size=V)
    n = np.full(T, 20_000, dtype=np.int64)
    rates = np.tile(p, (T, 1))
    rates[on:, :size] *= 4.0                      # a bundle arriving at week `on`
    rates[:, -20:] = 0.75                         # words as common as `the`
    X = rng.binomial(n[:, None], np.clip(rates, 0, 1)).astype(np.int64)
    vocab = [f"w{j}" for j in range(V)]

    A, base = normalise(X, n)
    assert np.allclose(A.mean(axis=0), 1.0, atol=1e-9), "normalisation is wrong"
    assert (A >= 0).all() and (base > 0).all()

    weight, profile, base, err = fit(X, n, k=4)
    out = pack(X, n, [f"w{t}" for t in range(T)], vocab, weight, profile)
    assert abs(sum(c["mass"] for c in out["components"]) - 1.0) < 1e-3, "mass is not a share"
    assert all(len(c["weight"]) == T for c in out["components"])
    assert all(len(w["pct"]) == T for c in out["components"] for w in c["words"])

    moved = {f"w{j}" for j in range(size)}
    hit = [c for c in out["components"]
           if len(moved & set(c["word_list"][:size + 6])) >= 4]
    assert hit, "the arriving bundle did not become a component"
    w = np.array(hit[0]["weight"])
    assert w[:on].mean() < 0.25 * w[on:].mean(), "its weight does not switch on"

    # L1 must buy sparsity, and the bundle above must survive it -- `weight` is the
    # penalised fit, so the switch-on assertion already covered that. What is deliberately
    # not asserted is the cost: on the real corpus L1 is free (error 892 either way), but
    # this fixture's normalised matrix is nearly rank-one, so its baseline error is ~2.7
    # and a relative bound on it measures the fixture rather than the penalty.
    plain, _, _, plain_err = fit(X, n, k=4, alpha_w=0.0)
    assert (weight == 0).mean() > (plain == 0).mean(), "L1 did not add sparsity"
    print(f"selftest: ok  (exact zeros {(plain == 0).mean():.0%} -> "
          f"{(weight == 0).mean():.0%}, error {plain_err:.2f} -> {err:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K, help=f"components (default {K})")
    ap.add_argument("--out", default="analysis.js")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X, n, weeks, vocab = build()
    weight, profile, base, err = fit(X, n, k=args.k)
    out = pack(X, n, weeks, vocab, weight, profile)

    if out["components"][0]["mass"] > 0.55:
        print(f"WARNING largest component holds {out['components'][0]['mass']:.0%} of "
              f"mass -- ALPHA_W ({ALPHA_W}) is likely too strong for a "
              f"{len(vocab)}-word vocabulary")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.ANALYSIS = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"error {err:.1f}, wrote {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)\n")
    for c in out["components"]:
        print(f"  {c['mass']:5.1%}  peak {c['peak_week']}  "
              f"{c['zero_weeks']:.0%} weeks at zero")
        print(f"         " + ", ".join(w["word"][:22] for w in c["words"][:8]))


if __name__ == "__main__":
    main()

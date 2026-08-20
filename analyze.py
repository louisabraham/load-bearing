"""Find groups of words that arrived on GitHub at the same time.

The model, and the whole of it. Let X be words by weeks, where X[v, t] counts every
appearance of word v in week t. Factorise it:

    X  ~=  W H          W: words by k, columns summing to 1
                        H: k by weeks, non-negative

Each column of W is a probability distribution over the vocabulary -- a way of writing.
Each row of H says how much of that way of writing was in the air each week. Because the
columns of W are normalised, the column sums of H are the week's word count, so
H[c, t] / sum_c H[c, t] is component c's share of everything written that week.

Nothing here knows what a model release is. The claim is only that if an assistant brings
a bundle of habits, that bundle is a rank-one piece of the matrix, and the factorisation
has to spend a component on it.

Run `python analyze.py` to write analysis.js, `--selftest` to check the invariants.
"""

import argparse
import glob
import re
import gzip
import json
import os
from collections import Counter
from datetime import date

import numpy as np
from sklearn.decomposition import NMF

# --------------------------------------------------------------------------- corpus

ANCHOR = date(2024, 1, 1)          # a Monday; weeks starting mid-week would straddle
                                   # two partial weekends and mix the author mix
WEEK_GLOB = "data/weeks/*.jsonl.gz"

# A word is a run of letters, digits, hyphens and underscores containing at least one
# letter -- so `load-bearing`, `snake_case` and `--all-targets` survive whole, while `/`,
# backtick, `:` and `>` are separators rather than characters a word may contain. Whole
# http(s) links are pulled out first and kept as single tokens, before that split can
# shred them. No stemming, no n-grams, no stopword list.
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}]+")
TAG_RE = re.compile(r"<[a-z/!][^<>]*>")   # html markup, not prose: `a > b` is not a tag
EM_DASH = "\u2014"                       # a word by fiat; see `tokens`
WORD_RE = re.compile(r"[a-z0-9_-]*[a-z][a-z0-9_-]*")
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

# ----------------------------------------------------------------------------- model

K = 16
LOSS = "kl"                        # "kl" or "l2"; see `fit` -- they trade separation
                                   # against a working L1
ALPHA_H = 2e-2                     # L1 on the weekly activations; see `fit`
MAX_ITER = 600
SEED = 0
WORDS_CHARTED = 16
WORDS_LISTED = 40
WORDS_MOST_USED = 12


def tokens(body):
    """Every appearance of every word in one document.

    Links are taken first, whole: `[bugbot](https://cursor.com/x)` yields `bugbot` and the
    link, where splitting on punctuation first would have produced `bugbot](https` and a
    trail of fragments. Those fragments were real: they used to rank among a component's
    most representative words.

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

    Requiring a letter drops what is left of numbers and rules: `27.49`, `589/1000`,
    `2025-06-24`, `-------`. The arrow and `+` go with them. The em dash is the one
    exception, taken before the split and counted as a word of its own -- it earns that by
    going from 0.0 appearances per 10,000 words in early 2024 to 123.0 in mid-2026, the
    sharpest single signal here.
    """
    body = body.lower()
    out = [m.group(0).rstrip(".,;:!?") for m in URL_RE.finditer(body)]
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
        w = w.strip("_").rstrip("-")
        if w and any(c.isalpha() for c in w):
            out.append(w)
    return out


def read_week(path):
    """One week's word counts and per-word document counts, repeats collapsed.

    Two documents with the identical set of words count once. This is about text, not
    authorship: one ordinary account once posted 147 copies of the same sentence inside a
    fortnight, 16% of it, and every word of its template moved with it. It applies inside
    the week and not across the window on purpose -- collapsing globally would make a
    template that runs for months look as though it started or stopped.

    The week is then cut off at a common number of documents. Sampling the same number of
    hours from every week does not give the same number of documents; volume swings by a
    factor of two. Text is overdispersed -- words cluster inside repositories -- so a rate
    computed on more documents comes out inflated rather than merely more precise, and
    busy weeks would outrank busy language. Document *length* still varies threefold even
    after this, which is why the model reports H as a share of the week rather than raw.
    """
    seen, kept, counts, docs = set(), 0, Counter(), Counter()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            t = tokens(json.loads(line)["body"])
            if len(set(t)) < MIN_WORDS:
                continue
            key = frozenset(t)
            if key in seen:
                continue
            seen.add(key)
            kept += 1
            if kept > DOCS_PER_WEEK:
                break
            counts.update(t)
            docs.update(key)          # once per document, for the breadth filter
    return counts, docs


def build(log=print):
    """X[word, week] -- every appearance, not one per document."""
    files = sorted(glob.glob(WEEK_GLOB))
    if not files:
        raise SystemExit(f"no weeks in {WEEK_GLOB} -- run `python fetch_week.py --all`")
    weeks = [os.path.basename(f)[:10] for f in files]

    per_week, docs = [], Counter()
    for f in files:
        counts, seen = read_week(f)
        per_week.append(counts)
        docs.update(seen)
    total = Counter()
    for c in per_week:
        total.update(c)
    vocab = sorted(w for w, n in total.items() if n >= MIN_TF and docs[w] >= MIN_DF)
    index = {w: j for j, w in enumerate(vocab)}
    log(f"{len(files)} weeks, {total.total():,} word appearances, {len(total):,} distinct, "
        f"{len(vocab):,} kept at >= {MIN_TF} appearances in >= {MIN_DF} documents")

    X = np.zeros((len(vocab), len(files)))
    for t, c in enumerate(per_week):
        for w, n in c.items():
            j = index.get(w)
            if j is not None:
                X[j, t] = n
    return X, weeks, vocab


def fit(X, k=K, alpha_h=ALPHA_H, seed=SEED, max_iter=MAX_ITER, loss=LOSS):
    """Factorise X into word distributions and weekly activations.

    **Why Kullback-Leibler and not squared error.** X holds counts and the columns of W are
    probability distributions over words; together that is a multinomial mixture, and KL is
    its likelihood. Squared error instead assumes Gaussian noise of constant variance,
    which counts do not have -- the variance of a count grows with its mean, so squared
    error treats a swing of 50 in a word appearing 200,000 times as equally surprising as a
    swing of 50 in a word appearing 60 times. It is also better on the only test that
    matters, at k=16 with this vocabulary:

                             register component        exact zeros in H
        KL                   13.5% of mass, 0.001 -> 0.725        0%
        squared error         7.3% of mass, 0.009 -> 0.420       22%

    **The cost, stated plainly: under KL the L1 on H does nothing.** KL needs the
    multiplicative solver, which approaches zero without reaching it, so no exact zeros.
    Worse, this parameterisation cancels the penalty outright. NMF fixes W H only up to a
    diagonal rescaling, and the normalisation below pins that scale *after* fitting -- so
    the optimiser can satisfy an L1 on H by shrinking H uniformly and inflating W, which
    costs it nothing, and the rescaling then undoes the shrinkage exactly. Measured:
    alpha_H from 0 to 10 moves sum(H) by 0.7% and the per-week shape of H by 0.0085.
    An L1 is only meaningful where the scale is not free.

    With `loss="l2"` the penalty does bite -- 22% of H exactly zero at alpha_H=0, 25% at
    0.02, 47% at 0.2 -- at the cost of the separation in the table above. The knob is here
    so that trade is one line, not a rewrite.

    W is rescaled after fitting so each column sums to 1, which fixes the free scale at the
    one place it carries meaning and pushes it into H, whose column sums then recover each
    week's word count.
    """
    kw = dict(n_components=k, init="nndsvda", alpha_H=alpha_h, l1_ratio=1.0,
              max_iter=max_iter, random_state=seed)
    model = (NMF(solver="mu", beta_loss="kullback-leibler", **kw) if loss == "kl"
             else NMF(solver="cd", **kw))
    W = model.fit_transform(X)                 # words by k
    H = model.components_                      # k by weeks
    scale = W.sum(axis=0)
    scale[scale == 0] = 1.0
    W, H = W / scale, H * scale[:, None]
    assert np.allclose(W.sum(axis=0), 1.0), "columns of W are not distributions"
    return W, H, float(model.reconstruction_err_)


def pack(X, weeks, vocab, W, H):
    """Shape the fit into the structure the page reads."""
    words_per_week = X.sum(axis=0)
    overall = X.sum(axis=1) / X.sum()                    # corpus word distribution
    # H is reported as it is, in word appearances. Because the columns of W sum to 1, a
    # column of H sums to that week's word count -- so H[c, t] is the number of appearances
    # in week t attributable to component c, and every component's curve is in the same
    # units. Not divided through by the week: that would hide how much was written, and
    # the weeks differ threefold in length even after the document cap.
    mass = H.sum(axis=1) / H.sum()
    per10k = 1e4 * X / np.maximum(words_per_week, 1)     # a word's rate, comparable

    def r(a, nd):
        return [round(float(v), nd) for v in a]

    components = []
    for c in np.argsort(-mass):
        p = W[:, c]
        lift = p / np.maximum(overall, 1e-12)
        # A component's most representative words are the ones whose probability under it
        # is furthest above their probability in the corpus as a whole. No support floor:
        # every word here already appears at least MIN_TF times, and flooring on
        # probability throws away exactly the rare-but-concentrated words this is for --
        # `load-bearing` ranks 24th in its component by lift and 6,062nd once floored.
        order = np.argsort(-lift)
        # kept as a second view, because it answers the other question: what a component
        # is mostly *made* of. Ranking by probability alone would list `the` under every
        # component, so this is the pointwise contribution to KL(component || corpus).
        used = np.argsort(-(p * np.log(np.maximum(lift, 1e-12))))
        components.append({
            "id": int(c),
            "mass": round(float(mass[c]), 4),
            "peak_week": weeks[int(np.argmax(H[c]))],
            "start": int(round(H[c][:8].mean())),        # appearances a week, first/last
            "end": int(round(H[c][-8:].mean())),         # two months
            "weight": [int(round(v)) for v in H[c]],     # absolute, in appearances
            "words": [{"word": vocab[j],
                       "prob": round(float(p[j]), 6),
                       "lift": round(float(lift[j]), 2),
                       "per10k": r(per10k[j], 2)}
                      for j in order[:WORDS_CHARTED]],
            "word_list": [vocab[j] for j in order[:WORDS_LISTED]],
            "most_used": [vocab[j] for j in used[:WORDS_MOST_USED]],
        })
    return {
        "generated": date.today().isoformat(),
        "source": "GitHub pull request descriptions, sampled from the search API",
        "weeks": weeks,
        "appearances": int(X.sum()),
        "words_per_week": [int(v) for v in words_per_week],
        "vocab": len(vocab),
        "k": len(components),
        "components": components,
    }


def selftest():
    """The invariants whose silent failure would invalidate the result."""
    rng = np.random.default_rng(0)
    V, T, on, size = 300, 60, 30, 12
    p = rng.uniform(0.2, 3.0, size=V)
    rate = np.tile(p[:, None], (1, T))
    rate[:size, on:] *= 6.0                        # a bundle arriving at week `on`
    rate[-20:, :] = 60.0                           # words as common as `the`
    X = rng.poisson(rate * 300).astype(float)
    vocab = [f"w{j}" for j in range(V)]

    W, H, err = fit(X, k=4, max_iter=400)
    assert np.allclose(W.sum(axis=0), 1.0), "columns of W must sum to 1"
    assert (W >= 0).all() and (H >= 0).all(), "factors must stay non-negative"

    out = pack(X, [f"w{t}" for t in range(T)], vocab, W, H)
    assert abs(sum(c["mass"] for c in out["components"]) - 1.0) < 1e-3
    assert all(len(c["weight"]) == T for c in out["components"])
    assert all(len(w["per10k"]) == T for c in out["components"] for w in c["words"])
    # W's columns summing to 1 means H's columns must recover the week's word count
    tot = np.sum([np.array(c["weight"]) for c in out["components"]], axis=0)
    want = X.sum(axis=0)
    assert np.allclose(tot, want, rtol=0.05), "H does not reconstruct the week's total"

    moved = {f"w{j}" for j in range(size)}
    hit = [c for c in out["components"]
           if len(moved & set(c["word_list"] + c["most_used"])) >= 4]
    assert hit, "the arriving bundle did not become a component"
    w = np.array(hit[0]["weight"], dtype=float)
    assert w[:on].mean() < 0.5 * w[on:].mean(), "its activation does not rise"

    assert err >= 0
    print(f"selftest: ok  (W columns sum to 1, H recovers each week's total, "
          f"bundle recovered at {hit[0]['mass']:.0%} of mass)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K, help=f"components (default {K})")
    ap.add_argument("--out", default="analysis.js")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X, weeks, vocab = build()
    W, H, err = fit(X, k=args.k)
    out = pack(X, weeks, vocab, W, H)

    if out["components"][0]["mass"] > 0.55:
        print(f"WARNING largest component holds {out['components'][0]['mass']:.0%} of "
              f"mass -- ALPHA_H ({ALPHA_H}) may be too strong for a "
              f"{len(vocab)}-word vocabulary")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.ANALYSIS = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"KL divergence {err:.1f}, wrote {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)\n")
    for c in out["components"]:
        print(f"  {c['mass']:5.1%}  peak {c['peak_week']}  "
              f"{c['start']:,} -> {c['end']:,} appearances a week")
        print("         " + ", ".join(w["word"][:20] for w in c["words"][:8]))
        print("         most used:   " + ", ".join(w[:20] for w in c["most_used"][:8]))


if __name__ == "__main__":
    main()

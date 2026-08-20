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

    python mixture.py               # writes mixture.js, read by mixture.html
    python mixture.py --selftest    # recovers a planted component from synthetic data
"""

import argparse
import glob
import gzip
import json
import os
from collections import Counter
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

import analyze                      # the corpus rules live there and are shared verbatim

K = 12
LAMBDA = 40.0                       # smoothness; see `fit_pi`
OUTER = 12                          # EM passes per restart
N_INIT = 10                         # restarts; see `fit_best`
SEED = 0
WORDS_LISTED = 40
WORDS_CHARTED = 16


# ------------------------------------------------------------------------- corpus

def documents(log=print):
    """One row per document. Same filters as analyze.py, applied per document.

    The other model aggregates each week into a bag of words; this one needs the documents
    themselves, because a document is what gets attributed to a component.
    """
    files = sorted(glob.glob(analyze.WEEK_GLOB))
    if not files:
        raise SystemExit(f"no weeks in {analyze.WEEK_GLOB} -- run `python fetch_week.py`")
    weeks = [os.path.basename(f)[:10] for f in files]

    docs, week_of, tf, df = [], [], Counter(), Counter()
    for t, f in enumerate(files):
        seen, kept, by_author = set(), 0, Counter()
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                toks = analyze.tokens(row["body"])
                if len(set(toks)) < analyze.MIN_WORDS:
                    continue
                key = frozenset(toks)
                if key in seen:
                    continue
                author = row.get("author") or ""
                if by_author[author] >= analyze.MAX_PER_AUTHOR:
                    continue
                by_author[author] += 1
                seen.add(key)
                kept += 1
                if kept > analyze.DOCS_PER_WEEK:
                    break
                c = Counter(toks)
                docs.append(c)
                week_of.append(t)
                tf.update(c)
                df.update(key)

    vocab = sorted(w for w, n in tf.items()
                   if n >= analyze.MIN_TF and df[w] >= analyze.MIN_DF)
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
    keep = np.asarray(X.sum(axis=1)).ravel() >= analyze.MIN_WORDS
    X, week_of = X[keep], np.asarray(week_of)[keep]
    log(f"{len(files)} weeks, {X.shape[0]:,} documents, {X.sum():,.0f} appearances, "
        f"{len(vocab):,} words")
    return X, week_of, weeks, vocab


# -------------------------------------------------------------------------- model

def fit_pi(C, lam=LAMBDA):
    """Prevalences: maximise sum C log pi - lam sum (pi_t - pi_{t-1})^2.

    Subject to each week summing to 1. Solved on a per-week softmax, so the constraint holds
    by construction and the problem is unconstrained in those coordinates -- L-BFGS handles
    it directly rather than needing a projection at every step.

    The smoothness term exists because weekly counts are noisy at a few hundred documents a
    week, not because a component is expected to be monotone. It binds, and for a structural
    reason worth contrasting with the L1 in analyze.py, which does not: there the columns of
    W are normalised *after* fitting, so an L1 on H could be satisfied by shrinking H and
    inflating W at no cost and the rescaling undid it exactly. Here pi sums to 1 in every
    week by construction, so the scale is not free and there is nothing to game. Swept on
    the corpus at k=12, total squared week-to-week change:

        lambda        0      10      40     200    1000
        roughness  1.870   1.765   1.517   0.909   0.333

    A 5.6-fold reduction, monotone, costing 0.002% of the log-likelihood at the far end.
    LAMBDA sits at the low end because the aim is to take the jitter off the curve rather
    than to flatten it.

    An L1 on pi itself would do nothing at all, for a third reason: on the simplex
    ||pi_t||_1 = 1 identically, so it is a constant with zero gradient. L1 induces sparsity
    by trading against magnitude, and on a simplex the magnitude is already spent.
    """
    T, K_ = C.shape

    def softmax(theta):
        z = theta - theta.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def neg(v):
        pi = softmax(v.reshape(T, K_))
        p = np.maximum(pi, 1e-12)
        d = np.diff(pi, axis=0)
        obj = (C * np.log(p)).sum() - lam * (d ** 2).sum()
        g = C / p
        g[1:] -= 2 * lam * d
        g[:-1] += 2 * lam * d
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


def fit(X, week_of, T, k=K, lam=LAMBDA, outer=OUTER, seed=SEED, log=print):
    """One EM run. Returns (W, pi, log-likelihood)."""
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

        pi = fit_pi(C, lam)                                   # 4. prevalences
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
             seed=SEED, log=print):
    """Fit `n_init` times from different seeds and keep the highest likelihood.

    One run is not enough: EM finds different local optima here, and the worst of them mix a
    component with something else and give it two thirds of the peak the good fits find. The
    likelihood tells them apart reliably, so the only cost is time -- about a second a
    restart.
    """
    best = None
    for i in range(n_init):
        W, pi, C, A, ll = fit(X, week_of, T, k, lam, outer, seed + i,
                              log=lambda *_: None)
        log(f"  restart {i + 1}/{n_init}  loglik {ll:,.0f}")
        if best is None or ll > best[-1]:
            best = (W, pi, C, A, ll)
    log(f"  kept loglik {best[-1]:,.0f}")
    return best


# --------------------------------------------------------------------------- out

def pack(X, week_of, weeks, vocab, W, pi, C, A, ll, lam):
    overall = np.asarray(X.sum(axis=0)).ravel()
    overall = overall / overall.sum()
    docs_per_week = np.bincount(week_of, minlength=len(weeks))
    words_per_week = np.zeros(len(weeks))
    np.add.at(words_per_week, week_of, np.asarray(X.sum(axis=1)).ravel())
    per10k = np.zeros((len(vocab), len(weeks)))
    for t in range(len(weeks)):
        sel = week_of == t
        if sel.any():
            per10k[:, t] = 1e4 * np.asarray(X[sel].sum(axis=0)).ravel() \
                / max(words_per_week[t], 1)

    # ordered by when each component peaks, so the stacked view reads left to right
    order = np.argsort([int(np.argmax(pi[:, c])) for c in range(W.shape[0])])
    comps = []
    for c in order:
        lift = W[c] / np.maximum(overall, 1e-12)
        rank = np.argsort(-lift)
        comps.append({
            "id": int(c),
            "share": round(float(pi[:, c].mean()), 5),
            "peak": round(float(pi[:, c].max()), 5),
            "peak_week": weeks[int(np.argmax(pi[:, c]))],
            "start_share": round(float(pi[:8, c].mean()), 5),
            "end_share": round(float(pi[-8:, c].mean()), 5),
            "prevalence": [round(float(v), 5) for v in pi[:, c]],
            "count": [round(float(v), 1) for v in C[:, c]],   # documents, absolute
            "appearances": [int(round(v)) for v in A[:, c]],   # absolute, length-weighted
            "words": [{"word": vocab[j], "lift": round(float(lift[j]), 2),
                       "prob": round(float(W[c][j]), 6),
                       "per10k": [round(float(v), 2) for v in per10k[j]]}
                      for j in rank[:WORDS_CHARTED]],
            "word_list": [vocab[j] for j in rank[:WORDS_LISTED]],
            "word_lift": [round(float(lift[j]), 2) for j in rank[:WORDS_LISTED]],
            # how long the tail is. The listed words are the head of a smooth decline, not a
            # natural group: lift falls from about 9 at rank 1 to about 6 at rank 80 with no
            # cliff anywhere, so any cut is arbitrary and the reader should see the numbers.
            "tail": {t: int((lift >= t).sum()) for t in (5, 3, 2)},
        })
    return {"generated": date.today().isoformat(), "weeks": weeks, "n_init": N_INIT,
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
    ap.add_argument("--out", default="mixture.js")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X, week_of, weeks, vocab = documents()
    W, pi, C, A, ll = fit_best(X, week_of, len(weeks), k=args.k, lam=args.lam,
                               outer=args.outer, n_init=args.n_init)
    out = pack(X, week_of, weeks, vocab, W, pi, C, A, ll, args.lam)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.MIXTURE = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"\nloglik {ll:,.0f}, wrote {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)\n")
    for c in out["components"]:
        print(f"  peaks {c['peak_week']}  mean {c['share']:6.1%}  peak {c['peak']:5.1%}  "
              f"{c['start_share']:.1%} -> {c['end_share']:.1%}")
        print("        " + ", ".join(w["word"][:20] for w in c["words"][:9]))


if __name__ == "__main__":
    main()

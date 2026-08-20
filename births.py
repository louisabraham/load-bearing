"""A mixture model where each way of writing has a birth week.

The alternative to penalising the weekly curve into sparsity: make the zero region a
parameter. Each component k has an unknown birth week tau_k and is *impossible* before it,
so no document from an earlier week can be attributed to it. Sparsity is then structural
rather than encouraged, and the thing we actually want to know -- when did this way of
writing start -- is estimated rather than eyeballed off a curve.

    W_k       a fixed distribution over the vocabulary
    tau_k     the birth week, unknown
    pi_tk     prevalence in week t, with sum_k pi_tk = 1 and pi_tk = 0 for t < tau_k

Generative, per document d written in week t(d):

    z_d ~ Categorical(pi_t)
    x_d | z_d = k ~ Multinomial(n_d, W_k)

After birth the only thing asked of the curve is smoothness, lam * sum_t (pi_tk -
pi_{t-1,k})^2, so a component may rise, fall or wander -- it just may not exist early.

Fitted by alternating four steps: attribute documents, refit the word distributions,
refit the prevalences under the birth constraint, then move each birth week to wherever
the regularised likelihood is highest.

    python births.py                # writes births.js, read by births.html
    python births.py --selftest     # recovers a planted birth week from synthetic data
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
LAMBDA = 40.0                       # smoothness after birth; see `fit_pi`
BIRTH_FRAC = 0.05                   # a component is born when it first reaches this much
BIRTH_SUSTAIN = 2                   # of its eventual peak, and holds it; see `birth_weeks`
OUTER = 12
N_INIT = 10                         # EM restarts; see `fit_best` -- one run is not enough
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

def _masked_softmax(theta, mask):
    out = np.zeros_like(theta)
    z = np.where(mask, theta, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    out = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)
    return out


def fit_pi(C, tau, lam=LAMBDA, T=None):
    """Prevalences: maximise sum C log pi - lam sum (pi_t - pi_{t-1})^2.

    Subject to each week summing to 1 and to pi_tk = 0 before birth. Solved on the free
    entries of a per-week softmax, so both constraints hold by construction and the problem
    is unconstrained in those coordinates -- L-BFGS then handles it directly rather than
    needing a projection at every step.

    The smoothness term is the only thing asked of the curve after birth. It exists because
    weekly counts are noisy at a few hundred documents a week, not because a component is
    expected to be monotone: nothing here stops one rising and falling again.

    Unlike the L1 in analyze.py, this penalty actually bites, and for a structural reason.
    There the columns of W are normalised *after* fitting, so the optimiser could satisfy an
    L1 on H by shrinking H and inflating W at no cost, and the rescaling undid it exactly.
    Here pi sums to 1 in every week by construction, so the scale is not free and there is
    nothing to game. Swept on the corpus at k=12, total squared week-to-week change:

        lambda        0      10      40     200    1000
        roughness  1.870   1.765   1.517   0.909   0.333

    A 5.6-fold reduction, monotone, and it costs 0.002% of the log-likelihood at the far
    end. The finding is untouched across the whole range: the one late birth stays at week
    89 and its peak within 1% of 67%. LAMBDA is set at the low end because the aim is to
    take the week-to-week jitter off the curve, not to flatten it.
    """
    T = T or C.shape[0]
    K_ = C.shape[1]
    mask = np.arange(T)[:, None] >= np.asarray(tau)[None, :]
    free = np.flatnonzero(mask.ravel())

    def unpack(v):
        theta = np.full(T * K_, -np.inf)
        theta[free] = v
        return _masked_softmax(theta.reshape(T, K_), mask)

    def neg(v):
        pi = unpack(v)
        p = np.maximum(pi, 1e-12)
        d = np.diff(pi, axis=0)
        obj = (C * np.log(p)).sum() - lam * (d ** 2).sum()
        g = C / p
        g[1:] -= 2 * lam * d
        g[:-1] += 2 * lam * d
        gt = pi * (g - (pi * g).sum(axis=1, keepdims=True))   # through the softmax
        return -obj, -gt.ravel()[free]

    v0 = np.zeros(free.size)
    res = minimize(neg, v0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 400, "maxfun": 500})
    return unpack(res.x), -res.fun


def responsibilities(logits, pi, week_of):
    """r_dk, and the total log-likelihood. Components not yet born get zero weight."""
    z = logits + np.log(np.maximum(pi[week_of], 1e-300))
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(axis=1, keepdims=True)
    return e / s, float((np.log(s) + m).sum())


def fit(X, week_of, T, k=K, lam=LAMBDA, outer=OUTER, seed=SEED, log=print):
    """Alternate the four steps. Returns (W, pi, tau, log-likelihood)."""
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
    tau = np.zeros(k, dtype=int)
    pi = np.full((T, k), 1.0 / k)

    ll = -np.inf
    for it in range(outer):
        logits = X @ np.log(np.maximum(W, 1e-12)).T          # 1. attribute
        r, ll = responsibilities(logits, pi, week_of)

        W = (r.T @ X) + 0.01                                  # 2. word distributions
        W = np.asarray(W)
        W /= W.sum(axis=1, keepdims=True)

        C = np.zeros((T, k))                                  # 3. weekly counts
        np.add.at(C, week_of, r)

        # 4 and 5 together: fit the curve with no birth constraint, read the births off it,
        # then refit under them. Re-deriving from the free fit each pass is what lets a
        # birth move earlier as well as later -- thresholding a masked curve could only
        # ever ratchet forward, so one early mistake would be permanent.
        free, _ = fit_pi(C, np.zeros(k, dtype=int), lam, T)
        tau = birth_weeks(free)
        pi, _ = fit_pi(C, tau, lam, T)
        log(f"  iter {it + 1:2d}  loglik {ll:,.0f}  "
            f"births {np.sort(tau).tolist()}")
    logits = X @ np.log(np.maximum(W, 1e-12)).T
    _, ll = responsibilities(logits, pi, week_of)
    return W, pi, tau, ll


def birth_weeks(pi, frac=BIRTH_FRAC, sustain=BIRTH_SUSTAIN):
    """Read each birth week off the curve: the first week a component reaches `frac` of its
    eventual peak and stays there for `sustain` weeks.

    This replaces searching over candidate births by likelihood, which cannot work: a later
    birth is a strictly tighter constraint, so it can only lower the likelihood, and the
    search always returns week zero. A threshold answers the question that was actually
    being asked -- when did this stop being negligible -- without needing a per-live-week
    cost invented to make the optimiser prefer lateness.

    Relative to the component's own peak rather than absolute, because components differ by
    two orders of magnitude in size. The sustain requirement is there so one noisy week
    cannot set a birth date.
    """
    T, k = pi.shape
    tau = np.zeros(k, dtype=int)
    for c in range(k):
        above = pi[:, c] >= frac * max(pi[:, c].max(), 1e-12)
        run, first = 0, 0
        for t in range(T):
            run = run + 1 if above[t] else 0
            if run >= sustain:
                first = t - sustain + 1
                break
        tau[c] = first
    return tau


def fit_best(X, week_of, T, k=K, lam=LAMBDA, outer=OUTER, n_init=N_INIT,
             seed=SEED, log=print):
    """Fit `n_init` times from different seeds, keep the highest likelihood, and report how
    much each birth week moved across the restarts.

    One run is not enough, and the reason is worth stating. EM finds different local optima
    here, and while the *component* is stable across them -- the same words, a peak between
    66% and 73% -- the *birth week* is not. Eight single runs put it anywhere from 2024-01-29
    to 2026-02-09, a spread of 23 months, because the birth threshold sits in the near-zero
    tail where a handful of documents decides whether a week clears it. Dropping 75
    documents of 47,373 moved it by 13 weeks.

    Likelihood selection fixes most of that. The three highest-likelihood runs of eight agree
    to within five weeks, and they are the runs that recover the component cleanly; the 2024
    outlier is the worst-likelihood fit, where the component is mixed with something else and
    peaks at 40% rather than 70%. Higher likelihood also means a later birth, consistently,
    because a better fit attributes the early tail elsewhere rather than to a component that
    had barely started.

    The spread across restarts is returned rather than discarded, because the birth week is
    the headline number here and quoting one to the week would be over-claiming.
    """
    runs = []
    for i in range(n_init):
        W, pi, tau, ll = fit(X, week_of, T, k, lam, outer, seed + i, log=lambda *_: None)
        runs.append((ll, W, pi, tau))
        log(f"  restart {i + 1}/{n_init}  loglik {ll:,.0f}  "
            f"latest birth week {int(tau.max())}")
    runs.sort(key=lambda r: -r[0])
    ll, W, pi, tau = runs[0]

    # match every other run's components to the best run's by profile similarity, so that
    # "this component's birth moved by N weeks" is a statement about the same component
    # over the better half of the restarts, not all of them: a fit rejected on likelihood
    # is not evidence about the parameter, and including it inflates the interval with a
    # solution we would not have used
    keep = runs[:max(2, (n_init + 1) // 2)]
    norm = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    spread = []
    for c in range(k):
        seen = []
        for _, W_i, _, tau_i in keep:
            n_i = W_i / np.maximum(np.linalg.norm(W_i, axis=1, keepdims=True), 1e-12)
            seen.append(int(tau_i[int(np.argmax(n_i @ norm[c]))]))
        spread.append((min(seen), max(seen)))
    log(f"  kept loglik {ll:,.0f}; birth intervals over the top {len(keep)} of "
        f"{n_init} restarts")
    return W, pi, tau, ll, spread


# --------------------------------------------------------------------------- out

def pack(X, week_of, weeks, vocab, W, pi, tau, ll, lam, spread=None):
    overall = np.asarray(X.sum(axis=0)).ravel()
    overall = overall / overall.sum()
    words_per_week = np.zeros(len(weeks))
    np.add.at(words_per_week, week_of, np.asarray(X.sum(axis=1)).ravel())
    per10k = np.zeros((len(vocab), len(weeks)))
    for t in range(len(weeks)):
        sel = week_of == t
        if sel.any():
            per10k[:, t] = 1e4 * np.asarray(X[sel].sum(axis=0)).ravel() \
                / max(words_per_week[t], 1)

    order = np.argsort(tau)                        # youngest last, oldest first
    spread = spread or [(int(t), int(t)) for t in tau]
    comps = []
    for c in order:
        lift = W[c] / np.maximum(overall, 1e-12)
        rank = np.argsort(-lift)
        comps.append({
            "id": int(c),
            "birth": weeks[int(tau[c])],
            "birth_index": int(tau[c]),
            # the same component's birth across EM restarts; the estimate is worth about
            # this much, not the week it happens to land on
            "birth_low": weeks[spread[c][0]],
            "birth_high": weeks[spread[c][1]],
            "birth_spread_weeks": spread[c][1] - spread[c][0],
            "share": round(float(pi[:, c].mean()), 5),
            "peak": round(float(pi[:, c].max()), 5),
            "peak_week": weeks[int(np.argmax(pi[:, c]))],
            "prevalence": [round(float(v), 5) for v in pi[:, c]],
            "words": [{"word": vocab[j], "lift": round(float(lift[j]), 2),
                       "prob": round(float(W[c][j]), 6),
                       "per10k": [round(float(v), 2) for v in per10k[j]]}
                      for j in rank[:WORDS_CHARTED]],
            "word_list": [vocab[j] for j in rank[:WORDS_LISTED]],
        })
    return {"generated": date.today().isoformat(), "weeks": weeks, "n_init": N_INIT,
            "documents": int(X.shape[0]), "appearances": int(X.sum()),
            "vocab": len(vocab), "k": len(comps), "lambda": lam,
            "loglik": round(ll, 1), "components": comps}


def selftest():
    """A planted birth week must be recovered."""
    rng = np.random.default_rng(0)
    T, V, D_per, born = 40, 60, 60, 22
    Wt = rng.dirichlet(np.full(V, 0.4), size=3)
    rows, cols, vals, week_of = [], [], [], []
    d = 0
    for t in range(T):
        for _ in range(D_per):
            live = [0, 1] if t < born else [0, 1, 2]
            k = live[rng.integers(len(live))]
            x = rng.multinomial(50, Wt[k])
            for j in np.flatnonzero(x):
                rows.append(d); cols.append(int(j)); vals.append(float(x[j]))
            week_of.append(t); d += 1
    X = csr_matrix((vals, (rows, cols)), shape=(d, V))
    W, pi, tau, ll = fit(X, np.array(week_of), T, k=3, outer=6, log=lambda *_: None)

    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-6), "prevalences must sum to 1"
    for c in range(3):
        assert (pi[:tau[c], c] == 0).all(), "a component is alive before its birth"
    # the planted latecomer is whichever component matches Wt[2]
    late = int(np.argmax([Wt[2] @ np.log(np.maximum(W[c], 1e-12)) for c in range(3)]))
    assert abs(tau[late] - born) <= 4, f"birth {tau[late]} not near planted {born}"
    assert tau[late] > max(tau[c] for c in range(3) if c != late), \
        "the latecomer is not the youngest"
    print(f"selftest: ok  (planted birth {born}, recovered {tau[late]}, "
          f"others {[int(tau[c]) for c in range(3) if c != late]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--outer", type=int, default=OUTER)
    ap.add_argument("--n-init", type=int, default=N_INIT, dest="n_init")
    ap.add_argument("--out", default="births.js")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X, week_of, weeks, vocab = documents()
    W, pi, tau, ll, spread = fit_best(X, week_of, len(weeks), k=args.k, lam=args.lam,
                                      outer=args.outer, n_init=args.n_init)
    out = pack(X, week_of, weeks, vocab, W, pi, tau, ll, args.lam, spread)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("window.BIRTHS = ")
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"\nloglik {ll:,.0f}, wrote {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)\n")
    for c in out["components"]:
        print(f"  born {c['birth']} (restarts {c['birth_low']}..{c['birth_high']})  "
              f"mean {c['share']:6.1%}  peak {c['peak']:5.1%} at {c['peak_week']}")
        print("        " + ", ".join(w["word"][:20] for w in c["words"][:9]))


if __name__ == "__main__":
    main()

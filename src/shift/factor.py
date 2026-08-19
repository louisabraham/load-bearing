"""Factorise the week x word matrix into a few components.

The model is the obvious one:

    rate(word v, week t) = sum over k of  weight(k, t) * profile(k, v)

A component `k` is a way of writing: `profile[k]` says which words it uses, fixed across
the whole window, and `weight[k]` says how much of it was in the air in each week,
shared by every word. Nothing here knows what a model release is. The claim is only that
if a released model brings a bundle of habits, that bundle is a rank-one piece of the
matrix, and the factorisation should have to spend a component on it.

Both factors are non-negative, because a rate and a share of a rate both are. That is
not tidiness: non-negativity is what makes the components additive pieces of the
observed rate rather than arbitrary directions, which is why they can be read at all. A
word that falls is still representable, as a component whose weight is high early and
low later.

Each word is divided by its own average rate before fitting, so the matrix it sees
averages one everywhere. This is the same model -- the word's average rate simply moves
into `profile` -- but it decides what the fit is allowed to care about. Without it,
squared error is dominated by the handful of words in half of all documents, and every
component is spent reproducing `the`.

On the L1 penalty, the answer turned out to be that it is not needed. Non-negative least
squares already puts exactly zero in about half the weight cells, which is the "nothing
before it existed" shape the penalty was meant to buy; and pushing L1 up does not sharpen
the components, it merges them -- the largest component's share of the fitted mass goes
from 34% at no penalty to 72% at 0.01 and 97% at 0.15, which is one component pretending
to be the whole corpus. The penalty is kept as a knob, and left near zero.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.decomposition import LatentDirichletAllocation, NMF

from .detect import nearest_release
from .fetch import week_start

K = 8               # components
L1 = 0.0            # L1 on the weekly weights; see the note above
MAX_ITER = 800


class Factors(NamedTuple):
    weight: np.ndarray     # (weeks, k) how much of component k was in week t
    profile: np.ndarray    # (k, words) component k's multiple of each word's average
    base: np.ndarray       # (words,) each word's average rate over the window
    vocab: list[str]
    err: float             # final reconstruction error


def relative(X: np.ndarray, n: np.ndarray, vocab: list[str],
             transform: str = "mean"):
    """Rates divided by each word's own average, and that average.

    `transform` compresses what is left, and it behaves as a purity dial rather than a
    quality knob:

    * "mean" -- the ratio itself. Widest coverage: 31% of mass, both phases of the
      register, including the tooling markers (`bugbot`, `opus`, `renovate` URLs).
    * "log", "sqrt" -- compress the ratio further. Narrower and purer: 12-13% of mass and
      only the later, rarer half -- `byte-identical`, `deliberately`, `fail-closed`,
      `refuses`, `genuinely`, `claimed` -- with no tool names or URLs at all. Suppressing
      the dynamic range suppresses the more frequent tooling words along with it.

    Multiplying by an inverse document frequency instead was tried and is mostly
    redundant with dividing by the mean; the sublinear part of tf-idf is what does the
    work. Computing idf over *weeks* is worse than useless here -- it rewards words
    present in few weeks, which are one-off template tokens rather than arriving
    expressions.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.nan_to_num(X / n[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    base = P.mean(axis=0)
    keep = base > 0
    A = P[:, keep] / base[keep]
    if transform == "log":
        A = np.log1p(A)
    elif transform == "sqrt":
        A = np.sqrt(A)
    elif transform != "mean":
        raise ValueError(f"unknown transform {transform!r}")
    return A, base[keep], [w for w, k in zip(vocab, keep) if k]


def fit_lda(X: np.ndarray, n: np.ndarray, vocab: list[str], k: int = K, seed: int = 0,
            max_iter: int = 60, scale: str = "relative",
            transform: str = "mean") -> Factors:
    """The same decomposition, fitted as Latent Dirichlet Allocation.

    LDA asks a different question from NMF. NMF asks what non-negative parts add up to
    the observed matrix; LDA asks that if each week's words were drawn from a mixture of
    topics, what mixture and what topics. Its Dirichlet priors push both the week's
    mixture and each topic's word distribution toward sparsity, which is what the L1
    penalty was meant to buy and here comes from the model rather than a penalty.

    `scale="relative"` divides each word by its own average first, exactly as `fit` does.
    Strictly this breaks the generative story -- an entry stops being a count of tokens
    and becomes a ratio, so the multinomial over words is no longer a sample from any
    word distribution and the priors lose their pseudo-count reading. It was worth
    testing anyway, and it is what makes the model useful here: on the counts LDA puts
    16% of its mass on the twenty commonest words and never separates the prose register
    at all; on the normalised matrix it puts 0.4% there and finds the register with 14%
    of mass, dated within two weeks of the NMF fit.

    Pruning frequent words instead does not work, which is worth recording because it is
    the obvious thing to try. Only twenty words exceed 25% document frequency here, and
    removing them simply promotes the next tier: the pruned fit spends 68% of its mass on
    `it, if, not, as, new, you, change, have`. There is no threshold that separates
    function words from content words, because the problem is the scale of the counts at
    every level, and normalising is what addresses it at every level.

    `scale="counts"` keeps the coherent generative form, for the comparison.
    """
    if scale == "relative":
        A, base, kept = relative(X, n, vocab, transform)
        A = A * 100.0          # pseudo-counts; LDA is scale-sensitive, the fit is not
    else:
        A = np.rint(np.nan_to_num(X, nan=0.0)).astype(np.float64)
        keep = A.sum(axis=0) > 0
        A, kept = A[:, keep], [w for w, m in zip(vocab, keep) if m]
        with np.errstate(divide="ignore", invalid="ignore"):
            base = np.nan_to_num(A.sum(axis=0) / max(n.sum(), 1))
    model = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=max_iter,
        random_state=seed,
    )
    weight = model.fit_transform(A)
    profile = model.components_ / model.components_.sum(axis=1, keepdims=True)
    return Factors(weight, profile, base, kept, float(model.perplexity(A)))


def fit(X: np.ndarray, n: np.ndarray, vocab: list[str], k: int = K, l1: float = L1,
        seed: int = 0, max_iter: int = MAX_ITER, transform: str = "mean") -> Factors:
    A, base, kept = relative(X, n, vocab, transform)
    model = NMF(
        n_components=k,
        init="nndsvda",              # plain nndsvd seeds zeros a solver cannot leave
        solver="cd",                 # coordinate descent, the one that reaches exact
        alpha_W=l1,                  # zeros; sklearn's W is the (weeks, k) factor
        l1_ratio=1.0,
        max_iter=max_iter,
        random_state=seed,
        tol=1e-5,
    )
    weight = model.fit_transform(A)
    return Factors(weight, model.components_, base, kept,
                   float(model.reconstruction_err_))


def mass(f: Factors) -> np.ndarray:
    """Share of the fitted matrix each component accounts for."""
    m = (f.weight.sum(axis=0)[:, None] * f.profile).sum(axis=1)
    return m / max(m.sum(), 1e-12)


def shapes(f: Factors, first: int = 0) -> pl.DataFrame:
    """One row per component: what its weight curve looks like over the window.

    `off` is the share of weeks where the weight is exactly zero -- the component is
    simply not there. `jump` is the largest one-week rise as a fraction of the curve's
    own peak, and `jump_week` is where it lands. A component with many off weeks and one
    large jump is era-shaped; a component that is on throughout with no jump is a
    background register, and there should be one of those.
    """
    W = f.weight
    peak = np.maximum(W.max(axis=0), 1e-12)
    step = np.diff(W, axis=0) / peak
    rise = step.argmax(axis=0) + 1
    live = [np.flatnonzero(W[:, j] > 0.05 * peak[j]) for j in range(W.shape[1])]
    cuts = [week_start(int(r) + first) for r in rise]
    near = [nearest_release(c) for c in cuts]
    return pl.DataFrame(
        {
            "k": np.arange(W.shape[1]),
            "mass": mass(f),
            "off": (W == 0).mean(axis=0),
            "jump": step.max(axis=0),
            "jump_week": cuts,
            "first_live": [week_start(int(v[0]) + first) if len(v) else None
                           for v in live],
            "last_live": [week_start(int(v[-1]) + first) if len(v) else None
                          for v in live],
            "release": [name for name, _ in near],
            "release_days": [days for _, days in near],
        }
    ).sort("jump", descending=True)


def characteristic(f: Factors, k: int, top: int = 15) -> pl.DataFrame:
    """The words component `k` owns, not merely the words it uses.

    Ranking by the profile alone would list whatever the component happens to lift most,
    including words every component has to lift. The share asks the useful question: of
    everything that made people write this word, how much came from this component?
    """
    contrib = f.weight.mean(axis=0)[:, None] * f.profile        # (k, words)
    total = np.maximum(contrib.sum(axis=0), 1e-12)
    share = contrib[k] / total
    order = np.argsort(-(contrib[k] * share))[:top]             # own it and lift it
    return pl.DataFrame(
        {
            "word": [f.vocab[j] for j in order],
            "share": share[order],
            "lift": f.profile[k][order],
            "base_pct": 100.0 * f.base[order],
        }
    )


def curve(f: Factors, k: int, first: int = 0, width: int = 42) -> list[str]:
    """The component's weight in every week, as text."""
    w = f.weight[:, k]
    hi = max(w.max(), 1e-12)
    return [f"  {week_start(t + first)}  {w[t] / hi:5.2f}  "
            f"{'#' * int(round(width * w[t] / hi))}" for t in range(len(w))]


def plot(f: Factors, path: str, first: int = 0, top: int = 9, title: str = "") -> str:
    """One panel per component: its weekly weight, labelled with the words it owns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    order = np.argsort(-mass(f))
    m = mass(f)
    dates = [week_start(t + first) for t in range(len(f.weight))]
    fig, axes = plt.subplots(len(order), 1, figsize=(11, 1.45 * len(order)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, order):
        peak = max(f.weight[:, k].max(), 1e-12)
        w = f.weight[:, k] / peak
        # each panel is scaled to its own peak so the shape is readable, which would
        # otherwise make a component holding nothing look as loud as the largest one --
        # so the fill fades with the component's share and the peak is printed
        ax.fill_between(dates, w, color="#2b6cb0",
                        alpha=0.20 + 0.65 * float(np.sqrt(m[k] / max(m.max(), 1e-12))),
                        linewidth=0)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        # truncate each word rather than the list: template components own single
        # tokens hundreds of characters long, and cutting the line would show one of them
        words = ", ".join(w if len(w) <= 24 else w[:23] + "\u2026"
                          for w in characteristic(f, k, top=top)["word"])
        ax.text(0.004, 0.93, f"k={k}  ·  {m[k]:.1%} of mass  ·  peak {peak:.3g}",
                transform=ax.transAxes, va="top", fontsize=8, weight="bold",
                color="#1a365d")
        ax.text(0.004, 0.60, words[:190], transform=ax.transAxes, va="top",
                fontsize=7.4, color="#2d3748", family="monospace")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

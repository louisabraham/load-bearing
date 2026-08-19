# Words that arrived together

Groups of words whose frequency on GitHub changed at the same time, found without being
told what to look for.

Three files:

| file | what it is |
|---|---|
| `fetch_week.py` | one `.jsonl.gz` per week of pull request descriptions, from GitHub's search API. Standard library only. |
| `analyze.py` | counts the words, factorises the week × word matrix, writes `analysis.js`. Needs `numpy` and `scikit-learn`. |
| `index.html` | reads `analysis.js`. No build step, no dependencies. Open it. |

```bash
pip install numpy scikit-learn
export GITHUB_TOKEN=$(gh auth token)

python fetch_week.py --all      # ~85 min; skip if data/weeks/ is populated
python analyze.py               # ~40 s
open index.html
```

`python fetch_week.py` with no arguments fetches only the weeks that have no file yet, so
it is what you run weekly. `python analyze.py --selftest` checks the invariants whose
silent failure would invalidate the result.

## The model

Let `X[v, t]` count every appearance of word `v` in week `t` — every appearance, not one
per document. Factorise it:

```
X  ≈  W H        W: words × k, each column summing to 1
                 H: k × weeks, non-negative
```

Each column of `W` is a probability distribution over the vocabulary — a way of writing.
Each row of `H` says how much of that way of writing was in the air each week. Because the
columns of `W` are normalised, the column sums of `H` recover the week's word count, so

```
H[c, t] / Σ_c H[c, t]
```

is genuinely component `c`'s **share of everything written that week**. That is why all
eight charts on the page share one axis instead of each being scaled to its own peak: the
shares sum to 1, so the components are directly comparable and you can watch one overtake
another.

NMF fixes `W H` only up to a diagonal rescaling — `W H = (W D)(D⁻¹ H)` for any positive
diagonal `D` — so normalising the columns of `W` pins that free scale at the one place it
carries meaning, and pushes it into `H`.

**The loss is Kullback–Leibler, not squared error.** `X` holds counts and the columns of
`W` are distributions over words; together that is a multinomial mixture, and KL is its
likelihood. It is also better on the only test that matters — the register's share of the
week rises from 0.003 to 0.747 under KL, against 0.021 to 0.582 under squared error. The
price is that KL needs the multiplicative solver, which updates by multiplication and so
approaches zero without reaching it: the slight L1 on `H` shrinks the quiet weeks rather
than zeroing them. That costs little, because the register averages 0.003 of the week over
the first two months anyway — the "not there yet" shape, written as a small number instead
of an exact zero.

### Two rankings, because there are two questions

The words charted and listed are ranked by `P(v|c) · log(P(v|c) / P(v))` — each word's
pointwise contribution to the divergence between the component and the corpus. A word
scores highly when the component uses it a lot **and** uses it more than average. Ranking
by probability alone lists `the` under every component, since every component has to
reproduce `the`.

The *distinctive* words are ranked by lift alone, `P(v|c) / P(v)`, floored on probability so
a one-off cannot win and on lift itself so a background component does not fill the list
with `or` and `are` at a lift of 1.1. Components show fewer than twelve where fewer qualify.

An earlier version of this counted one appearance per document and needed each word divided
by its own average across the window before fitting — without that, every fit spent itself
on `the`. This formulation does not need it, because the interpretation step does that work
instead: `W`'s columns are *allowed* to be dominated by common words, since they are
distributions over real text, and the lift term is what recovers what is characteristic.

## Why not GH Archive

Because it no longer works. Its feed has carried almost only `PushEvent` since mid-2025: a
complete hour of 2024-08-12 holds 13,555 `IssueCommentEvent` against 86 for the same hour
of 2026-08-10, and polling `/events` in August 2026 returns 97 `PushEvent` out of the 100
most recent events. Measured from the files, the archive carried 3,000–10,000 issue
comments an hour through 2025-10, then 1,590 in 2026-03, 866 in 2026-06 and 77 in 2026-07.

The cause is upstream, in GitHub's Events API. Its own tracker carries
[#310 "Drastic Drop Off in Events After 2025-05-23"](https://github.com/igrigorik/gharchive.org/issues/310),
open since July 2025 with no reply, and
[community discussion #178788](https://github.com/orgs/community/discussions/178788)
traces the same loss to "a GitHub Event API outage propagated downstream" — the identical
gaps appear in OSSInsight, which reads the API directly. GitHub has published no fix and no
alternative, so every bulk mirror inherits it: BigQuery `githubarchive`, ClickHouse GH
Explorer, the Kaggle and Hugging Face copies. GHTorrent has been dead since 2021, and
Software Heritage archives real git history but exports once a year.

## How the sampling works

Each week contributes the same number of randomly placed five-minute windows of newly
opened pull requests. *Uniform* across the window, so a difference between two weeks cannot
come from having looked harder at one of them; *random* inside the week, so the sample is
of the whole week rather than a chosen slice of the clock.

Two filters are pushed into the query, and together they take a hundred-item page from 43
usable documents to 97:

- **Four Apps excluded.** `-author:app/{pull,dependabot,renovate,github-actions}` — 90% of
  App-authored bodies. `-author:app/*` is rejected with a 422, so there is no way to say
  "no apps", and other App accounts stay in on purpose: some of the clearest agent-written
  prose on GitHub is App-authored.
- **Empty bodies excluded**, and 45% of pull requests have none. There is no emptiness
  qualifier — `-body:""` is a 422, `has:body` and `-in:body` are silently ignored, and
  `body:*` cuts 94% rather than 45% because it is a text match on the asterisk. Requiring
  any one of ten function words *in the body* does it exactly. Note the qualifier is
  repeated per term: `in:body` does **not** distribute over an OR group, so
  `(the OR a) in:body` matches titles and lets empty bodies back in.

Pull request descriptions were chosen over comments and issue bodies because they have the
largest dynamic range — measured on one expression, ×2,322 against ×866 for issue bodies
and ×208 for comments. They were the quietest surface before assistants arrived.

## Counting

A word is a run of non-space characters, lowercased, with surrounding punctuation removed.
That is the only normalisation: no stemming, no n-grams, no stopword list. Every appearance
counts, so a word used three times in one description contributes three. Purely numeric
tokens are dropped — the calendar advances every week, so a bare `10` or `2026` arrives and
departs on a schedule of its own (`apr` went 0.05% → 9% of documents at the end of one
March; month abbreviations are words and stay, so that one is an artifact to read past).

Two documents with the identical word set count once, within each week — one ordinary account once posted 147 copies of the same sentence inside
a fortnight, 16% of it, and every word of its template moved with it. That collapse is
deliberately per-week and not global: collapsing across the window would make a template
running for months look as though it started or stopped.

Every week is then cut off at the same number of documents. Sampling the same number of
hours does not give the same number of documents, and because text is overdispersed — words
cluster inside repositories — a rate computed on more documents comes out inflated rather
than merely more precise. Without it, busy weeks outrank busy language. Document *length* still varies threefold
after this, which is exactly why the model reports `H` as a share of the week rather than
raw.

## Two things to hold onto

The corpus only contains pull requests whose author wrote a description, and **that
condition loosens over the window** — empty descriptions fall from about a third to about a
seventh. A rate measured per pull request therefore rises both because descriptions use a
word more and because more pull requests have descriptions at all.

The components come from the model; the word curves do not. Each word chart is the raw
weekly share of documents containing that word, so only the *choice* of which words to show
is the model's. Sparklines are scaled to their own peak, printed beside each one, so their
shapes are comparable but their heights are not.

## What was here before

Roughly 6,800 lines across two packages, being the record of working out which method and
which source answer the question — a GH Archive ingester, MinHash template mining,
changepoint detection, clustering, release alignment, Bayesian date estimation, an LDA
path, and the parameter sweeps behind the choices above. It is all at the
`archive/full-pipeline` tag. The changepoint scan is the one worth knowing about: on every
corpus it was run against it found tool deployments rather than language, because comparing
two windows at a time cannot see a gradual multi-month shift, while a bot changing its
template produces exactly the jump it looks for.

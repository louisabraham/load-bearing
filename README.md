# Words that arrived together

Groups of words whose frequency on GitHub changed at the same time, found without being
told what to look for.

Three files:

| file | what it is |
|---|---|
| `fetch_week.py` | one `.jsonl.gz` per week of pull request descriptions, from GitHub's search API. Standard library only. |
| `analyze.py` | counts the words, factorises the week × word matrix, writes `analysis.js`. Needs `numpy` and `scikit-learn`. |
| `index.html` | reads `analysis.js`. No build step, no dependencies. Open it. Every component appears as a miniature on one shared axis; picking one shows it in full, and the choice is in the URL hash. |

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

**Why Kullback–Leibler and not squared error.** `X` holds counts and the columns of `W` are
probability distributions over words; together that is a multinomial mixture, and KL is its
likelihood. Squared error instead assumes Gaussian noise of constant variance, which counts
do not have — the variance of a count grows with its mean, so squared error treats a swing
of 50 in a word appearing 200,000 times as equally surprising as a swing of 50 in a word
appearing 60 times. Measured at `k = 16` on this vocabulary:

| loss | the register component | exact zeros in `H` |
|---|---|---|
| **KL** | **13.8% of mass, 0.001 → 0.732** | 0% |
| squared error | 7.3% of mass, 0.009 → 0.420 | 22% |

**And the cost, stated plainly: under KL the L1 on `H` does nothing.** Two reasons. KL needs
the multiplicative solver, which approaches zero without reaching it, so there are no exact
zeros. And this parameterisation cancels the penalty outright: NMF fixes `W H` only up to a
diagonal rescaling, and normalising `W`'s columns pins that scale *after* fitting — so the
optimiser can satisfy an L1 on `H` by shrinking `H` uniformly and inflating `W`, which costs
it nothing, and the rescaling then undoes the shrinkage exactly. Measured: `alpha_H` from 0
to 10 moves `sum(H)` by 0.7% and the per-week *shape* of `H` by 0.0085. **An L1 is only
meaningful where the scale is not free.**

With `LOSS = "l2"` the penalty does bite — 22% of `H` exactly zero at `alpha_H = 0`, 25% at
0.02, 47% at 0.2 — at the cost of the separation in the table above. That trade is one
constant, not a rewrite.

### Two rankings, because there are two questions

A component's **most representative** words — the sixteen charted and the forty listed —
are those with the highest ratio of probability under the component to probability in the
corpus:

```
lift(v, c) = P(v | c) / P(v)
```

That ratio is what makes a word belong to a component rather than to English. Its
counterpart, the twelve words a component is most **made of**, ranks by
`P(v|c) · log(P(v|c) / P(v))` — the pointwise contribution to the divergence between
component and corpus, which is why `the` can appear there and never in the first list.

**Two floors keep the lift ranking honest.** A word needs 45 total appearances *and*, 
separately, must appear in 25 distinct documents. Appearances alone are not breadth:
`multi-draw` appears 101 times inside a single description, `m₀` 140 times, and both were
ranking among a component's most representative words, because a ratio cannot tell a
widespread word from a word someone repeated. Both ceilings are set by `load-bearing`
itself — 51 appearances across 45 documents — which under this ranking comes out 8th of
40 in the component that rises through 2026.

There is deliberately no *probability* floor. Flooring on probability throws away exactly
the rare-but-concentrated words the ratio is for: `load-bearing` ranks 8th with no floor
and 6,062nd with one at the 80th percentile.

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

**What counts as a word.** A run of letters, digits, hyphens and underscores containing at
least one letter — so `load-bearing`, `snake_case` and `--all-targets` survive whole, while
`/`, backtick, `:` and `>` are separators rather than characters a word may contain. No
stemming, no n-grams, no stopword list. Every appearance counts, so a word used three times
in one description contributes three.

Order matters, and each step exists because of what the previous one broke:

1. **Links first, whole.** `[bugbot](https://cursor.com/x)` gives `bugbot` and the link.
   Splitting on punctuation first produced `bugbot](https` and a trail of fragments, and
   those fragments were ranking among components' most representative words.
2. **Then HTML tags, whole.** Splitting them character by character turned
   `<sup>reviewed</sup>` into `sup, reviewed, sup` and made `li`, `br`, `td` and `href` six
   of one component's twelve commonest words. The pattern requires a letter or slash after
   the bracket, so `a > b` in prose is not mistaken for markup.
3. **Then everything else splits** on any character a word may not contain, which handles
   what markdown creates without needing to know about it: `srcset="…"` gives `srcset`,
   `height="28` gives `height`, `*emphasis*` needs nothing because `*` is a separator.
4. **Then trim the edges.** `_other example_` needs its underscores trimmed, since an
   underscore is allowed *inside* a word. A trailing hyphen goes for the same reason; a
   leading one stays, so `--all-targets` is not quietly turned into `all-targets`.

Requiring a letter drops what is left of numbers and rules — `27.49`, `589/1000`,
`2025-06-24`, `-------` — along with the arrow and `+`. **The em dash is the one exception**,
taken before the split and counted as a word of its own. It earns that: 0.0 appearances per
10,000 words in early 2024 against 123.0 in mid-2026, the sharpest single signal here. It is
counted separately rather than added to the word characters because it is as often unspaced
as spaced, and inside the character class `foo—bar` would become one token instead of three.

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

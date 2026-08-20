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
**`H[c, t]` is a count: the number of word appearances in week `t` attributable to component
`c`.** Measured against the corpus, that identity holds to within 0.73%.

`H` is reported as it is, not divided through by the week. Dividing would hide how much was
written, and the weeks differ threefold in length even after the document cap. Component 1
is the case that makes the point: as a share of the week it falls from 33% to 14%, but in
appearances it barely moves — 9,996 a week to 8,631. It did not shrink; the corpus grew
around it. Meanwhile the component that rises goes from 36 appearances a week to 43,703.

Every component's curve is therefore in the same units, which is why all sixteen charts on
the page share one axis instead of each being scaled to its own peak.

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

## An alternative model, in `mixture.py`

Same corpus, a different shape of question. Rather than factorising a week × word matrix,
attribute each *document* to one of `k` ways of writing:

```
W_k                   a fixed distribution over the vocabulary
pi_tk                 how much of week t was written that way, sum_k pi_tk = 1
z_d ~ Cat(pi_t)       which component wrote document d
x_d ~ Mult(n_d, W_k)  its words
```

The only thing asked of a prevalence curve is smoothness, `lambda * sum_t (pi_tk -
pi_{t-1,k})^2` — nothing requires a component to rise, to fall, or to be absent early. Fitted
by EM, restarted ten times and keeping the highest likelihood, because EM finds different
local optima here and the worst of them mix a component with something else and give it two
thirds of the peak the good fits find.

```bash
python mixture.py           # writes mixture.js
open mixture.html           # stacked, absolute and as a share
```

**That penalty works, where the L1 in `analyze.py` does not** — and the difference is
structural rather than a matter of coefficient. There `W`'s columns are normalised *after*
fitting, so an L1 on `H` can be satisfied by shrinking `H` and inflating `W` at no cost, and
the rescaling undoes it exactly. Here `pi` sums to 1 in every week by construction, so the
scale is not free and there is nothing to game. Swept at `k = 12`, total squared week-to-week
change:

| `lambda` | 0 | 10 | 40 | 200 | 1000 |
|---|---|---|---|---|---|
| roughness | 1.870 | 1.765 | 1.517 | 0.909 | **0.333** |

A 5.6-fold reduction, monotone, costing 0.002% of the log-likelihood at the far end. An L1 on
`pi` *itself* would do nothing for a third reason: on the simplex `||pi_t||_1 = 1` identically,
so it is a constant with zero gradient — L1 induces sparsity by trading against magnitude, and
on a simplex the magnitude is already spent.

### How a word relates to a component

No word is *assigned* to a component. `W_kv = P(word v | component k)` is positive for every
pair, so every word belongs a little to all of them. What the page shows is a **ranking**, by
how much more probable a word is under one component than in the corpus at large:

```
lift(v, k) = P(v | k) / P(v)
```

For `load-bearing` at `k = 12` the ranking is not close, which is worth showing because it is
not guaranteed:

| component | `P(word｜comp)` | lift | its rank there |
|---|---|---|---|
| 3 | 6.89 × 10⁻⁵ | **6.58** | **1st of 6,207** |
| 0 | 1.50 × 10⁻⁶ | 0.14 | 5,491st |
| the other ten | ≤ 1.1 × 10⁻⁷ | ≤ 0.01 | 1,887th – 5,825th |

One component gives it 46 times the probability of the next best, and it is that component's
single most representative word. But that is a fact about this word. A word with lift near 1
everywhere belongs to no component in particular, and the ranking will still put it 40th
somewhere — the list is a top-40, not a test.

### How many components

`--k` sets it, `--out` names the file, and `mixture.html?k=N` reads whichever fit you ask for:

```bash
for k in 4 6 8 12 16 24 32; do
  python mixture.py --k $k --n-init 6 --out mixture-k$k.js
done
```

Training likelihood rises with k because more parameters always fit better, so it cannot
choose. Held-out likelihood can: fit on 90% of documents, score the other 10%. Pushed far
enough, it turns over.

| `k` | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | **128** | 192 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | −9.458 | −9.302 | −9.240 | −9.180 | −9.097 | −9.045 | −8.969 | −8.922 | −8.862 | −8.806 | −8.739 |
| **held out** | −9.481 | −9.338 | −9.291 | −9.246 | −9.181 | −9.149 | −9.110 | −9.095 | −9.083 | **−9.078** | −9.103 |

**The data supports about 128 components**, an order of magnitude more than the default. One
caveat that runs the right way: k ≥ 96 was fitted with two restarts against six for the small
k, so the large fits are handicapped and the true optimum may be higher still. The turnover is
real regardless, because 128 and 192 had the same two restarts and 192 is worse.

`mixture.html?k=128` shows it. Above twenty components a full-width row each is unreadable —
at 128 it would run to twenty-five thousand pixels — so the page switches to a grid of small
multiples, six across, each with its mean, its peak and its four most representative words.
Five of the 128 grow: `firewall, workbench, sha-256, publication, hermes, idempotency`;
`--all-targets, --workspace, reproduces, cwd, envelope`; `reconnect, one-shot, drain,
reconcile, 24h, forever`; `leg, asserted, refuses, killed, refusal, throws, nobody`; and
`ancestor, backfill, tensor, dataclass, torch`. The register at k = 12 is these, plus the
tooling half, plus more.

**And this is where `load-bearing` stops owning a component.** Its lift and its rank in the
component that gives it most, at each k:

| `k` | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 128 |
|---|---|---|---|---|---|---|---|---|
| best lift | 2.4 | 4.3 | 6.5 | 6.1 | 7.2 | 8.6 | 9.1 | **26.2** |
| its rank there | 6th | **1st** | 3rd | **1st** | **1st** | 19th | **1st** | 19th |
| next-best lift | 0.00 | 0.01 | 0.15 | 0.01 | 0.02 | 4.07 | 0.44 | **12.6** |

At k = 6 to 32 one component holds it almost exclusively — the gap to the runner-up is 21 to
610-fold, so the word belongs somewhere. At k = 128 the lift is four times higher, because
narrower components can concentrate a word much harder, but the gap collapses to 2.1-fold and
it appears with lift above 4 in at least six components: `shim, pilot, stdio, mcp` (26.2),
`leg, asserted, refuses, killed` (12.6), `reconnect, one-shot, drain` (9.1), and on. The word
is real at every resolution; what it *belongs to* is only well defined at coarse ones.

That is the trade in one line. **`k = 12` is a legibility choice, not a statistical one** — twelve rows a reader can scan,
against 128 nobody will read. What the smaller k buys is a summary; what it costs is resolution.

What the sweep does show, without depending on any scoring choice, is the register's condition
at each k:

| `k` | pieces | the biggest piece's top words |
|---|---|---|
| 4 | 1 | `refusal, subagent, nixpkgs-update…` — 39% of the corpus, diluted with everything else |
| 6 | 1 | `load-bearing, optimole, imgbot, octocat` |
| 8 | 1 | `[webkit-url], byte-identical, load-bearing, ews` |
| 12 | 1 | `load-bearing, --all-targets, byte-identical, seam` |
| 16 | 1 | `load-bearing, 2100s, dhi, 600s` |
| 24 | **2** | splits; pieces stay identifiable |
| 32 | **2** | `load-bearing, genuine, carries, latent` |

Three regimes: below 6 the register is absorbed into a large component along with unrelated
vocabulary; from 6 to 16 it is one undivided component with `load-bearing` its most
representative word at 6, 12, 16 and 32; from about 24 it splits in two without dissolving.

**And the split at k = 32 is worth reading, because it is not a degradation.** The two pieces
are different registers that arrived five months apart:

| | prose, peaks 2026-08-10 | tooling, peaks 2026-03-09 |
|---|---|---|
| share of the week | 0.0% → 42.2% | 0.1% → 17.5% |
| lift 1 → 40 | 9.1 → 6.7 | 11.9 → 6.5 |
| words | `load-bearing, genuine, carries, latent, lands, folded, seam, refuses, drives, framing, byte-identical, deliberately, identically, surfaced, survives, refusal, honest, reproduces, proves, verdict, measured, defects, untouched, inert, asserting, holds` | `pythonpath, py_compile, workbench, -q, --filter, worktrees, --test, --lib, --all-features, compileall, modulenotfounderror, cjs, --check, --workspace, runbook, unittest, subcommands, jsonl, sha-256, --locked, --all-targets, redaction, fail-closed, handoff, pytest, cargo, harness, orchestration, mvp, governance, gpt-5, [chatgpt-url]` |

One is the vocabulary of asserting what is true about a change; the other is command-line flags
and agent scaffolding. At `k = 12` they are one component, and the tooling half is why
`--all-targets` sits third in its word list.

### The word lists are a window, not a set

Forty is arbitrary, and the numbers say so plainly. Lift declines smoothly with rank and there
is no cliff to cut at — for the prose component at `k = 32`, rank 1 is 9.1× and rank 80 is
still 6.0×, with **187 words above 5×, 803 above 3× and 1,743 above 2×**. So a top-40 is the
head of a long tail rather than a closed group.

The page therefore prints each word's lift beside it, and a line under every row saying how far
the tail runs. Nothing about the model changes; what changes is that a reader can see the list
is a window and where it was cut.

**A retraction.** An earlier version of this table scored each k by how many of 22 "marker
words" it recovered, and reported that the count peaked at 12. That metric was circular: the 22
words were chosen by reading the k = 12 output, so it measured agreement with k = 12 rather
than quality of fit. Held-out likelihood is the non-circular version, and it does not favour 12.

### Absolute, not share### Absolute, not share

`mixture.html` reports absolute counts everywhere, and the two stacked views side by side are
the argument for it. The corpus caps documents at 350 a week, so the document count is flat by
construction (it only varies 329 to 350) — but the words inside them are not capped and swing
**3.2-fold**, from 23,597 a week to 75,541. Descriptions got longer. So the absolute view is in
word appearances, and it shows total volume nearly tripling with one component driving all of
it; the share view shows the same data with the volume divided out, where a band can shrink
because the corpus grew around it rather than because it shrank.

Of twelve components, two end the window at least twice the size they started. The larger goes
from 0.2% to 63.8% of the week, and its most representative word is **`load-bearing`**,
followed by `seam, byte-identical, lands, refuses, --all-targets, genuine, folded, adversarial`.

### What was removed, and why

An earlier version gave each component an unknown **birth week** and held its prevalence at
exactly zero before then, so the empty stretch was a parameter rather than a shape a penalty
was asked to produce. The constraint worked, and a planted birth was recovered exactly on
synthetic data. It was still removed, because **on this corpus the birth week was not
identified**: single runs put one component's birth anywhere across a 23-month range, and
dropping 75 documents of 47,373 moved it thirteen weeks. The threshold that defined a birth sat
in the near-zero tail, where a handful of documents decides whether a week clears it.
Likelihood selection over restarts narrowed it to about four months, but a curve that starts
near zero says the same thing without claiming a date. It is in the git history if wanted.

Also worth recording, since it is checkable rather than a matter of taste: **the weekly total
throws nothing away.** Given the responsibilities, the M-step for `pi_t` depends on the
documents only through the column sums, so the weekly count is the *sufficient statistic*.
Three assumptions behind that were measured — 84% of documents have a maximum responsibility
above 0.9 and only 2.1% are genuinely split, so one-source-per-document is fair and an
admixture model would be solving a problem this corpus does not have; every document
contributes exactly 1.0 to the count regardless of length, so the shortest half supplies 50% of
it; and two documents from the same repository agree on their component 63% of the time against
16% by chance (ICC 0.56), but the design effect is only 1.17 because the three-per-author cap
already keeps clusters tiny.

## Why not GH Archive## Why not GH Archive

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

1. **Links first, each collapsed to its domain.** `[bugbot](https://cursor.com/x)` gives
   `bugbot` and `[cursor-url]`. Splitting on punctuation first produced `bugbot](https` and
   a trail of fragments, and those fragments ranked among components' most representative
   words. Keeping links whole was little better: a tool that puts a per-item link in every
   description gets one word per *item* instead of one word, and Snyk's vulnerability links
   alone were the top words of eight of sixteen components. `[snyk-url]`, `[claude-url]`,
   `[github-url]` say the useful thing in one token that can clear the frequency floors.
   The registrable domain is taken as the second-to-last label — wrong for `example.co.uk`,
   right for everything that turns up here.
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

Snyk advisory identifiers collapse the same way, `snyk-js-axios-6144788` to `[snyk-id]`,
because they are the same problem one level down: 1,401 distinct tokens, 113 of them past
the floors, between them occupying seven of sixteen components. Afterwards, none. The
trailing run of digits is what tells an identifier from `snyk-top-banner`. CVE and GHSA ids
have the same shape and are left alone — five and one of them clear the floors.

Requiring a letter drops what is left of numbers and rules — `27.49`, `589/1000`,
`2025-06-24`, `-------` — along with the arrow and `+`. **The em dash is the one exception**,
taken before the split and counted as a word of its own. It earns that: 0.0 appearances per
10,000 words in early 2024 against 123.0 in mid-2026, the sharpest single signal here. It is
counted separately rather than added to the word characters because it is as often unspaced
as spaced, and inside the character class `foo—bar` would become one token instead of three.

**No author may contribute more than three documents to a week.** This is what finds
mass-produced descriptions without a blocklist, and it works because they concentrate by
*author* rather than by repository: `copilot` wrote 197 of the 198 descriptions carrying
GitHub's coding-agent survey link, across 192 repositories, and `vercel[bot]` wrote all 85
carrying one particular CVE. It catches what the `[bot]` suffix misses — `copilot`,
`pyup-bot`, `scala-steward` and `regro-cf-autotick-bot` are ordinary logins — and it applies
to humans on the same terms, which is why it is a cap and not an exclusion. Across the
corpus 36,503 authors write 48,086 documents, so a cap of three costs 4.4% of them. It does
nothing about Snyk, whose 3,714 descriptions come from 2,197 authors because each
repository's integration runs under its own login; that needed the identifier collapse
above.

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

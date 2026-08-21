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

**The loss is squared error, and the argument against it is worth knowing.** `X` holds counts
and the columns of `W` are distributions over words, which together make a multinomial mixture
— so Kullback–Leibler is the likelihood and squared error is not. Squared error assumes
Gaussian noise of constant variance, which counts do not have: the variance of a count grows
with its mean, so it treats a swing of 50 in a word appearing 200,000 times as equally
surprising as the same swing in a word appearing 60. That objection is correct and it is
overruled by two things the output showed.

First, **KL folds a vendor's footer into the register.** Under KL the rising component's top
four representative words are `cursor.com` links, which inflates its mass and muddies what the
component is; squared error separates the links from the prose. Second, **KL cancels the L1**.
It needs the multiplicative solver, which approaches zero without reaching it, and worse: `W H`
is fixed only up to a diagonal rescaling, so normalising `W`'s columns *after* fitting lets the
optimiser satisfy an L1 on `H` by shrinking `H` and inflating `W` at no cost, and the rescaling
undoes it exactly. Measured, `alpha_H` from 0 to 10 moves `sum(H)` by 0.7% and the per-week
shape of `H` by 0.0085. **An L1 is only meaningful where the scale is not free.**

Under squared error it bites: 22% of `H` exactly zero at `alpha_H = 0`, 25% at 0.02, 47% at
0.2. `LOSS = "kl"` switches back, one constant.

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
itself — 51 appearances across 45 documents — which under this ranking comes out 24th of 40
in the component that rises through 2026.

There is deliberately no *probability* floor. Flooring on probability throws away exactly
the rare-but-concentrated words the ratio is for: `load-bearing` ranks in the top 40 with no
floor and 6,062nd with one at the 80th percentile.

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

The only thing asked of a prevalence curve is smoothness — `lambda * K² * sum_t (pi_tk -
pi_{t-1,k})²`, the `K²` making one `lambda` correct at every `k`, for the reason below.
Nothing requires a component to rise, to fall, or to be absent early. Fitted
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
scale is not free. An L1 on `pi` itself would still do nothing: on the simplex
`||pi_t||_1 = 1` identically, a constant with zero gradient.

### How many components

`--k` sets it, `--out` names the file, and `mixture.html?k=N` reads whichever fit you ask for:

```bash
for k in 4 6 8 12 16 24 32 128; do
  python mixture.py --k $k --n-init 4 --out mixture-k$k.js
done
```

Training likelihood rises with k because more parameters always fit better, so it cannot
choose. Held-out likelihood can: fit on 90% of documents, score the other 10%.

| `k` | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | **128** | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | −9.313 | −9.243 | −9.189 | −9.106 | −9.054 | −8.970 | −8.923 | −8.862 | −8.807 | −8.741 | −8.685 |
| **held out** | −9.357 | −9.294 | −9.254 | −9.187 | −9.151 | −9.108 | −9.091 | −9.077 | **−9.073** | −9.094 | −9.083 |
| gap | 0.044 | 0.052 | 0.065 | 0.081 | 0.097 | 0.138 | 0.168 | 0.215 | 0.266 | 0.354 | 0.397 |

**The data supports around 128 components**, an order of magnitude more than the default, and
that answer did not move when `lambda` was tuned — it was 128 before as well. The
train-to-held-out gap widens monotonically from 0.044 to 0.397 bits, which is overfitting
arriving steadily; the held-out curve turns over once the gap outruns the gain.

Two caveats. `k ≥ 96` was fitted with two restarts against three or four for the smaller k, so
the large fits are handicapped and the optimum may be slightly higher. And 192 scores worse
than 256, which is impossible for a well-fitted sequence — restart noise at those sizes is now
comparable to the differences being resolved, so "around 128" is as precise as this gets.

`mixture.html?k=128` displays it. Above twenty components a full-width row each is unreadable
— at 128 it would run to twenty-five thousand pixels — so the page switches to a grid of small
multiples, six across, each with its mean, its peak and its four most representative words.

**This is also where `load-bearing` stops owning a component.** Its lift, and its rank in the
component that gives it most, at each k:

| `k` | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 128 |
|---|---|---|---|---|---|---|---|---|
| best lift | 2.4 | 4.3 | 6.5 | 6.1 | 7.2 | 8.6 | 9.1 | **26.2** |
| its rank there | 6th | **1st** | 3rd | **1st** | **1st** | 19th | **1st** | 19th |
| next-best lift | 0.00 | 0.01 | 0.15 | 0.01 | 0.02 | 4.07 | 0.44 | **12.6** |

From k = 6 to 32 one component holds it almost exclusively — the gap to the runner-up is 21 to
610-fold. At k = 128 the lift is four times higher, because narrower components can concentrate
a word much harder, but the gap collapses to 2.1-fold and it appears above lift 4 in at least
six components. The word is real at every resolution; what it *belongs to* is only well defined
at coarse ones. **So `k = 12` is a legibility choice, not a statistical one** — twelve rows a
reader can scan, against 128 nobody will read.

**`lambda` is set by held-out likelihood, and made scale-free in `k`.** The difference is
penalised relative to `1/K` rather than absolutely, because without that the right `lambda`
moves by two orders of magnitude with `k` for a purely mechanical reason: prevalences sum to
one, so a typical `pi` is about `1/K` and a typical squared difference about `1/K²`. Held-out
likelihood puts the optimum at 5,000 for `k = 12` and 500,000 for `k = 128` — and
`(128/12)² × 5,000 = 568,889`, one grid step away. So the entire k-dependence is that factor,
and absorbing `K²` leaves one constant that is right at both: `5,000/144 = 34.7` and
`500,000/16,384 = 30.5`. Set to 32, it reproduces both per-k optima exactly, −9.2944 and
−9.0728 bits per word.

| `lambda` at k=128 | 0 | 1,000 | 25,000 | 100,000 | **500,000** | 2 M | 10 M |
|---|---|---|---|---|---|---|---|
| held out | −9.0782 | −9.0770 | −9.0742 | −9.0734 | **−9.0728** | −9.0738 | −9.0760 |
| train | −8.8055 | −8.8053 | −8.8064 | −8.8070 | −8.8072 | −8.8079 | −8.8099 |

Train getting worse while held-out gets better is the regularisation signature, and it only
appears at the larger `k`, where there are 17,536 prevalences to fit rather than 1,644. At
`k = 12` held-out is flat to four decimals across four orders of magnitude, so there the
penalty is free rather than helpful — worth taking anyway, since it cuts roughness twentyfold
for nothing.

Held-out likelihood cannot see over-smoothing, so the shape was checked separately. At
`k = 12` the register rises 0.4% → 65.4% at the old default and 0.3% → 63.8% at the new one,
but only 0.4% → **47.0%** at a hundred times that — the peak dragged down toward the early
weeks. The chosen value is the largest that leaves the shape alone.

### Absolute, not share

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

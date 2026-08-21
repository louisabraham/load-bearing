# The load-bearing vocabulary of Claude

Groups of words whose frequency in GitHub pull request descriptions changed at the same time,
found without being told what to look for. One of them was 1.6% of the corpus at the start of
2025 and is 62% of it by the middle of 2026.

**[louisabraham.github.io/load-bearing](https://louisabraham.github.io/load-bearing/)**

| file | what it is |
|---|---|
| `fetch_day.py` | one request a day to GitHub's search API, one `data/days/YYYY-MM-DD.jsonl`. Standard library only. |
| `analyze.py` | reads the days, groups them into whole weeks, fits the model, writes `analysis.js`. Needs `numpy`, `scipy`, `numba`. |
| `index.html` | reads `analysis.js`. No build step. Open it. |
| `.github/workflows/daily.yml` | does all of the above, daily, and commits the result. |

```bash
pip install numpy scipy numba
export GITHUB_TOKEN=$(gh auth token)

python fetch_day.py                  # yesterday, one request
python fetch_day.py --backfill 30    # and the last 30 days, if missing
python analyze.py                    # ~8 s end to end
python analyze.py --selftest         # the invariants, on synthetic data
open index.html
```

Current state: **598 collected days, 588 of them in 84 whole weeks** (2025-01-06 to 2026-08-10),
51,280 descriptions, 5,564,813 word appearances, 7,059 words above the floors.

---

# The whole process

Six steps. Everything below is either measured or labelled as a judgement call, and
[§7](#7-every-arbitrary-choice) lists every number in the code that could have been chosen
differently, including the two that were chosen by looking at the answer.

## 1. Why GH Archive cannot be used

The natural source for this is the public archive of GitHub's event stream. It stopped working.

Since mid-2025 its feed carries almost only `PushEvent`. A complete hour of 2024-08-12 holds
13,555 `IssueCommentEvent` against 86 for the same hour of 2026-08-10, and polling `/events`
in August 2026 returned 97 `PushEvent` out of the 100 most recent. Measured from the files
themselves, the archive carried three to ten thousand issue comments an hour through 2025-10,
then 1,590 in 2026-03, 866 in 2026-06 and 77 in 2026-07. Pull requests and issues fell with
them. Pushes survive at full volume but carry no text, GitHub having
[removed the commit array](https://github.blog/changelog/2025-08-08-upcoming-changes-to-github-events-api-payloads/)
from the payload in October 2025.

The cause is upstream, in GitHub's Events API. Its tracker carries
[#310 "Drastic Drop Off in Events After 2025-05-23"](https://github.com/igrigorik/gharchive.org/issues/310),
open since July 2025 with no maintainer reply, and
[community discussion #178788](https://github.com/orgs/community/discussions/178788) traces the
same loss to "a GitHub Event API outage propagated downstream" — the identical gaps appear in
OSSInsight, which reads the API directly rather than through the archive. So no mirror repairs
it: [BigQuery `githubarchive`](https://www.gharchive.org/),
[ClickHouse GH Explorer](https://ghe.clickhouse.tech/) and the Kaggle and Hugging Face copies
all read the same feed. GHTorrent has been dead since 2021, and
[Software Heritage](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html)
archives real git history but exports once a year. GitHub has published no fix and no
alternative.

What works is the search API, because `created:` accepts timestamps rather than only dates —
so a window can be minutes wide, narrow enough to enumerate rather than sample, and each
response carries the full body.

**This was found the hard way, and it is the reason for the rewrite.** An earlier version of
this project, built on the archive, reported `load-bearing` in 17 documents. That was wrong by
a factor of 158 — the comments had disappeared from the feed, not from GitHub. Every number
this project had produced up to that point was measuring the archive's decay.

## 2. How the data is collected

What works is GitHub's **search** API, for one specific reason: `created:` accepts timestamps
and not only dates. A window can therefore be minutes wide — narrow enough that a single page
of a hundred results *enumerates* it rather than samples it — and every response carries the
full body text.

So: **one randomly placed five-minute window of newly opened pull requests, every single day.**

- **Seeded on the date**, so the whole corpus is reproducible from its dates alone. Asking for
  a given day returns the window it returned the first time.
- **One immutable file per day**, `data/days/YYYY-MM-DD.jsonl`, about 140 kB, committed and
  never rewritten. The repository's history *is* the history of the sample.
- **Days collect, weeks analyse.** A hundred descriptions is too thin to compare against
  another hundred, so seven days make a week.
- Roughly 50 MB a year, left uncompressed so it can be read and grepped in place.

Two filters are pushed into the query itself, and together they take a page from 43 usable
descriptions to 97:

- **Four Apps excluded.** `-author:app/{pull,dependabot,renovate,github-actions}` — 90% of
  App-authored bodies. `-author:app/*` is rejected with a 422, so there is no way to say "no
  apps", and other App accounts stay in on purpose: some of the clearest agent-written prose on
  GitHub is App-authored.
- **Empty bodies excluded**, 45% of all pull requests. There is no emptiness qualifier —
  `-body:""` is a 422, `has:body` and `-in:body` are silently ignored, and `body:*` cuts 94%
  rather than 45% because it is a text match on the asterisk. Requiring any one of ten function
  words *in the body* does it exactly. The qualifier must be repeated per term: `in:body` does
  **not** distribute over an OR group, so `(the OR a) in:body` matches titles and lets empty
  bodies back in.

**One honest limit.** Every window comes back a full page, in 2025 and 2026 alike, so a window
holding more than a hundred is truncated to its earliest hundred. The placement is uniformly
random, so what is sampled is still everything created in a uniformly random interval — but
that interval is *effectively narrower* when GitHub is busier. The sample is uniform in time,
not in the fraction of each period it captures.

**A latent trap, documented because it nearly bit.** The files are written with
`ensure_ascii=False`, so the corpus contains 2 unescaped `U+2028` and 4 unescaped `U+0085`.
Python's `str.splitlines()` breaks on both; iterating the file object does not. The shipped
reader iterates, and is correct. Anything that reaches for `.splitlines()` on this data will
silently shred six rows into fragments.

## 3. How the data is cleaned

### What counts as a word

A run of letters, digits, slashes, hyphens and underscores containing at least one letter — so
`load-bearing`, `snake_case`, `--all-targets` and `src/main` survive whole, while backtick, `:`
and `>` are separators. No stemming, no n-grams, no stopword list. Every appearance counts, so
a word used three times in one description contributes three. Median description: 65 words.

**Order matters, and each step exists because of what the previous one broke:**

1. **Links first, collapsed to their domain.** `[bugbot](https://cursor.com/x)` gives `bugbot`
   and `[cursor-url]`. Splitting on punctuation first produced `bugbot](https` and a trail of
   fragments that ranked among components' most characteristic words. Keeping links whole was
   little better: a tool that puts a per-item link in every description earns one word per
   *item*, and Snyk's vulnerability links alone were the top words of eight of the sixteen
   components then in use.
2. **Then HTML tags, whole.** Splitting them character by character turned
   `<sup>reviewed</sup>` into `sup, reviewed, sup` and made `li`, `br`, `td` and `href` six of
   one component's twelve commonest words. The pattern requires a letter or slash after the
   bracket, so `a > b` in prose is not mistaken for markup.
3. **Then everything else splits** on any character a word may not contain, which handles what
   markdown creates without knowing markdown exists: `srcset="…"` gives `srcset`, `*emphasis*`
   needs no rule because `*` is a separator.
4. **Then trim the edges.** `_other example_` needs its underscores trimmed, since an
   underscore is legal *inside* a word; a leading hyphen stays, so `--all-targets` is not
   turned into `all-targets`.

Snyk advisory identifiers collapse to `[snyk-id]` for the same reason links collapse to their
domain: 1,401 distinct ones between them occupied seven components.

Requiring a letter drops numbers and rules — `27.49`, `589/1000`, `2025-06-24`, `-------` —
along with the arrow and `+`. **The em dash is the one deliberate exception**, taken before the
split and counted as a word of its own. It earns that: 0.2 appearances per 10,000 words over
the first four weeks of the corpus against 129.4 over the last four.

### What gets thrown away

**Identical word sets collapse, within each week.** This is about text, not authorship: one
ordinary human account posted 147 copies of one sentence in a fortnight, 16% of it. Collapsing
inside the week rather than across the window means a template running for months contributes
one description to every week alike, which is a level and not a change.

**No author may contribute more than three descriptions to a week.** This finds mass-produced
text without a blocklist, because it concentrates by *author* rather than by repository:
`copilot` wrote 197 of the 198 descriptions carrying GitHub's coding-agent survey link, across
192 repositories, and `vercel[bot]` wrote all 85 carrying one particular CVE. It catches what
the `[bot]` suffix misses — `copilot`, `pyup-bot`, `scala-steward` and `regro-cf-autotick-bot`
are ordinary logins — and applies to humans on the same terms, which is why it is a cap and not
an exclusion.

### Three floors on a word, and the third is about people

A word needs **45 total appearances**, **25 distinct descriptions**, and **20 distinct
accounts**.

The first two count documents, and a bot's template clears them easily, because the per-week
author cap bounds an account to three a week and a template that runs for sixteen months is
under that cap every single week. Counting *accounts* separates them at a glance:

| word | descriptions | distinct accounts |
|---|---|---|
| `proprosed` | 190 | **16** — three bots supply 144 |
| `pipelineruns` | 252 | **18** — the same three |
| `load-bearing` | 92 | **91** |
| `seam` | 136 | **132** |

A word 91 people reached for is a word; a word in 190 descriptions from 16 accounts is one
document written 190 times. The floor costs 109 of 7,168 words and removes both — including a
misspelling of "proposed" that had been sitting at rank two of the published list.

There is deliberately **no probability floor**: that would throw away exactly the
rare-but-concentrated words the ranking exists to find. Appearances alone are not breadth
either — `multi-draw` appears 101 times inside a single description.

### Rare words are not pruned, and that was tested

Raising the floors would shrink the vocabulary, so it was worth checking whether they could go.
They cannot. Sweeping them upward on one shared count matrix, so the comparison is controlled:

| tf / df | words | the leading component's top words |
|---|---|---|
| **45 / 25** | 7,267 | survived, load-bearing, quietly, refusal, pre-fix, halves |
| 100 / 50 | 4,670 | 17 of the previous top 40 survive |
| 250 / 100 | 2,655 | 7 of 40 |
| 500 / 200 | 1,605 | **0 of 40** — clippy, cargo, --check, uv, bun |
| 1000 / 400 | 930 | 1 of 40 — generic function words |

The weekly *shape* survives all of it (`r` = 0.87 to 0.99). The component's *identity* does not:
by 500/200 the largest component is Rust tooling and this page would be about something else.
And the only motive for pruning was speed, of which there is none to gain — the fit is a
fraction of a second at every vocabulary size.

### Whole weeks only

There is **no cap on a week's size**, and there used to be one. Weeks were once thinned to a
common count, on the argument that text is overdispersed — words cluster inside repositories —
so a rate computed on more descriptions comes out inflated rather than merely more precise. The
argument was sound; the premise stopped holding. Weeks then came from five bulk windows and
their sizes swung by more than a factor of two. Collection is now one window a day and every
window returns a full page, so a week is 700 descriptions by construction and the cap was
discarding half the corpus to enforce something already true.

**Part-weeks at either end are dropped outright.** This matters most at the trailing end and it
matters every day: collection runs each morning, so the newest week is almost always
half-collected, and it is the week everything leans on. The analysis therefore ends at the last
*complete* week and can lag by up to six days, which is the right trade at weekly resolution.
`ANCHOR` is 2024-12-30 because that is the Monday the first week of 2025 begins on, but that is
a labelling choice and not a reason to collect 2024, so collection starts 2025-01-01 and the
resulting five-day first week is dropped along with the trailing one.

## 4. What the model is

Each of `k` components is a fixed probability distribution over the vocabulary — one way of
writing. **One mixture covers the whole window**, and each description is taken to be drawn
from one of the components:

```
W_k                    a distribution over the vocabulary,  sum_v W_vk = 1
pi_k                   how much of the window was written that way,  sum_k pi_k = 1
z_d ~ Cat(pi)          which component wrote description d
x_d ~ Mult(n_d, W_kz)  its words
```

**There is no `t` anywhere in that.** The model has no per-week parameter, so it has nothing
that could describe a trend and no freedom to place one. Every curve produced is attribution
instead — each description assigned by its words alone, the weeks added up afterwards:

```
r_dk = pi_k prod_v W_vk^x_dv / sum_j pi_j prod_v W_vj^x_dv     the responsibility
C_tk = sum over descriptions d written in week t of r_dk        every curve on the page
```

`C` is a sum over fitted responsibilities and not itself a fitted quantity. It was never
optimised toward any shape. If a component rises, the rise is in what people wrote, because
there is nowhere else for it to be.

### This used to be the ablation

An earlier version fitted a mixture *per week*, `pi_tk`, with a smoothness penalty
`lambda K² sum_t (pi_tk - pi_{t-1,k})²` on how fast it could move, and `lambda` chosen by
held-out likelihood. Running that model with one mixture for the whole window was meant as a
check on whether the smoothing had drawn the trend. **The rise survived the check unchanged** —
so the per-week version's extra parameters, 84 × 8 of them plus a penalty weight to tune, were
machinery that bought a readable curve and the suspicion that the model had drawn it. The check
became the model, and `lambda` went with it. There is now nothing to regularise.

## 5. How the model is trained

Expectation-maximisation: attribute the descriptions, refit the word distributions, refit the
mixture. Repeat **to convergence**, from eight different starting points, keeping the highest
likelihood.

### Convergence, not a pass count

There is no iteration count. EM stops when a pass improves the log-likelihood by less than
`1e-6` of itself — about 35 passes here — because a fixed count is a number that has to be
guessed, and the wrong guess is invisible:

| passes | log-likelihood | the headline share |
|---|---|---|
| 6 | -36,913,436 | **43.2%** |
| 12 | -36,889,990 | **62.1%** |
| 24 | -36,882,749 | 61.7% |
| 48 | -36,881,620 | 61.6% |
| 96 | -36,881,615 | 61.6% |

The old setting was a fixed 12, which overstated the headline by half a point while looking
perfectly converged. Tightening the criterion to `1e-8` doubles the work to 79 passes and moves
nothing.

### Restarts are still needed, and convergence does not replace them

The natural hope is that converging properly would make the starting point stop mattering. It
does not. Across 16 seeds:

| | fixed 12 passes | converged 1e-6 | converged 1e-8 |
|---|---|---|---|
| log-likelihood spread | 0.386% | 0.398% | 0.392% |
| headline share range | 36.0 – 63.4% | 37.3 – 63.2% | 35.7 – 63.1% |
| `load-bearing` in top 5 | 13/16 | **15/16** | 15/16 |

**The spread is between distinct local optima, not between half-finished runs.** Converging
harder just lands each seed more precisely in its own basin. That distinction matters, because
it is the difference between "fit it longer" and "fit it more times", and only the second one
helps.

So the restarts stay, and they earn it: fitting once, at seed 0 and `k = 16`, put
`[transifex-url]` and `transifex` at ranks one and two of the published word list — a
translation service's boilerplate welded onto the prose, which is what a mixed local optimum
looks like from outside. The likelihood barely separates the runs while the answer moves by
nearly a factor of two, so the winning run is worth finding, and it is cheap to find: it
appears at the second restart and thirty further restarts never beat it. Eight is generous.

Do not read the published share as a bound in either direction. At `k = 16` the likeliest fit
happened to be the one that split the register most finely and so reported the *smallest* share
of any seed; at `k = 8` it reports near the top of the range. It is the likeliest fit's figure
and nothing more.

### Speed

About eight seconds end to end, on 51,280 descriptions and 7.2 million token occurrences.

| stage | time | share |
|---|---|---|
| **reading the corpus** | **4.6 s** | |
| — `tokens()`, the regex work | 2.4 s | 53% |
| — interning words to integer ids | 0.8 s | 18% |
| — building and deduplicating the matrix | 0.9 s | 21% |
| — reading files and `json.loads` | 0.4 s | 9% |
| **fitting** | **2.7 s** | |
| — one EM pass over 3.3 M nonzeros | 0.013 s | |
| — one fit, to convergence (~35 passes) | 0.33 s | |
| — eight restarts | 2.7 s | |
| import, numba cache load, writing `analysis.js` | ~1 s | |

Four techniques account for it.

**The model has no per-week parameter.** This is the largest factor by a wide margin and it is a
property of the model rather than an optimisation: with one mixture for the whole window there is
no inner optimisation inside an EM pass, so a pass is two sparse products and a softmax. A pass
costs 13 milliseconds.

**The EM sweep is one fused numba kernel.** The obvious formulation builds a `D × k` matrix of
logits, softmaxes it, then multiplies it back against the sparse matrix — three passes over the
data and two dense intermediates the size of the corpus. `_em_sweep` visits each description
once: its logits go into a length-`k` scratch array, are softmaxed in place, and are spent
immediately on the word totals, the weekly counts and the likelihood. Nothing `D`-sized is
allocated. It parallelises over contiguous *blocks* of descriptions, so each thread owns one
slice of every accumulator and no two threads ever touch the same cell — `threads × k × V`
floats, six megabytes at sixteen threads, and no atomics.

The parallelism is the whole of the benefit, and it is worth knowing how much:

| one EM pass | time | vs numpy |
|---|---|---|
| pure numpy — `_em_sweep_numpy`, 13 lines | 32.5 ms | — |
| numba, 1 thread | 50.4 ms | 0.6× |
| numba, 4 threads (what CI has) | 17.5 ms | 1.9× |
| numba, 16 threads | 7.7 ms | 4.3× |

`_em_sweep_numpy` states the same computation in thirteen readable lines and the selftest asserts
the two agree to eight decimal places on every run, because a hand-written parallel reduction is
exactly the kind of code that is wrong in ways tests written against its own output cannot see.

**Counting is sparse-matrix work, not dictionary work.** Words become integers as they are read.
Building the matrix collapses duplicate `(description, word)` pairs in C and sorts each row,
which is precisely the deduplication the filters need — so the distinct-word counts, the
per-week word-set keys and the document frequencies all fall out of a matrix that had to be built
anyway, and a second `(author, word)` matrix gives the distinct-account counts the same way.

**EM stops on convergence.** A pass that improves the log-likelihood by less than `1e-6` of
itself ends the run, which is about 35 passes. See the table above for why a fixed count is not
good enough.

Two small things in the tokeniser matter more than they look, because they run seven million
times: the Snyk-identifier pattern is guarded by a `startswith` so it is attempted a few hundred
times instead of once per token, and there is no letter test, because `WORD_RE` already requires
a letter and trimming only ever removes `_`, `/` and `-`.

The ranking is deterministic: ties in lift break on the word itself, so two builds of the same
corpus are byte-identical and the daily commit does not churn on words that score the same.

### What the selftest guarantees

Run before every publish, and the daily job stops if it fails: the mixture sums to one, the
weekly counts reconstruct each week's document total, the appearance counts reconstruct it too,
the numba kernel matches numpy to 1e-8, and **a planted component is recovered from synthetic
data** — rising from 0.000 to 0.350 at the week it was planted, even though the model has no
way to represent time.

## 6. How the results are displayed

### Choosing which component to show

**The largest one across the last four weeks.** Nothing is selected on how much it grew. A
month rather than a week because a week is 700 descriptions, and the subject of the whole page
should not turn on which of two close components led across one of them.

Growth thresholds used to do the choosing, and it was fragile: at a 1% start one fit rejected
its own largest component, which had gone from 1.06% to 40.35%, for beginning six hundredths of
a point too high.

Those thresholds survive but no longer select — they **check**. Picking the biggest component
says nothing about whether it arrived, and arriving is what the page claims, so the claim is
tested against the component actually chosen: **under 2% of the first eight weeks, at or above
20% of the last eight.** A test may be a round number in a way a selector may not, because
nothing is being ranked and there is no runner-up to exclude unfairly. If it fires, the page
should not be published from that fit, and CI stops.

**An earlier version asserted a growth ratio and a clean gap, and it was wrong.** It required
every component growing 100-fold to be ten times clear of everything that did not, and CI
caught it failing on the very first run — one extra day of data moved the largest non-arrival
from 4.8× to 69×, an eight-fold swing from a hundred descriptions. A ratio of `end/start`
explodes when the start is near zero, so it ranks a component starting at 0.07% above one
starting at 0.3% for no good reason, and is unstable at exactly the point it matters. And the
gap requirement encoded an assumption the data does not support: growth is a continuum here, so
a threshold cuts through the middle of it and no gap can exist.

### The stacked chart

Each week is normalised to its own total, so the bands fill the height and it reads as
composition. That is safe here in a way it would not have been earlier: every day contributes
one full page of descriptions, so the denominator is near constant and no band can appear to
shrink merely because the week around it grew. The arrival sits at the bottom, where a rising
floor is easier to follow than a shape squeezed between others. Gridlines are dashed and in
ink; the band separators are hairlines in the page's ground colour — a distinction that had to
be made deliberately, because when both were the same the gridlines read as band boundaries
cutting through the arrival.

### The word wall

Ranked by lift, measured against the mixture of every **other** component weighted by its share
of appearances — not against the whole corpus:

```
lift_k(v) = W_vk / [ sum_{j != k} m_j W_vj / sum_{j != k} m_j ]
```

**The exclusion is doing enormous work.** The component is now most of the recent weeks, so
dividing by the whole corpus would compare its vocabulary mostly against itself.
`load-bearing` scores **3.75× against the whole corpus and 3,613× against everything that is
not this component** — a thousandfold difference, from one choice of denominator.

Size and shade follow the *logarithm* of that multiple, because it spans three orders of
magnitude — 3,613× down to 4.12× across the thousand words shown — and on a linear ramp every
word past the first dozen would sit at the minimum.

### Hovering

Hovering any word replaces the chart above with that one word's own weekly appearances, and the
chart follows down the list. **Those curves are not the model's.** Each is the raw weekly count
of a word, so only the *choice* of which words to show is the model's doing. `?w=word` links
straight to one.

## 7. Every arbitrary choice

Asked directly, and worth answering completely: `k` is not the only number here that could have
been different. **Two were chosen by looking at the answer**, and both are marked.

| constant | value | how it was chosen |
|---|---|---|
| `K` | 8 | **chosen on the outcome** — the coarsest setting putting `load-bearing` in the top 5 |
| `MIN_TF` | 45 | **chosen on the outcome** — see below |
| `MIN_AUTHORS` | 20 | measured, but a thin margin: bots at 16 and 18, real words at 91 and 132 |
| `EXCLUDE_APPS` | 4 apps | measured — 90% of App-authored bodies |
| `N_INIT` | 8 | measured — the winner appears at restart 2 and 32 never beat it |
| `TOL` | 1e-6 | measured — 1e-8 doubles the work and changes nothing |
| `LEAD_WINDOW` | 4 weeks | judgement — "a month", to stop one week deciding the subject |
| `LEAD_START`, `LEAD_END` | 2%, 20% | round numbers, wide margins, and they only *check* |
| `MIN_DF` | 25 | judgement — breadth, paired with `MIN_TF` |
| `MAX_PER_AUTHOR` | 3 | **arbitrary** |
| `MIN_WORDS` | 5 | **arbitrary** |
| `WORDS_LISTED` | 40 | **arbitrary**, and the code says so |
| `WORDS_LEAD` | 1000 | **arbitrary** round number |
| first/last 8 weeks | 8 | **arbitrary** averaging window for the arrival check |
| `WINDOW_S` | 300 s | **consequential** — see below |
| ten prose terms | 10 | **arbitrary**, though only their union matters |
| `SNYK_ID_RE` digits | 4+ | **arbitrary** |
| `SEED` | 0 | arbitrary and immaterial — best-of-8 spans seeds 0–7 |

**`MIN_TF = 45` was picked by looking at the answer.** `load-bearing` had 51 appearances on the
corpus of the day, so 45 let it through and 60 would not have. That is the same species of
choice as `K` and deserves the same label. It is no longer binding on the title word — the
corpus has grown and it now has 101 appearances, clearing the floor by more than twice over —
but it still shapes the list: `throwaway`, third in the published top five, has 55 appearances,
and a floor at 60 would drop it.

**`WINDOW_S = 300` is more consequential than it looks.** Five minutes was chosen so a window
would fit in one page of a hundred results. It no longer does — every window comes back full,
in 2025 and 2026 alike — so the sampler *truncates* rather than enumerates, and "a five-minute
window" is really "the first hundred pull requests after a random instant". A narrower window
would enumerate honestly at the cost of fewer descriptions a day. The uniform placement means
this is not a bias in *time*; it is a varying effective width.

### How many components, and why that is not a neutral choice

`k = 8`, and the number was chosen so that `load-bearing` — the word this page is named after —
would rank among the five most characteristic words of the arriving component. It does, at rank
1, in 15 of 16 starting seeds.

| k | seeds with `load-bearing` in top 5 |
|---|---|
| **8** | **15 of 16** |
| 16 | 2 of 8 (it ranks 45th at the best seed) |
| 24 | 3 of 8 |
| 32 | 1 of 8 |
| 48 | 0 of 8 |

**That is selection on the outcome, and it cannot also be evidence for the outcome.** Held-out
likelihood prefers far more components than eight — it kept improving to about `k = 128` before
turning over at 192. A coarser model lumps together registers a finer one separates, which is
exactly why one word can come to dominate it. What eight buys is a page whose title matches its
own top line. What it costs is that no ranking here may be read as having been discovered: the
vocabulary is real and the rise is real, but the *order* was tuned until a chosen word came
first.

The finding itself does not depend on it. A component going from near nothing to a large share
of the week, with these words, is there at every `k` from 6 to 48. Only the ranking of
individual words within it moves.

**Retracted: "marker recovery".** This is the second time this project has chosen `k` by
looking at the answer. The first was an accident: an earlier version scored each setting by how
many of 22 marker words it reproduced, and those 22 had been chosen by reading the output at
`k = 12`. It was a measure of agreement with itself, dressed as validation. Held-out likelihood
replaced it. That one is retracted; this one is disclosed, because it was asked for
deliberately.

## 8. Caveats to carry

**On the title.** Words naming Claude are elevated inside this component — `claude` at 5.05×
and its link at 5.68× — while Cursor sits at 1.31×, ChatGPT at 1.05×, Codex at 1.06× and
Copilot at 0.62×, at or below the baseline. But `gpt-5` is elevated further than any of them,
at 10.12×. The register is far more strongly associated with Claude than with most assistants,
and it is not Claude's alone.

**A confound in the denominator.** The corpus only contains pull requests whose author wrote a
description, and that condition loosens over the window — empty descriptions fall from about a
third to about a seventh. A rate measured per pull request therefore rises both because
descriptions use a word more and because more pull requests have descriptions at all. Every
figure here is a share of descriptions *that exist*, which sidesteps that without resolving it.

**The sampling is uniform in time, not in coverage.** See `WINDOW_S` above.

**And the obvious one.** None of this observes an assistant writing anything. It observes a way
of writing becoming common, and the words in it are words people associate with one. The
correlation is strong; the causation is inferred.

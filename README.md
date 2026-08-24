# The load-bearing vocabulary of Claude

Groups of words whose frequency in GitHub pull request descriptions changed at the same time,
found without being told what to look for. One of them was 1.3% of the corpus at the start of
2025 and is 60% of it by the middle of 2026.

**[louisabraham.github.io/load-bearing](https://louisabraham.github.io/load-bearing/)**

| file | what it is |
|---|---|
| `fetch_day.py` | one request a day to GitHub's search API, one `data/days/YYYY-MM-DD.jsonl`. Standard library only. |
| `analyze.py` | reads the days, groups them into whole weeks, fits the model, writes `analysis.js`. Needs `numpy` and `scipy`. |
| `index.html` | reads `analysis.js`. No build step. Open it. |
| `.github/workflows/daily.yml` | does all of the above, daily, and commits the result. |

```bash
pip install numpy scipy
export GITHUB_TOKEN=$(gh auth token)

python fetch_day.py                  # yesterday, one request
python fetch_day.py --backfill 30    # and the last 30 days, if missing
python analyze.py                    # ~11 s end to end
python analyze.py --selftest         # the invariants, on synthetic data
open index.html
```

Current state: **600 collected days, 595 of them in 85 whole weeks** (2025-01-06 to 2026-08-17),
51,964 descriptions, 5,705,560 word appearances, 7,180 words above the floors.

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

Each of `k` **ways of writing** is a fixed distribution over the vocabulary, and every
description is assigned to exactly one of them: the one it is closest to, under the divergence
that belongs to word counts.

$$W_c \;\text{a distribution over the } V \text{ words},\qquad p_d = x_d / n_d \;\text{one description}$$

$$z_d \;=\; \arg\min_c \; n_d \, \mathrm{KL}(p_d \,\|\, W_c), \qquad W_c \;\propto \sum_{d\,:\,z_d = c} x_d$$

Each centre is the middle of what it was given, which is that cluster's KL-centroid. This is
k-means with KL in place of squared distance — Bregman hard clustering — and the $n_d$ weight is
the only trace of counting left in it: a long description pulls its centre harder than a short
one, which is what makes this KL k-means over *descriptions* rather than over word-frequency
vectors.

Nothing is ever evaluated as a divergence. Since

$$x_d \cdot \log W_c \;=\; -\,n_d\left(\mathrm{KL}(p_d \,\|\, W_c) + H(p_d)\right)$$

and $H(p_d)$ does not vary with $c$, the nearest centre is simply the largest $x_d \cdot \log W_c$,
and the whole assignment step is one sparse product against the corpus.

**There is no `t` anywhere in that.** One set of centres covers the whole window, so the fit has
no per-week parameter — nothing that could describe a trend and no freedom to place one. Every
curve the page draws is attribution instead, each description placed by its words alone and the
weeks counted up afterwards:

$$C_{tc} \;=\; \#\{\,d \;:\; t(d) = t,\; z_d = c\,\}$$

`C` is a count of whole descriptions and not a fitted quantity. It was never optimised toward
any shape. If a way of writing rises, the rise is in what people wrote, because there is nowhere
else for it to be.

**And the assignment is hard.** A description belongs to one way of writing rather than partly
to several, so a weekly curve is a count of descriptions and can be read as one.

## 5. How the model is trained

Greedy k-means++ under KL to place the centres, then Lloyd's algorithm to a fixed point. Once.

### Seeding

D² sampling with the multinomial's own divergence in place of squared distance: the first centre
is a random description, and each next one is drawn with probability proportional to its
divergence from the nearest centre already chosen. **Greedy**, meaning three candidates are drawn
each time and the one that reduces the total cost most is kept — which is what
`scikit-learn`'s `k-means++` does by default, at 2 + ⌊log k⌋ = 4 candidates for this `k`.

Only the first term of

$$n_d \, \mathrm{KL}(p_d \,\|\, W_c) \;=\; \sum_v x_{dv} \log \frac{x_{dv}}{n_d} \;-\; x_d \cdot \log W_c$$

depends on the description alone, so it is computed once and the sampling costs one sparse
product per candidate — 24 of them, about a tenth of a second.

Seeding this way rather than from random descriptions is what makes a single run defensible. It
is also the only reason there is no `n_init`.

### Iteration

Assign, recentre, repeat, and **stop when no description changes hands**. That is an exact fixed
point, so there is no tolerance to choose and no pass count to guess — the two settings a
k-means implementation usually has to invent. Thirty-six passes on this corpus.

A pseudo-count of 0.01 is added to every centre before normalising, so no centre gives a word
probability zero and no description can be infinitely far from anywhere.

### One run, and what that costs

There are no restarts, and the honest statement of the price is this: **the figure on the page is
one run's figure, and the seed moves it.** Fitting the same corpus from 32 different starting
points:

| | across 32 runs |
|---|---|
| where the arriving component ends | 29.8% to 62.9%, median 49.0% |
| `load-bearing`'s rank in its words | first in 21, top five in 28, top forty in 31 |
| the arrival check | passes in 21 |
| total cost | spread of 0.93% |
| agreement on the weekly *shape* | mean `r` = 0.985 between any two runs |
| agreement on *which* descriptions | mean F1 = 0.744, and as low as 0.13 |

So the thing the page is about is not in doubt — every run finds a way of writing that arrives,
almost every run puts the title word at or near the top of it, and the curves lie on top of each
other. Where exactly it ends is one run's answer.

**Restarts would not fix that, and it is worth being precise about why.** Cost is what a restart
would choose on, and cost is nearly independent of everything the page reports: across those 32
runs its correlation with the final share is **+0.03**, and picking the cheapest of eight runs
leaves the median share where one run left it. What restarts do buy is that the arrival check
stops firing — it passes for 65% of single runs, 94% of best-of-four, 99% of best-of-eight — so
they are worth about eighteen seconds if a daily job that occasionally refuses to publish is
worse than one that always does. They are not worth anything as a route to a truer number.

`SEED` is therefore a real choice, and it is listed as one in §7.

### Speed

About eleven seconds end to end, on 51,964 descriptions and 5.7 million word appearances,
four threads.

| stage | time |
|---|---|
| reading the corpus — regexes, interning, building the matrix | 8.4 s |
| seeding and fitting, 36 passes | 2.4 s |
| ranking words, packing, writing `analysis.js` | 0.1 s |

**There is no `numba` any more, and that is a measurement rather than a preference.** The whole
of the arithmetic is two sparse products per pass, which `scipy` already does in C:

| one pass over 3.3 M nonzeros | time |
|---|---|
| `scipy` sparse products, what the code does | 55 ms |
| hand-written parallel `numba` kernel, 4 threads | 27 ms |

That is 2.0×, and it buys 1.0 s across the 36 passes. A cold JIT compile of that kernel costs
1.5 s, and `__pycache__` is not committed, so the daily job pays it on **every** run — the
kernel was a net loss of half a second on the machine that actually runs it, and a dependency,
and forty lines of index arithmetic that had to be checked against a numpy reference to be
believed. On a warm cache it wins a second; the CI cache is never warm.

Two small things in the tokeniser matter more than they look, because they run seven million
times: the Snyk-identifier pattern is guarded by a `startswith` so it is attempted a few hundred
times instead of once per token, and there is no letter test, because `WORD_RE` already requires
a letter and trimming only ever removes `_`, `/` and `-`.

The ranking is deterministic: ties in lift break on the word itself, so two builds of the same
corpus are byte-identical and the daily commit does not churn on words that score the same.

### What the selftest guarantees

Run before every publish, and the daily job stops if it fails: the centres are distributions, the
weekly counts are whole numbers and reconstruct each week's description total, the appearance
counts reconstruct it too, and **a planted way of writing is recovered from synthetic data** —
rising from 0.000 to 0.350 at the week it was planted, even though the model has no way to
represent time.

## 6. How the results are displayed

### Choosing which component to show

**The largest one across the last four weeks.** Nothing is selected on how much it grew. A
month rather than a week because a week is 700 descriptions, and the subject of the whole page
should not turn on which of two close components led across one of them.

Two thresholds sit beside that choice, and they **check** rather than select. Picking the biggest component
says nothing about whether it arrived, and arriving is what the page claims, so the claim is
tested against the component actually chosen: **under 2% of the first eight weeks, at or above
20% of the last eight.** A test may be a round number in a way a selector may not, because
nothing is being ranked and there is no runner-up to exclude unfairly. If it fires, the page
should not be published from that fit, and CI stops.

They are stated as two absolute shares rather than as a growth ratio because `end/start`
explodes when the start is near zero: it would rank a component starting at 0.07% above one
starting at 0.3% for no good reason, and be least stable exactly where it matters.

### Whether it is "still growing"

The page says the component is still growing, and that sentence is read off the data rather than
typed into the markup: `analyze.py` fits a least-squares line to the component's observed weekly
share over the last 12 weeks and reports the slope, and the page phrases itself from the sign.
Currently **+1.3 points a week**, over a stretch running 49.3% to 66.4%. If it ever flattens the
page will say it has levelled off instead, without anyone editing it — a claim that can go stale
should not be a string constant.

Note that the last eight weeks alone are noisy around 60% and would not support the claim on
their own; twelve weeks is what makes the slope clear of the week-to-week scatter.

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

Ranked by lift, measured against every **other** component weighted by its share
of appearances — not against the whole corpus:

$$\mathrm{lift}_k(v) = W_{vk} \Big/ \frac{\sum_{j \neq k} m_j W_{vj}}{\sum_{j \neq k} m_j}$$

where $m_j$ is component $j$'s share of all word appearances.

**The exclusion is doing enormous work.** The component is now most of the recent weeks, so
dividing by the whole corpus would compare its vocabulary mostly against itself.
`load-bearing` scores **3.76× against the whole corpus and 4,062× against everything that is
not this component** — a thousandfold difference, from one choice of denominator.

Size and shade follow the *logarithm* of that multiple, because it spans three orders of
magnitude — 4,062× down to 4.07× across the thousand words shown — and on a linear ramp every
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
| `K` | 8 | **chosen on the outcome** — see below |
| `MIN_TF` | 45 | **chosen on the outcome** — see below |
| `MIN_AUTHORS` | 20 | measured, but a thin margin: bots at 16 and 18, real words at 91 and 132 |
| `EXCLUDE_APPS` | 4 apps | measured — 90% of App-authored bodies |
| `SEED` | 0 | **consequential** — there is one run, and the seed moves the headline; see §5 |
| `TRIALS` | 3 | judgement — k-means++ candidates per centre; `scikit-learn` uses 4 at this `k` |
| `SMOOTH` | 0.01 | **arbitrary** pseudo-count, so no centre gives a word probability zero |
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
| `MAX_PASSES` | 200 | a runaway guard — the fixed point comes at 36 |

**`MIN_TF = 45` was picked by looking at the answer.** `load-bearing` had 51 appearances on the
corpus of the day, so 45 let it through and 60 would not have. That is the same species of
choice as `K` and deserves the same label. It is no longer binding on the title word — the
corpus has grown and it now has 103 appearances, clearing the floor by more than twice over —
but it still shapes the list: `throwaway`, fourth in the published top five, has 53 appearances,
and a floor at 60 would drop it.

**`WINDOW_S = 300` is more consequential than it looks.** Five minutes was chosen so a window
would fit in one page of a hundred results. It no longer does — every window comes back full,
in 2025 and 2026 alike — so the sampler *truncates* rather than enumerates, and "a five-minute
window" is really "the first hundred pull requests after a random instant". A narrower window
would enumerate honestly at the cost of fewer descriptions a day. The uniform placement means
this is not a bias in *time*; it is a varying effective width.

### How many components, and why that is not a neutral choice

`k = 8`, and the number was chosen so that `load-bearing` — the word this page is named after —
would rank among the most characteristic words of the arriving component. It does, at rank 1.

| k | rank of `load-bearing` | the arrival check |
|---|---|---|
| 4 | 2 | fails — the largest component starts at 7.7% |
| 6 | 1 | passes |
| **8** | **1** | **passes** |
| 12 | 1 | passes |
| 16 | 4 | passes |
| 24 | 4 | passes |
| 32 | 36 | fails — nothing ends above 20% |
| 48 | 48 | fails |

**That is selection on the outcome, and it cannot also be evidence for the outcome.** Nothing in
the fit can choose `k` for you: the total cost falls monotonically as `k` rises — 13.7 M at four
centres, 13.0 M at eight, 11.9 M at forty-eight — as it must, because more centres can only be
closer. A coarser setting lumps together ways of writing that a finer one separates, which is
exactly why one word can come to dominate one of them. What eight buys is a page whose title
matches its own top line. What it costs is that no ranking here may be read as having been
discovered: the vocabulary is real and the rise is real, but the *order* was tuned until a
chosen word came first.

The finding itself does not depend on it. A way of writing going from near nothing to most of
the recent weeks, with these words, is there at every `k` from 6 to 24. Only the ranking of
individual words within it moves, and past 24 the register is split finely enough that no single
piece of it is large enough to be called the subject of the page.

**Retracted: "marker recovery".** This is the second time this project has chosen `k` by looking
at the answer. The first was an accident: an earlier version scored each setting by how many of
22 marker words it reproduced, and those 22 had been chosen by reading the output at `k = 12`. It
was a measure of agreement with itself, dressed as validation. That one is retracted; this one is
disclosed, because it was asked for deliberately.

## 8. Caveats to carry

**On the title.** Words naming Claude are elevated inside this component — `claude` at 5.24×
and its link at 5.67× — while Cursor sits at 1.36×, ChatGPT at 1.29×, Codex at 0.81× and
Copilot at 0.61×, at or near the baseline. But `gpt-5` is elevated further than any of them,
at 10.89×. The register is far more strongly associated with Claude than with most assistants,
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

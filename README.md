# The load-bearing vocabulary of Claude

GitHub pull request descriptions, grouped by the words they are written with rather than by
anything they were told to look for: eight ways of writing, and every description belongs to
one of them. One of the eight was 1.0% of the corpus at the start of 2025 and is 45% of it by
the middle of 2026.

**[louisabraham.github.io/load-bearing](https://louisabraham.github.io/load-bearing/)**

None of this observes an assistant writing anything. It observes a way of writing becoming
common, and the words in it are words people associate with one.

| file | what it is |
|---|---|
| `fetch_day.py` | one request a day to GitHub's search API, one `data/days/YYYY-MM-DD.jsonl`. Standard library only. |
| `analyze.py` | reads the days, groups them into whole weeks, fits the model, writes `analysis.js`. Needs `numpy` and `scipy`. |
| `index.html` | reads `analysis.js`. One board, one screen: the figures, the stack, a word's own history, the thousand words. No build step. Open it. |
| `tests/test_page.py` | what the page must keep doing, driven in a real browser. Every test is a bug it once had. |
| `.github/workflows/daily.yml` | does all of the above, daily, and commits the result. |

```bash
pip install numpy scipy
export GITHUB_TOKEN=$(gh auth token)

python fetch_day.py                  # yesterday, one request
python fetch_day.py --backfill 30    # and the last 30 days, if missing
python analyze.py                    # ~30 s end to end
python analyze.py --selftest         # the invariants, on synthetic data
open index.html

pre-commit install                   # optional: ruff and the html formatter, on commit

uv pip install pytest-playwright     # the page's own tests, in a real browser
pytest tests -q
```

Current state: **602 collected days, 595 of them in 85 whole weeks** (2025-01-06 to 2026-08-17),
47,464 descriptions, 5,024,747 word appearances, 6,846 words above the floors.

---

# The whole process

Everything below is either measured or labelled as a judgement call, and
[§7](#7-the-arbitrary-choices) lists the numbers that could have been chosen differently,
including the two that were chosen by looking at the answer.

## 1. Why GH Archive cannot be used

The natural source for this is the public archive of GitHub's event stream, and it stopped
working. Since mid-2025 its feed carries almost only `PushEvent`: a complete hour of 2024-08-12
holds 13,555 `IssueCommentEvent` against 86 for the same hour of 2026-08-10, and the archive
carried three to ten thousand issue comments an hour through 2025-10, then 866 in 2026-06 and 77
in 2026-07. Pushes survive at full volume but carry no text, GitHub having
[removed the commit array](https://github.blog/changelog/2025-08-08-upcoming-changes-to-github-events-api-payloads/)
from the payload in October 2025.

The cause is upstream, in GitHub's Events API —
[#310 "Drastic Drop Off in Events After 2025-05-23"](https://github.com/igrigorik/gharchive.org/issues/310)
has been open since July 2025 with no maintainer reply, and the identical gaps appear in
OSSInsight, which reads the API directly. So no mirror repairs it: BigQuery, ClickHouse and the
Kaggle and Hugging Face copies all read the same feed.

**This was found the hard way, and it is the reason for the rewrite.** An earlier version of this
project, built on the archive, reported `load-bearing` in 17 documents. That was wrong by a factor
of 158 — the comments had disappeared from the feed, not from GitHub.

## 2. How the data is collected

What works is GitHub's **search** API, for one specific reason: `created:` accepts timestamps and
not only dates. A window can therefore be minutes wide — narrow enough that a single page of a
hundred results *enumerates* it rather than samples it — and every response carries the full body
text.

So: **one randomly placed five-minute window of newly opened pull requests, every single day.**

- **Seeded on the date**, so the whole corpus is reproducible from its dates alone.
- **One immutable file per day**, `data/days/YYYY-MM-DD.jsonl`, about 140 kB, committed and never
  rewritten. The repository's history *is* the history of the sample.
- **Days collect, weeks analyse.** A hundred descriptions is too thin to compare against another
  hundred, so seven days make a week.

Two filters are pushed into the query itself, and together they take a page from 43 usable
descriptions to 97: four Apps excluded — `pull`, `dependabot`, `renovate`, `github-actions`, 90%
of App-authored bodies — and empty bodies excluded, 45% of all pull requests. There is no
emptiness qualifier in the search API; requiring any one of ten function words *in the body* does
it exactly.

**One honest limit.** Every window comes back a full page, so a window holding more than a hundred
is truncated to its earliest hundred. The placement is uniformly random, so what is sampled is
still everything created in a uniformly random interval — but that interval is *effectively
narrower* when GitHub is busier. The sample is uniform in time, not in the fraction of each period
it captures.

## 3. How the data is cleaned

**A word** is a run of letters, digits, slashes, hyphens and underscores containing at least one
letter, so `load-bearing`, `snake_case`, `--all-targets` and `src/main` survive whole. No stemming,
no n-grams, no stopword list. Every appearance counts, so a word used three times in one
description contributes three. Median description: 65 words.

Links collapse to their domain and HTML tags are taken whole, because the alternative ranked
fragments: splitting on punctuation first produced `bugbot](https` among a component's most
characteristic words, and splitting tags character by character made `li`, `br`, `td` and `href`
six of one component's twelve commonest. Snyk advisory identifiers collapse to `[snyk-id]` for the
same reason — 1,401 distinct ones occupied seven components between them.

**The em dash is the one deliberate exception** to requiring a letter, taken before the split and
counted as a word of its own. It earns that: 0.2 appearances per 10,000 words over the first four
weeks of the corpus against 132.4 over the last four.

### What gets thrown away

**Accounts that are not people are dropped by the shape of their name**: anything ending `[bot]`
or `-bot`, and `copilot`, which is an agent posting under an ordinary login. The query can only
exclude Apps one slug at a time and the four it names are four of thousands — 1,042 such accounts
are in the collected corpus, and dropping them costs 7,901 rows, 13% of what was collected. The
arrival survives it, which is the point of dropping them: with every bot- and agent-named account
gone, a cluster still goes from under 1% to about 45%, and `load-bearing` is still first in it.

Beyond that, identical word sets collapse within each week — one ordinary human account posted 147
copies of one sentence in a fortnight — and **no author may contribute more than three
descriptions to a week**. That catches mass-produced text written from accounts that look human,
and it applies to humans on the same terms, which is why it is a cap and not an exclusion.

### Three floors on a word, and the third is about people

A word needs **45 total appearances**, **25 distinct descriptions**, and **20 distinct accounts**.
The first two count documents, and a bot's template clears them easily. Counting *accounts*
separates them at a glance:

| word | descriptions | distinct accounts |
|---|---|---|
| `pullrequest` | 201 | **17** |
| `mq` | 194 | **10** |
| `600s` | 103 | **16** |
| `load-bearing` | 93 | **92** |
| `seam` | 137 | **134** |

A word 92 people reached for is a word; a word in 201 descriptions from 17 accounts is one document
written 201 times. The floor costs 42 of 6,888 words — it used to cost 109 of 7,168, and the
difference is the bot rule above doing the same job earlier: the two words this table used to
open with, a misspelling of "proposed" and `pipelineruns`, no longer reach the floors at all.

There is deliberately **no probability floor**: that would throw away exactly the
rare-but-concentrated words the ranking exists to find.

### Whole weeks only

A week is 700 descriptions by construction — one window a day, every window a full page — so there
is no cap on a week's size and none is needed. **Part-weeks at either end are dropped outright**,
which matters every day: collection runs each morning, so the newest week is almost always
half-collected, and it is the week everything leans on. The analysis ends at the last *complete*
week and can lag by up to six days, which is the right trade at weekly resolution.

## 4. What the model is

Each of `k` **ways of writing** is a fixed distribution over the vocabulary, and every description
is assigned to exactly one of them: the one it is closest to, under the divergence that belongs to
word counts.

```math
W_c \;\text{a distribution over the } V \text{ words},\qquad p_d = x_d / n_d \;\text{one description}
```

```math
z_d \;=\; \arg\min_c \; n_d \, \mathrm{KL}(p_d \,\|\, W_c), \qquad W_c \;\propto \sum_{d\,:\,z_d = c} x_d
```

Each centre is the middle of what it was given, which is that cluster's KL-centroid. This is
k-means with KL in place of squared distance — Bregman hard clustering — and the $n_d$ weight is
the only trace of counting left in it: a long description pulls its centre harder than a short one,
which is what makes this KL k-means over *descriptions* rather than over word-frequency vectors.

Nothing is ever evaluated as a divergence. Since

```math
x_d \cdot \log W_c \;=\; -\,n_d\left(\mathrm{KL}(p_d \,\|\, W_c) + H(p_d)\right)
```

and $H(p_d)$ does not vary with $c$, the nearest centre is simply the largest $x_d \cdot \log W_c$,
and the whole assignment step is one sparse product against the corpus.

**There is no `t` anywhere in that.** One set of centres covers the whole window, so the fit has no
per-week parameter — nothing that could describe a trend and no freedom to place one. Every curve
the page draws is attribution instead, each description placed by its words alone and the weeks
counted up afterwards:

```math
C_{tc} \;=\; \#\{\,d \;:\; t(d) = t,\; z_d = c\,\}
```

`C` is a count of whole descriptions and not a fitted quantity. It was never optimised toward any
shape. If a way of writing rises, the rise is in what people wrote, because there is nowhere else
for it to be. **And the assignment is hard**: a description belongs to one way of writing rather
than partly to several, so a weekly curve is a count of descriptions and can be read as one.

## 5. How the model is trained

Greedy k-means++ under KL to place the centres, then Lloyd's algorithm to a fixed point. Assign,
recentre, and **stop when no description changes hands** — an exact fixed point, so there is no
tolerance to choose and no pass count to guess, which are the two settings a k-means
implementation usually has to invent. Twenty-six passes on this corpus, about fifteen seconds end to
end. A pseudo-count of 0.01 on every centre keeps any word from having probability zero, so no
description is infinitely far from anywhere.

### Eight restarts, and a retry

Eight fits from eight seeds, and the cheapest is published. **The restarts are not there to find a
better answer**, and it is worth being precise about that: cost is the only quality measure this
model has, and across 32 runs its correlation with the share the page reports is **+0.03**. The
cheapest of eight is not a truer fit than the first of one.

They are there so that the daily job publishes something. A single fit passes the arrival check
about two times in three; the cheapest of eight passes it ninety-nine times in a hundred. If the
cheapest of a batch still does not arrive, the whole batch is run again from eight fresh seeds, up
to four batches, and then nothing is published and the job stops.

**That retry conditions the fit on the check.** On a day one happens, `LEAD_START` and `LEAD_END`
select rather than test, and the published start share is under 2% by construction. So the evidence
that the component arrived is not that the published fit passed — it is the rate at which
*unconditioned* fits arrive at all. Over 32 single runs of the same corpus:

| | across 32 single fits |
|---|---|
| the arrival check | passes in 21 |
| where the component ends | 36.3% to 63.8%, median 45.5% |
| `load-bearing`'s rank in its words | first in 15, top five in 29, top forty in 32 |
| total cost | spread of 1.15% |
| agreement on the weekly *shape* | mean `r` = 0.991 between any two runs |
| agreement on *which* descriptions | mean F1 = 0.770, and as low as 0.46 |

So the thing the page is about is not in doubt: every one of those runs finds a way of writing that
ends between 30% and 63% of the recent weeks, almost every one puts the title word at or near the
top of it, and any two of them draw the same weekly shape. **Where exactly it ends is one fit's
answer**, and `SEED` is listed as a real choice in §7 for that reason.

### What the selftest guarantees

Run before every publish, and the daily job stops if it fails: the centres are distributions, the
weekly counts are whole numbers and reconstruct each week's description total, and **a planted way
of writing is recovered from synthetic data** — rising from 0.000 to 0.350 at the week it was
planted, even though the model has no way to represent time.

## 6. How the results are displayed

**The component shown is the largest across the last four weeks.** Nothing is selected on how much
it grew, and a month rather than a week because a week is 700 descriptions and the subject of the
whole page should not turn on which of two close components led across one of them.

Two thresholds sit beside that choice, and they **check** rather than select: **under 2% of the
first eight weeks, at or above 20% of the last eight**. Picking the biggest component says nothing
about whether it arrived, and arriving is what the page claims. If the check fires, CI stops and
nothing is published. They are two absolute shares rather than a growth ratio because `end/start`
explodes when the start is near zero, and would be least stable exactly where it matters.

**"Still growing" is read off the data rather than typed into the markup.** `analyze.py` fits a
least-squares line to the component's observed weekly share over the last 12 weeks and the page
phrases itself from the sign — currently **+0.8 points a week**, over a stretch running 37.9% to
47.1%. A claim that can go stale should not be a string constant.

**The stack is normalised per week**, so the bands fill the height and it reads as composition;
the arrival sits at the bottom, where a rising floor is easier to follow than a shape squeezed
between others.

**The words are ranked by counting, not by the fit.** The assignment is hard, so every appearance
of a word belongs to exactly one component, and the counts below partition its occurrences — as do
the two totals they are measured against:

```math
\mathrm{ratio}(v) = \frac{x^{\,\text{in}}_v \big/ N^{\,\text{in}}}
                        {\max(x^{\,\text{out}}_v,\ 1) \big/ N^{\,\text{out}}}
```

where $N^{\text{in}}$ is every appearance of every word inside the component and $N^{\text{out}}$
is every appearance outside it. **Each side is divided by its own size**, which is what makes this
a ratio of two frequencies rather than of two counts.

A ratio of raw counts — which is what this was — carries the component's size inside it. The
component holds 1,054,188 of the corpus's 5,024,747 appearances, a fifth, so dividing its count of
a word by the four fifths written outside scaled every one of its words down by the same 3.77; the
component holding a fortieth of the corpus had its own scaled down by 40. Those numbers were
comparable neither between components nor with the "× more frequent" the page prints them as. The
ranking *within* a component does not move, because the correction is one factor per component and
the same for every word in it. What moves is what the number means.

The floor of one is for the words that are never written outside this component at all, which is
most of the top of the list: without it their frequency outside is zero and the ratio a division by
zero rather than their own count. A word that *is* written outside is divided by what it was
actually written, not by that plus a pseudo-count.

**The comparison is against everything that is not this component**, rather than against the whole
corpus. The component is close to half of the descriptions of the recent weeks, so a corpus-wide
baseline would be made mostly of its own writing and would compare it against itself.
`load-bearing` is written 98 times inside it and three times outside — 93 per million words inside
against 0.76 per million outside, which is 123×. Size follows the *logarithm* of the ratio, which
runs from there down to 3.25× across the thousand words shown.

Choosing a word replaces the chart beside it with that word's own history, **as a rate**:
appearances per million words written that week. The corpus is not the same size from one week to
the next — 37,036 words in the thinnest week and 128,589 in the fattest — so a curve of raw counts
draws the corpus growing wherever it draws the word arriving, and nothing in the picture separates
the two. **Those curves are still not the model's**: each is a count of one word divided by the
week that held it, so only the *choice* of which words to show is the model's doing.

## 7. The arbitrary choices

`k` is not the only number here that could have been different. **Two were chosen by looking at the
answer**, and both are marked.

| constant | value | how it was chosen |
|---|---|---|
| `K` | 8 | **chosen on the outcome** — the number was picked so that `load-bearing` would rank first among the arriving component's words |
| `MIN_TF` | 45 | **chosen on the outcome** — see below |
| `SEED` | 0 | **consequential** — the seed moves the headline; see §5 |
| `WINDOW_S` | 300 s | **consequential** — see below |
| `MIN_AUTHORS` | 20 | measured, but a thin margin: templates at 10 and 17, real words at 92 and 134 |
| `BOT_SUFFIX`, `BOT_LOGIN` | `[bot]`, `-bot`, `copilot` | judgement — the accounts a login says are not people; 13% of the collected rows |
| `EXCLUDE_APPS` | 4 apps | measured — 90% of App-authored bodies |
| `N_INIT` | 8 | measured — a single fit publishes 2 times in 3, the cheapest of eight 99 in 100 |
| `LEAD_WINDOW` | 4 weeks | judgement — "a month", to stop one week deciding the subject |
| `LEAD_START`, `LEAD_END` | 2%, 20% | round numbers, wide margins, and they only *check* |
| `MAX_PER_AUTHOR` | 3 | **arbitrary** |
| `WORDS_LEAD` | 1000 | **arbitrary** round number |

**`MIN_TF = 45` was picked by looking at the answer.** `load-bearing` had 51 appearances on the
corpus of the day, so 45 let it through and 60 would not have. That is the same species of choice
as `K` and deserves the same label. It is not binding on the title word today — the corpus has
grown and it now has 101 appearances, clearing the floor by more than twice over — but it still
shapes the list: `throwaway`, 41st of the thousand published, has 52 appearances, and a floor at
60 would drop it.

**`WINDOW_S = 300` is more consequential than it looks.** Five minutes was chosen so a window would
fit in one page of a hundred results. It does not, in 2025 or 2026 — every window comes back full —
so the sampler *truncates* rather than enumerates, and "a five-minute window" is really "the first
hundred pull requests after a random instant". A narrower window would enumerate honestly at the
cost of fewer descriptions a day. The uniform placement means this is not a bias in *time*; it is a
varying effective width.

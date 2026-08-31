# The load-bearing vocabulary of Claude

GitHub pull request descriptions, grouped by the words they are written with rather than by
anything they were told to look for: ten ways of writing, and every description belongs to one of
them. One of the ten was 0.7% of the corpus at the start of 2025 and is 39% of it by the middle of
2026.

**[louisabraham.github.io/load-bearing](https://louisabraham.github.io/load-bearing/)**

| file | what it is |
|---|---|
| `fetch_day.py` | ten requests a day to GitHub's search API, one `data/days/YYYY-MM-DD.jsonl`. Standard library only. |
| `analyze.py` | reads the days, groups them into whole weeks, fits the model, writes `analysis.js` and `model.js`. Needs `numpy`, `scipy` and `numba`. |
| `index.html` | reads `analysis.js`. One board, one screen: the figures, the stack, a word's own history, the thousand words. No build step. Open it. |
| `detect.html` | reads `model.js`. Paste a description, and the same fit says whether it is the arriving way of writing. Runs in the page; §7. |
| `model.js` | the whole of the fit written down — every word, all ten numbers — in 304 kB. Nothing but the classifier reads it. |
| `tests/` | what the two pages must keep doing, driven in a real browser. Every test is a bug one of them once had. |
| `.github/workflows/daily.yml` | does all of the above daily, commits the corpus here, publishes the page to `gh-pages`. |

```bash
pip install numpy scipy numba
export GITHUB_TOKEN=$(gh auth token)

python fetch_day.py                  # yesterday, ten requests
python fetch_day.py --backfill 30    # and the last 30 days, if missing
python analyze.py                    # ~50 s on twelve cores
python analyze.py --selftest         # the invariants, on synthetic data
open index.html                      # the board
open detect.html                     # the same fit, asked about one text

pre-commit install                   # optional: ruff and the html formatter, on commit
uv pip install pytest-playwright && pytest tests -q
```

Current state: **603 collected days, 595 of them in 85 whole weeks** (2025-01-06 to 2026-08-17),
461,121 descriptions, 51,079,244 word appearances, 19,798 words above the floor.

---

## 1. Why GH Archive cannot be used

The natural source is the public archive of GitHub's event stream, and it stopped working. Since
mid-2025 the feed carries almost only `PushEvent` — a complete hour of 2024-08-12 holds 13,555
`IssueCommentEvent` against 86 for the same hour of 2026-08-10 — and pushes carry no text, GitHub
having [removed the commit array](https://github.blog/changelog/2025-08-08-upcoming-changes-to-github-events-api-payloads/)
in October 2025. The cause is upstream in the Events API:
[#310](https://github.com/igrigorik/gharchive.org/issues/310) has been open since July 2025 with no
maintainer reply, and the same gaps appear in OSSInsight, which reads the API directly. No mirror
repairs it, because they all read the same feed.

**This was found the hard way.** An earlier version of this project, built on the archive, reported
`load-bearing` in 17 documents. That was wrong by a factor of 158: the comments had disappeared from
the feed, not from GitHub.

## 2. How the data is collected

What works is GitHub's **search** API, for one reason: `created:` accepts timestamps and not only
dates, so a window can be minutes wide and every response carries the full body text.

**Ten five-minute windows a day, one drawn from each 2.4 hours of it.** The ten starts are drawn to
the second and one per block, which is not fussiness: they used to be multiples of five minutes,
which is exactly the granularity a cron schedule fires on, so every window opened at an instant
when scheduled automation opens pull requests. Blocks also keep the ten from clumping and make it
impossible for two to overlap. The draw is seeded on the date, so the whole corpus is reproducible
from its dates alone, and each day is one immutable file of about 1.4 MB, committed and never
rewritten. The repository's history *is* the history of the sample.

Two filters go into the query itself and together take a page from 43 usable descriptions to 97:
four Apps excluded by name — `pull`, `dependabot`, `renovate`, `github-actions`, which are 90% of
App-authored bodies — and empty bodies excluded, 45% of all pull requests. There is no emptiness
qualifier in the search API; requiring any one of ten function words *in the body* does it exactly.

**One honest limit.** A day of 2026 holds some 460,000 pull requests matching the query, a
five-minute window about 1,250, and a page is a hundred — so a window is truncated to its earliest
hundred and this samples rather than enumerates. Nothing here could enumerate a day: the search API
returns at most 1,000 results per query however many matched. Uniform placement means this is not a
bias in *time*; the effective width just narrows as GitHub gets busier. A day can also come in
short, and 116 of the 603 do — mostly early 2025, when a window did not always fill its page, but
two days hold 900 because one of their ten windows returns nothing at all and returns nothing again
when asked twice. Those are holes in GitHub's own index, and the day is written short rather than
patched.

**The corpus and the site live on different branches.** A published Pages site may be no larger
than 1 GB and the corpus grows 1.4 MB a day, so the daily run commits the day here and pushes only
`index.html`, `analysis.js` and `.nojekyll` — a quarter of a megabyte — to `gh-pages`. The corpus
keeps its history because its history is the point; the site does not need one.

## 3. How the data is cleaned

**A word** is a run of letters, digits, slashes, hyphens and underscores containing at least one
letter, so `load-bearing`, `snake_case`, `--all-targets` and `src/main` survive whole. No stemming,
no n-grams, no stopword list. Links collapse to their domain and HTML tags are taken whole, because
splitting on punctuation first put `bugbot](https` and `href` among a component's most
characteristic words. The em dash is the one deliberate exception to requiring a letter, and it
earns it: 0.2 appearances per 10,000 words in early 2024 against 123 in mid-2026. Median
description: 65 words.

**What gets thrown away.** Accounts that are not people, by the shape of the login — anything
ending `[bot]` or `-bot`, plus `copilot` — which is 3,784 accounts and 13.2% of collected rows.
Identical word sets within a week, because one ordinary human account posted 147 copies of one
sentence in a fortnight. And **no author may contribute more than three descriptions to a week**,
which catches mass-produced text from accounts that look human and applies to humans on the same
terms, which is why it is a cap and not an exclusion.

### One floor on a word, and it counts people

A word is in the vocabulary when **50 distinct accounts** have written it. That is the only floor.
There were three — 45 appearances, 25 descriptions, 20 accounts — and two of them were doing
nothing this one does not do better, because counting appearances cannot tell a shared word from
one document written two hundred times:

| word | appearances | descriptions | accounts | |
|---|---|---|---|---|
| `store-path` | 242 | 242 | **2** | dropped |
| `mq` | 569 | 533 | **36** | dropped |
| `load-bearing` | 1,011 | 905 | **848** | kept |
| `seam` | 1,849 | 1,247 | **1,135** | kept |

A word 848 people reached for is a word; a word in 242 descriptions from 2 accounts is one document
written 242 times. **The number is set on a property of the method, not on the answer**: it is the
least restrictive floor at which two independent fits agree on half of their top twenty words.
Agreement rises with the floor all the way up, so there is no optimum to find — only a rate of
return, and a rule that picks a point on it for a stated reason. It costs coverage: 19,798 words of
the 2.1 million in the corpus, where the old three floors kept 26,113.

**Whole weeks only.** Seven days of ten windows is 7,000 descriptions collected and about 5,300
after the filters, so weeks are the same size by construction and need no cap. Part-weeks at either
end are dropped outright, which matters daily — collection runs each morning, so the newest week is
almost always half-collected, and it is the week everything leans on.

## 4. What the model is

Each of `k` **ways of writing** is a fixed distribution over the vocabulary, and every description
is assigned to exactly one of them: the one it is closest to, under the divergence that belongs to
word counts.

```math
z_d \;=\; \arg\min_c \; n_d \, \mathrm{KL}(p_d \,\|\, W_c), \qquad W_c \;\propto \sum_{d\,:\,z_d = c} x_d
```

Each centre is the middle of what it was given — that cluster's KL-centroid. This is k-means with
KL in place of squared distance, and the $n_d$ weight is the only trace of counting left in it: a
long description pulls its centre harder than a short one. Nothing is ever evaluated as a
divergence, because $x_d \cdot \log W_c = -n_d(\mathrm{KL}(p_d \| W_c) + H(p_d))$ and $H(p_d)$ does
not vary with $c$, so the nearest centre is the largest $x_d \cdot \log W_c$ and the assignment step
is one sparse product against the corpus.

**There is no `t` anywhere in that.** One set of centres covers the whole window, so the fit has no
per-week parameter — nothing that could describe a trend and no freedom to place one. Every curve
the page draws is attribution instead: each description placed by its words alone, the weeks counted
up afterwards. If a way of writing rises, the rise is in what people wrote, because there is nowhere
else for it to be.

## 5. How the model is trained

Greedy k-means++ under KL, then Lloyd's algorithm to an exact fixed point — stop when no
description changes hands, so there is no tolerance to choose and no pass count to guess. Eight
fits from eight seeds, and the cheapest is published. **The restarts are not there to find a better
answer**: cost correlates +0.03 with the share the page reports. They are there so the daily job
publishes something.

**What the page claims is that the component arrived**, so two thresholds check it rather than
select it: under 2% of the first eight weeks, at or above 20% of the last eight. Picking the biggest
component says nothing about whether it arrived. If a batch fails the check, it runs again from
fresh seeds; if four batches fail, nothing is published and the job stops. That retry would
condition the fit on its own check, which is why the evidence is the rate at which *unconditioned*
fits arrive: 31 of 32 single fits of this corpus, and in 1 of the 32 the leading component came out
mixed with another. **Where exactly it ends is one fit's answer**, and `SEED` is listed below for
that reason.

The selftest runs before every publish and stops the job if it fails: the centres are
distributions, the weekly counts are whole numbers that reconstruct each week's total, and a planted
way of writing is recovered from synthetic data — 0.000 to 0.350 at the week it was planted, though
the model has no way to represent time.

## 6. How the results are displayed

**The component shown is the largest across the last four weeks** — a month rather than a week, so
the subject of the page does not turn on which of two close components led across one of them.
"Still growing" is read off the data rather than typed into the markup: a least-squares line over
the last 12 weeks, currently **+1.2 points a week**.

**The words are ranked by counting, not by the fit.** The assignment is hard, so every appearance
belongs to exactly one component and the counts partition it:

```math
\mathrm{ratio}(v) = \frac{x^{\,\text{in}}_v \big/ N^{\,\text{in}}}
                        {\left(x^{\,\text{out}}_v + \tfrac{1}{2}\,\texttt{MIN\_AUTHORS}\right) \big/ N^{\,\text{out}}}
```

Each side is divided by its own size, which makes this a ratio of two frequencies rather than of two
counts. **The pseudo-count in the denominator is the difference between a ranking and a lottery.**
Without it the top of the list was decided by counts of two to seven in 42 million: a word written
three times outside beat one written 158 times, on a difference no larger than its own noise, and
both are too rare for anyone to have noticed. Half of `MIN_AUTHORS` is the fewest appearances a word
in this vocabulary can have, halved — the honest prior for *written outside less often than can be
measured*. A word never written outside then scores in proportion to what it was written **inside**,
so the top is ordered by frequency among a component's exclusive words rather than by the accident
of a tiny divisor.

The comparison is against everything that is not this component, not against the whole corpus,
which would compare the component against itself. `load-bearing` is written 929 times inside and
82 outside — 39×, the top of the list — and size on the page follows the logarithm of the ratio,
from there down to 5× at the thousandth word.

Choosing a word replaces the chart with that word's own history **as a rate**: appearances per
million words written that week. The corpus is not the same size from one week to the next —
380,404 words in the thinnest, 1,406,687 in the fattest — so a curve of raw counts would draw the
corpus growing wherever it draws the word arriving.

## 7. The model is also a classifier

Nothing is added to the fit to make one. The assignment step is $\arg\max_c x_d \cdot \log W_c$,
and $x_d \cdot \log W_c$ is the log-likelihood of the description under a multinomial that draws
every word from $W_c$, up to a coefficient that does not vary with $c$. Normalise the ten and they
are a posterior:

```math
P(c \mid x) \;=\; \frac{\prod_v W_c[v]^{\,x_v}}{\sum_{c'} \prod_v W_{c'}[v]^{\,x_v}}
```

which is multinomial naive Bayes with these centres as its class-conditional distributions.
`SMOOTH` is what makes it usable on text the fit never saw: no centre gives any word probability
zero, so one unexpected word cannot zero a whole component.
**[detect.html](https://louisabraham.github.io/load-bearing/detect.html)** does that arithmetic in
the browser and asks one question of it — is this the component that arrived, or is it one of the
other nine? Nothing is uploaded; the model is a file beside the page. A GitHub link can be pasted
instead of the text — a pull request, an issue, or any of the three kinds of comment — and the
browser fetches it from GitHub's API itself, which answers a page directly and sixty times an hour
without a login. A link that loads goes into the page's own address as `?url=`, so a reading can be
sent to somebody and read back on the way in.

**No prior.** The components are not equally big — the arriving one is 8% of the window and 39% of
the last month — and there is no $\pi_c$ in the formula above because weighting by those shares
would answer *what does a description of 2025 and 2026 usually look like* over the top of the
question the box actually asks, which is about the text in it. Left out, the ten start level and
only the words move them. It is a choice and not an absence of one: with a prior, six words that
say almost nothing come back as the shares; without one, they come back as two to one, which is
what having almost nothing to go on should look like.

**The answer saturates, and the page says so rather than hiding it.** Every word counts as a
separate piece of evidence, so sixty of them in agreement put the odds past anything a percentage
can usefully print, and anything longer than a line or two comes back certain. The page caps what
it will claim at *over 99%* for that reason, and under the answer it prints the words themselves —
sized by how much each moved it, red for the ones that make it this vocabulary and black for the
ones that argue against, in that order. The distance between two components is a sum over the words
of the text and over nothing else, so the strip is the whole of the reason rather than a selection
from it.

**It reads vocabulary. It does not read authorship.** The ten components are ways of writing,
fitted with no label of any kind, and the corpus behind them is pull request descriptions — a text
that is not one is being measured against a ruler made for something else.

### What is shipped, and what the shipping costs

The board draws 150 words a component. The classifier needs all of them: ten numbers for each of
the 20,309 words, which is 4 MB of JSON. It is 304 kB instead, written as text one character at a
time, and the arithmetic on the other end is a lookup and an add.

**The vocabulary, 174 kB of words, is 78 kB.** Sorted, each word is stored as how much of its
predecessor it repeats and then the rest of itself — the trie of the vocabulary written out in the
order a walk of it visits, where the shared prefix is the path already climbed. The repeat count is
a capital letter, and that is why it can run without separators: a word is lowercased before it is
ever counted, so the capital that opens a word is also what ends the one before it.

**Ten numbers a word, where nine would do.** The posterior depends only on the differences: it is
$1 / (1 + \sum_c \exp((L_c - L_0) \cdot x))$, so one of the ten rows is redundant and the model
is exactly expressible in nine. It is shipped as ten because nine is *bigger*. The ten rows are
$\log(1 + M/\texttt{SMOOTH})$, which is exactly zero wherever a component never wrote the word — a
quarter of the entries — and fits one character for 90% of the rest. The nine differences have no
such floor and spread over twice the range: 7% of them are zero and 29% need a second character.
Measured at the same precision, ten numbers cost 11.02 characters a word and nine cost 11.65. The
approximations that would really shrink it are not the same model: pooling the other nine into one
distribution costs 12% of the corpus its answer, 3% even after the best single correction, and a
logistic regression fitted to imitate the fit — one number a word — still differs on 1%.

**The weights, 203,090 numbers, are 226 kB.** What is stored is not $\log W$ but
$E = \log(1 + M/\texttt{SMOOTH})$, where $M$ is the appearances that component holds of that word
— because $E$ is *exactly* zero wherever the word is absent, which a quarter of the entries are,
and the rest of $\log W$ is one number per component that the page adds once per word of the text.
Each $E$ is one character from an alphabet of 92. The first code says the word is absent; the next
80 are a grid over the crowded bottom of the range; the eleven above those are an escape, and that
character with the next names a point on a grid nine times finer over the sparse top, where the
commonest words live and where a coarse step is paid once per appearance. A seventh of the present
entries take the second character.

A *uniform* grid, which is the opposite of what fitting a quantiser is usually for, and it is worth
saying why. At one character a number and no escape, a uniform grid misplaces 0.8% of the corpus
and a grid fitted to the distribution of $E$ misplaces 7 to 10% of it. What matters here is not the
average error but the largest one: a word written half a million times is consulted in every
description and its error is systematic, not noise. A fitted grid spends its levels where the
values are crowded, which is the bottom, and leaves the top coarse. The escape does that job the
other way round.

**What the precision costs, measured against the exact centres over the whole corpus:** no entry is
more than 0.034 nats out; 0.054% of descriptions land on a different component, and those are ties —
the median gap between their top two is 0.04 nats against 15.6 across the corpus; 99.1% of
descriptions have every reported probability within 0.02 of the exact one, and the worst is 0.25.
`analyze.py --selftest` reads the file back and fails the build if either half of it stops
reconstructing the fit.

**The page has its own copy of the tokeniser**, in JavaScript, and that is the one part of this
with nothing structural keeping it honest: a page that split a word differently would classify a
text the corpus never contained, and would look exactly as confident as a right answer.
`tests/test_detect.py` runs both over the same strings — links, tags, the em dash, the trimming,
advisory identifiers — and fails if they ever disagree.

## 8. The arbitrary choices

Everything above is either measured or a judgement call. These are the numbers that could have been
different, and one of them was chosen by looking at the answer.

| constant | value | how it was chosen |
|---|---|---|
| `K` | 10 | **chosen on the outcome** — see below |
| `SEED` | 0 | **consequential** — the seed moves the headline; §5 |
| `WINDOW_S` | 300 s | **consequential** — see below |
| `MIN_AUTHORS` | 50 | measured — the least restrictive floor at which two fits agree on half their top twenty; §3 |
| `MAX_PER_AUTHOR` | 3 | **arbitrary** |
| `N_INIT` | 8 | insurance — a single fit publishes 31 times in 32, so this is margin |
| `LEAD_WINDOW` | 4 weeks | judgement — "a month", to stop one week deciding the subject |
| `LEAD_START`, `LEAD_END` | 2%, 20% | round numbers, wide margins, and they only *check* |
| `WORDS_LEAD` | 1000 | **arbitrary** round number |
| `BOT_SUFFIX`, `BOT_LOGIN` | `[bot]`, `-bot`, `copilot` | judgement — what a login says is not a person |
| `ESCAPE_AT`, `SPLIT` | 80, 10 nats | measured — the pair that costs 0.054% of assignments at 304 kB; §7 |

**`K = 10` was chosen on the outcome, but from a window rather than a preference.** Below ten the
component is a mixture: at `k = 8` the leading component's own top twenty carries WebKit, nixos and
CSS vocabulary in 7 of 32 fits, and the arrival check fails in 10 of them. Above fourteen the
component splits until the pieces fall under the 20% the check asks for — 6 of 8 fits arrive at
`k = 16`, 1 of 8 at `k = 24`. Ten and twelve are both inside the window; ten keeps the title word
at the top of the list and reports a fuller share, twelve makes the arrival unanimous. Cost cannot
settle it: training cost falls with every added centre and held-out cost is still falling at
`k = 64`, so the corpus would happily support far more clusters than a reader can look at.

**`WINDOW_S = 300` is not the sample size it looks like.** Five minutes was chosen so a window
would fit in one page of a hundred results. It does not, in 2025 or 2026, so the sampler truncates
rather than enumerates and the width is doing almost nothing — it is a floor that guarantees a full
page in the thinnest era of the corpus. `WINDOWS = 10` is what sets the sample size, and it is set
against a limit: ten pages is 1.4 MB a day, which is about the most the repository can take.

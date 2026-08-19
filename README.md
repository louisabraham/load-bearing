# load-bearing

Unsupervised detection of expressions whose use on GitHub rises unusually fast,
and of the *bundles* of expressions that rise together.

The premise is that a model generation leaves not one catchphrase but a set of
habits that spread at the same time. A single phrase rising is weak evidence —
plenty of words rise for plenty of reasons. Several independent phrases rising in
the same month, across unrelated repositories, is a much stronger signal, and it
is what this project looks for.

The output is a continuous *LLM-era expression score*, never a human-versus-model
verdict. Humans who read model output pick up its phrasing too.

## Two pipelines

`shift` is the small one, and the one to read first. `lbdetect` is the earlier and
much larger pipeline — cleaning battery, n-gram families, expression clustering,
date estimation — kept for its findings; nothing in `shift` depends on it.

## Where the data comes from, and why not GH Archive

GH Archive is no longer a usable source, and the cause is upstream of it.

Measured from the files themselves — comment count in a fixed byte window, scaled by the
hour-file's true size — the feed carried 3,000–10,000 issue comments an hour from 2024-01
through 2025-10, then decayed: ~3,900/hour that winter, 1,590 in 2026-03, 1,076 in
2026-04, 866 in 2026-06, **77 in 2026-07**. A complete hour of 2024-08-12 holds 13,555
`IssueCommentEvent` against 86 for the same hour of 2026-08-10. Issues, pull requests and
reviews fell with them. Pushes survive at full volume but carry no text, GitHub having
removed the commit array from the payload in [October 2025](https://github.blog/changelog/2025-08-08-upcoming-changes-to-github-events-api-payloads/).

It is documented, though not by GitHub. GH Archive's own tracker carries
[#310 "Drastic Drop Off in Events After 2025-05-23"](https://github.com/igrigorik/gharchive.org/issues/310)
(open since July 2025, no maintainer reply) and
[#320 "WatchEvent capture rate has degraded significantly since June 2025"](https://github.com/igrigorik/gharchive.org/issues).
On GitHub's community forum,
[discussion #178788](https://github.com/orgs/community/discussions/178788) traces the
same loss to "a GitHub Event API outage propagated downstream, not an OpenDigger parsing
issue" — and the identical gaps appear in OSSInsight, which reads the Events API directly
rather than through the archive. **GitHub has published no fix and no alternative.**

The upstream break is still live. Polling `/events` in August 2026 returns 97 `PushEvent`
out of the 100 most recent events, with one `IssueCommentEvent`. So no mirror repairs it:
[BigQuery `githubarchive`](https://www.gharchive.org/),
[ClickHouse GH Explorer](https://ghe.clickhouse.tech/), the Kaggle and Hugging Face
copies and `librariesio/github-firehose` all read the same feed. GHTorrent has been dead
since 2021. [Software Heritage](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html)
archives real git history rather than events, but publishes its export **once a year**.
`open-index/open-github-issues` uses the REST and GraphQL APIs, but covers seven
repositories.

### What works: GitHub's search API, in narrow random windows

`apifetch.py` samples GitHub directly. Three properties make a clean sample possible:

* `created:` accepts **timestamps**, not just dates, so a window can be minutes wide;
* a window that narrow holds few enough items to be **enumerated** rather than sampled;
* the response carries the **full body**, so one request yields a hundred documents.

Each week gets the same number of randomly placed windows — uniform across the window,
random inside each week, and authoritative, because it is GitHub answering about its own
present state rather than a replay of a broken feed. A window wider than one page is
truncated to its earliest hundred items, which is not a bias in time: the placement is
uniformly random, so what is sampled is still "everything created in a uniformly random
interval", the interval's effective width varying as GitHub's volume does.

**The budget is requests, not bandwidth.** Search is capped at 30 requests/minute, so one
hour is 1,800 requests and at most **180,000 documents** — bandwidth buys nothing. 137
weeks × 10 windows is 1,370 requests, about 48 minutes; 13 windows/week is the ceiling
for a one-hour budget. (The non-search core limit is 5,000/hour, so a two-stage design
that samples issues by search and then pulls their comment threads could reach ~500k
documents/hour, at the cost of sampling repositories rather than documents.)

```bash
export GITHUB_TOKEN=$(gh auth token)
shift api-fetch --per-week 5 --kind issue      # ~85 min, ~67k documents
shift build --source api --cap 300             # same counting path as the archive
```

### Which surface

Measured with `shift track` on `"load-bearing"`, per 10k of each surface:

| surface | 2025 baseline | Aug 2026 | dynamic range |
|---|---|---|---|
| **PR bodies** | **0.039** | 91.0 | **x2322** |
| issue bodies | 0.114 | 98.9 | x866 |
| comments | 0.653 | 135.8 | x208 |

Comments carry the highest absolute rate and the worst signal-to-noise: their baseline is
seventeen times the pull request baseline, which is bot templates and conversational
filler. Pull request descriptions were the quietest surface before assistants arrived —
they used to be `fix typo` and `bump deps` — so the change registers largest there. The
two prose surfaces converge as it saturates: issue bodies used this vocabulary four times
more often than PR bodies in 2025-12, and 1.1x more by 2026-08.

Pull requests also sample better. There are 18.4M a month against 4.7M issues, so a
random window is a finer slice of time, and a five-minute window holds ~4,200 of them
against ~840 issues — every request fills its hundred-item page, where 2024 issue windows
returned only ~75.

The cost is severe and it is not the bots. On 4,600 sampled PR bodies with the App
exclusions already applied, **45.5% are completely empty**, another 9.1% are stubs
(`autogenerated pr`, `Body of PR`, `test`, `Fixes #23`), 3.0% are App-authored residue,
and 43.2% are usable. Nearly half of all pull requests are opened with no description.

Worse, that share is not stable. The empty rate falls from 26-33% in 2024 to 13-15% in
2026, and the usable share rises from 23% to 58% — a 2.5x drift in the sampling frame.
Issue bodies over the same window sit at 74% usable for four straight half-years and
reach 84% only at the very end.

That drift inflates the apparent advantage. A rate measured per pull request rises both
because descriptions use the word more and because more pull requests have descriptions
at all. Dividing the x2322 range by the 2.5x frame drift leaves about x930 — which is
the x866 measured on issues. **Conditioned on the item having a description, the two
surfaces show the same change.** Pull request descriptions are the more interesting
object of study, since their arrival is itself part of the phenomenon; issue bodies are
the steadier frame for measuring it.

**Excluding bots in the query.** `-author:app/NAME` works, one slug at a time;
`-author:app/*` is rejected with a 422, so there is no way to say "no apps". There is no latency penalty.

On **pull requests** it is worth doing: 35% of bodies are App-authored, and `pull`,
`dependabot`, `renovate` and `github-actions` are 90% of them. Excluding those four took
a sample page from 29 bot documents to 1, and from 45 usable to 63 — a 40% yield gain,
which matters because the budget is requests. On **issues** the same list matters much
less, since only 6.4% are App-authored there; `github-actions` alone takes that to 2.1%,
eight apps to 1.3%, and it takes forty to reach 92%.

**Empty bodies cannot be excluded at all.** `-body:""` is a 422; `has:body` and
`-in:body` are silently ignored and return the identical total. `body:*` does cut the
result set, but by 94% rather than the 22% that are actually empty — it is a text match
on the asterisk, not an emptiness test, and using it would bias the sample toward
whatever writes asterisks. Empty and near-empty bodies are dropped at count time by
`MIN_WORDS`, where they cost a page slot but not correctness.

`--exclude-apps` writes to its own directory, because excluded and unexcluded windows
enumerate different populations. Corpora are named by source string: `api` and `api:pr`
for the plain samples, `api-noapps:pr` for the excluded one. `build --drop-bots` removes
bot documents at analysis time instead, and keeps them on disk.

## `shift` — abrupt shifts in comment vocabulary

One uniform stream, whitespace words, one statistic.

**Data.** `IssueCommentEvent` bodies from GH Archive, 2024-01 onward — the only
prose-bearing event type still present in every month of the window
(`PullRequestEvent` bodies stop in 2025-11, `PushEvent` commit arrays in 2025-10).

Every week contributes the same number of hours, drawn uniformly at random from its
168 without replacement. *Uniformly* is across the window: the same sampling effort
in 2024-01 as in 2026-07, so a difference between two periods cannot come from
having looked harder at one of them. *At random* is inside the week: no fixed
hour-of-day, so the sample represents the whole week rather than a chosen slice of
the clock. The draw is a truncated permutation seeded on the week index, which makes
it reproducible and makes deepening free — a larger `hours_per_week` extends the
sample and refetches nothing. A missing hour is never substituted from a neighbour;
that would buy volume by breaking the draw, so it just leaves that week thinner,
which the per-week normalisation handles. Each hour is read up to a fixed byte cap,
taken from the opening of the hour because a gzip stream cannot be decoded from an
arbitrary offset — the same slice from every hour, so it is a constant rather than a
drift.

The week is the sampling unit, not the comparison unit. Sampling weekly is what lets
the two-week comparison windows slide one week at a time.

**Counting.** A word is a run of non-space characters, lowercased, with surrounding
punctuation removed. That is the only normalisation: no stemming, no n-grams, no
stopword list. Each word is counted at most once per document, so `X[t, v]` is the
number of documents in week `t` containing word `v`, and a single loquacious comment
cannot move a rate however often it repeats itself.

Every author counts, bots included — machine-written comments are a large and growing
part of what GitHub prose *is*, and excluding them would remove the clearest carrier
of the change being looked for. The `author` column is kept in the shards, so
filtering stays a decision made at analysis time (`build --drop-bots`) rather than one
baked into the data.

**Equal documents per week.** Sampling the same number of *hours* from every week does
not give the same number of *documents*: the archive's own comment volume swings by a
factor of two across the window, and in mid-2026 it collapses by a factor of twenty.
That matters because real text is overdispersed — words cluster inside repositories and
threads — so a z-score computed on more documents comes out *inflated* rather than
merely more precise, and a boundary score would rank the busy stretches of the archive
above the busy stretches of language. Every week is therefore thinned to a common
number of documents by a seeded random subsample (`build --cap M`, 0 to disable), which
is what makes two boundaries in different years comparable at all. Weeks that cannot
reach the target stay short, and their boundaries fail the `--min-docs` gate.

Two documents with the identical set of words count once, inside each week. This is
about text, not authorship: one ordinary human account ran a mass-close script and
posted 147 copies of one sentence in a fortnight — 16% of that fortnight's documents
— and every word of its template moved with it. Exact-set equality costs a hash
lookup, so no near-duplicate machinery is involved; and collapsing inside the week
rather than across the window means a template that runs for months contributes one
document to every week alike, which is a level and not a change.

**Detection.** The test runs at **every week boundary**, and each test pools **two
weeks on each side**: weeks *k−2, k−1* against *k, k+1*. Two weeks a side buys enough
documents for a word-level comparison; testing every week means the pair of windows
slides one week at a time, so a change is located to the week rather than to whichever
fortnight it happened to fall inside. Neighbouring tests share three quarters of their
data, so their scores are correlated — a real change reads as a short run of elevated
weeks, and the peak of the run is the estimate.

Each boundary becomes one vector of z-scores, one per word. "Did a
group of words change abruptly here?" is then a question about the length of that
vector: under no change the z are roughly standard normal, so

```
S = (Σ z² − (V−1)) / √(2(V−1))
```

is the same quantity in standard-deviation units for any vocabulary size. Real text is
overdispersed — words cluster inside repositories and threads — so instead of
modelling that, `S` is calibrated against its own median and MAD across the other
boundaries in the window. The z-vector then answers the second question for free: the
words that moved are its large components, and how many are large (`n_up`, `n_down`)
distinguishes one word from a whole register.

The z is a difference of **log-odds** with the median difference across the vocabulary
subtracted, and that detail carries most of the weight. Document frequency depends on
document length: if comments simply get longer, every common word appears in more of
them, and a raw rate comparison reports `to`, `of` and `is` all doubling — verbosity
wearing the costume of vocabulary. Longer documents shift every word's log-odds by
roughly the same additive amount, so subtracting the median removes exactly the shared
part. What was removed is reported per boundary as `common shift` rather than silently
discarded.

Because a rate can rise for reasons other than a change in how people write, the words
actually reported get a second look — how many distinct repositories carried each, and
what share of its documents came from an App account. A language change is spread
across repositories; a tool deployment is not, and shows up as bot share instead. Both
are computed on the handful of words in the report rather than across the whole matrix,
so they cost nothing.

```bash
shift fetch-data --hours-per-week 5  # resumable; re-run deeper without refetching
shift build                          # week x word document-frequency matrix
shift scan                           # boundaries ranked by S
shift movers 2025-06-02              # the words that moved at one week boundary
shift word load-bearing              # one word's trajectory, week by week
shift report                         # out/shift/report.md
```

`build --hours N` uses the first N draws of each week, which is how to get a balanced
matrix while a fetch is still running: draw N exists for every week or for none, so a
prefix is thinner but not skewed toward one end of the window.

`plot` draws every component — its weekly weight curve and the words it owns — to a PNG.
`--model lda` fits Latent Dirichlet Allocation on the same matrix, `--model lda-counts`
on the raw counts.

The comparison turned out to be about the **input, not the model**. Measured by whether a
fit separates the prose register at all, and how much mass lands on the twenty commonest
words:

| fit | stopword mass | register |
|---|---|---|
| NMF, per-word-normalised rates | 0.4% | found, 31% of mass |
| NMF, raw rates (Frobenius) | 14.4% | not found |
| NMF, counts (KL loss) | 14.5% | not found |
| LDA, counts | 15.6% | not found |
| LDA, counts, >25%-DF words pruned | — | not found |
| LDA, per-word-normalised | 0.4% | found, 14% of mass |

**tf-idf was tried and its two halves behave very differently.** Computing idf over
*weeks* is worse than useless: it rewards words present in few weeks, which are one-off
template tokens rather than arriving expressions, and stopword mass barely moves (14.4% to
13.0%). Doc-level idf helps (to 3.5%) but still does not separate the register. What works
is the *sublinear* half — `log1p(tf) x idf` finds it with 11% of mass. And since dividing
by the mean already does what idf attempts, the useful ingredient is the compression
alone: `log1p(rate/mean)` and `sqrt(rate/mean)` both find it with 12-13% of mass.

Compression turns out to be a **purity dial**, not a quality knob:

| transform | mass | 50% of peak | what the component contains |
|---|---|---|---|
| `mean` | 31% | 2026-04-13 | both phases, incl. `bugbot`, `opus`, `renovate` URLs |
| `sqrt` | 12% | 2026-06-29 | `byte-identical`, `deliberately`, `fail-closed`, `refuses` |
| `log` | 13% | 2026-06-29 | same, plus `adversarial`, `claimed`, `genuinely` |

Suppressing the dynamic range suppresses the more frequent tooling markers along with it,
leaving only the rarest prose — no tool names, no URLs. `--transform` exposes the dial.

Both models find the register on normalised input, within two weeks of each other on the date;
neither finds it on counts. Normalising each word by its own average is what buys a rare
word an equal voice, and the register lives among rare words.

Pruning frequent words instead does not work, which is worth recording because it is the
obvious thing to try. Only twenty words exceed 25% document frequency here, and removing
them promotes the next tier — the pruned fit spends 68% of its mass on `it, if, not, as,
new, you, change, have`. No threshold separates function words from content words, because
the problem is the scale of the counts at every level.

Feeding LDA normalised input does break its generative story: an entry is no longer a
count of tokens, so the multinomial over words is not a sample from any word distribution
and the Dirichlet priors lose their pseudo-count reading. That is a theoretical objection
and it costs nothing measurable here.

`word` is the check every boundary score needs. A genuine arrival is a step that
holds; a flood is one tall week. The boundary score cannot tell them apart, and the
trajectory can.

Release dates are annotations only — printed next to a detected boundary so it can
be read against what shipped nearby, not used to score anything.

## `lbdetect` — what the original pipeline does

1. Streams GitHub issue/PR prose from [GH Archive](https://data.gharchive.org) into cleaned monthly shards.
2. Counts word 1–3-grams, hyphenated compounds, typography markers and ~50 rhetorical constructions, by **document frequency**.
3. Finds each expression's sharpest level shift and scores how large, fast, significant and persistent it was.
4. Measures breadth and confounders for a shortlist, and drops what fails.
5. Clusters the survivors by *when* they moved, fitting a shared latent curve per cluster.
6. Compares cluster timing to a curated release timeline — against a placebo null.
7. Scores text, and estimates when an undated text was written.

## Install

```bash
uv venv && uv pip install -e .
```

## Use

```bash
lbdetect ingest --pilot            # ~2 GB, validates the pipeline end to end
lbdetect ingest                    # full plan, ~31 GB
lbdetect all                       # every analysis stage on what is ingested

lbdetect score --file comment.md
lbdetect score "this assumption is load-bearing" --date 2025-06
lbdetect date --file undated.md
```

Stages are separate commands (`templates`, `pass-a`, `vocab`, `pass-b`,
`emergence`, `breadth`, `cluster`, `atlas`, `align`, `validate`) so the expensive
ingest and counting passes are not repeated when only the analysis changes.
Everything is resumable: shards and per-period counts are cached on disk.

Outputs land in `out/` (`atlas.md`, `atlas.csv`, `validation.md`, cluster plots)
and `data/artifacts/` (parquet).

## Design decisions that carry the result

**The same predicate produces numerator and denominator.** Every frequency is
`documents containing e / eligible documents`, where "eligible" is one function
(`textclean.eligible`) applied identically to both. Otherwise frequencies move
with the document mix rather than with language.

**Machine-authored text is excluded, and counted.** The machine-authored share of
GitHub prose in this corpus goes from 7% (2018) to 37% (2025) to 53% (2026). Left
in, it would swamp everything: the project would measure how many review bots
comment on GitHub, not how people write. The bot flag is recomputed at analysis
time from the author login, never read from the shard, because the list of AI
review tools grows faster than any snapshot of it.

**Boilerplate is mined, not listed.** Whole-document de-duplication misses the
damaging case: a banner inserted into text a human really wrote. Lines that recur
verbatim across many repositories are mined per period and stripped, with digit
runs masked so `reviewed 67 of 69 files` and `reviewed 3 of 4 files` collapse to
one template. Lines under five tokens are exempt, so `lgtm` survives as language.

**Confounders are removed before clustering, not after.** Bot templates and
migration imports co-emerge perfectly by construction. Filtering afterwards leaves
them dominating the clusters; the largest cluster in an early run was 209 members
of market-research spam.

**Candidate selection pools into calendar windows.** A per-period document-frequency
floor makes visibility depend on that period's volume, so with uneven coverage a
term needs a far higher rate to be noticed in a thin month — biasing discovery
against exactly the era of interest.

**A decline is not an emergence.** The best changepoint is a maximum even for an
expression that only ever fell, so the score is gated on the shift being upward.
Declines stay in the table because a rise-then-fall profile is what pins down a
narrow date.

**Dating uses rate *ratios*, not raw probabilities.** Absence is genuinely
informative, but over a fixed feature set it is swamped by document length: a
two-sentence comment lacks almost every expression whenever it was written. Using
absence dated recent documents years early, with error growing the more recent the
document. See `dating.Dater._posterior`.

## Dating LLM-generated text

`lbdetect date-model` is the piece aimed at generated text specifically: given a
passage of model-written prose, put a posterior on *when it was generated*.

The likelihood has to come from the right population. `P(expression | period)`
estimated on general GitHub prose describes how developers write; dating model
output needs how *generated* text read in that month, and a phrase can be
ubiquitous in model output while staying rare in human comments. So the estimator
is fitted on the material the rest of the pipeline throws away: review bots and
coding agents (Copilot, CodeRabbit, Sourcery, Gemini Code Assist, Codex, Devin
and others) write dated, unambiguously LLM-generated prose. `llmcorpus.py` treats
that as both the training distribution and a labelled test set.

Three things keep the evaluation honest:

- **Split on repository**, so near-identical reviews of the same pull request
  cannot land on both sides.
- **Strip tool boilerplate**, or the model identifies the tool from its banner and
  recovers the date from that tool's popularity curve — which would not transfer
  to a pasted paragraph of model prose.
- **Leave-one-tool-out**, which asks the harder question the use case implies:
  does the language carry the date for a generator the model has never seen?

Measured on the corpus in this repository (38 months, 2023-06 to 2026-07):

| | MAE | baseline | skill |
|---|---|---|---|
| expression-level | 4.8 months | 6.5 | 1.35× |
| cluster-level | 4.9 months | 6.5 | 1.34× |
| held out: Copilot | 4.6 | 9.1 | 2.0× |
| held out: CodeRabbit | 6.4 | 5.3 | 0.8× |
| held out: Gemini Code Assist | 8.7 | 9.5 | 1.1× |

Coverage is ~100% (nearly every generated document contains at least one known
expression), against 55% on general prose. Read this as modest but real skill
within the fitted window, and unreliable generalisation to an unseen generator:
one held-out tool beats its baseline by 2×, another does worse than guessing.

Two failure modes were found and fixed while building this, both of which had
made the numbers look worse than the method deserved:

- A **rise-only lexicon biases dates late.** Selecting expressions purely on growth
  leaves no early-period markers, so every feature a document can contain argues
  for a later date. Error grew monotonically with age (2018 documents landed 84
  months late). The vocabulary now admits declines as well — `swing`, not `growth`.
- **No-evidence documents were silently dated to the first period.** A uniform
  posterior has no argmax, and `argmax` returned index 0, so short documents with
  no known expression read as a confident "earliest month in the corpus". They are
  now reported as no-evidence and excluded from the error, with coverage stated.

## Reading the validation output

`lbdetect validate` writes `out/validation.md`. It is built to be able to fail.

- **placebo_releases** — the important one. Null A circularly shifts the release
  calendar inside the observed window; null B scatters the changepoints. A plain
  (non-circular) shift pushes the calendar out of the data range and loses by
  construction, which made an early version report p=0.000 for a result that does
  not hold. The test also reports whether it is *underpowered*: with twenty
  releases in a sixty-month window every date is ~1.5 months from something.
- **pre_era_placebo** — runs the detector on pre-2022 data alone. Language changed
  before LLMs too, and this is the noise floor any claim must clear.
- **temporal_backtest** — asks whether a rise found before a cutoff *holds* after
  it, not whether it keeps rising: a phrase that entered the language plateaus.
  Windows either side of the cutoff are equal length, or the comparison just
  measures which years were sampled most.
- **repo_holdout**, **cluster_stability** — replication on disjoint repositories,
  and membership under perturbation.

## Limits

- GH Archive's 2026 data is degraded: issue/PR events fall ~95% while PushEvent
  volume holds. Recent months are low-power, not low-usage.
- Sensitivity is set by corpus volume. ~90 eligible documents per megabyte
  downloaded; rare expressions need the full plan, not the pilot.
- Release dates are annotations and weak priors. Releases weeks apart are grouped
  into generations and not separated.
- A high score means a text uses expressions that spread in a period. It is not
  evidence about who or what wrote it.

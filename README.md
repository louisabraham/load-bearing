# The load-bearing vocabulary of Claude

Groups of words whose frequency in GitHub pull request descriptions changed at the same time,
found without being told what to look for. One of them barely existed at the start of 2025 and
is 62% of the corpus by the middle of 2026.

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

python fetch_day.py                       # yesterday, one request
python fetch_day.py --backfill 30         # and the last 30 days, if missing
python analyze.py                         # ~12 s, eight restarts included
open index.html
```

`python analyze.py --selftest` checks the invariants whose silent failure would invalidate the
result: that the mixture sums to one, that the weekly counts reconstruct each week's document
total, that a planted component is recovered from synthetic data even though the model has no
way to represent time, and that the hand-written numba kernel agrees with a plain numpy
statement of the same computation to eight decimal places.

## How the corpus is collected

**Days are the unit of collection, weeks the unit of analysis.** CI makes one request a day —
a single randomly placed five-minute window of newly opened pull requests — and commits the
result as an immutable file under `data/days`. Nothing rewrites an earlier day, so the history
of the repository is the history of the sample. The files are left uncompressed so they can be
read and grepped in place: about 140 kB a day, and roughly 50 MB a year. A hundred descriptions
is too thin to compare against another hundred, so analysis groups seven days into a week.

Weeks run from the first present to the last with no gaps, so a week that was never collected
appears as an empty week rather than being quietly closed up and shifting everything after it.

The window is seeded on the date, so the choice is reproducible and a re-run fetches the same
window. A window holding more than one page is truncated to its earliest hundred items, which
is not a bias in time: the placement is uniformly random, so what is sampled is still
everything created in a uniformly random interval, its effective width varying with GitHub's
volume.

Two filters are pushed into the query, and together they take a page from 43 usable
descriptions to 97:

- **Four Apps excluded.** `-author:app/{pull,dependabot,renovate,github-actions}` — 90% of
  App-authored bodies. `-author:app/*` is rejected with a 422, so there is no way to say "no
  apps", and other App accounts stay in on purpose: some of the clearest agent-written prose
  on GitHub is App-authored.
- **Empty bodies excluded**, and 45% of pull requests have none. There is no emptiness
  qualifier — `-body:""` is a 422, `has:body` and `-in:body` are silently ignored, and
  `body:*` cuts 94% rather than 45% because it is a text match on the asterisk. Requiring any
  one of ten function words *in the body* does it exactly. Note the qualifier is repeated per
  term: `in:body` does **not** distribute over an OR group, so `(the OR a) in:body` matches
  titles and lets empty bodies back in.

The corpus in this repository begins **2025-01** and every day of it was collected the same
way — one window, seeded on the date, one request. The first version seeded the pre-CI period
from a bulk collection of five windows per week, written as if each week had been sampled on its
Monday; that was replaced by refetching every day individually, so there is no seam between the
history and what CI adds each morning. The whole corpus is reproducible from the dates alone:
`fetch_day.py DATE` returns the same window it returned the first time.

`ANCHOR` is 2024-12-30 because that is the Monday the first week of 2025 begins on, but that is
a labelling choice and not a reason to collect 2024, so collection starts on 2025-01-01. That
leaves the first week holding five days, and the last week holding however many mornings have
run — so both part-weeks are dropped from the analysis and the 598 collected days become 588
analysed ones across 84 whole weeks.

### Why not GH Archive

Because it stopped working. Since mid-2025 its feed carries almost only `PushEvent`: a
complete hour of 2024-08-12 holds 13,555 `IssueCommentEvent` against 86 for the same hour of
2026-08-10, and polling `/events` in August 2026 returns 97 `PushEvent` out of the 100 most
recent. Measured from the files, the archive carried three to ten thousand issue comments an
hour through 2025-10, then 1,590 in 2026-03, 866 in 2026-06 and 77 in 2026-07. Pull requests
and issues fell with them; pushes survive at full volume but carry no text, GitHub having
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

## What counts as a word

A run of letters, digits, slashes, hyphens and underscores containing at least one letter — so
`load-bearing`, `snake_case`, `--all-targets` and `src/main` survive whole, while backtick,
`:` and `>` are separators. No stemming, no n-grams, no stopword list. Every appearance counts,
so a word used three times in one description contributes three.

Order matters, and each step exists because of what the previous one broke:

1. **Links first, collapsed to their domain.** `[bugbot](https://cursor.com/x)` gives `bugbot`
   and `[cursor-url]`. Splitting on punctuation first produced `bugbot](https` and a trail of
   fragments, and those fragments ranked among components' most representative words. Keeping
   links whole was little better: a tool that puts a per-item link in every description gets
   one word per *item*, and Snyk's vulnerability links alone were the top words of eight of the
   sixteen components then in use.
2. **Then HTML tags, whole.** Splitting them character by character turned `<sup>reviewed</sup>`
   into `sup, reviewed, sup` and made `li`, `br`, `td` and `href` six of one component's twelve
   commonest words. The pattern requires a letter or slash after the bracket, so `a > b` in
   prose is not mistaken for markup.
3. **Then everything else splits** on any character a word may not contain, which handles what
   markdown creates without knowing about it: `srcset="…"` gives `srcset`, `*emphasis*` needs
   nothing because `*` is a separator.
4. **Then trim the edges.** `_other example_` needs its underscores trimmed, since an
   underscore is allowed *inside* a word; a leading hyphen stays, so `--all-targets` is not
   turned into `all-targets`.

Snyk advisory identifiers collapse to `[snyk-id]` for the same reason links collapse to their
domain: 1,401 distinct tokens between them occupied seven of the sixteen components then in
use.

Requiring a letter drops numbers and rules — `27.49`, `589/1000`, `2025-06-24`, `-------` —
along with the arrow and `+`. **The em dash is the one exception**, taken before the split and
counted as a word of its own. It earns that: 0.2 appearances per 10,000 words over the first
four weeks of the corpus against 129.4 over the last four.

## Counting

A word counts once per description per appearance, and the model sees the counts. Two
descriptions with the identical word set collapse to one, within each week — this is about text,
not authorship: one ordinary human account posted 147 copies of one sentence in a fortnight, 16%
of it. Collapsing inside the week rather than across the window means a template running for
months contributes one description to every week alike, which is a level and not a change.

**No author may contribute more than three descriptions to a week.** This finds mass-produced
text without a blocklist, because it concentrates by *author* rather than by repository:
`copilot` wrote 197 of the 198 descriptions carrying GitHub's coding-agent survey link, across
192 repositories, and `vercel[bot]` wrote all 85 carrying one particular CVE. It catches what the
`[bot]` suffix misses — `copilot`, `pyup-bot`, `scala-steward` and `regro-cf-autotick-bot` are
ordinary logins — and applies to humans on the same terms, which is why it is a cap and not an
exclusion. What it does *not* catch is a template that runs for months at three a week; that is
what the distinct-account floor is for.

**There is no cap on the size of a week, and there used to be.** Weeks were once thinned to a
common count, on the argument that text is overdispersed — words cluster inside repositories — so
a rate computed on more descriptions comes out inflated rather than merely more precise. The
argument was sound; the premise stopped holding. Weeks then came from five bulk windows and their
sizes swung by more than a factor of two, which is what made thinning worth its cost. Collection
is now one window a day and every window returns a full page of a hundred, so a week is 700
descriptions by construction and the cap was discarding half the corpus to enforce something
already true.

**Part-weeks at either end are dropped entirely.** This matters most at the trailing end and it
matters every day: collection runs each morning, so the newest week is almost always
half-collected, and it is the week everything leans on. The page ends at the last complete week
and can therefore lag by up to six days, which is the right trade at weekly resolution.

## The model

Each of `k` components is a fixed probability distribution over the vocabulary — one way of
writing. **One mixture covers the whole window**, and each description is taken to be drawn
from one of the components:

```
W_k                 a distribution over the vocabulary,  sum_v W_vk = 1
pi_k                how much of the window was written that way,  sum_k pi_k = 1
z_d ~ Cat(pi)       which component wrote description d
x_d ~ Mult(n_d, W_k)  its words
```

**There is no `t` anywhere in that.** The model has no per-week parameter, so it has nothing
that could describe a trend and no freedom to place one. Every curve on the page is attribution
instead — each description assigned by its words alone, the weeks added up afterwards:

```
C_tk = sum over descriptions d written in week t of r_dk
```

which is a sum over fitted responsibilities and not a fitted quantity. It was never optimised
toward any shape. If a component rises, the rise is in what people wrote, because there is
nowhere else for it to be.

### This used to be the ablation

An earlier version fitted a mixture *per week*, `pi_tk`, with a smoothness penalty
`lambda K² sum_t (pi_tk - pi_{t-1,k})²` on how fast it could move, and `lambda` tuned by
held-out likelihood. Running that model with one mixture for the whole window was meant as a
check on whether the smoothing had drawn the trend. The rise survived the check unchanged — so
the per-week version's extra parameters, 84 × 8 of them plus a penalty weight to tune, were
machinery that bought a readable curve and the suspicion that the model had drawn it. The check
became the model, and `lambda` went with it. There is now nothing to regularise.

### Fitting

EM: attribute descriptions, refit the word distributions, refit the mixture, twelve passes.
Eight restarts from different starting points, keeping the highest likelihood.

**The restarts are not optional, and that was measured rather than assumed.** Fitting once, at
seed 0 and `k = 16`, put `[transifex-url]` and `transifex` at ranks one and two of the
register's most characteristic words — a translation service's boilerplate welded onto the
prose, which is what a mixed local optimum looks like from outside. Across 16 seeds at the
settings actually used:

```
loglik              best -36,889,990   worst -37,032,477   spread 0.39%
end share           min 36.0%   median 55.0%   max 63.4%
load-bearing rank   1 nine times, 2 twice, 3 twice, then 40, 243, 565
```

The likelihood barely separates the runs while the answer moves by nearly a factor of two, so
the winning run is worth finding — and it is easy to find, appearing at the second restart with
30 further restarts never beating it. Do not read the published share as a bound: at `k = 16`
the likeliest fit happened to be the one that split the register most finely and so reported the
*smallest* share of any seed, but at `k = 8` it reports 62.1% against a median of 55.0%, near
the top. It is the likeliest fit's figure and nothing more.

### numba

The EM sweep is one `njit(parallel=True)` kernel that fuses the E step with the M step's
sufficient statistics. The obvious formulation builds a `D × k` matrix of logits, softmaxes it,
and multiplies it back against the sparse matrix — three passes and two dense intermediates the
size of the corpus. The kernel visits each description once: logits into a length-`k` scratch
array, softmaxed in place, spent immediately on the word totals, the weekly counts and the
likelihood. Nothing `D`-sized is allocated.

It parallelises over contiguous blocks of descriptions rather than over descriptions, so each
thread owns one slice of every accumulator and no two threads touch the same cell; the caller
sums the leading axis. That costs `threads × k × V` floats — six megabytes at sixteen threads,
a thousandth of what the dense logits would have cost — and buys freedom from atomics.

A full run went from **313 seconds to 12**, and the fit itself is now under a second, which is
what made the restart sweep and every other measurement in this file affordable enough to
actually run. `_em_sweep_numpy` states the same sweep in six lines of numpy and the selftest
holds the kernel to it: a hand-written parallel reduction is exactly the kind of code that is
wrong in ways tests written against its own output cannot see.

### Two rankings, and why lift excludes the component itself

A component's most representative words are those most more probable inside it than in
**everything that is not it**:

```
lift_k(v) = W_vk / [ sum_{j != k} m_j W_vj / sum_{j != k} m_j ]
```

The exclusion is doing enormous work here, more than it was when this was written. Dividing by
the whole corpus understates a large component's own words, because the component is now 62% of
the recent weeks — its vocabulary would be compared mostly against itself. `load-bearing` scores
**3.70× against the whole corpus and 3,535× against everything that is not this component.**
A thousandfold difference, from one choice of denominator.

There is deliberately no probability floor: flooring throws away exactly the
rare-but-concentrated words the ratio is for. Instead three frequency floors — see below.

### Three floors, and the third one is about people

A word needs 45 total appearances, 25 distinct descriptions, **and 20 distinct accounts.**

The first two count documents, and a bot's template clears them easily. `proprosed` — a
misspelling of "proposed" inside one Red Hat Konflux template — reaches 190 descriptions, and
`pipelineruns` 252, because the per-week author cap bounds an account to three descriptions a
week and a template running for sixteen months is under that cap every single week. Counting
accounts separates them at a glance:

| word | descriptions | distinct accounts |
|---|---|---|
| `proprosed` | 190 | **16** (three bots supply 144) |
| `pipelineruns` | 252 | **18** (the same three) |
| `load-bearing` | 92 | **91** |
| `seam` | 136 | **132** |

A word 91 people reached for is a word; a word in 190 descriptions from 16 accounts is one
document written 190 times. The floor costs 109 of 7,168 words and removes both, along with the
typo that had been sitting at rank 2 of the headline list.

Appearances alone are not breadth either — `multi-draw` appears 101 times inside a single
description — because a ratio cannot tell a widespread word from one someone repeated.

### Rare words are not pruned, and that was tested

Raising the frequency floors would shrink the vocabulary, so it was worth checking whether they
could go. They cannot. Sweeping them upward on one shared count matrix, so the comparison is
controlled:

```
tf/df       words   load-bearing's component, top words
45/25       7,267   survived, load-bearing, quietly, refusal, pre-fix, halves
100/50      4,670   17 of the previous top 40 survive
250/100     2,655   7 of 40
500/200     1,605   0 of 40 -- clippy, cargo, --check, uv, bun: a different subject entirely
1000/400      930   1 of 40 -- generic function words
```

The weekly *shape* survives all of it (`r` = 0.87 to 0.99). The component's *identity* does not:
by 500/200 the largest component is Rust tooling and the page would be about something else. And
the only motive for pruning was speed, of which there is none to gain — the fit is 0.1 to 0.3
seconds at every vocabulary size, because numba made the corpus load and the compile dominate.
All cost, no benefit.

### Choosing the component, and checking it

**The component the page is about is simply the largest one of the last four weeks.** Nothing is
selected on how much it grew. A month rather than a week because a week is 700 descriptions and
the subject of the whole page should not turn on which of two close components led across one of
them.

Growth thresholds used to do the selecting, and that was fragile. At a 1% start one fit rejected
its own largest component, 1.06% → 40.35%, for beginning six hundredths of a point too high.

`LEAD_START` and `LEAD_END` survive but no longer select — they **check**. Picking the biggest
component says nothing about whether it arrived, and the page's headline claim is that it
arrived, so that claim is tested against the component actually chosen: under 2% of the first
eight weeks, at or above 20% of the last eight. A test may be a round number in a way a selector
may not, because nothing is being ranked and there is no runner-up to exclude unfairly. If it
fires, the page should not be published from that fit.

**An earlier version asserted a growth ratio and a clean gap, and it was wrong.** It required
every component growing 100-fold to be ten times clear of everything that did not, and CI caught
it failing on the very first run — one extra day of data moved the largest non-arrival from 4.8×
to 69×, an eight-fold swing from a hundred descriptions. Two things were wrong. A ratio of
`end/start` explodes when the start is near zero, so it ranks a component starting at 0.07% above
one starting at 0.3% for no good reason and is unstable at exactly the point it matters. And the
gap requirement encoded an assumption the data does not support: growth is a continuum here, so a
threshold cuts through the middle of it and no gap can exist.

The test reads observed weekly shares rather than the fitted mixture — which is now the only
thing it could read, since the mixture has no weekly component at all.

## How many components: eight, and why that is not a neutral choice

`k = 8`, and **the number was chosen so that `load-bearing` — the word this page is named after —
would rank among the five most characteristic words of the arriving component.** It does, in 13 of
16 starting seeds, at rank 1 in nine of them. At `k = 16` it ranks 45th; at 24, third; at 32,
first; at 48, nowhere near.

**That is selection on the outcome, and it cannot also be evidence for the outcome.** Held-out
likelihood prefers far more components than eight — it kept improving to about `k = 128` before
turning over at 192. A coarser model lumps together registers a finer one separates, which is
exactly why one word can come to dominate it. What eight buys is a page whose title matches its
own top line. What it costs is that no ranking here may be read as having been discovered: the
vocabulary is real and the rise is real, but the *order* was tuned until a chosen word came
first.

This is the second time this project has picked `k` by looking at the answer. The first time was
an accident and is retracted below. This time it was asked for deliberately, so it is disclosed
instead.

**Retracted: "marker recovery".** An earlier version chose `k` by counting how many of 22 marker
words each setting reproduced — but those 22 were chosen by reading the output at `k = 12`. It
was a measure of agreement with itself, dressed as validation. Held-out likelihood replaced it.

The finding does not depend on the choice of `k`. A component going from near nothing to a large
share of the week, with these words, is there at every `k` from 6 to 48. Only the ranking of
individual words within it moves.

## On the title

Words naming Claude are elevated inside this component — `claude` at 5.08× and its link at 5.71×
— while Cursor sits at 1.31×, ChatGPT at 1.08×, Codex at 1.06× and Copilot at 0.61×, at or below
the baseline. But `gpt-5` is elevated further than any of them, at 8.56×. The register is far
more strongly associated with Claude than with most assistants, and it is not Claude's alone.

## Three caveats to carry

The corpus only contains pull requests whose author wrote a description, and that condition
loosens over the window — empty descriptions fall from about a third to about a seventh. A rate
measured per pull request therefore rises both because descriptions use a word more and because
more pull requests have descriptions at all.

The components come from the model; the word curves do not. Each is the raw weekly count of that
word's appearances, so only the *choice* of which words to show is the model's.

Every window returns a full page of 100 results, in 2025 and 2026 alike, which means the
five-minute window is *effectively narrower* in busier periods. The sample is uniform in time
but not in the fraction of each period it captures.

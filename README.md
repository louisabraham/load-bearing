# The load-bearing vocabulary of Claude

Groups of words whose frequency in GitHub pull request descriptions changed at the same time,
found without being told what to look for. One of them barely existed at the start of 2025 and
is about a third of the corpus by the middle of 2026.

**[louisabraham.github.io/load-bearing](https://louisabraham.github.io/load-bearing/)**

| file | what it is |
|---|---|
| `fetch_day.py` | one request a day to GitHub's search API, one `data/days/YYYY-MM-DD.jsonl`. Standard library only. |
| `analyze.py` | reads the days, groups them into weeks, fits the model, writes `analysis.js`. Needs `numpy`, `scipy`, `scikit-learn`. |
| `index.html` | reads `analysis.js`. No build step. Open it. `?v=nol2` and `?v=flat` load the ablations on the same page. |
| `.github/workflows/daily.yml` | does all of the above, daily, and commits the result. |

```bash
pip install numpy scipy scikit-learn
export GITHUB_TOKEN=$(gh auth token)

python fetch_day.py                       # yesterday, one request
python fetch_day.py --backfill 30         # and the last 30 days, if missing
python analyze.py                         # ~15 s
python analyze.py --lam 0 --out analysis-nol2.js
python analyze.py --flat --out analysis-flat.js
open index.html
```

`python analyze.py --selftest` checks the invariants whose silent failure would invalidate the
result: that the prevalences sum to one, that the weekly counts reconstruct each week's
document total, that a planted component is recovered, and that the smoothness penalty does
something.

## How the corpus is collected

**Days are the unit of collection, weeks the unit of analysis.** CI makes one request a day —
a single randomly placed five-minute window of newly opened pull requests — and commits the
result as an immutable file under `data/days`. Nothing rewrites an earlier day, so the history
of the repository is the history of the sample. The files are left uncompressed so they can be
read and grepped in place: about 210 kB a day, and roughly 77 MB a year. A hundred descriptions is too thin to compare
against another hundred, so analysis groups seven days into a week.

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
   one word per *item*, and Snyk's vulnerability links alone were the top words of eight of
   sixteen components.
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
domain: 1,401 distinct tokens between them occupied seven of sixteen components.

Requiring a letter drops numbers and rules — `27.49`, `589/1000`, `2025-06-24`, `-------` —
along with the arrow and `+`. **The em dash is the one exception**, taken before the split and
counted as a word of its own. It earns that: 0.0 appearances per 10,000 words in early 2024
against 123.0 in mid-2026.

## Counting

**No author may contribute more than three descriptions to a week.** This finds mass-produced
text without a blocklist, because it concentrates by *author* rather than by repository:
`copilot` wrote 197 of the 198 descriptions carrying GitHub's coding-agent survey link, across
192 repositories, and `vercel[bot]` wrote all 85 carrying one particular CVE. It catches what
the `[bot]` suffix misses — `copilot`, `pyup-bot`, `scala-steward` and `regro-cf-autotick-bot`
are ordinary logins — and applies to humans on the same terms, which is why it is a cap and not
an exclusion.

Two descriptions with the identical word set count once, within each week. This is about text,
not authorship: one ordinary human account posted 147 copies of one sentence in a fortnight,
16% of it. Collapsing inside the week rather than across the window means a template running
for months contributes one description to every week alike, which is a level and not a change.

Every week is then cut off at the same number of descriptions, because text is overdispersed —
words cluster inside repositories — so a rate computed on more descriptions comes out inflated
rather than merely more precise.

## The model

Each of `k` components is a fixed probability distribution over the vocabulary — one way of
writing. Every week has a mixture over those `k`, and each description is taken to be drawn
from one of them:

```
W_k                   a distribution over the vocabulary,  sum_v W_vk = 1
pi_tk                 how much of week t was written that way,  sum_k pi_tk = 1
z_d ~ Cat(pi_t)       which component wrote description d
x_d ~ Mult(n_d, W_k)  its words
```

Fitted by EM, restarted ten times and keeping the highest likelihood, because EM finds
different local optima here and the worst of them mix a component with something else.

### Smoothing

The only thing asked of a prevalence curve is smoothness, `lambda * K² * sum_t (pi_tk -
pi_{t-1,k})²`. Nothing requires a component to rise, to fall, or to be absent early.

The `K²` makes one `lambda` correct at any `k`, and it is not a fudge: prevalences sum to one,
so a typical `pi` is about `1/K` and a typical squared difference about `1/K²`. Held-out
likelihood — fit on 90% of descriptions, score the other 10% — put the optimum at 5,000 for
`k = 12` and 500,000 for `k = 128`, and `(128/12)² × 5,000 = 568,889`, one grid step away. So
the whole `k`-dependence is that factor, and absorbing it leaves a single constant right at
both: `5,000/144 = 34.7` and `500,000/16,384 = 30.5`. Set to 32, it reproduces both per-`k`
optima exactly.

### Two rankings, and why lift excludes the component itself

A component's most representative words are those most more probable inside it than in
**everything that is not it**:

```
lift_k(v) = W_vk / [ sum_{j != k} m_j W_vj / sum_{j != k} m_j ]
```

The exclusion matters. Dividing by the whole corpus understates a large component's own words,
because by the end of the window one component is a third of everything written — its
vocabulary was being compared partly against itself. `load-bearing` scores 6.95× that way and
273× this way.

There is deliberately no probability floor: flooring throws away exactly the
rare-but-concentrated words the ratio is for. Instead two frequency floors, a word needing 45
total appearances *and* 25 distinct documents. Appearances alone are not breadth —
`multi-draw` appears 101 times inside a single description — because a ratio cannot tell a
widespread word from one someone repeated.

### The arrival assertion

`analyze.py` asserts that at least one component started under 1% of the first eight weeks and
ended at or above 20% of the last eight. If that fails, the claim this page is built on has
stopped being true and the page should not be published from that fit.

**An earlier version asserted a growth ratio and a clean gap, and it was wrong.** It required
every component growing 100-fold to be ten times clear of everything that did not, and CI
caught it failing on the very first run — one extra day of data moved the largest non-arrival
from 4.8× to 69×, an eight-fold swing from a hundred descriptions. Two things were wrong. A
ratio of `end/start` explodes when the start is near zero, so it ranks a component starting at
0.07% above one starting at 0.3% for no good reason and is unstable at exactly the point it
matters. And the gap requirement encoded an assumption the data does not support: growth is a
continuum here, with one component at 569× and another at 69×, so a threshold at 100 cuts
through the middle of it and no gap can exist. Two absolute shares are the honest test, and
they name the claim directly.

The test reads observed weekly shares rather than the fitted `pi`. It is what the data says, it
coincides with `pi` wherever `pi` was free to follow it, and it is the only thing that means
anything in the ablation below, where `pi` is held constant.

## How many components

`k = 16` is a **legibility choice, not a statistical one.** Training likelihood rises with `k`
because more parameters always fit better, and held-out likelihood keeps improving to about
`k = 128` before turning over at 192 — an order of magnitude more than the default. Sixteen is
what a reader can hold. What the smaller `k` buys is a summary; what it costs is resolution,
and at `k = 32` the register visibly splits into prose and command-line tooling.

The finding does not depend on the choice: a component going from near nothing to roughly a
third of the week, with these words, is there at every `k` from 6 to 32.

## The ablations

Two are kept, and `index.html?v=nol2` and `?v=flat` show them on the same page — the page reads
its own `variant` field for the banner, so nothing has to be kept in step by hand. Separate HTML
files were tried and fell behind: one still said "137 weeks" after the corpus was cut to 86.

**`--lam 0`, no smoothing.** The finding survives untouched: two components still start under
0.1% and end at 68% of the last week between them. What `lambda` buys is a readable curve —
roughness 1.82 against 0.06 — and not the result. Worth having as the page you can point at when
someone asks whether the smoothing made the shape.

**`--flat`, one mixture for the whole window.** The model fits a single mixture over components
for all weeks at once, so it has no way to represent time at all and the weekly curves it
produces are purely observed. A component still goes from under 1% to 27% of the last week. It
fails the arrival test and the page says so, but the rise is plain with the model given no
freedom to fit it. **That is the strongest available evidence that the pattern is in the words
and not in the fitting.**

Three others were run and removed, and their results are worth recording. Smoothing 150× too
strong still finds the component but drags its peak down, which is what over-smoothing looks
like. Hard assignment — each description to its single best component — costs 13,000 nats out of
31.6 million, as the responsibilities predict, since 84% of descriptions already concentrate
above 0.9 on one. And LDA, letting each *word* pick its own component rather than each
description, costs four million nats and its leading component blends a vendor's footer with the
prose the document-level model keeps apart. They are in the git history.

## On the title

Words naming Claude are elevated inside this component, `claude` at 4.35× and its link at
4.48×, while Cursor, ChatGPT, Codex and Copilot sit at or below the baseline. But `gpt-5` is
elevated further, at 5.20×. The register is far more strongly associated with Claude than with
most assistants, and it is not Claude's alone.

## Two caveats to carry

The corpus only contains pull requests whose author wrote a description, and that condition
loosens over the window — empty descriptions fall from about a third to about a seventh. A rate
measured per pull request therefore rises both because descriptions use a word more and because
more pull requests have descriptions at all.

The components come from the model; the word curves do not. Each is the raw weekly count of
that word's appearances, so only the *choice* of which words to show is the model's.

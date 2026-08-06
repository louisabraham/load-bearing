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

## What it does

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

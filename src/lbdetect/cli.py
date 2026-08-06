"""Command line interface.

Stages are separate commands because the expensive ones (ingest, pass A/B) should
not be re-run when only the analysis changes. `lbdetect all` chains them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from . import breadth as breadth_mod
from . import clustering, config as C, dating, emergence, ingest, releasealign
from . import report as report_mod
from . import llmcorpus, provenance, scoring, series, templates, validate
from .util import read_json, write_json

app = typer.Typer(add_completion=False, help=__doc__)

CLUSTERS = C.ARTIFACTS / "clusters.pkl"
EMERGENCE = C.ARTIFACTS / "emergence.parquet"
BREADTH = C.ARTIFACTS / "breadth.parquet"
LEXICON = C.ARTIFACTS / "lexicon.parquet"


def _save_clusters(clusters: list[clustering.Cluster]) -> None:
    import pickle

    CLUSTERS.write_bytes(pickle.dumps(clusters))


def _load_clusters() -> list[clustering.Cluster]:
    import pickle

    if not CLUSTERS.exists():
        return []
    return pickle.loads(CLUSTERS.read_bytes())


# ------------------------------------------------------------------------ ingest

@app.command("ingest")
def cmd_ingest(
    workers: int = 12,
    pilot: bool = typer.Option(False, help="small byte-capped run"),
    start: str = f"{C.START[0]}-{C.START[1]:02d}",
    end: str = f"{C.END[0]}-{C.END[1]:02d}",
    max_mb: int = typer.Option(-1, help="cap MB per hour-file; -1 keeps the plan default"),
    commit_sample: int = 0,
):
    """Download and clean GitHub prose into monthly document shards."""
    plan = C.PILOT_PLAN if pilot else C.DEFAULT_PLAN
    if max_mb >= 0:
        plan = C.SamplingPlan(plan.baseline_hours, plan.era_hours,
                              plan.sparse_hours, max_mb << 20)
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    res = ingest.run((y0, m0), (y1, m1), plan, workers=workers,
                     commit_sample=commit_sample)
    typer.echo(json.dumps(res, indent=2))


@app.command("coverage")
def cmd_coverage():
    """Documents, tokens, repositories and authors per period."""
    cov = ingest.coverage()
    typer.echo(cov.to_pandas().to_string(index=False) if _has_pandas()
               else str(cov))
    report_mod.coverage_plot(cov.rename({"docs": "docs"}), C.OUT / "coverage.png")
    typer.echo(f"\nwrote {C.OUT / 'coverage.png'}")


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------ series

@app.command("templates")
def cmd_templates(workers: int = 12, min_tokens: int = 5, min_repos: int = 5,
                  min_frac: float = 3e-4, show: int = 15):
    """Mine boilerplate lines that recur across repositories, so they can be
    stripped before counting."""
    t = templates.mine(workers=workers, min_tokens=min_tokens, min_repos=min_repos,
                       min_frac=min_frac)
    for ln in sorted(t, key=len, reverse=True)[: max(0, show)]:
        typer.echo(f"  {ln[:140]}")


@app.command("pass-a")
def cmd_pass_a(min_df: int = series.MIN_DF_PERIOD, workers: int = 12,
               char_ngrams: bool = False):
    """Count every feature per period and keep what clears the floor."""
    meta = series.pass_a(min_df=min_df, use_char=char_ngrams, workers=workers)
    typer.echo(f"periods={meta.height} docs={meta['docs'].sum():,}")


@app.command("vocab")
def cmd_vocab(max_terms: int = 80_000, min_peak_df: int = 10,
              min_peak_rate: float = 3e-4, min_growth: float = 2.0,
              window_months: int = 12):
    """Choose the candidate expressions worth an exact recount."""
    v = series.build_vocab(max_terms, min_peak_df, min_peak_rate, min_growth,
                           window_months)
    typer.echo(f"vocab={v.height:,}")


@app.command("pass-b")
def cmd_pass_b(workers: int = 12, char_ngrams: bool = False):
    """Recount candidates exactly in every period."""
    typer.echo(json.dumps(series.pass_b(use_char=char_ngrams, workers=workers)))


# --------------------------------------------------------------------- discovery

@app.command("emergence")
def cmd_emergence(min_docs: int = 1500, standardize: bool = True):
    """Locate and score each expression's level shift."""
    s = series.Series.load()
    if standardize:
        s = s.standardize()
    em = emergence.analyze(s, min_docs=min_docs)
    em.write_parquet(EMERGENCE)
    typer.echo(em.head(25).select(
        ["term", "cp_period", "pre_rate", "post_rate", "log_growth", "binom_z",
         "core_score"]).__str__())


@app.command("cluster")
def cmd_cluster(top_n: int = 1200, threshold: float = 0.55, min_size: int = 3,
                min_families: int = 3, min_docs: int = 1500,
                min_penalty: float = typer.Option(
                    0.5, help="drop expressions whose confounder penalty is below this "
                              "before clustering (requires `breadth` to have run)")):
    """Group expressions whose changes happen together.

    Confounders are removed *before* clustering, not after. Bot templates and
    migration imports co-emerge perfectly by construction, so leaving them in
    produces clusters that are real but uninteresting, and they crowd out the
    linguistic bundles the study is looking for.
    """
    s = series.Series.load().standardize()
    em = pl.read_parquet(EMERGENCE)
    if BREADTH.exists():
        br = pl.read_parquet(BREADTH)
        flagged = breadth_mod.flag_confounders(em, br)
        before = flagged.height
        em = (flagged.filter(pl.col("confounder_penalty") >= min_penalty)
              .sort("adj_score", descending=True))
        typer.echo(f"confounder filter: {em.height}/{before} expressions survive "
                   f"penalty >= {min_penalty}")
    top = em.head(top_n)
    terms = [t for t in top["term"].to_list() if t in s.index]
    sub = s.subset(np.array([s.index[t] for t in terms]))
    g, gp = emergence.growth_matrix(sub, min_docs=min_docs)
    fam = dict(zip(top["term"], top["family"]))
    cps = dict(zip(top["term"], top["cp_period"]))
    clusters = clustering.build(g, terms, [fam[t] for t in terms],
                                [cps[t] for t in terms], gp,
                                threshold=threshold, min_size=min_size,
                                min_families=min_families)
    _save_clusters(clusters)
    clustering.to_frame(clusters).write_parquet(C.ARTIFACTS / "cluster_members.parquet")
    clustering.latent_frame(clusters).write_parquet(C.ARTIFACTS / "cluster_latent.parquet")
    for c in clusters[:12]:
        typer.echo(json.dumps(c.summary()))


@app.command("breadth")
def cmd_breadth(top_n: int = 600, workers: int = 10):
    """Repositories, authors, contexts and confounders for the shortlist."""
    em = pl.read_parquet(EMERGENCE).head(top_n)
    terms = em["term"].to_list()
    cps = dict(zip(em["term"], em["cp_period"]))
    br = breadth_mod.compute(terms, cps, series.available_periods(), workers=workers)
    br.write_parquet(BREADTH)
    typer.echo(br.head(20).select(
        ["term", "df_post", "repo_spread", "top_repo_share", "ai_repo_share"]).__str__())


@app.command("atlas")
def cmd_atlas(top_plots: int = 8):
    """Assemble the expression atlas and the cluster report."""
    em = pl.read_parquet(EMERGENCE)
    br = pl.read_parquet(BREADTH) if BREADTH.exists() else None
    if br is not None:
        em = breadth_mod.flag_confounders(em, br)
    members = (pl.read_parquet(C.ARTIFACTS / "cluster_members.parquet")
               if (C.ARTIFACTS / "cluster_members.parquet").exists()
               else pl.DataFrame(schema={"term": pl.Utf8}))
    atlas = report_mod.build_atlas(em, members, None)
    atlas.write_parquet(report_mod.ATLAS)

    clusters = _load_clusters()
    align = releasealign.align_clusters(clusters)
    align.write_parquet(C.ARTIFACTS / "release_alignment.parquet")

    s = series.Series.load()
    md = ["# Expression atlas\n", report_mod.atlas_markdown(atlas), "\n\n# Clusters\n",
          report_mod.cluster_markdown(clusters, align)]
    if align.height:
        md += ["\n\n# Clusters by release generation\n",
               "```\n" + str(releasealign.generation_table(align)) + "\n```"]
    (C.OUT / "atlas.md").write_text("\n".join(md))
    atlas.drop("contexts").write_csv(C.OUT / "atlas.csv") if "contexts" in atlas.columns \
        else atlas.write_csv(C.OUT / "atlas.csv")

    for c in clusters[:top_plots]:
        report_mod.plot_cluster(c, C.OUT / f"cluster_{c.cid}.png")
    top_terms = atlas.head(10)["term"].to_list()
    report_mod.plot_expression(s, top_terms, C.OUT / "top_expressions.png",
                               "highest-weighted expressions")
    lex = scoring.Lexicon.from_atlas(atlas, s)
    lex.save(LEXICON)
    typer.echo(f"atlas rows={atlas.height} clusters={len(clusters)} -> {C.OUT}")


@app.command("align")
def cmd_align(draws: int = 2000):
    """Compare cluster timing to the release calendar, with a placebo null."""
    clusters = _load_clusters()
    align = releasealign.align_clusters(clusters)
    typer.echo(str(align))
    res = releasealign.placebo_test(clusters, n_draws=draws)
    write_json(C.ARTIFACTS / "placebo.json", res)
    typer.echo(json.dumps(res, indent=2))


# -------------------------------------------------------------------- validation

@app.command("validate")
def cmd_validate(cutoff: str = "2024-06", top_n: int = 250, min_docs: int = 1500,
                 workers: int = 8, skip_slow: bool = False):
    """Run the validation suite and write a report."""
    s = series.Series.load()
    em = pl.read_parquet(EMERGENCE)
    out: dict = {}
    out["temporal_backtest"] = validate.temporal_backtest(s, cutoff, top_n, min_docs)
    out["pre_era_placebo"] = validate.pre_era_placebo(s, min_docs=min_docs)
    out["cluster_stability"] = validate.cluster_stability(s, em, top_n=min(top_n, 300),
                                                          min_docs=min_docs)
    out["placebo_releases"] = releasealign.placebo_test(_load_clusters())
    if not skip_slow:
        terms = em.head(min(top_n, 200))["term"].to_list()
        cps = dict(zip(em["term"], em["cp_period"]))
        out["repo_holdout"] = validate.repo_holdout(
            terms, series.available_periods(), cps, workers=workers)
        # the dater must be built from clustered expressions, or the cluster-level
        # estimator degenerates into the expression-level one with every term its
        # own singleton cluster and the two results become trivially identical
        cl = {}
        clustered: list[str] = []
        if (C.ARTIFACTS / "cluster_members.parquet").exists():
            cm = pl.read_parquet(C.ARTIFACTS / "cluster_members.parquet")
            cl = {t: int(c) for t, c in zip(cm["term"], cm["cid"])}
            clustered = [t for t in cm["term"].to_list()]
        # selected for temporal information, not emergence rank
        # candidates must pass the same confounder filter as everything else:
        # selecting on temporal information alone happily picks spam and template
        # terms, which are the most time-localised features in the corpus
        cand = em
        if BREADTH.exists():
            cand = (breadth_mod.flag_confounders(em, pl.read_parquet(BREADTH))
                    .filter(pl.col("confounder_penalty") >= 0.45))
        lex_terms = dating.informative_terms(
            s, k=2000, min_docs=min_docs,
            candidates=list(dict.fromkeys(clustered + cand["term"].to_list())))
        dater = dating.Dater(s, lex_terms, min_docs=min_docs, cluster_of=cl)
        # sample across the whole timeline: evaluating on the final period alone
        # measures nothing, and that period is also the thinnest
        avail = [p for p in series.available_periods() if p in dater.periods]
        picks = avail[:: max(1, len(avail) // 12)][:12]
        frames = [d for p in picks if (d := series.load_period(p)).height]
        if frames:
            docs = pl.concat(frames, how="vertical_relaxed")
            out["date_prediction"] = dating.evaluate(dater, docs, sample=600)
            out["date_prediction_clusters"] = dating.evaluate(
                dater, docs, use_clusters=True, sample=600)
    write_json(C.ARTIFACTS / "validation.json", out)
    (C.OUT / "validation.md").write_text(
        "# Validation\n\n```json\n" + json.dumps(out, indent=2, default=str) + "\n```\n")
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("provenance")
def cmd_provenance(start: str = "2024-01", end: str = "2026-07", hours: int = 1,
                   max_mb: int = 6, workers: int = 6):
    """Measure declared AI assistance (Co-Authored-By trailers) over time.

    Scans the archive directly and tallies; stores no documents. Reports a floor,
    not an estimate: only assistance that the author's tool declares is visible.
    """
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    slots = []
    for (y, m) in C.months((y0, m0), (y1, m1)):
        slots.extend(C.sample_hours(y, m, hours))
    typer.echo(f"scanning {len(slots)} hours, {max_mb}MB each")
    df = provenance.scan(slots, max_bytes=max_mb << 20, workers=workers)
    df.write_parquet(C.ARTIFACTS / "provenance.parquet")
    typer.echo(str(df))
    typer.echo(f"\nwrote {C.ARTIFACTS / 'provenance.parquet'}")


@app.command("date-model")
def cmd_date_model(k: int = 1500, min_docs: int = 600, holdout: float = 0.35,
                   start: str = "2023-06", leave_tool_out: bool = True,
                   min_llm_docs: int = 12):
    """Fit and evaluate the generation-date estimator for LLM-written text.

    Likelihoods are estimated on LLM-authored prose rather than general GitHub
    prose, because that is the population being dated.
    """
    s = series.Series.load()
    em = pl.read_parquet(EMERGENCE)
    cand = em
    if BREADTH.exists():
        cand = (breadth_mod.flag_confounders(em, pl.read_parquet(BREADTH))
                .filter(pl.col("confounder_penalty") >= 0.45))
    terms = list(dict.fromkeys(cand["term"].to_list() + em.head(3000)["term"].to_list()))
    terms = [t for t in terms if t in s.index][:20000]

    periods = [p for p in series.available_periods() if p >= start]
    docs = llmcorpus.corpus(periods)
    if docs.height == 0:
        typer.echo("no LLM-authored documents in range")
        raise typer.Exit(1)
    typer.echo(f"LLM-authored corpus: {docs.height:,} documents, "
               f"{docs['tool'].n_unique()} tools, {len(periods)} periods")

    # rank the candidate expressions by temporal information *within* this corpus
    full = llmcorpus.build_series(docs, terms, periods)
    sel = dating.informative_terms(full, k=k, min_docs=min_llm_docs, min_total=15)
    typer.echo(f"selected {len(sel)} dating expressions")

    train, test = llmcorpus.split_by_repo(docs, holdout=holdout)
    cl = {}
    if (C.ARTIFACTS / "cluster_members.parquet").exists():
        cm = pl.read_parquet(C.ARTIFACTS / "cluster_members.parquet")
        cl = {t: int(c) for t, c in zip(cm["term"], cm["cid"])}
    out = {
        "expression_level": llmcorpus.evaluate(train, test, sel, periods,
                                              min_docs=min_llm_docs),
        "cluster_level": llmcorpus.evaluate(train, test, sel, periods,
                                            use_clusters=True, cluster_of=cl,
                                            min_docs=min_llm_docs),
    }
    if leave_tool_out:
        out["leave_one_tool_out"] = llmcorpus.leave_one_tool_out(
            docs, sel, periods, min_docs=min_llm_docs)
    write_json(C.ARTIFACTS / "llm_date_model.json", out)
    pl.DataFrame({"term": sel}).write_parquet(C.ARTIFACTS / "llm_date_terms.parquet")
    typer.echo(json.dumps(out, indent=2, default=str))


# ------------------------------------------------------------------------ scoring

@app.command("score")
def cmd_score(text: str = typer.Argument(None), file: Path = None,
              date: str = typer.Option(None, help="YYYY-MM for the date-aware score"),
              reference: str = "pre_llm"):
    """Score a text for LLM-era expressions."""
    if file:
        text = file.read_text()
    if not text:
        text = typer.get_text_stream("stdin").read()
    atlas = pl.read_parquet(report_mod.ATLAS)
    s = series.Series.load()
    lex = scoring.Lexicon.from_atlas(atlas, s)
    res = scoring.score_text(text, lex)
    typer.echo(res.explain())
    if date:
        typer.echo("\ndate-aware (log-likelihood ratio vs " + reference + "):")
        typer.echo(scoring.score_text_dated(text, date, lex, reference).explain())


@app.command("date")
def cmd_date(text: str = typer.Argument(None), file: Path = None,
             clusters: bool = typer.Option(True, help="use cluster-level likelihood"),
             top_n: int = 400, min_docs: int = 1500):
    """Estimate when an undated text was written."""
    if file:
        text = file.read_text()
    if not text:
        text = typer.get_text_stream("stdin").read()
    s = series.Series.load()
    atlas = pl.read_parquet(report_mod.ATLAS)
    terms = atlas.head(top_n)["term"].to_list()
    cl = {}
    if "cid" in atlas.columns:
        cl = {t: int(c) for t, c in zip(atlas["term"], atlas["cid"]) if c is not None}
    d = dating.Dater(s, terms, min_docs=min_docs, cluster_of=cl)
    est = d.estimate_by_cluster(text) if clusters else d.estimate(text)
    typer.echo(est.explain())


# ---------------------------------------------------------------------- pipeline

@app.command("all")
def cmd_all(workers: int = 12, min_df: int = 3,
            min_docs: int = 600, top_n: int = 900, breadth_n: int = 1500,
            min_peak_df: int = 10, min_peak_rate: float = 3e-4,
            min_growth: float = 2.0, min_penalty: float = 0.45,
            cutoff: str = "2024-06", skip_ingest: bool = True,
            skip_templates: bool = False):
    """Run every analysis stage on whatever has been ingested.

    Breadth runs *before* clustering: the confounder measures it produces are what
    keep bot templates and migration imports -- which co-emerge perfectly by
    construction -- from crowding out the linguistic bundles.
    """
    if not skip_ingest:
        cmd_ingest(workers=workers)
    if not skip_templates:
        cmd_templates(workers=workers, show=0)
    cmd_pass_a(min_df=min_df, workers=workers)
    cmd_vocab(min_peak_df=min_peak_df, min_peak_rate=min_peak_rate,
              min_growth=min_growth, max_terms=80_000)
    cmd_pass_b(workers=workers)
    cmd_emergence(min_docs=min_docs)
    cmd_breadth(top_n=breadth_n, workers=max(2, workers - 2))
    cmd_cluster(top_n=top_n, min_docs=min_docs, min_penalty=min_penalty)
    cmd_atlas()
    cmd_align()
    cmd_validate(cutoff=cutoff, min_docs=min_docs, workers=max(2, workers - 4))


if __name__ == "__main__":
    app()

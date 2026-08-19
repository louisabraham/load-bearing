"""shift -- uniform GitHub comment sample, weekly word counts, abrupt shifts."""

from __future__ import annotations

import numpy as np
import polars as pl
import typer

from . import counts, detect, factor, fetch
from . import track as tracker  # the command below takes the name `track`

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def fetch_data(hours_per_week: int = fetch.HOURS_PER_WEEK, cap_mb: int = fetch.CAP_MB,
               seed: int = fetch.SEED, workers: int = 12):
    """Download the sample: `hours_per_week` random hours from every week.

    Resumable, and deepening is free: the draw is a truncated permutation, so a
    larger `hours_per_week` extends the sample and refetches nothing.
    """
    typer.echo(fetch.run(hours_per_week=hours_per_week, cap_mb=cap_mb, seed=seed,
                         workers=workers))


@app.command()
def api_fetch(per_week: int = 10, kind: str = "issue", seed: int = 0,
              window_s: int = 300, token: str = "", rate: float = 26.0,
              workers: int = 5, exclude_apps: bool = False,
              require_prose: bool = False):
    """Sample issues or PRs directly from GitHub, bypassing the archive entirely.

    Each week gets `per_week` randomly placed short windows, enumerated from the search
    API. Needs GITHUB_TOKEN (or --token). The budget is requests per minute, not
    bandwidth: at 26/min one hour buys about 1,560 windows, so 137 weeks x 10 windows
    takes roughly 53 minutes.

    --exclude-apps drops the four Apps that dominate the surface, and --require-prose
    requires any of a few function words in the body, which is the only way to exclude
    empty bodies. Together they take a hundred-item page from 43 usable documents to 97.
    Each filter combination writes to its own directory, because each enumerates a
    different population.
    """
    from . import apifetch
    typer.echo(apifetch.run(per_week=per_week, kind=kind, seed=seed,
                            window_s=window_s, token=token or None, rate=rate,
                            workers=workers,
                            exclude_apps=apifetch.EXCLUDE_APPS if exclude_apps else (),
                            prose_terms=apifetch.PROSE_TERMS if require_prose else (),
                            log=lambda m: typer.echo(m)))


@app.command()
def coverage(hours: int = fetch.HOURS_PER_WEEK, seed: int = fetch.SEED,
             draws_only: bool = False):
    """What of the sample is on disk, and how many documents each week has.

    Reported per draw rather than as one total, because a draw is the unit that keeps
    the sample balanced: draw d is present for every week or for none, and only
    complete draws should go into a matrix.
    """
    T = fetch.n_weeks()
    by_bin = counts.groups(hours, seed)
    typer.echo(f"{sum(len(g) for g in by_bin.values())} of {hours * T} "
               f"sampled hours present over {T} weeks")
    for d in range(hours):
        have = sum(1 for k in range(T)
                   if fetch.path(*fetch.draws(k, d + 1, seed)[d]).exists())
        typer.echo(f"  draw {d}: {have:3d}/{T}" + ("  complete" if have == T else ""))
    if draws_only:
        return
    if counts.MATRIX.exists():
        m = counts.load()
        for i in range(len(m.n)):
            bar = "#" * int(40 * m.n[i] / max(m.n.max(), 1))
            typer.echo(f"  {fetch.week_start(i)}  {m.n[i]:7,}  {bar}")


@app.command()
def build(min_df: int = counts.MIN_DF, hours: int = fetch.HOURS_PER_WEEK,
          seed: int = fetch.SEED, cap: int = counts.DOCS_PER_WEEK,
          drop_bots: bool = False, source: str = "archive"):
    """Count words once per document into a week x word matrix.

    --hours N uses the first N draws of each week, which is how to get a balanced
    matrix while a fetch is still running: draw N exists for every week or for none.

    --cap M thins every week to M documents so that boundaries in different years are
    comparable; 0 disables it and lets the source's own volume swings through.

    --source api uses the sample taken from GitHub's search API instead of the archive,
    which is the only one of the two that still holds prose after 2025.
    """
    typer.echo(counts.build(min_df=min_df, hours=hours, seed=seed, cap=cap,
                            drop_bots=drop_bots, source=source,
                            log=lambda m: typer.echo(m)))


@app.command()
def scan(top: int = 12, min_docs: int = detect.MIN_DOCS, half: int = detect.HALF):
    """Rank week boundaries by how much the word distribution moved."""
    m = counts.load()
    frame, _ = detect.scan(m.X, m.n, half=half, min_docs=min_docs)
    with pl.Config(tbl_rows=200, tbl_width_chars=160, float_precision=1):
        typer.echo(frame.sort("cut"))
        typer.echo("\ntop boundaries:")
        typer.echo(frame.filter("usable").sort("shift", descending=True).head(top))


@app.command()
def movers(cut: str, k: int = 25, falling: bool = False,
           min_docs: int = detect.MIN_DOCS, half: int = detect.HALF):
    """Words that moved most at one boundary, given as its date (YYYY-MM-DD)."""
    m = counts.load()
    frame, z = detect.scan(m.X, m.n, half=half, min_docs=min_docs)
    row = frame.filter(pl.col("cut").cast(pl.Utf8) == cut)
    if row.is_empty():
        raise typer.BadParameter(f"no boundary at {cut}; see `scan`")
    i = int(row["i"][0])
    before, at, after = detect.window(row["cut"][0], half)
    typer.echo(f"boundary {at}: {before}..{at} vs {at}..{after}  "
               f"shift={row['shift'][0]:.1f}  n_up={row['n_up'][0]}")
    with pl.Config(tbl_rows=200, tbl_width_chars=140, float_precision=3):
        typer.echo(detect.movers(z, m.X, m.n, m.vocab, i, k=k, rising=not falling, half=half))


@app.command()
def word(word: str):
    """One word's document frequency in every week.

    A boundary score says something moved; this says what the move looked like. A
    genuine arrival is a step that holds, a flood is a single tall week, and the two
    are indistinguishable from the boundary score alone.
    """
    m = counts.load()
    if word not in m.vocab:
        raise typer.BadParameter(
            f"{word!r} is not in the vocabulary (document frequency < {m.min_df})")
    j = m.vocab.index(word)
    pct = 100.0 * m.X[:, j] / np.maximum(m.n, 1)
    hi = pct.max()
    typer.echo(f"{word!r}: peak {hi:.2f}% of documents in a week")
    for i in range(len(m.n)):
        bar = "#" * int(round(46 * pct[i] / hi)) if hi > 0 else ""
        typer.echo(f"  {fetch.week_start(i)}  {pct[i]:6.2f}%  "
                   f"{m.X[i, j]:5d}/{m.n[i]:<6d} {bar}")


@app.command()
def track(term: str, start: str = "2024-01", end: str = "2026-08",
          surface: str = "comment", token: str = ""):
    """Count one expression per month using GitHub's own search.

    The archive cannot answer this: since mid-2025 its feed carries almost only
    PushEvent and no prose at all. Surfaces are comment, issue, pr, title, commit --
    or a comma-separated list of them. Set GITHUB_TOKEN to go three times faster.
    """
    ys, ms = (int(x) for x in start.split("-"))
    ye, me = (int(x) for x in end.split("-"))
    frames = [
        tracker.monthly(term, (ys, ms), (ye, me), sf.strip(), token or None,
                        log=lambda m: typer.echo(m))
        for sf in surface.split(",")
    ]
    df = pl.concat(frames)
    typer.echo("")
    for sf, part in df.group_by("surface", maintain_order=True):
        unit = tracker.SURFACES[sf[0]].unit
        hi = max(part["per_10k"].fill_null(0).max() or 0.0, 1e-9)
        typer.echo(f"{term!r} in {sf[0]} (per 10k {unit})")
        for r in part.iter_rows(named=True):
            v = r["per_10k"] or 0.0
            typer.echo(f"  {r['month']}  {r['hits']!s:>8}  {v:8.2f}  "
                       f"{'#' * int(round(42 * v / hi))}")


@app.command()
def components(k: int = factor.K, l1: float = factor.L1, last: int = 0, top: int = 12,
               show: int = -1):
    """Factorise the matrix: rate(word, week) = sum of weight(week) x profile(word).

    Each component is a way of writing -- a fixed set of words and a weekly weight
    curve shared by all of them. A component that is off and then on is what a bundle
    of habits arriving looks like.

    --last N fits on the first N weeks only, which matters because GH Archive's prose
    dies during 2026; --show K prints one component's whole weekly curve.
    """
    m = counts.load()
    end = last or len(m.n)
    f = factor.fit(m.X[:end], m.n[:end], m.vocab, k=k, l1=l1)
    typer.echo(f"weeks 0-{end - 1}, {m.n[:end].sum():,} documents, "
               f"{len(f.vocab):,} words, error {f.err:.1f}")
    if show >= 0:
        for line in factor.curve(f, show):
            typer.echo(line)
        typer.echo("")
    for r in factor.shapes(f).sort("mass", descending=True).iter_rows(named=True):
        rel = f" | {r['release']} {r['release_days']:+d}d" if r["release"] else ""
        typer.echo(f"k={r['k']}  mass={r['mass']:.2f} off={r['off']:.2f} "
                   f"jump={r['jump']:.2f} live {r['first_live']}..{r['last_live']}{rel}")
        typer.echo("   " + ", ".join(factor.characteristic(f, r["k"], top=top)["word"]))


@app.command()
def plot(k: int = factor.K, model: str = "nmf", transform: str = "mean",
         last: int = 0, top: int = 9, out: str = "out/shift/components.png"):
    """Draw every component: its weekly weight curve and the words it owns.

    --transform log or sqrt compresses the normalised matrix further, which narrows the
    register to its rarest, purest half. --model lda fits Latent Dirichlet Allocation on
    the same normalised matrix;
    --model lda-counts fits it on the raw counts, which is the coherent generative form
    and the one that fails to separate the prose register.
    """
    from pathlib import Path

    m = counts.load()
    end = last or len(m.n)
    if model.startswith("lda"):
        f = factor.fit_lda(m.X[:end], m.n[:end], m.vocab, k=k, transform=transform,
                           scale="counts" if model == "lda-counts" else "relative")
    else:
        f = factor.fit(m.X[:end], m.n[:end], m.vocab, k=k, transform=transform)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    label = {"lda": "LDA (normalised)", "lda-counts": "LDA (counts)"}.get(model, "NMF")
    factor.plot(f, out, top=top,
                title=f"{label}, k={k}, transform={transform} — "
                      "weekly weight and characteristic words")
    typer.echo(f"wrote {out}  ({label}, k={k}, "
               f"{'perplexity' if model.startswith('lda') else 'error'} {f.err:.1f})")


@app.command()
def report(top: int = 6, k: int = 20, min_docs: int = detect.MIN_DOCS,
           half: int = detect.HALF, hours: int = fetch.HOURS_PER_WEEK,
           seed: int = fetch.SEED, cap: int = counts.DOCS_PER_WEEK,
           source: str = "archive", out: str = "out/shift/report.md"):
    """Write the whole finding: coverage, ranked boundaries, words, repo support."""
    from pathlib import Path

    m = counts.load()
    frame, z = detect.scan(m.X, m.n, half=half, min_docs=min_docs)
    best = frame.filter("usable").sort("shift", descending=True).head(top)

    # movers and their repo breadth first: how many distinct repositories carry the
    # words that moved is the sharpest available check on whether a boundary is a
    # language change or one busy project, so it belongs in the summary table and
    # not only in the per-boundary detail
    detail = []
    for r in best.iter_rows(named=True):
        mv = detect.movers(z, m.X, m.n, m.vocab, r["i"], k=k, half=half)
        sup = counts.support(detect.weeks_after(r["i"], half), mv["word"].to_list(),
                             hours=hours, seed=seed, cap=cap, source=source)
        detail.append((r, mv, sup,
                       int(np.median([sup[w].repos for w in mv["word"]])),
                       float(np.median([sup[w].bot_share for w in mv["word"]]))))

    lines = [
        "# Abrupt shifts in GitHub comment vocabulary",
        "",
        "- source: " + ("GitHub search API, issue bodies, all authors"
                         if source == "api" else
                         "`IssueCommentEvent` bodies, GH Archive, all authors"),
        f"- window: {fetch.week_start(0)} to {fetch.week_start(len(m.n))} "
        f"({len(m.n)} weeks)",
        f"- documents: {m.n.sum():,} (min {m.n.min():,}, "
        f"median {int(np.median(m.n)):,}, max {m.n.max():,} per week)",
        f"- vocabulary: {len(m.vocab):,} words, document frequency >= {m.min_df}",
        f"- sample: {hours} " + ("5-minute windows enumerated from each week"
                                  if source == "api" else
                                  "hours drawn at random from each week")
        + f" (seed {seed})",
        f"- test: every week boundary, {half} weeks pooled on each side",
        f"- each week thinned to {cap} documents so boundaries are comparable"
        if cap else "- weeks not thinned: document counts vary with the archive",
        f"- boundaries scored: {frame['usable'].sum()} of {len(frame)} "
        f"(both sides >= {min_docs:,} documents)",
        "",
        "## Ranked boundaries",
        "",
        "Windows overlap, so neighbouring boundaries share three quarters of their",
        "data and their scores are correlated: a real change reads as a short run of",
        "elevated weeks, and the peak of the run is the estimate.",
        "",
        "`common shift` is the odds ratio the median word moved by — mostly document",
        "length, and removed from every z. `repos` and `bot` are medians over the top",
        f"{k} risers: how many distinct repositories carried each word, and what share",
        "of its documents came from an App account. A language change is spread across",
        "repositories; a tool deployment is not, and shows up as bot share instead.",
        "",
        "| cut | shift | up | down | common shift | repos | bot | docs before | "
        "docs after | nearest release | days |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, _, _, breadth, botshare in detail:
        lines.append(
            f"| {r['cut']} | {r['shift']:.1f} | {r['n_up']} | {r['n_down']} | "
            f"x{r['common']:.2f} | {breadth} | {botshare:.0%} | "
            f"{r['docs_before']:,} | {r['docs_after']:,} | {r['release'] or '--'} | "
            f"{format(r['release_days'], '+d') if r['release'] else ''} |"
        )

    for r, mv, sup, _, _ in detail:
        lines += [
            "",
            f"## {r['cut']}  (shift {r['shift']:.1f}, " + (
                f"nearest release {r['release']} {r['release_days']:+d} days)"
                if r["release"] else f"no release within {detect.NEAR_DAYS} days)"),
            "",
            "%s .. %s against %s .. %s" % (
                detect.window(r["cut"], half)[0], r["cut"], r["cut"],
                detect.window(r["cut"], half)[2]),
            "",
            "| word | z | % before | % after | x | repos after | bot share |",
            "|---|---|---|---|---|---|---|",
        ]
        for w in mv.iter_rows(named=True):
            lines.append(
                f"| `{w['word']}` | {w['z']:.1f} | {w['pct_before']:.3f} | "
                f"{w['pct_after']:.3f} | {w['ratio']:.2f} | "
                f"{sup[w['word']].repos} | {sup[w['word']].bot_share:.0%} |"
            )

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    typer.echo(f"wrote {p}")


if __name__ == "__main__":
    app()

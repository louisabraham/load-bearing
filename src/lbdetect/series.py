"""Build expression x period frequency series.

Three passes of decreasing width and increasing detail, because the wide passes
must stay cheap:

* **Pass A** counts every feature in every period and keeps only what clears a
  per-period document-frequency floor. Output: a candidate vocabulary.
* **Pass B** recounts the candidates exactly in every period, including the ones
  where they were too rare to survive pass A. Output: a dense count matrix. The
  pre-emergence rate has to be exact, because it is the denominator of the
  growth ratio.
* **Pass C** (``breadth.py``) adds repositories, authors, contexts and
  occurrence counts for a shortlist only.

One rule governs the whole module: the predicate that selects documents for the
numerator also produces the denominator. Otherwise frequencies move with the
document mix rather than with language.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from . import dedupe, templates, textclean
from .ngrams import features
from .util import write_json

DF_A = C.SERIES / "pass_a"
VOCAB = C.SERIES / "vocab.parquet"
MATRIX = C.SERIES / "matrix.npz"
PERIODS_META = C.SERIES / "periods.parquet"

MIN_DF_PERIOD = 15  # a feature must appear in this many docs in a period to survive pass A


# --------------------------------------------------------------------- documents

def available_periods(freq: str = "M") -> list[str]:
    return sorted(p.name for p in C.DOCS.iterdir() if p.is_dir() and any(p.glob("*.parquet")))


def _concat_shards(files: list[Path]) -> pl.DataFrame:
    """Concatenate shards written by different versions of the ingester.

    Columns are added over time -- `assist` came later than the first shards --
    and a plain concat fails on the mismatch. Missing columns are filled rather
    than forcing a re-download of everything already fetched.
    """
    from .ingest import SCHEMA

    frames = []
    for f in files:
        d = pl.read_parquet(f)
        for col, dt in SCHEMA.items():
            if col not in d.columns:
                d = d.with_columns(pl.lit(None, dt).alias(col))
        # concat matches on position, not name, so a filled-in column appended at
        # the end would line up against a different column in the newer shards
        extra = [c for c in d.columns if c not in SCHEMA]
        frames.append(d.select([*SCHEMA.keys(), *extra]))
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def load_period(period: str, freq: str = "M", apply_templates: bool = True) -> pl.DataFrame:
    """Eligible, de-duplicated, non-bot documents of one period.

    Duplicates, templates and machine authors are dropped here rather than
    down-weighted later: a template rolling out across repositories mimics a
    genuine broad rise, and a review bot writes more prose than any human.

    The bot flag is recomputed from the author login instead of read from the
    shard, so that extending the bot list does not require re-downloading.
    """
    files = sorted((C.DOCS / period).glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    df = _concat_shards(files)
    df = df.with_columns(is_bot=textclean.bot_expr("author"))
    df = df.filter(~pl.col("is_bot") & pl.col("artifact").is_in(C.PROSE_ARTIFACTS))
    if df.height == 0:
        return df
    df = df.unique(subset=["doc_id"], keep="first")
    if apply_templates:
        tmpl = templates.load()
        if tmpl:
            df = df.with_columns(
                text=pl.col("text").map_elements(
                    lambda t: templates.strip(t, tmpl), return_dtype=pl.Utf8)
            ).filter(pl.col("text").str.len_chars() > 0)
            if df.height == 0:
                return df
    df = dedupe.annotate(df)
    return df.filter(~pl.col("is_dup") & ~pl.col("is_template")
                     & ~pl.col("is_templated_author"))


def period_meta(period: str, df: pl.DataFrame) -> dict:
    return {
        "period": period,
        "docs": df.height,
        "tokens": int(df["n_tokens"].sum()) if df.height else 0,
        "repos": int(df["repo_id"].n_unique()) if df.height else 0,
        "authors": int(df["author"].n_unique()) if df.height else 0,
    }


def exclusion_stats(period: str) -> dict:
    """What each filter removed, per period.

    Worth reporting rather than hiding: the machine-authored share of GitHub prose
    is itself a finding, and a rising exclusion rate is the most likely
    explanation for an apparent change in human language.
    """
    files = sorted((C.DOCS / period).glob("*.parquet"))
    if not files:
        return {"period": period, "raw": 0}
    raw = _concat_shards(files)
    raw = raw.unique(subset=["doc_id"], keep="first")
    n_raw = raw.height
    n_bot = int(raw["is_bot"].sum())
    prose = raw.filter(~pl.col("is_bot") & pl.col("artifact").is_in(C.PROSE_ARTIFACTS))
    ann = dedupe.annotate(prose) if prose.height else prose
    out = {
        "period": period,
        "raw": n_raw,
        "known_bot": n_bot,
        "bot_share": round(n_bot / n_raw, 4) if n_raw else 0.0,
    }
    if ann.height:
        out |= {
            "templated_author": int(ann["is_templated_author"].sum()),
            "duplicate": int(ann["is_dup"].sum()),
            "cross_repo_template": int(ann["is_template"].sum()),
            "machine_share": round(
                (n_bot + int(ann["is_templated_author"].sum())) / n_raw, 4),
        }
    return out


# ------------------------------------------------------------------------ pass A

def _pass_a_period(period: str, min_df: int, use_char: bool) -> dict:
    out = DF_A / f"{period}.parquet"
    meta_path = DF_A / f"{period}.meta.json"
    if out.exists() and meta_path.exists():
        import json

        return json.loads(meta_path.read_text())

    df = load_period(period)
    meta = period_meta(period, df)
    counter: Counter[str] = Counter()
    for text in df["text"].to_list():
        counter.update(features(text, use_char))
    keep = [(t, c) for t, c in counter.items() if c >= min_df]
    keep.sort(key=lambda x: -x[1])
    pl.DataFrame(
        {"term": [k[0] for k in keep], "df": [k[1] for k in keep]},
        schema={"term": pl.Utf8, "df": pl.Int32},
    ).write_parquet(out, compression="zstd")
    meta["kept_terms"] = len(keep)
    meta["total_terms"] = len(counter)
    meta["min_df"] = min_df  # the censoring floor, needed to bound growth honestly
    write_json(meta_path, meta)
    return meta


def pass_a(periods: list[str] | None = None, min_df: int = MIN_DF_PERIOD,
           use_char: bool = False, workers: int = 12, log=print) -> pl.DataFrame:
    DF_A.mkdir(parents=True, exist_ok=True)
    periods = periods or available_periods()
    log(f"pass A: {len(periods)} periods, min_df={min_df}, char_ngrams={use_char}")
    metas = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_pass_a_period, p, min_df, use_char): p for p in periods}
        for i, fut in enumerate(as_completed(futs), 1):
            m = fut.result()
            metas.append(m)
            if i % 10 == 0 or i == len(periods):
                log(f"  {i}/{len(periods)} periods")
    meta = pl.DataFrame(metas).sort("period")
    meta.write_parquet(PERIODS_META)
    log(f"  docs={meta['docs'].sum():,} tokens={meta['tokens'].sum():,}")
    return meta


# ----------------------------------------------------------------- vocabulary

def window_bounds(periods: list[str], window_months: int = 12,
                  equal_docs: bool = False, n_windows: int = 6
                  ) -> list[tuple[str, str]]:
    """Contiguous windows for pooled candidate selection.

    Calendar windows by default. Equal-document windows sound fairer but are not:
    when one era holds most of the corpus it swallows several years of a thinner
    era into a single window, and a rise inside that window becomes invisible.
    Rates already handle unequal volume; what pooling must preserve is time
    resolution.
    """
    if not periods:
        return []
    if equal_docs:
        meta = pl.read_parquet(PERIODS_META).sort("period")
        d = dict(zip(meta["period"].to_list(), meta["docs"].to_list()))
        tot = sum(d.get(p, 0) for p in periods)
        if tot == 0:
            return [(periods[0], periods[-1])]
        target = tot / n_windows
        out, start, acc = [], periods[0], 0
        for i, p in enumerate(periods):
            acc += d.get(p, 0)
            if acc >= target and i < len(periods) - 1:
                out.append((start, p))
                start, acc = periods[i + 1], 0
        out.append((start, periods[-1]))
        return out
    return [(periods[i], periods[min(i + window_months - 1, len(periods) - 1)])
            for i in range(0, len(periods), window_months)]


def build_vocab(
    max_terms: int = 120_000,
    min_peak_df: int = 25,
    min_peak_rate: float = 1e-4,
    min_growth: float = 2.5,
    window_months: int = 12,
    log=print,
) -> pl.DataFrame:
    """Shortlist features worth an exact recount.

    Selection works on pooled windows rather than single periods. A per-period
    floor makes visibility depend on that period's document count, so with uneven
    coverage a term needs a far higher rate to be noticed in a thin month than in
    a fat one -- which would bias discovery against exactly the era of interest.
    Pooling into equal-document windows removes that asymmetry.

    Counts below the pass A floor are censored, not zero, so an absent term is
    charged the floor. That understates growth rather than inventing it.
    """
    meta = pl.read_parquet(PERIODS_META).sort("period")
    periods = meta["period"].to_list()
    docs = dict(zip(periods, meta["docs"].to_list()))
    windows = window_bounds(periods, window_months)
    log(f"vocab windows: {', '.join(f'{a}..{b}' for a, b in windows)}")

    win_of = {}
    win_docs = [0] * len(windows)
    for wi, (a, b) in enumerate(windows):
        for p in periods:
            if a <= p <= b:
                win_of[p] = wi
                win_docs[wi] += docs.get(p, 0)

    # pooled document frequency per term per window
    df_win: dict[str, np.ndarray] = {}
    # how much of each window's volume sat in periods where the term was censored
    censored: list[int] = [0] * len(windows)
    for p in periods:
        f = DF_A / f"{p}.parquet"
        if not f.exists() or not docs.get(p):
            continue
        wi = win_of[p]
        t = pl.read_parquet(f)
        for term, d in zip(t["term"].to_list(), t["df"].to_list()):
            arr = df_win.get(term)
            if arr is None:
                arr = np.zeros(len(windows))
                df_win[term] = arr
            arr[wi] += d

    wd = np.array(win_docs, dtype=float)
    wd[wd == 0] = np.nan
    # the floor actually used in pass A, not the module default
    used_min_df = (int(meta["min_df"].max()) if "min_df" in meta.columns
                   else MIN_DF_PERIOD)
    floor_rate = used_min_df / np.nanmedian(
        np.array([docs[p] for p in periods if docs.get(p)], dtype=float)
    )

    rows = []
    for term, arr in df_win.items():
        rates = arr / wd
        peak_i = int(np.nanargmax(rates))
        trough_i = int(np.nanargmin(rates))
        peak_r = float(rates[peak_i])
        if arr[peak_i] < min_peak_df or peak_r < min_peak_rate:
            continue
        # Variation in *either* direction qualifies. Selecting only risers leaves
        # the lexicon with no early-period markers, which biases date estimation
        # systematically late: every feature a document can contain is evidence
        # for a later period. Expressions that rose and then faded are the most
        # informative of all for pinning down a narrow range.
        trough_eff = max(float(rates[trough_i]), floor_rate * 0.25)
        swing = peak_r / trough_eff
        if swing < min_growth:
            continue
        base_r = float(np.nanmin(rates[: max(1, peak_i)])) if peak_i > 0 else peak_r
        growth = peak_r / max(base_r, floor_rate * 0.25)
        rows.append((term, peak_r, base_r, growth, swing, int(arr[peak_i]),
                     windows[peak_i][0], "rise" if trough_i < peak_i else "decline",
                     int((arr > 0).sum())))

    v = pl.DataFrame(
        rows,
        schema=["term", "peak_rate", "base_rate", "prelim_growth", "swing",
                "peak_df", "peak_window", "direction", "windows_seen"],
        orient="row",
    ).sort("swing", descending=True)
    if v.height > max_terms:
        v = v.head(max_terms)
    v.write_parquet(VOCAB)
    log(f"vocab: {v.height:,} candidates from {len(df_win):,} features above the floor")
    return v


# ------------------------------------------------------------------------ pass B

_VOCAB_CACHE: dict[str, dict[str, int]] = {}


def _vocab_index() -> dict[str, int]:
    if "v" not in _VOCAB_CACHE:
        terms = pl.read_parquet(VOCAB)["term"].to_list()
        _VOCAB_CACHE["v"] = {t: i for i, t in enumerate(terms)}
    return _VOCAB_CACHE["v"]


def _pass_b_period(period: str, use_char: bool
                   ) -> tuple[str, np.ndarray, np.ndarray, dict]:
    """Counts per expression *per artifact type*, plus documents per artifact.

    Stratifying by artifact is what makes the series comparable over time. The
    mix drifts heavily -- issue comments fall from 35% to 12% of this corpus while
    review comments rise -- and in 2026 pull-request descriptions vanish from the
    archive entirely. Without stratification every expression commoner in review
    comments than in descriptions shows a jump exactly where the mix moved.
    """
    idx = _vocab_index()
    arts = list(C.PROSE_ARTIFACTS)
    apos = {a: i for i, a in enumerate(arts)}
    counts = np.zeros((len(idx), len(arts)), dtype=np.int32)
    docs_a = np.zeros(len(arts), dtype=np.int64)
    df = load_period(period)
    for text, artifact in zip(df["text"].to_list(), df["artifact"].to_list()):
        a = apos.get(artifact)
        if a is None:
            continue
        docs_a[a] += 1
        for f in features(text, use_char):
            j = idx.get(f)
            if j is not None:
                counts[j, a] += 1
    return period, counts, docs_a, period_meta(period, df)


def pass_b(periods: list[str] | None = None, use_char: bool = False,
           workers: int = 12, log=print) -> dict:
    periods = periods or available_periods()
    terms = pl.read_parquet(VOCAB)["term"].to_list()
    log(f"pass B: {len(terms):,} terms x {len(periods)} periods")
    arts = list(C.PROSE_ARTIFACTS)
    mat = np.zeros((len(terms), len(periods), len(arts)), dtype=np.int32)
    docs_a = np.zeros((len(periods), len(arts)), dtype=np.int64)
    pos = {p: i for i, p in enumerate(periods)}
    metas = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_pass_b_period, p, use_char) for p in periods]
        for i, fut in enumerate(as_completed(futs), 1):
            p, counts, da, meta = fut.result()
            mat[:, pos[p], :] = counts
            docs_a[pos[p], :] = da
            metas.append(meta)
            if i % 10 == 0 or i == len(periods):
                log(f"  {i}/{len(periods)} periods")
    meta = pl.DataFrame(metas).sort("period")
    meta.write_parquet(PERIODS_META)
    np.savez_compressed(
        MATRIX,
        counts=mat.sum(axis=2),
        counts_by_artifact=mat,
        docs_by_artifact=docs_a,
        artifacts=np.array(arts, dtype=object),
        terms=np.array(terms, dtype=object),
        periods=np.array(periods, dtype=object),
        docs=meta["docs"].to_numpy(),
    )
    log(f"  matrix {mat.shape[:2]} x {len(arts)} artifacts, "
        f"nonzero cells {int((mat.sum(axis=2) > 0).sum()):,}")
    return {"terms": len(terms), "periods": len(periods)}


# ------------------------------------------------------------------------ loading

class Series:
    """The count matrix plus the denominators, with rate helpers."""

    def __init__(self, counts: np.ndarray, terms: list[str], periods: list[str],
                 docs: np.ndarray, counts_by_artifact: np.ndarray | None = None,
                 docs_by_artifact: np.ndarray | None = None,
                 artifacts: list[str] | None = None):
        self.counts = counts
        self.terms = terms
        self.periods = periods
        self.docs = docs.astype(float)
        self.index = {t: i for i, t in enumerate(terms)}
        self.counts_by_artifact = counts_by_artifact
        self.docs_by_artifact = docs_by_artifact
        self.artifacts = artifacts

    @classmethod
    def load(cls, path: Path = MATRIX) -> "Series":
        z = np.load(path, allow_pickle=True)
        return cls(
            z["counts"],
            [str(t) for t in z["terms"]],
            [str(p) for p in z["periods"]],
            z["docs"],
            z["counts_by_artifact"] if "counts_by_artifact" in z else None,
            z["docs_by_artifact"] if "docs_by_artifact" in z else None,
            [str(a) for a in z["artifacts"]] if "artifacts" in z else None,
        )

    def standardize(self) -> "Series":
        """Re-express counts as if every period had the corpus-average artifact mix.

        Direct standardisation: the rate within each artifact type is estimated
        separately, then recombined with weights fixed at the corpus average. A
        period's own mix therefore cannot move any expression's rate, which is what
        makes months comparable when the archive's coverage of artifact types
        changes -- as it does severely in 2026, where pull-request descriptions are
        absent altogether.

        Where an artifact is missing from a period, its weight is redistributed
        over the types that are present; this is the best available comparison, not
        a perfect one, and periods missing a large share of the mix stay suspect.
        """
        if self.counts_by_artifact is None or self.docs_by_artifact is None:
            return self
        da = self.docs_by_artifact.astype(float)  # (periods, artifacts)
        w = da.sum(axis=0)
        w = w / w.sum() if w.sum() else w
        present = da > 0
        # renormalise the fixed weights over the strata available in each period
        wp = np.where(present, w[None, :], 0.0)
        norm = wp.sum(axis=1, keepdims=True)
        wp = np.divide(wp, norm, out=np.zeros_like(wp), where=norm > 0)
        rate_a = np.divide(
            self.counts_by_artifact.astype(float), da[None, :, :],
            out=np.zeros(self.counts_by_artifact.shape, dtype=float),
            where=da[None, :, :] > 0,
        )
        std_rate = (rate_a * wp[None, :, :]).sum(axis=2)  # (terms, periods)
        # back to an effective count on the same scale, so the binomial test and
        # every downstream threshold keep their meaning
        eff = std_rate * self.docs[None, :]
        return Series(eff, self.terms, self.periods, self.docs,
                      self.counts_by_artifact, self.docs_by_artifact, self.artifacts)

    def artifact_coverage(self) -> np.ndarray:
        """Share of the corpus-average artifact mix actually present per period."""
        if self.docs_by_artifact is None:
            return np.ones(len(self.periods))
        da = self.docs_by_artifact.astype(float)
        w = da.sum(axis=0)
        w = w / w.sum() if w.sum() else w
        return (np.where(da > 0, w[None, :], 0.0)).sum(axis=1)

    @property
    def rates(self) -> np.ndarray:
        """Document-frequency share per period, with empty periods left as NaN so
        low-coverage months cannot masquerade as zero usage."""
        d = np.where(self.docs > 0, self.docs, np.nan)
        return self.counts / d

    def rate_of(self, term: str) -> np.ndarray:
        return self.rates[self.index[term]]

    def usable_mask(self, min_docs: int = 3000,
                    min_artifact_coverage: float = 0.9) -> np.ndarray:
        """Periods that can be compared: enough documents, and an intact artifact mix.

        The coverage gate matters as much as the volume gate. From 2025-11 this
        archive stops carrying pull-request descriptions, so roughly 15% of the
        usual mix is simply absent. Standardisation redistributes that weight but
        cannot invent the missing stratum, and the result was 21 of 30 clusters
        placing their shared changepoint in the first month of the gap -- an
        artifact of the collector, reported as a language event.

        Dropping those months costs the most recent window, which is worth saying
        out loud, but they were also the thinnest in the corpus.
        """
        ok = self.docs >= min_docs
        if self.docs_by_artifact is not None:
            ok = ok & (self.artifact_coverage() >= min_artifact_coverage)
        return ok

    def subset(self, term_idx: np.ndarray) -> "Series":
        return Series(
            self.counts[term_idx], [self.terms[i] for i in term_idx],
            self.periods, self.docs,
            self.counts_by_artifact[term_idx] if self.counts_by_artifact is not None else None,
            self.docs_by_artifact, self.artifacts,
        )

    def aggregate(self, k: int = 3) -> "Series":
        """Pool every `k` consecutive periods (k=3 gives quarters).

        Correlating growth series needs each period to carry enough documents that
        its rate is not dominated by sampling noise; where monthly coverage is
        thin, pooling first is what makes the correlations mean anything.
        """
        if k <= 1:
            return self
        n = (len(self.periods) // k) * k
        if n == 0:
            return self
        c = self.counts[:, :n].reshape(self.counts.shape[0], -1, k).sum(axis=2)
        d = self.docs[:n].reshape(-1, k).sum(axis=1)
        labels = [self.periods[i] for i in range(0, n, k)]
        return Series(c, self.terms, labels, d)

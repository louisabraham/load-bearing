"""Pass C: breadth, contexts and confounders for a shortlist of expressions.

Emergence in the series says *when* something changed. This module asks whether
the change is a change in how people write, and the answers are what separate a
linguistic habit from the four things that imitate one:

* one loud repository or author (concentration)
* a template or bot that survived cleaning (repetition of identical contexts)
* a new product, library or CVE name (a proper noun, not a phrase)
* a single technical topic or an AI-only corner of GitHub (topical capture)

Everything here is measured on the post-emergence window, because that is the
population that produced the signal. Measuring breadth over all time would let a
long pre-history dilute a concentrated burst.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl

from . import config as C
from . import dedupe
from .ngrams import CONSTRUCTIONS, TYPO_MARKERS, features
from .series import load_period
from .util import month_index

CONTEXT_WINDOW = 60  # characters either side of a hit
SIG_PER_PERIOD = 6  # document signatures kept per expression per period
SIG_TOTAL = 48  # ...and in total, for the cohesion estimate


def cohesion(signatures: list) -> float | None:
    """Mean pairwise Jaccard among the documents that contain an expression.

    A genuine turn of phrase turns up in documents that otherwise have nothing in
    common, so cohesion is near zero. A high value means the expression marks a
    *kind of document* rather than a way of writing: spam with a fixed skeleton, an
    issue template, a migration import, a tool's output. This catches all of those
    with one number and no list of known offenders, including the ones that defeat
    whole-document de-duplication because their variable parts differ.
    """
    import numpy as _np

    if len(signatures) < 4:
        return None
    m = _np.array(signatures[:SIG_TOTAL])
    n = m.shape[0]
    total, pairs = 0.0, 0
    for i in range(n - 1):
        eq = (m[i + 1 :] == m[i]).mean(axis=1)
        total += float(eq.sum())
        pairs += eq.size
    return total / pairs if pairs else None

# Repositories whose topic is AI itself: an expression confined to these is
# probably shop-talk about models, not a habit picked up from using them.
AI_REPO = re.compile(
    r"(?:^|[-_/])(?:llm|llms|gpt|chatgpt|openai|anthropic|claude|gemini|bard|copilot|"
    r"langchain|llamaindex|ollama|huggingface|transformers|diffusers|stable-?diffusion|"
    r"comfyui|autogpt|agentic|rag|embeddings?|vllm|deepseek|qwen|mistral|"
    r"prompt|prompts|prompting|aider|cursor|codex|mcp)(?:$|[-_/])",
    re.I,
)


def _term_matcher(term: str):
    """A callable testing whether one cleaned document contains `term`.

    Word n-grams are matched on the canonical token stream, so surface variants
    ("load-bearing" / "load bearing") both count -- the same rule used when the
    series were built, otherwise breadth would be measured on a different
    population than emergence.
    """
    if term.startswith("CONSTR:"):
        pat = CONSTRUCTIONS[term[7:]]
        return lambda text: bool(pat.search(text.lower()))
    if term.startswith("TYPO:"):
        pat = TYPO_MARKERS[term[5:]]
        return lambda text: bool(pat.search(text))
    if term.startswith("HYPH:"):
        needle = term[5:]
        return lambda text: needle in text.lower()
    if term.startswith("CHR:"):
        needle = term[4:]
        return lambda text: needle in re.sub(r"\s+", " ", text.lower())
    needle = term

    def match(text: str) -> bool:
        from .ngrams import tokenize

        toks = tokenize(text)
        n = len(needle.split())
        if n == 1:
            return needle in toks
        joined = " ".join(toks)
        return needle in joined

    return match


_SENT_END = ".!?\n:;"


def _cap_pattern(term: str) -> re.Pattern | None:
    """Match a word n-gram in the original text, tolerating hyphen/space variants."""
    words = term.split()
    if not words or not all(w.isalpha() for w in words):
        return None
    return re.compile(r"\b" + r"[\s\-_]+".join(re.escape(w) for w in words) + r"\b",
                      re.I)


def capitalization(text: str, pat: re.Pattern) -> tuple[int, int]:
    """(mid-sentence occurrences, of which capitalised).

    A phrase that is nearly always capitalised mid-sentence is a name -- a
    product, library or project -- not a turn of phrase. This distinguishes them
    without a list of products, which matters because the whole point is to catch
    ones nobody has heard of yet. Sentence-initial hits are excluded because
    English capitalises those regardless.
    """
    occ = cap = 0
    for m in pat.finditer(text):
        j = m.start() - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        if j < 0 or text[j] in _SENT_END:
            continue  # sentence-initial tells us nothing
        occ += 1
        if m.group(0)[0].isupper():
            cap += 1
    return occ, cap


def _contexts(text: str, term: str, out: list[str], cap: int) -> None:
    if len(out) >= cap:
        return
    low = text.lower()
    probe = term.split(":", 1)[-1].replace("-", " ") if ":" in term else term
    i = low.find(probe if probe in low else probe.split()[0])
    if i < 0:
        i = 0
    lo = max(0, i - CONTEXT_WINDOW)
    out.append(text[lo : i + len(probe) + CONTEXT_WINDOW].replace("\n", " ").strip())


def _period_breadth(period: str, terms: list[str], ctx_cap: int) -> dict:
    """Per-term aggregates for one period. Runs in a worker process."""
    df = load_period(period)
    if df.height == 0:
        return {"period": period, "rows": [], "docs": 0}

    wanted = set(terms)
    cap_pats = {t: p for t in terms if (p := _cap_pattern(t)) is not None}
    cap_occ: Counter = Counter()
    cap_hit: Counter = Counter()
    hasher = dedupe.MinHasher(32)
    sigs: dict[str, list] = defaultdict(list)  # term -> sample of doc signatures
    # One pass over documents extracting the full feature set is far cheaper than
    # one pass per term, so long as the shortlist fits in a set.
    repos: dict[str, set[int]] = defaultdict(set)
    authors: dict[str, set[str]] = defaultdict(set)
    repo_counts: dict[str, Counter] = defaultdict(Counter)
    artifacts: dict[str, Counter] = defaultdict(Counter)
    dfreq: Counter = Counter()
    ai_hits: Counter = Counter()
    ctxs: dict[str, list[str]] = defaultdict(list)

    cols = df.select("text", "repo", "repo_id", "author", "artifact").iter_rows()
    for text, repo, repo_id, author, artifact in cols:
        hits = features(text) & wanted
        if not hits:
            continue
        is_ai = bool(AI_REPO.search(repo or ""))
        sig = None
        for t in hits:
            if len(sigs[t]) < SIG_PER_PERIOD:
                if sig is None:
                    sig = hasher.signature(text)
                if sig is not None:
                    sigs[t].append(sig)
            dfreq[t] += 1
            repos[t].add(repo_id)
            authors[t].add(author)
            repo_counts[t][repo_id] += 1
            artifacts[t][artifact] += 1
            if is_ai:
                ai_hits[t] += 1
            p = cap_pats.get(t)
            if p is not None:
                o, c = capitalization(text, p)
                cap_occ[t] += o
                cap_hit[t] += c
            if len(ctxs[t]) < ctx_cap:
                _contexts(text, t, ctxs[t], ctx_cap)

    rows = []
    for t, d in dfreq.items():
        rc = repo_counts[t]
        top = rc.most_common(1)[0][1] if rc else 0
        rows.append(
            {
                "term": t,
                "period": period,
                "df": d,
                "n_repos": len(repos[t]),
                "n_authors": len(authors[t]),
                "top_repo_share": top / d if d else 0.0,
                "ai_repo_share": ai_hits[t] / d if d else 0.0,
                "cap_occ": cap_occ[t],
                "cap_hit": cap_hit[t],
                "sigs": sigs[t],
                "artifacts": dict(artifacts[t]),
                "contexts": ctxs[t],
            }
        )
    return {"period": period, "rows": rows, "docs": df.height}


def compute(
    terms: list[str],
    cp_periods: dict[str, str],
    periods: list[str],
    workers: int = 10,
    ctx_cap: int = 6,
    log=print,
) -> pl.DataFrame:
    """Breadth per expression, aggregated over its own post-emergence window."""
    log(f"breadth: {len(terms)} expressions over {len(periods)} periods")
    per_period: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_period_breadth, p, terms, ctx_cap) for p in periods]
        for i, fut in enumerate(as_completed(futs), 1):
            per_period.append(fut.result())
            if i % 10 == 0 or i == len(periods):
                log(f"  {i}/{len(periods)} periods")

    agg: dict[str, dict] = {
        t: {
            "df_post": 0, "df_pre": 0, "top_share": [], "ai_share": [],
            "artifacts": Counter(), "contexts": [], "sigs": [],
        }
        for t in terms
    }
    for block in per_period:
        p = block["period"]
        for r in block["rows"]:
            t = r["term"]
            a = agg.get(t)
            if a is None:
                continue
            post = month_index(p) >= month_index(cp_periods.get(t, "2099-01"))
            if post:
                a["df_post"] += r["df"]
                a["top_share"].append((r["top_repo_share"], r["df"]))
                a["ai_share"].append((r["ai_repo_share"], r["df"]))
                a["cap_occ"] = a.get("cap_occ", 0) + r.get("cap_occ", 0)
                a["cap_hit"] = a.get("cap_hit", 0) + r.get("cap_hit", 0)
                if len(a["sigs"]) < SIG_TOTAL:
                    a["sigs"].extend(r.get("sigs", [])[: SIG_TOTAL - len(a["sigs"])])
                a["artifacts"].update(r["artifacts"])
                a["n_repos_sum"] = a.get("n_repos_sum", 0) + r["n_repos"]
                a["n_authors_sum"] = a.get("n_authors_sum", 0) + r["n_authors"]
                a["max_repos"] = max(a.get("max_repos", 0), r["n_repos"])
                if len(a["contexts"]) < ctx_cap:
                    a["contexts"].extend(r["contexts"][: ctx_cap - len(a["contexts"])])
            else:
                a["df_pre"] += r["df"]

    def wmean(pairs: list[tuple[float, int]]) -> float:
        w = sum(n for _, n in pairs)
        return sum(v * n for v, n in pairs) / w if w else 0.0

    rows = []
    for t, a in agg.items():
        arts = a["artifacts"]
        tot_art = sum(arts.values()) or 1
        rows.append(
            {
                "term": t,
                "df_post": a["df_post"],
                "df_pre": a["df_pre"],
                # repositories summed over periods double-counts a repo that uses
                # the phrase in several months; reported as a spread indicator,
                # not a distinct count
                "repo_spread": a.get("n_repos_sum", 0),
                "author_spread": a.get("n_authors_sum", 0),
                "max_repos_in_period": a.get("max_repos", 0),
                "top_repo_share": wmean(a["top_share"]),
                "ai_repo_share": wmean(a["ai_share"]),
                # only meaningful with enough mid-sentence occurrences to estimate
                "capitalized_share": (a.get("cap_hit", 0) / a["cap_occ"]
                                      if a.get("cap_occ", 0) >= 8 else None),
                "doc_cohesion": cohesion(a["sigs"]),
                "artifact_entropy": _entropy([v / tot_art for v in arts.values()]),
                "dominant_artifact": max(arts, key=arts.get) if arts else "",
                "contexts": a["contexts"],
            }
        )
    return pl.DataFrame(rows)


def _entropy(ps: list[float]) -> float:
    """Normalised Shannon entropy of the artifact mix. Low means the expression
    lives in one kind of document, which is a topical signature more than a
    stylistic one."""
    ps = [p for p in ps if p > 0]
    if len(ps) <= 1:
        return 0.0
    h = -sum(p * np.log(p) for p in ps)
    return float(h / np.log(len(C.PROSE_ARTIFACTS)))


PROPER_NOUN = re.compile(r"^(?:[a-z]+\d|\d|v\d|cve|gh|pr)\b|\d{3,}")


def flag_confounders(
    em: pl.DataFrame,
    br: pl.DataFrame,
    max_top_repo_share: float = 0.25,
    max_ai_share: float = 0.5,
    min_repo_spread: int = 15,
    max_cohesion: float = 0.18,
) -> pl.DataFrame:
    """Attach confounder flags and a penalty multiplier.

    Nothing is deleted: a flagged expression stays in the atlas with its reason
    recorded, because the point of the atlas is to be inspectable. The penalty
    only affects ranking and the weight the text scorer gives it.
    """
    df = em.join(br, on="term", how="left")
    df = df.with_columns(
        [
            pl.col("top_repo_share").fill_null(1.0),
            pl.col("ai_repo_share").fill_null(0.0),
            pl.col("repo_spread").fill_null(0),
            pl.col("author_spread").fill_null(0),
            pl.col("artifact_entropy").fill_null(0.0),
        ]
        + ([pl.col("capitalized_share")] if "capitalized_share" in df.columns
           else [pl.lit(None, pl.Float64).alias("capitalized_share")])
        + ([pl.col("doc_cohesion")] if "doc_cohesion" in df.columns
           else [pl.lit(None, pl.Float64).alias("doc_cohesion")])
    )
    df = df.with_columns(
        flag_concentrated=pl.col("top_repo_share") > max_top_repo_share,
        flag_narrow=pl.col("repo_spread") < min_repo_spread,
        flag_ai_only=pl.col("ai_repo_share") > max_ai_share,
        flag_single_artifact=pl.col("artifact_entropy") < 0.35,
        # the documents containing it are near-copies of one another: this marks a
        # genre of document, not a way of writing
        flag_cohesive_docs=pl.col("doc_cohesion").fill_null(0.0) > max_cohesion,
        # a name, not a phrase: usually capitalised where English would not require it
        flag_proper_noun=(pl.col("capitalized_share").fill_null(0.0) > 0.6)
        | pl.col("term").map_elements(
            lambda t: bool(PROPER_NOUN.search(t)), return_dtype=pl.Boolean
        ),
    )
    # Artifact spread is graded rather than thresholded. An expression confined to
    # a single artifact type is a genre marker: the clearest case in this corpus is
    # market-research spam filed as issue bodies, which is broad across
    # repositories and authors and so defeats every other breadth test, but never
    # appears in a review comment or a reply. A binary flag with a mild penalty let
    # it through; the entropy itself is the more honest signal.
    artifact_factor = 0.25 + 0.75 * pl.col("artifact_entropy").clip(0, 1)
    penalty = (
        pl.when(pl.col("flag_concentrated")).then(0.45).otherwise(1.0)
        * pl.when(pl.col("flag_narrow")).then(0.35).otherwise(1.0)
        * pl.when(pl.col("flag_ai_only")).then(0.4).otherwise(1.0)
        * artifact_factor
        * pl.when(pl.col("flag_proper_noun")).then(0.35).otherwise(1.0)
        * pl.when(pl.col("flag_cohesive_docs")).then(0.2).otherwise(1.0)
    )
    return df.with_columns(confounder_penalty=penalty).with_columns(
        adj_score=pl.col("core_score") * pl.col("confounder_penalty")
    )

"""Detect declared AI assistance in raw GitHub text.

Some authors tell you. Coding agents append a trailer to the commits and pull
requests they help write -- ``Co-Authored-By: Claude <noreply@anthropic.com>``,
``🤖 Generated with [Claude Code]``, ``Co-authored-by: Copilot`` -- and that is a
*declared label* on text submitted under a human account. It is the only ground
truth this project has for the middle category: not a bot writing on its own, and
not unassisted human prose, but a person shipping model-written text as their own.

Two rules make this usable:

* Detection runs on the **raw** text, before cleaning. The cleaner strips these
  trailers deliberately, since counting them would just measure how many people
  use the tool rather than how anyone writes.
* The marker is then removed from the text and kept only as a column. If the
  trailer stayed in the document, every expression in an assisted commit would
  correlate perfectly with the marker and the label would be circular.

The label has a known bias and it is not small: it identifies people who *declare*
assistance, in tools whose default is to declare it. Undeclared assistance is
invisible here, so prevalence is a floor, never an estimate of how much AI-written
text is on GitHub.
"""

from __future__ import annotations

import re

# Ordered: the first match wins, so specific tools precede the generic trailer.
MARKERS: list[tuple[str, re.Pattern]] = [
    ("claude_code", re.compile(
        r"co-?authored-by:\s*claude|noreply@anthropic\.com"
        r"|generated\s+with\s+\[?claude\s+code|claude\.ai/code", re.I)),
    ("copilot", re.compile(
        r"co-?authored-by:\s*(?:github\s+)?copilot|copilot-swe-agent@", re.I)),
    ("codex", re.compile(
        r"co-?authored-by:\s*(?:openai\s+)?codex|chatgpt\.com/codex"
        r"|generated\s+with\s+\[?codex", re.I)),
    ("cursor", re.compile(r"co-?authored-by:\s*cursor(?:agent)?|cursoragent@", re.I)),
    ("devin", re.compile(r"co-?authored-by:\s*devin|devin-ai-integration", re.I)),
    ("aider", re.compile(r"co-?authored-by:\s*aider|aider\s*\(", re.I)),
    ("gemini", re.compile(
        r"co-?authored-by:\s*gemini|generated\s+with\s+\[?gemini", re.I)),
    ("windsurf", re.compile(r"co-?authored-by:\s*windsurf|codeium", re.I)),
    ("cline", re.compile(r"co-?authored-by:\s*(?:cline|roo\s*code)", re.I)),
    ("other_ai", re.compile(
        # named LLM tools only. An earlier version matched any co-author line
        # containing "ai" or "bot", which made this category 18x larger than
        # Claude Code's -- almost entirely renovate, dependabot, pre-commit-ci and
        # github-actions, none of which write prose.
        r"co-?authored-by:[^\n]*\b(?:coderabbitai|greptile|sourcery-ai|sweep-ai|"
        r"korbit|qodo|ellipsis-dev|cubic-dev-ai|codeant-ai|bito-ai|charliehelps|"
        r"tabnine|continue-dev|amazon-q|codewhisperer|junie|jules)\b"
        r"|generated\s+(?:with|by)\s+(?:an?\s+)?(?:ai|llm)\b", re.I)),
]

# Automation that co-authors commits without writing prose. Excluded from every
# assistance category so prevalence means what it says.
NON_LLM_BOT = re.compile(
    r"co-?authored-by:[^\n]*\b(?:renovate|dependabot|pre-commit-ci(?:-lite)?|"
    r"github-actions|ti-chi-bot|weblate|transifex|imgbot|allcontributors|"
    r"snyk-bot|mergify|semantic-release|whitesource|pyup-bot|scala-steward|"
    r"restyled-io|crowdin|lingohub)\b", re.I)

ANY = re.compile(r"co-?authored-by:|generated\s+(?:with|by)|🤖", re.I)


def detect(raw: str) -> str:
    """Return the declaring tool, or '' when nothing is declared.

    Cheap guard first: the vast majority of documents contain no trailer at all,
    and this runs on every document during ingest.
    """
    if not raw or not ANY.search(raw):
        return ""
    for name, pat in MARKERS:
        if pat.search(raw):
            return name
    return ""


def is_non_llm_bot_coauthor(raw: str) -> bool:
    """A co-author trailer naming automation that does not write prose."""
    return bool(raw) and bool(NON_LLM_BOT.search(raw))


def has_human_coauthor(raw: str) -> bool:
    """A Co-authored-by trailer that names no known AI tool.

    Kept separate so that ordinary human pair-programming trailers are not
    counted as assistance, and so the AI share can be expressed against all
    trailers rather than against all documents.
    """
    if not raw or "co-auth" not in raw.lower().replace("coauth", "co-auth"):
        return False
    return not detect(raw) and not is_non_llm_bot_coauthor(raw)


# ------------------------------------------------------------------ prevalence

def scan_slot(slot: tuple[int, int, int, int], max_bytes: int = 0,
              timeout: int = 90) -> dict:
    """Tally declared-assistance markers in one archive hour.

    Deliberately separate from document ingest, and deliberately not subject to
    the prose eligibility gate. Prevalence asks "what share of commits declare
    assistance", which must not depend on whether the commit message happens to
    be long enough or English enough to be worth counting n-grams in -- most
    commit messages are neither, and that is where the trailer lives.

    Streams and tallies; stores nothing.
    """
    import orjson

    from .ingest import _fetch, _iter_lines, _url

    y, m, d, h = slot
    try:
        blob = _fetch(_url(y, m, d, h), timeout=timeout, max_bytes=max_bytes)
    except Exception as ex:
        return {"period": f"{y:04d}-{m:02d}", "error": type(ex).__name__}

    totals: dict[str, int] = {}
    assist: dict[str, int] = {}
    coauth: dict[str, int] = {}

    def tally(kind: str, raw: str) -> None:
        if not raw:
            return
        totals[kind] = totals.get(kind, 0) + 1
        tool = detect(raw)
        if tool:
            assist[f"{kind}|{tool}"] = assist.get(f"{kind}|{tool}", 0) + 1
        elif is_non_llm_bot_coauthor(raw):
            assist[f"{kind}|non_llm_bot"] = assist.get(f"{kind}|non_llm_bot", 0) + 1
        elif has_human_coauthor(raw):
            coauth[kind] = coauth.get(kind, 0) + 1

    for line in _iter_lines(blob):
        if len(line) < 40:
            continue
        head = line[:120]
        if b'"type":"PushEvent"' in head:
            try:
                e = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            for cm in (e.get("payload") or {}).get("commits") or []:
                tally("commit", cm.get("message") or "")
        elif b'"type":"PullRequestEvent"' in head:
            try:
                e = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            p = e.get("payload") or {}
            if p.get("action") != "opened":
                continue
            pr = p.get("pull_request") or {}
            tally("pr", f"{pr.get('title') or ''}\n{pr.get('body') or ''}")
        elif b'"type":"IssuesEvent"' in head:
            try:
                e = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            p = e.get("payload") or {}
            if p.get("action") != "opened":
                continue
            iss = p.get("issue") or {}
            if iss.get("pull_request"):
                continue
            tally("issue", f"{iss.get('title') or ''}\n{iss.get('body') or ''}")

    return {"period": f"{y:04d}-{m:02d}", "slot": f"{y:04d}-{m:02d}-{d:02d}-{h:02d}",
            "totals": totals, "assist": assist, "human_coauthor": coauth,
            "bytes": len(blob)}


def scan(slots: list[tuple[int, int, int, int]], max_bytes: int = 0,
         workers: int = 6, log=print) -> "object":
    """Scan many hours and return a per-period prevalence table."""
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import polars as pl

    per: dict[str, dict] = defaultdict(
        lambda: {"totals": defaultdict(int), "assist": defaultdict(int),
                 "human_coauthor": defaultdict(int)})
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(scan_slot, s, max_bytes) for s in slots]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r.get("error"):
                continue
            acc = per[r["period"]]
            for k, v in r["totals"].items():
                acc["totals"][k] += v
            for k, v in r["assist"].items():
                acc["assist"][k] += v
            for k, v in r["human_coauthor"].items():
                acc["human_coauthor"][k] += v
            if done % 5 == 0 or done == len(slots):
                log(f"  {done}/{len(slots)} slots scanned")

    rows = []
    for period, acc in sorted(per.items()):
        row = {"period": period}
        for kind, n in acc["totals"].items():
            row[f"n_{kind}"] = n
        tools: dict[str, int] = defaultdict(int)
        for key, n in acc["assist"].items():
            kind, tool = key.split("|")
            row[f"{kind}_{tool}"] = n
            tools[tool] += n
        for tool, n in tools.items():
            row[f"total_{tool}"] = n
        for kind, n in acc["human_coauthor"].items():
            row[f"{kind}_human_coauthor"] = n
        rows.append(row)
    return pl.DataFrame(rows).fill_null(0).sort("period")

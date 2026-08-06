"""Reduce a GitHub artifact to the prose its author actually wrote.

Everything that is not free prose is stripped rather than counted: code blocks,
logs and tracebacks, URLs, quoted replies, HTML, and bot boilerplate. Each of
these has its own drift over time (CI formats change, bots get rewritten), and
leaving them in would produce "emerging expressions" that are really template
churn.

The module also reports *why* text was dropped, so ingestion can record the
proportion of a document that was code or log rather than silently deleting it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ------------------------------------------------------------------ patterns

FENCED = re.compile(r"(?:^|\n)\s*(?:```|~~~)[^\n]*\n.*?(?:\n\s*(?:```|~~~)|\Z)", re.S)
UNCLOSED_FENCE = re.compile(r"(?:^|\n)\s*(?:```|~~~).*\Z", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
DETAILS = re.compile(r"<details.*?</details>", re.S | re.I)
HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]{0,400})?/?>")
URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
INLINE_CODE = re.compile(r"`[^`\n]{1,200}`")
QUOTE_LINE = re.compile(r"^\s*>.*$", re.M)
MENTION = re.compile(r"(?<![\w/])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})\b")
ISSUE_REF = re.compile(r"(?<![\w])(?:[\w.-]+/[\w.-]+)?#\d+\b")
SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
HEXNUM = re.compile(r"\b0x[0-9a-fA-F]+\b")
LONG_NUM = re.compile(r"\b\d[\d.,:_-]{3,}\b")
CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*", re.M)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
HRULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.M)
HEADER_HASH = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
WS = re.compile(r"[ \t\u00a0]+")
NL = re.compile(r"\n{2,}")

# A line that looks like machine output rather than prose.
LOGLIKE = re.compile(
    r"""^\s*(?:
        (?:\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})            # leading timestamp
      | (?:\d{2}:\d{2}:\d{2}[.,]?\d*)                       # clock
      | (?:at\s+[\w.$<>]+\([^)]*\))                         # java/js stack frame
      | (?:File\s+"[^"]+",\s+line\s+\d+)                    # python frame
      | (?:Traceback\s*\(most\srecent\scall\slast\))
      | (?:\s*\#\d+\s+0x[0-9a-f]+)                          # gdb frame
      | (?:[A-Z]{0,8}(?:ERROR|WARN|WARNING|INFO|DEBUG|TRACE|FATAL)\b[:\]\s])
      | (?:[\w./-]+:\d+(?::\d+)?[:\s])                      # file:line:col
      | (?:\$\s+\S)                                         # shell prompt
      | (?:[+-]{3}\s+[ab]?/)                                # diff header
      | (?:@@\s*-\d)                                        # diff hunk
      | (?:\s*[|+-]{2,}\s*$)
      | (?:npm\s+ERR!|yarn\s+error|E:\s|W:\s)
    )""",
    re.X,
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*[._][A-Za-z0-9_.]+|[a-z]+[A-Z][A-Za-z0-9]*")
_PUNCT_HEAVY = re.compile(r"[{}();=<>\[\]/\\|]")

BOT_SUFFIXES = ("[bot]", "-bot", "bot", "-ci", "[robot]")
BOT_LOGINS = {
    "dependabot", "dependabot-preview", "renovate", "renovate-bot", "greenkeeper",
    "codecov", "coveralls", "snyk-bot", "github-actions", "mergify", "stale",
    "allcontributors", "imgbot", "pre-commit-ci", "sonarcloud", "netlify",
    "vercel", "azure-pipelines", "travis-ci", "circleci", "appveyor", "houndci-bot",
    "semantic-release-bot", "whitesource-bolt-for-github", "pyup-bot", "scala-steward",
    "gitpod-io", "deepsource-autofix", "restyled-io", "sourcery-ai", "codecov-commenter",
    "release-drafter", "changeset-bot", "cla-bot", "cla-assistant", "linux-foundation-easycla",
    "sweep-ai", "korbit-ai", "coderabbitai", "gemini-code-assist", "graphite-app",
    "claude", "devin-ai-integration", "copilot-pull-request-reviewer", "cursor",
    # GitHub's own review assistant posts as plain "copilot" with no [bot] suffix
    "copilot", "copilot-swe-agent", "chatgpt-codex-connector", "codex",
    "qodo-merge-pro", "pr-agent", "ellipsis-dev", "greptile-apps", "cubic-dev-ai",
}

# Boilerplate that bots and tools append to human-authored text. These lines are
# removed even from non-bot documents, because a human running a tool inherits
# its phrasing and we would otherwise score the tool, not the person.
BOILERPLATE = re.compile(
    r"""(?:
        ^.*generated\s+(?:with|by)\s+\[?(?:claude\s+code|cursor|copilot|devin|aider|codex).*$
      | ^\s*co-authored-by:.*$
      | ^\s*signed-off-by:.*$
      | ^.*\bthis\s+(?:pr|pull\s+request|issue|comment)\s+was\s+(?:auto|created\s+auto|generated).*$
      | ^\s*(?:bumps?|updates?)\s+\[?[\w./@-]+\]?\s+from\s+[\w.+-]+\s+to\s+[\w.+-]+.*$
      | ^.*\bdependabot\b.*$
      | ^\s*<!--.*-->\s*$
      | ^.*\bcodecov\b.*\breport\b.*$
      | ^\s*\*?\s*(?:merging|patch)\s+#\d+\s+into\s+\w+.*$
      | ^.*please\s+(?:review|check)\s+the\s+(?:following\s+)?(?:checklist|changes\s+below).*$
      | ^\s*(?:closes?|fixes|resolves)\s+#\d+\s*$
      | ^\s*🤖.*$
      | ^.*\bstale\b.*\b(?:inactivity|no\s+recent\s+activity)\b.*$
      | ^\s*\[!\w+\].*$
    )""",
    re.X | re.M | re.I,
)

# Function words used as a cheap English test. A language filter matters here:
# the non-English share of GitHub drifts over time and would otherwise move
# every frequency series.
EN_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "it", "this", "that", "for", "on", "with", "as", "but", "not", "have", "has",
    "do", "does", "did", "you", "i", "we", "they", "if", "when", "would", "should",
    "can", "could", "there", "and", "or", "so", "my", "me", "your", "what", "why",
}


@dataclass
class Cleaned:
    text: str
    n_tokens: int
    code_ratio: float  # share of original characters removed as code/log
    had_code: bool
    is_english: bool
    n_chars_raw: int


def _strip_loglike(text: str) -> tuple[str, int]:
    """Drop runs of machine-output lines. Isolated matches are kept, since prose
    legitimately mentions things like `foo.py:12`."""
    lines = text.split("\n")
    flags = [bool(LOGLIKE.match(ln)) or _is_codey(ln) for ln in lines]
    keep, removed, run = [], 0, 0
    for ln, f in zip(lines, flags):
        if f:
            run += 1
        else:
            run = 0
        keep.append(None if run >= 2 else ln)
        if run >= 2:
            removed += len(ln)
    # also remove the first line of any run of >= 2
    out = []
    for i, ln in enumerate(keep):
        if ln is not None and i + 1 < len(keep) and keep[i + 1] is None and flags[i]:
            removed += len(ln)
            continue
        if ln is not None:
            out.append(ln)
    return "\n".join(out), removed


def _is_codey(line: str) -> bool:
    s = line.strip()
    if len(s) < 3:
        return False
    if len(s) > 200 and " " not in s[:80]:
        return True
    punct = len(_PUNCT_HEAVY.findall(s))
    words = s.split()
    if not words:
        return False
    if punct >= 4 and punct / max(1, len(s)) > 0.06:
        return True
    if len(_IDENT.findall(s)) >= 3 and punct >= 2:
        return True
    return False


def clean(raw: str, max_chars: int = 20000) -> Cleaned:
    if not raw:
        return Cleaned("", 0, 0.0, False, False, 0)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    n_raw = len(raw)
    if n_raw > max_chars:
        raw = raw[:max_chars]

    t = raw
    before = len(t)
    t, n_fence = FENCED.subn(" \n", t)
    t = UNCLOSED_FENCE.sub(" \n", t)
    t = DETAILS.sub(" \n", t)
    t = HTML_COMMENT.sub(" ", t)
    code_removed = before - len(t)

    t = BOILERPLATE.sub(" ", t)
    t = QUOTE_LINE.sub(" ", t)
    t = TABLE_ROW.sub(" ", t)
    t = HRULE.sub(" ", t)
    t = MD_IMAGE.sub(" ", t)
    t = MD_LINK.sub(r"\1", t)
    t = URL.sub(" ", t)
    t = INLINE_CODE.sub(" ", t)
    t = HTML_TAG.sub(" ", t)
    t = CHECKBOX.sub(" ", t)
    t = HEADER_HASH.sub(" ", t)

    t, log_removed = _strip_loglike(t)
    code_removed += log_removed

    t = MENTION.sub(" ", t)
    t = ISSUE_REF.sub(" ", t)
    t = SHA.sub(" ", t)
    t = HEXNUM.sub(" ", t)
    t = LONG_NUM.sub(" ", t)

    t = WS.sub(" ", t)
    t = NL.sub("\n", t)
    t = "\n".join(ln.strip() for ln in t.split("\n"))
    t = t.strip()

    toks = t.lower().split()
    n_tok = len(toks)
    latin = sum(ch.isascii() and ch.isalpha() for ch in t)
    alpha = sum(ch.isalpha() for ch in t)
    latin_ratio = latin / alpha if alpha else 0.0
    head = {w.strip(".,!?;:'\"()") for w in toks[:60]}
    is_en = latin_ratio > 0.7 and len(head & EN_STOP) >= 2

    return Cleaned(
        text=t,
        n_tokens=n_tok,
        code_ratio=min(1.0, code_removed / max(1, n_raw)),
        had_code=n_fence > 0,
        is_english=is_en,
        n_chars_raw=n_raw,
    )


def is_bot_login(login: str | None) -> bool:
    if not login:
        return True
    lo = login.lower()
    if lo in BOT_LOGINS:
        return True
    return any(lo.endswith(s) for s in BOT_SUFFIXES)


def bot_expr(col: str = "author"):
    """The same test as `is_bot_login`, as a polars expression.

    Used to recompute the flag at analysis time. The ingested value is a snapshot
    of whatever the bot list contained on the day of download, and that list grows
    every time a new review tool appears -- so the stored column must never be
    trusted for filtering.
    """
    import polars as pl

    a = pl.col(col).str.to_lowercase()
    cond = a.is_null() | (a == "") | a.is_in(sorted(BOT_LOGINS))
    for s in BOT_SUFFIXES:
        cond = cond | a.str.ends_with(s)
    return cond


MIN_TOKENS = 8


def eligible(c: Cleaned) -> bool:
    """Whether a cleaned document counts toward the denominator.

    The same predicate must gate both numerator and denominator, or expression
    frequencies will drift with document-mix changes rather than language.
    """
    return c.n_tokens >= MIN_TOKENS and c.is_english and c.code_ratio < 0.9

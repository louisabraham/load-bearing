"""Stream GH Archive hour-files into cleaned document shards.

Raw archive files are never written to disk: each worker downloads one hour into
memory, filters lines with a byte-level prefilter before any JSON parsing, cleans
the surviving prose, and writes a small Parquet shard. This keeps peak disk in
the low hundreds of MB even though tens of GB flow through.

The archive has genuine gaps (404) and truncated files; both are tolerated, and
a missing slot falls back to a nearby hour so monthly volume stays stable.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import polars as pl

from . import config as C
from .textclean import clean, eligible, is_bot_login

UA = "lbdetect/0.1 (research; longitudinal language change)"
BASE = "https://data.gharchive.org"

# Byte prefilter: `type` is the second key of every line, so testing the head of
# the line avoids parsing the ~85% of events that carry no prose.
WANTED_TYPES = {
    b'"type":"IssuesEvent"': "issue",
    b'"type":"IssueCommentEvent"': "issue_comment",
    b'"type":"PullRequestEvent"': "pr",
    b'"type":"PullRequestReviewCommentEvent"': "pr_review_comment",
    b'"type":"PullRequestReviewEvent"': "pr_review",
    b'"type":"CommitCommentEvent"': "commit_comment",
    b'"type":"PushEvent"': "commit_msg",
}
def _probes(commit_sample: int) -> list[tuple[bytes, str]]:
    """PushEvent is 60-90% of the archive; probing for it when commit messages
    are not wanted would mean parsing those lines for nothing."""
    return [
        (k, v) for k, v in WANTED_TYPES.items()
        if v != "commit_msg" or commit_sample > 0
    ]

SCHEMA = {
    "doc_id": pl.Utf8,
    "ts": pl.Utf8,
    "period": pl.Utf8,
    "artifact": pl.Utf8,
    "repo": pl.Utf8,
    "repo_id": pl.Int64,
    "author": pl.Utf8,
    "is_bot": pl.Boolean,
    "n_tokens": pl.Int32,
    "code_ratio": pl.Float32,
    "text": pl.Utf8,
}


@dataclass
class ShardStats:
    slot: str
    ok: bool
    n_events: int = 0
    n_docs: int = 0
    bytes: int = 0
    error: str = ""


def _url(y: int, m: int, d: int, h: int) -> str:
    return f"{BASE}/{y:04d}-{m:02d}-{d:02d}-{h}.json.gz"


def _fetch(url: str, timeout: int = 90, max_bytes: int = 0) -> bytes:
    headers = {"User-Agent": UA}
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _iter_lines(blob: bytes):
    """Decompress incrementally and yield lines, tolerating truncation."""
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b""
    pos = 0
    step = 1 << 22
    while pos < len(blob):
        try:
            chunk = dec.decompress(blob[pos : pos + step])
        except zlib.error:
            break
        pos += step
        if not chunk:
            continue
        buf += chunk
        if b"\n" in buf:
            *lines, buf = buf.split(b"\n")
            yield from lines
    if buf:
        yield buf


def _extract(line: bytes, artifact: str, commit_sample: int) -> list[dict]:
    try:
        e = orjson.loads(line)
    except orjson.JSONDecodeError:
        return []
    p = e.get("payload") or {}
    action = p.get("action")
    repo = (e.get("repo") or {}).get("name") or ""
    repo_id = int((e.get("repo") or {}).get("id") or 0)
    actor = (e.get("actor") or {}).get("login")
    ts = e.get("created_at") or ""
    eid = str(e.get("id") or "")

    raws: list[tuple[str, str, str | None]] = []  # (artifact, raw_text, author)

    if artifact == "issue":
        if action != "opened":
            return []
        iss = p.get("issue") or {}
        if iss.get("pull_request"):
            return []  # PullRequestEvent covers these with a body
        raws.append(("issue", f"{iss.get('title') or ''}\n{iss.get('body') or ''}",
                     (iss.get("user") or {}).get("login") or actor))
    elif artifact == "issue_comment":
        if action != "created":
            return []
        c = p.get("comment") or {}
        iss = p.get("issue") or {}
        kind = "pr_comment" if iss.get("pull_request") else "issue_comment"
        raws.append((kind, c.get("body") or "", (c.get("user") or {}).get("login") or actor))
    elif artifact == "pr":
        if action != "opened":
            return []
        pr = p.get("pull_request") or {}
        raws.append(("pr", f"{pr.get('title') or ''}\n{pr.get('body') or ''}",
                     (pr.get("user") or {}).get("login") or actor))
    elif artifact in ("pr_review_comment", "commit_comment"):
        if action not in (None, "created"):
            return []
        c = p.get("comment") or {}
        raws.append((artifact, c.get("body") or "", (c.get("user") or {}).get("login") or actor))
    elif artifact == "pr_review":
        r = p.get("review") or {}
        if not r.get("body"):
            return []
        raws.append(("pr_review", r.get("body") or "", (r.get("user") or {}).get("login") or actor))
    elif artifact == "commit_msg":
        if commit_sample <= 0:
            return []
        commits = p.get("commits") or []
        if not commits:
            return []
        # deterministic subsample keyed on event id so pushes are sampled evenly
        try:
            if int(eid[-4:] or 0) % commit_sample != 0:
                return []
        except ValueError:
            return []
        for cm in commits[:3]:
            raws.append(("commit_msg", cm.get("message") or "",
                         (cm.get("author") or {}).get("name") or actor))

    out = []
    for i, (kind, raw, author) in enumerate(raws):
        if not raw or len(raw) < 12:
            continue
        cl = clean(raw)
        if not eligible(cl):
            continue
        out.append(
            {
                "doc_id": f"{eid}:{i}",
                "ts": ts,
                "period": ts[:7],
                "artifact": kind,
                "repo": repo,
                "repo_id": repo_id,
                "author": (author or "").lower(),
                "is_bot": is_bot_login(author),
                "n_tokens": cl.n_tokens,
                "code_ratio": cl.code_ratio,
                "text": cl.text[:6000],
            }
        )
    return out


def ingest_slot(slot: tuple[int, int, int, int], commit_sample: int = 0,
                allow_alternates: bool = True, max_bytes: int = 0,
                timeout: int = 90, slot_budget: float = 420.0) -> ShardStats:
    """Download one hour-file and write its shard. Returns stats, never raises."""
    y, m, d, h = slot
    tag = f"{y:04d}-{m:02d}-{d:02d}-{h:02d}"
    out_dir = C.DOCS / f"{y:04d}-{m:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.parquet"
    if out.exists():
        try:
            n = pl.scan_parquet(out).select(pl.len()).collect().item()
            return ShardStats(tag, True, n_docs=n)
        except Exception:
            out.unlink(missing_ok=True)

    candidates = [slot] + (C.alternates(*slot) if allow_alternates else [])
    blob, used, err = None, None, ""
    # A per-slot deadline as well as a per-request timeout. With only the latter,
    # eight fallback candidates at a 300s timeout let a single slot occupy a worker
    # for forty minutes when the connection degrades, which stalls the whole run
    # instead of skipping the slot and moving on.
    deadline = time.monotonic() + slot_budget
    for cand in candidates[:8]:
        if time.monotonic() > deadline:
            err = err or "slot budget exhausted"
            break
        try:
            blob = _fetch(_url(*cand), timeout=timeout, max_bytes=max_bytes)
            used = cand
            break
        except urllib.error.HTTPError as ex:
            err = f"http {ex.code}"
            if ex.code == 404:
                continue  # genuine archive gap: try the next candidate
        except Exception as ex:  # network hiccup, DNS, timeout
            err = f"{type(ex).__name__}"
    if blob is None:
        return ShardStats(tag, False, error=err or "unavailable")

    rows, n_events = [], 0
    probes = _probes(commit_sample)
    for line in _iter_lines(blob):
        if len(line) < 40:
            continue
        head = line[:120]
        for probe, artifact in probes:
            if probe in head:
                n_events += 1
                rows.extend(_extract(line, artifact, commit_sample))
                break

    df = pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)
    # pid in the temp name: two concurrent runs must not write the same scratch
    # file, or a reader can pick up interleaved bytes.
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=7)
    tmp.replace(out)
    return ShardStats(f"{used[0]:04d}-{used[1]:02d}-{used[2]:02d}-{used[3]:02d}",
                      True, n_events=n_events, n_docs=len(df), bytes=len(blob))


def _slots(plan: C.SamplingPlan, start, end) -> list[tuple[int, int, int, int]]:
    """Slots ordered breadth-first across months, not chronologically.

    Every month gets its first hour before any month gets its second, so an
    interrupted run still covers the whole timeline. Chronological order would
    spend the entire budget on the baseline years and leave the LLM era empty --
    the one arrangement that makes partial data useless here.
    """
    per_month = [C.sample_hours(y, m, plan.hours_for(y, m)) for (y, m) in C.months(start, end)]
    slots = []
    for rank in range(max((len(s) for s in per_month), default=0)):
        for month_slots in per_month:
            if rank < len(month_slots):
                slots.append(month_slots[rank])
    return slots


def run(start=C.START, end=C.END, plan: C.SamplingPlan = C.DEFAULT_PLAN,
        workers: int = 12, commit_sample: int = 0, log=print) -> dict:
    """Ingest the whole configured window. Resumable: existing shards are kept."""
    slots = _slots(plan, start, end)
    cap = f"{plan.max_bytes >> 20}MB/file" if plan.max_bytes else "whole files"
    log(f"ingest: {len(slots)} hour-slots, {workers} workers, {cap}, "
        f"commit_sample={commit_sample}")
    done = failed = docs = events = nbytes = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(ingest_slot, s, commit_sample, True, plan.max_bytes): s
                for s in slots}
        for fut in as_completed(futs):
            st = fut.result()
            done += 1
            if st.ok:
                docs += st.n_docs
                events += st.n_events
                nbytes += st.bytes
            else:
                failed += 1
            if done % 25 == 0 or done == len(slots):
                log(f"  {done}/{len(slots)} slots  docs={docs:,}  "
                    f"prose_events={events:,}  fetched={nbytes/1e9:.1f}GB  failed={failed}")
    return {"slots": len(slots), "docs": docs, "failed": failed, "bytes": nbytes}


def corpus() -> pl.LazyFrame:
    """All ingested documents as one lazy frame."""
    return pl.scan_parquet(str(C.DOCS / "*" / "*.parquet"))


def coverage() -> pl.DataFrame:
    """Documents per period and artifact -- the denominator sanity check."""
    return (
        corpus()
        .filter(~pl.col("is_bot"))
        .group_by("period")
        .agg(
            pl.len().alias("docs"),
            pl.col("n_tokens").sum().alias("tokens"),
            pl.col("repo_id").n_unique().alias("repos"),
            pl.col("author").n_unique().alias("authors"),
        )
        .sort("period")
        .collect()
    )

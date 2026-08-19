"""Sample GitHub pull request descriptions, one file per week.

Not GH Archive. Its feed has carried almost only `PushEvent` since mid-2025: a complete
hour of 2024-08-12 holds 13,555 `IssueCommentEvent` against 86 for the same hour of
2026-08-10, and polling `/events` today returns 97 `PushEvent` out of 100. The cause is
upstream, in GitHub's Events API -- its own tracker carries "Drastic Drop Off in Events
After 2025-05-23", open since July 2025 with no reply -- so every bulk mirror inherits it
and none can be repaired. GitHub has published no fix and no alternative.

The search API can be sampled cleanly instead, because of three properties together:

  * `created:` accepts timestamps, not just dates, so a window can be minutes wide;
  * a window that narrow holds few enough items to be enumerated rather than sampled;
  * the response carries the full body, so one request yields a hundred documents.

Every week gets the same number of randomly placed windows: uniform across the window, so
a difference between two weeks cannot come from having looked harder at one; random inside
the week, so the sample represents the whole week rather than a chosen slice of the clock.

    export GITHUB_TOKEN=$(gh auth token)
    python fetch_week.py          # fetch every week with no file yet
    python fetch_week.py --all    # backfill the window (~85 minutes)
"""

import argparse
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

ANCHOR = date(2024, 1, 1)          # a Monday
WINDOW_S = 300                     # seconds per window
WINDOWS_PER_WEEK = 5               # 137 x 5 requests at ~7/min is about 85 minutes
SLOTS = 7 * 24 * 60 * 60 // WINDOW_S
PER_PAGE = 100
PAUSE = 2.3                        # see the note in `fetch_week`
OUT = "data/weeks"
UA = "wordshift/1.0 (research; longitudinal language change)"

# Excluded in the query, one slug at a time. `-author:app/*` is rejected with a 422, so
# there is no way to say "no apps". These four are 90% of App-authored pull request bodies
# and cut a sample page from 29 bot documents to 1; adding five more changed the total by
# nothing. Other App accounts stay in on purpose -- some of the clearest agent-written
# prose on GitHub is App-authored, and dropping it makes the register diffuse.
EXCLUDE_APPS = ("pull", "dependabot", "renovate", "github-actions")

# The only way to exclude empty bodies, and 45% of pull requests have none. There is no
# emptiness qualifier: `-body:""` is a 422, `has:body` and `-in:body` are silently ignored
# and return the identical total, and `body:*` cuts 94% rather than the 45% that are empty
# because it is a text match on the asterisk. Requiring any one of a few function words in
# the body does the job exactly -- zero empty bodies in every era tested, 2024-02 to
# 2026-07, and a page goes from 43 usable documents to 97.
#
# The qualifier is repeated per term deliberately: `in:body` does NOT distribute over an OR
# group, so `(the OR a) in:body` matches titles and lets empty bodies back in.
PROSE_TERMS = ("the", "a", "to", "of", "and", "in", "is", "for", "that", "with")


def n_weeks(today=None):
    return ((today or date.today()) - ANCHOR).days // 7


def week_start(k):
    return ANCHOR + timedelta(days=7 * k)


def windows(k):
    """Start times of the windows sampled from week `k`.

    A shuffled slot list truncated to length N, rather than N independent draws, so that
    raising WINDOWS_PER_WEEK extends the sample instead of replacing it and a deeper run
    refetches nothing. Seeding on the week index makes it reproducible and independent
    between weeks.
    """
    slots = list(range(SLOTS))
    random.Random(k).shuffle(slots)
    start = datetime.combine(week_start(k), datetime.min.time(), timezone.utc)
    return [start + timedelta(seconds=s * WINDOW_S) for s in slots[:WINDOWS_PER_WEEK]]


def query(start):
    hi = start + timedelta(seconds=WINDOW_S)
    return (f"created:{start:%Y-%m-%dT%H:%M:%SZ}..{hi:%Y-%m-%dT%H:%M:%SZ} is:pr"
            + "".join(f" -author:app/{a}" for a in EXCLUDE_APPS)
            + " (" + " OR ".join(f"{t} in:body" for t in PROSE_TERMS) + ")")


def search(q, token):
    """One search request, with backoff on the secondary rate limit."""
    url = ("https://api.github.com/search/issues?advanced_search=true&sort=created"
           f"&order=asc&per_page={PER_PAGE}&q={urllib.parse.quote(q)}")
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as ex:
            if ex.code not in (403, 429):
                return None
            time.sleep(20 * (attempt + 1))
        except Exception:
            time.sleep(8)          # IncompleteRead happens on large responses
    return None


def fetch_week(k, token, log=print):
    """Fetch one week into `data/weeks/{date}.jsonl.gz`. Returns documents written.

    Serial, one request every PAUSE seconds. GitHub's REST guidance asks for serial
    requests per credential, and five threads stalled behind the secondary rate limit
    rather than going five times faster. A search query takes five to ten seconds, so the
    real ceiling is about seven requests a minute and the 30/min cap is never approached.

    A window holding more than one page is truncated to its earliest hundred items. That is
    not a bias in time: the placement is uniformly random, so what is sampled is still
    everything created in a uniformly random interval, its effective width varying as
    GitHub's volume does.
    """
    path = os.path.join(OUT, f"{week_start(k)}.jsonl.gz")
    if os.path.exists(path):
        return None

    rows, missed = [], 0
    for w in windows(k):
        data = search(query(w), token)
        time.sleep(PAUSE)
        if data is None:
            missed += 1
            continue
        for it in data.get("items", []):
            body = it.get("body") or ""
            rows.append({
                "ts": it.get("created_at") or "",
                # the search result names the repo only by API url; the last two
                # segments are owner and name
                "repo": "/".join((it.get("repository_url") or "").split("/")[-2:]),
                "author": ((it.get("user") or {}).get("login") or "").lower(),
                "body": body[:8000],
            })
    if missed:
        # not written, so a later run retries the whole week rather than leaving a thin one
        log(f"  {week_start(k)}  {missed} of {WINDOWS_PER_WEEK} windows failed, skipping")
        return 0

    os.makedirs(OUT, exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for row in sorted(rows, key=lambda r: r["ts"]):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    log(f"  {week_start(k)}  {len(rows)} documents")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true",
                    help="every week since 2024-01-01, not just the missing ones")
    ap.add_argument("--token", default="")
    args = ap.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        sys.exit("set GITHUB_TOKEN, e.g. export GITHUB_TOKEN=$(gh auth token)")

    total = n_weeks()
    todo = [k for k in range(total)
            if args.all or not os.path.exists(
                os.path.join(OUT, f"{week_start(k)}.jsonl.gz"))]
    have = total - len([k for k in todo
                        if not os.path.exists(
                            os.path.join(OUT, f"{week_start(k)}.jsonl.gz"))])
    print(f"{total} weeks since {ANCHOR}, {have} already on disk, "
          f"{len(todo)} to fetch (~{len(todo) * WINDOWS_PER_WEEK * PAUSE / 60:.0f} min)")

    written = 0
    for k in todo:
        got = fetch_week(k, token)
        if got:
            written += got
    print(f"wrote {written:,} documents")


if __name__ == "__main__":
    main()

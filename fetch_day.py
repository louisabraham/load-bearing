"""Fetch one day's sample of GitHub pull request descriptions. One request, one file.

Designed to run once a day from CI, so the corpus grows by a single append-only file that is
committed to the repository. A day's file is written once and never touched again.

Not from GH Archive. Since mid-2025 its feed carries almost only PushEvent -- a complete hour
of 2024-08-12 holds 13,555 IssueCommentEvent against 86 for the same hour of 2026-08-10, and
polling /events in August 2026 returns 97 PushEvent out of the 100 most recent. The cause is
upstream, in GitHub's Events API: its tracker carries "Drastic Drop Off in Events After
2025-05-23" (gharchive.org issue 310), open since July 2025 with no maintainer reply, and a
GitHub community discussion (178788) traces the same loss to "a GitHub Event API outage
propagated downstream" -- the identical gaps appear in OSSInsight, which reads the API
directly. So no mirror repairs it, and GitHub has published no fix.

What works is the search API, because `created:` accepts timestamps and not merely dates. A
window can therefore be minutes wide, narrow enough to enumerate rather than sample, and the
response carries the full body. One randomly placed five-minute window a day is one request.

    GITHUB_TOKEN=... python fetch_day.py              # yesterday, if not already there
    GITHUB_TOKEN=... python fetch_day.py 2026-08-20   # a particular day
    GITHUB_TOKEN=... python fetch_day.py --backfill 30
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

OUT = "data/days"
WINDOW_S = 300                      # seconds; ~100 pull requests in 2026, one page
SLOTS = 24 * 60 * 60 // WINDOW_S    # five-minute slots in a day
PER_PAGE = 100
UA = "load-bearing/1.0 (research; longitudinal language change)"

# Excluded in the query itself, one slug at a time -- `-author:app/*` is a 422, so there is no
# way to say "no apps". These four are 90% of App-authored pull request bodies and cut a page
# from 29 bot documents to 1. Other App accounts stay in on purpose: some of the clearest
# agent-written prose on GitHub is App-authored.
EXCLUDE_APPS = ("pull", "dependabot", "renovate", "github-actions")

# The only way to exclude empty bodies, and 45% of pull requests have none. There is no
# emptiness qualifier: `-body:""` is a 422, `has:body` and `-in:body` are silently ignored and
# return the identical total, and `body:*` cuts 94% rather than 45% because it is a text match
# on the asterisk. Requiring any one of a few function words in the body does it exactly --
# zero empty bodies in every era tested -- and takes a page from 43 usable documents to 97.
#
# The qualifier is repeated per term deliberately: `in:body` does NOT distribute over an OR
# group, so `(the OR a) in:body` matches titles and lets empty bodies back in.
PROSE_TERMS = ("the", "a", "to", "of", "and", "in", "is", "for", "that", "with")


def window(day):
    """The five-minute window sampled from `day`, seeded on the date so it is reproducible."""
    slot = random.Random(day.toordinal()).randrange(SLOTS)
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    return start + timedelta(seconds=slot * WINDOW_S)


def path(day):
    return os.path.join(OUT, f"{day.isoformat()}.jsonl.gz")


def search(start, token):
    """One search request. Backs off on the secondary rate limit; returns None on failure."""
    hi = start + timedelta(seconds=WINDOW_S)
    q = (f"created:{start:%Y-%m-%dT%H:%M:%SZ}..{hi:%Y-%m-%dT%H:%M:%SZ} is:pr"
         + "".join(f" -author:app/{a}" for a in EXCLUDE_APPS)
         + " (" + " OR ".join(f"{t} in:body" for t in PROSE_TERMS) + ")")
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
                print(f"  http {ex.code}", file=sys.stderr)
                return None
            time.sleep(20 * (attempt + 1))
        except Exception as ex:
            print(f"  {type(ex).__name__}, retrying", file=sys.stderr)
            time.sleep(8)
    return None


def fetch(day, token):
    """Write one day's file. Returns documents written, or None if it already exists.

    A window holding more than one page is truncated to its earliest hundred items. That is
    not a bias in time: the placement is uniformly random, so what is sampled is still
    everything created in a uniformly random interval, its effective width varying with
    GitHub's volume.
    """
    if os.path.exists(path(day)):
        return None
    data = search(window(day), token)
    if data is None:
        return -1
    rows = [{
        "ts": it.get("created_at") or "",
        # the search result names the repo only by API url; the last two segments are
        # owner and name
        "repo": "/".join((it.get("repository_url") or "").split("/")[-2:]),
        "author": ((it.get("user") or {}).get("login") or "").lower(),
        "body": (it.get("body") or "")[:8000],
    } for it in data.get("items", [])]

    os.makedirs(OUT, exist_ok=True)
    tmp = path(day) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for row in sorted(rows, key=lambda r: r["ts"]):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path(day))       # never leave a half-written day behind
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("day", nargs="?", help="YYYY-MM-DD; default yesterday, UTC")
    ap.add_argument("--backfill", type=int, default=0,
                    help="also fetch this many earlier days that are missing")
    ap.add_argument("--token", default="")
    args = ap.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        sys.exit("set GITHUB_TOKEN, e.g. export GITHUB_TOKEN=$(gh auth token)")

    # yesterday by default: today is still in progress, and a window drawn from the part of it
    # that has not happened yet would come back empty
    last = date.fromisoformat(args.day) if args.day \
        else datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [last - timedelta(days=i) for i in range(args.backfill + 1)]

    total = skipped = failed = 0
    for day in days:
        n = fetch(day, token)
        if n is None:
            skipped += 1
        elif n < 0:
            failed += 1
            print(f"{day}  failed")
        else:
            total += n
            print(f"{day}  {n} descriptions")
        if n is not None and len(days) > 1:
            time.sleep(2.5)          # 30 search requests a minute, authenticated
    print(f"wrote {total:,} descriptions; {skipped} already present, {failed} failed")
    return 1 if failed and not total else 0


if __name__ == "__main__":
    sys.exit(main())

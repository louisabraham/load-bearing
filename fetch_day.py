"""Fetch one day's sample of GitHub pull request descriptions. Ten requests, one file.

Designed to run once a day from CI, so the corpus grows by a single append-only file that is
committed to the repository. A day's file is written once and never touched again, and is left
uncompressed so that it can be read and grepped in place -- about 1.4 MB a day.

Not from GH Archive. Since mid-2025 its feed carries almost only PushEvent -- a complete hour
of 2024-08-12 holds 13,555 IssueCommentEvent against 86 for the same hour of 2026-08-10, and
polling /events in August 2026 returns 97 PushEvent out of the 100 most recent. The cause is
upstream, in GitHub's Events API: its tracker carries "Drastic Drop Off in Events After
2025-05-23" (gharchive.org issue 310), open since July 2025 with no maintainer reply, and a
GitHub community discussion (178788) traces the same loss to "a GitHub Event API outage
propagated downstream" -- the identical gaps appear in OSSInsight, which reads the API
directly. So no mirror repairs it, and GitHub has published no fix.

What works is the search API, because `created:` accepts timestamps and not merely dates. A
window can therefore be minutes wide, and the response carries the full body. Ten five-minute
windows a day, one drawn from each 2.4 hours of it, are ten requests and about a thousand
descriptions.

That is a sample and not a census, and the numbers say so plainly: a day of 2026 holds some
460,000 pull requests matching the query, a five-minute window of it about 1,250, and a page is
a hundred. Enumeration is not on the table at any width -- the search API returns at most 1,000
results for a query however many matched -- so the design is to place the windows honestly
rather than to pretend to completeness.

    GITHUB_TOKEN=... python fetch_day.py              # yesterday, if not already there
    GITHUB_TOKEN=... python fetch_day.py 2026-08-20   # a particular day
    GITHUB_TOKEN=... python fetch_day.py --backfill 30
    GITHUB_TOKEN=... python fetch_day.py --backfill 999 --refetch   # every day, to full depth
"""

import argparse
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
# The corpus begins here, and a sweep that would walk back past it stops instead. Extending the
# corpus backwards is a separate decision from filling out the days already in it: the first week
# of 2025 is the one every trend on the page is measured from.
START = date(2025, 1, 1)
WINDOW_S = 300  # seconds a window spans; ~1,250 pull requests in 2026, ~130 in early 2025
WINDOWS = 10  # windows a day: ten pages, about a thousand descriptions
BLOCK_S = 24 * 60 * 60 // WINDOWS  # 8,640 -- the 2.4 hours each window is drawn from
MIN_GAP_S = 60 / 28  # a floor under the gap between requests; the bucket allows thirty a minute
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


def windows(day):
    """The windows sampled from `day`, seeded on the date so the corpus follows from its dates.

    One window per equal block of the day, its start drawn to the second inside that block.
    Stratified rather than drawn freely across the whole day, for two reasons.

    The grid was the more serious one. Starts used to be multiples of five minutes, which is
    exactly the granularity a cron schedule fires on, so every window opened on one of the 288
    instants at which scheduled automation opens pull requests -- and since a window is
    truncated to its first hundred, the sample was the twenty-five seconds after such an
    instant. A start drawn to the second cannot align with anything.

    The other is arithmetic: ten free draws from 288 slots collide on about one day in seven,
    and a collision is a window sampled twice and a day short of its thousand. Blocks make that
    impossible, and spread the ten windows across the day's hours rather than letting them clump.

    A block is 2.4 hours and a window five minutes, and the draw leaves the window strictly
    inside its block, so no two windows of a day overlap and no pull request is sampled twice.
    """
    r = random.Random(day.toordinal())
    midnight = datetime.combine(day, datetime.min.time(), timezone.utc)
    return [
        midnight + timedelta(seconds=b * BLOCK_S + r.randrange(BLOCK_S - WINDOW_S))
        for b in range(WINDOWS)
    ]


def path(day):
    return os.path.join(OUT, f"{day.isoformat()}.jsonl")


def thin(p):
    """Whether the file at `p` came from the sampler that took one window a day.

    Nothing is stamped in a day's file and nothing needs to be: one window is one page, so a
    file of more than a page can only have come from the sampler that takes ten. That makes a
    sweep of the corpus restartable -- the days it has already deepened are skipped on the way
    back through -- which matters when the sweep is hours long. A genuinely tiny day would be
    read as thin and collected twice; it costs ten requests and writes the same rows.
    """
    with open(p, encoding="utf-8") as fh:
        return sum(1 for _ in fh) <= PER_PAGE


_last_request = 0.0


def pace():
    """Hold the request rate under the search bucket's ceiling, and no lower than it has to.

    One of these requests takes about four and a half seconds to come back -- the query carries
    ten `in:body` terms and a sort -- which is already half the ceiling's rate, so across a sweep
    this sleeps for nothing at all. It is here for the day the API gets quick, and it is a floor
    under the gap rather than a fixed sleep after each request: a fixed one would be hours of tax
    across the corpus for a limit that is nowhere near being reached.
    """
    global _last_request
    wait = MIN_GAP_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


_bucket_said = False


def bucket(headers):
    """Say what the search bucket allows, once a run, from the response's own headers.

    Worth a line in the log because the documentation does not settle it: the search endpoints
    carry a ceiling of thirty a minute, and the token CI hands the job carries a per-repository
    one of its own, and which of the two is in force decides how long a sweep of the whole
    corpus takes. The header is the answer, and it costs nothing to read.
    """
    global _bucket_said
    if _bucket_said:
        return
    _bucket_said = True
    print(
        f"search bucket: limit {headers.get('x-ratelimit-limit')},"
        f" {headers.get('x-ratelimit-remaining')} left,"
        f" {MIN_GAP_S:.1f}s the floor between requests",
        file=sys.stderr,
        flush=True,
    )


def full(data):
    """Whether a 200 actually carried the page it says it carried.

    `incomplete_results` is the API saying the query timed out and the items are a part of the
    answer; taking them would be a thin window indistinguishable from a real one. It is the only
    thing checked here, and deliberately so.

    A short page is NOT that. Search drops items it cannot resolve, so 94 of 100 is ordinary,
    and a window can be genuinely empty: two days in this corpus hold 900 descriptions because
    one of their ten windows returns nothing, and re-querying says the same. Those are holes in
    GitHub's index rather than failures of the request -- on 2025-11-02 the whole 08:00 hour
    holds four matching pull requests against 62,498 for the day, and on 2026-07-24 the five
    minutes from 19:36 hold none inside an hour holding 7,568. A day is written short and says
    so, which is the honest record of what could be read.
    """
    if data.get("incomplete_results"):
        print(f"  {len(data.get('items', []))} items, incomplete, retrying", file=sys.stderr)
        return False
    return True


def search(start, token):
    """One search request. Backs off on the secondary rate limit; returns None on failure."""
    hi = start + timedelta(seconds=WINDOW_S)
    q = (
        f"created:{start:%Y-%m-%dT%H:%M:%SZ}..{hi:%Y-%m-%dT%H:%M:%SZ} is:pr"
        + "".join(f" -author:app/{a}" for a in EXCLUDE_APPS)
        + " ("
        + " OR ".join(f"{t} in:body" for t in PROSE_TERMS)
        + ")"
    )
    url = (
        "https://api.github.com/search/issues?advanced_search=true&sort=created"
        f"&order=asc&per_page={PER_PAGE}&q={urllib.parse.quote(q)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        },
    )
    for attempt in range(4):
        try:
            pace()
            with urllib.request.urlopen(req, timeout=90) as r:
                bucket(r.headers)
                data = json.loads(r.read())
        except urllib.error.HTTPError as ex:
            if ex.code not in (403, 429):
                print(f"  http {ex.code}", file=sys.stderr)
                return None
            time.sleep(20 * (attempt + 1))
            continue
        except Exception as ex:
            print(f"  {type(ex).__name__}, retrying", file=sys.stderr)
            time.sleep(8)
            continue
        if full(data):
            return data
        time.sleep(8)
    return None


def fetch(day, token):
    """Write one day's file from its ten windows. Returns documents written, or -1 on failure.

    A day is written whole or not at all. A file short of a window would be a thin day that
    reads as a full one, and nothing downstream could tell the two apart -- where absent, it is
    simply collected again by the next run, which is what the corpus already expects of a day
    that failed.

    A window holding more than one page is truncated to its earliest hundred items. That is
    not a bias in time: the placement is uniformly random, so what is sampled is still
    everything created in ten uniformly random intervals, their effective width varying with
    GitHub's volume.
    """
    rows = []
    for start in windows(day):
        data = search(start, token)
        if data is None:
            return -1
        rows += [
            {
                "ts": it.get("created_at") or "",
                # the search result names the repo only by API url; the last two segments are
                # owner and name
                "repo": "/".join((it.get("repository_url") or "").split("/")[-2:]),
                "author": ((it.get("user") or {}).get("login") or "").lower(),
                "body": (it.get("body") or "")[:8000],
            }
            for it in data.get("items", [])
        ]

    os.makedirs(OUT, exist_ok=True)
    # the pid is in the name so that two sweeps over different stretches of the corpus cannot
    # interleave their writes into one temporary file where they meet
    tmp = f"{path(day)}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in sorted(rows, key=lambda r: r["ts"])
        )
    os.replace(tmp, path(day))  # never leave a half-written day behind
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("day", nargs="?", help="YYYY-MM-DD; default yesterday, UTC")
    ap.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="also fetch this many earlier days that are missing",
    )
    ap.add_argument(
        "--refetch",
        action="store_true",
        help="also rewrite days present but collected at one window, before the depth was ten",
    )
    ap.add_argument("--token", default="")
    args = ap.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        sys.exit("set GITHUB_TOKEN, e.g. export GITHUB_TOKEN=$(gh auth token)")

    # yesterday by default: today is still in progress, and a window drawn from the part of it
    # that has not happened yet would come back empty
    last = (
        date.fromisoformat(args.day)
        if args.day
        else datetime.now(timezone.utc).date() - timedelta(days=1)
    )
    days = [d for d in (last - timedelta(days=i) for i in range(args.backfill + 1)) if d >= START]

    total = skipped = failed = 0
    for day in days:
        if os.path.exists(path(day)) and not (args.refetch and thin(path(day))):
            skipped += 1
            continue
        n = fetch(day, token)
        if n < 0:
            failed += 1
            print(f"{day}  failed", flush=True)
        else:
            total += n
            print(f"{day}  {n} descriptions", flush=True)
    print(f"wrote {total:,} descriptions; {skipped} already present, {failed} failed")
    return 1 if failed and not total else 0


if __name__ == "__main__":
    sys.exit(main())

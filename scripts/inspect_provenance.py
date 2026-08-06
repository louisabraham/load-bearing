"""Print the actual trailer lines that matched, to check for false positives.

A prevalence number is worthless without knowing what the pattern caught.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

import orjson

from lbdetect.ingest import _fetch, _iter_lines, _url
from lbdetect.provenance import ANY, MARKERS, detect

TRAILER = re.compile(r"^.*co-?authored-by:.*$|^.*generated\s+(?:with|by).*$",
                     re.I | re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="2025-06-12-14")
    ap.add_argument("--max-mb", type=int, default=6)
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()
    y, m, d, h = (int(x) for x in a.slot.split("-"))

    blob = _fetch(_url(y, m, d, h), max_bytes=a.max_mb << 20)
    by_label: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    n = 0
    for line in _iter_lines(blob):
        if b'"type":"PushEvent"' not in line[:120]:
            continue
        try:
            e = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        for cm in (e.get("payload") or {}).get("commits") or []:
            msg = cm.get("message") or ""
            n += 1
            if not ANY.search(msg):
                continue
            label = detect(msg) or "unlabelled_trailer"
            for t in TRAILER.findall(msg):
                by_label[label][t.strip()[:110]] += 1

    print(f"{n:,} commit messages scanned\n")
    for label in sorted(by_label, key=lambda k: -sum(by_label[k].values())):
        total = sum(by_label[label].values())
        print(f"=== {label}: {total} trailer lines")
        for txt, c in by_label[label].most_common(a.show):
            print(f"  {c:>5}  {txt}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

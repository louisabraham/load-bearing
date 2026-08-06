"""Entry point for the ingest run.

A real module rather than `python -c`, because macOS spawns worker processes and
they must be able to import the parent module.
"""

from __future__ import annotations

import argparse
import sys
import time

from lbdetect import config as C
from lbdetect import ingest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--commit-sample", type=int, default=0,
                    help="keep 1 in N push events as commit messages (0 = skip)")
    ap.add_argument("--start", default=f"{C.START[0]}-{C.START[1]:02d}")
    ap.add_argument("--end", default=f"{C.END[0]}-{C.END[1]:02d}")
    ap.add_argument("--pilot", action="store_true",
                    help="small byte-capped run for validating the pipeline")
    ap.add_argument("--baseline-hours", type=int)
    ap.add_argument("--era-hours", type=int)
    ap.add_argument("--sparse-hours", type=int)
    ap.add_argument("--max-mb", type=int, help="cap bytes read per hour-file (0 = whole file)")
    a = ap.parse_args()

    def parse(s: str) -> tuple[int, int]:
        y, m = s.split("-")
        return int(y), int(m)

    base = C.PILOT_PLAN if a.pilot else C.DEFAULT_PLAN
    plan = C.SamplingPlan(
        baseline_hours=a.baseline_hours if a.baseline_hours is not None else base.baseline_hours,
        era_hours=a.era_hours if a.era_hours is not None else base.era_hours,
        sparse_hours=a.sparse_hours if a.sparse_hours is not None else base.sparse_hours,
        max_bytes=(a.max_mb << 20) if a.max_mb is not None else base.max_bytes,
    )
    t = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t:7.1f}s] {msg}", flush=True)

    res = ingest.run(parse(a.start), parse(a.end), plan,
                     workers=a.workers, commit_sample=a.commit_sample, log=log)
    log(f"DONE {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

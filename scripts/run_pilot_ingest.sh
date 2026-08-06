#!/usr/bin/env bash
# Pilot ingest: fill the months that are missing, concentrating the byte budget
# on the LLM era. The 2018-01..2020-01 baseline is already ingested at full
# depth, so nothing is re-bought there.
#
# ~133 hour-slots x 16MB = ~2.1GB. Sensitivity lands around 1e-3 document
# frequency; the full plan (scripts/run_ingest.py with no cap) reaches ~1e-5.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W="${WORKERS:-10}"
MB="${MAX_MB:-16}"

# thin coverage across the remaining baseline, just enough to avoid a gap
$PY scripts/run_ingest.py --start 2020-02 --end 2021-12 \
    --baseline-hours 1 --era-hours 1 --max-mb "$MB" --workers "$W"

# the window where emergence is expected: twice the depth
$PY scripts/run_ingest.py --start 2022-01 --end 2026-07 \
    --era-hours 2 --sparse-hours 2 --max-mb "$MB" --workers "$W"

echo "PILOT INGEST COMPLETE"

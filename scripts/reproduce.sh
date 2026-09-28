#!/usr/bin/env bash
# TwinBench reproduction helpers. Run from repository root.
set -euo pipefail
PY="${PYTHON:-python}"
STEP="${1:-tables}"

case "$STEP" in
  tables)
    PYTHONUTF8=1 "$PY" analysis/generate_tables.py --write
    PYTHONUTF8=1 "$PY" analysis/verify_paper_numbers.py
    ;;
  figures)
    PYTHONUTF8=1 "$PY" analysis/figures/make_all.py
    ;;
  tests)
    PYTHONUTF8=1 "$PY" -m pytest tests -q
    ;;
  all)
    PYTHONUTF8=1 "$PY" -m pytest tests -q
    PYTHONUTF8=1 "$PY" analysis/verify_paper_numbers.py
    PYTHONUTF8=1 "$PY" analysis/figures/make_all.py
    ;;
  *)
    echo "Usage: $0 {tables|figures|tests|all}" >&2
    exit 2
    ;;
esac

"""Shared pytest fixtures + path bootstrap for the TwinBench test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SKIP_PINOCCHIO: bool = os.environ.get("SKIP_PINOCCHIO", "") == "1"

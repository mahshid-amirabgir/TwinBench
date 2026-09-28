"""
make_all.py — rebuild every manuscript figure in one command
==================================================================
Runs every generator under analysis/figures/ in sequence, each as a fresh
subprocess (so one figure's failure doesn't take down the rest, and each
gets a clean matplotlib state). All read only committed data/*.csv -- no
new simulation.

Usage (run from the repository root):
    python analysis/figures/make_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

GENERATORS = [
    "make_F1_pipeline.py",
    "make_F3_trajectories.py",
    "make_F7_kendall.py",
    "make_figA_ranking_bars.py",
    "make_figB_tracking.py",
    "make_methodology_threshold.py",
    "make_S1_workspace_feasibility.py",   # requires data/workspace_feasibility.csv
    "make_S2_uncertainty_degradation.py",
    "make_mismatch_robustness.py",
]


def main():
    failures = []
    for name in GENERATORS:
        script = HERE / name
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        if not script.exists():
            print(f"  [SKIP] {script} does not exist yet")
            failures.append(name)
            continue
        result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  [FAIL] exit code {result.returncode}")
            failures.append(name)

    print(f"\n{'='*70}")
    if failures:
        print(f"DONE with {len(failures)} failure(s)/skip(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"DONE — all {len(GENERATORS)} figures regenerated.")


if __name__ == "__main__":
    main()

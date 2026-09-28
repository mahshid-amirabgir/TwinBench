"""Generate workspace feasibility figure."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROBOT_ORDER = ["kr6", "rs006l", "mpx1950", "ur5e", "ur10e", "tx60"]
LABEL = {"kr6": "KR6", "rs006l": "RS006L", "mpx1950": "MPX1950",
         "ur5e": "UR5e", "ur10e": "UR10e", "tx60": "TX60"}
SCENE_LABEL = {"open_reach": "Open Reach", "shelf_reach": "Shelf Reach"}
SCENE_COLOR = {"open_reach": "#3b82f6", "shelf_reach": "#f59e0b"}


def main():
    df = pd.read_csv(ROOT / "data" / "workspace_feasibility.csv")
    n_points = {"open_reach": int(df[df.scene == "open_reach"]["n_points"].iloc[0]),
                "shelf_reach": int(df[df.scene == "shelf_reach"]["n_points"].iloc[0])}
    seeds = int(df["seeds"].iloc[0])
    ik_restarts = int(df["ik_restarts"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ROBOT_ORDER))
    width = 0.35
    for i, scene in enumerate(["open_reach", "shelf_reach"]):
        vals = [df[(df.robot == r) & (df.scene == scene)]["pct_feasible"].iloc[0] for r in ROBOT_ORDER]
        offs = [xi + (i - 0.5) * width for xi in x]
        ax.bar(offs, vals, width, label=SCENE_LABEL[scene], color=SCENE_COLOR[scene])
        for xi, v in zip(offs, vals):
            ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[r] for r in ROBOT_ORDER])
    ax.set_ylabel("Workspace feasibility (%)")
    ax.set_ylim(0, 108)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    min_row = df.loc[df.pct_feasible.idxmin()]

    def describe_at_max(df, max_pct):
        """Group every (robot, scene) cell at exactly max_pct into a phrase
        like 'UR5e and UR10e in both scenes, and RS006L in Shelf Reach' --
        built from the data every time, not a hardcoded robot list, so it
        can't silently go stale if a future re-run changes which cells hit
        the maximum."""
        at_max = df[df.pct_feasible == max_pct]
        by_robot = at_max.groupby("robot")["scene"].apply(set)
        both_scenes = sorted([r for r, s in by_robot.items() if len(s) == 2],
                             key=ROBOT_ORDER.index)
        parts = []
        if both_scenes:
            names = " and ".join(LABEL[r] for r in both_scenes)
            parts.append(f"{names} in both scenes")
        for r, s in by_robot.items():
            if len(s) == 1:
                scene = next(iter(s))
                parts.append(f"{LABEL[r]} in {SCENE_LABEL[scene]}")
        return ", and ".join(parts) if parts else "no cell"

    max_pct = float(df.pct_feasible.max())
    max_desc = describe_at_max(df, max_pct)

    ax.set_title(
        f"Workspace feasibility per (robot, scene) — {n_points['open_reach']:,} candidate "
        f"targets/robot in Open Reach, {n_points['shelf_reach']:,} in Shelf Reach "
        f"({seeds} seeds, {ik_restarts} IK restarts)\n"
        f"Range: {min_row.pct_feasible:.1f}% ({LABEL[min_row.robot]}, {SCENE_LABEL[min_row.scene]}) "
        f"to {max_pct:.0f}% ({max_desc})", fontsize=10)

    fig.tight_layout()
    out = ROOT / "figures" / "S1_workspace_feasibility.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"SAVED {out}")
    print(df.sort_values("pct_feasible").to_string(index=False))
    print(f"\nTrue minimum: {LABEL[min_row.robot]} / {SCENE_LABEL[min_row.scene]} = {min_row.pct_feasible:.3f}%")
    print(f"True maximum: {max_pct:.1f}% ({max_desc})")


if __name__ == "__main__":
    main()

"""Generate uncertainty-degradation figure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ORDER = ["pdg", "ctc_smc", "ctc_pd", "pbc"]
LABEL = {"pdg": "PD+g", "ctc_smc": "CTC-SMC", "ctc_pd": "CTC-PD", "pbc": "PBC"}
COLOR = {"pdg": "#3b82f6", "ctc_pd": "#ef4444", "ctc_smc": "#a855f7", "pbc": "#f59e0b"}
PARAMS = ["light", "nominal", "heavy"]


def main():
    df = pd.read_csv(ROOT / "data" / "all_results_datasheet.csv")
    df = df[df.ik_ok.astype(bool)]

    means = df.groupby(["controller", "params"])["RMSE_cart_norm"].mean().unstack()[PARAMS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for ctrl in ORDER:
        ax.plot(PARAMS, means.loc[ctrl, PARAMS], "o-", color=COLOR[ctrl],
                label=LABEL[ctrl], linewidth=2, markersize=7)
    ax.set_ylabel(r"$RMSE_{norm}$")
    ax.set_title("(a) Absolute tracking error by uncertainty level")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    x = np.arange(len(ORDER))
    width = 0.35
    light_pct = [(means.loc[c, "light"] - means.loc[c, "nominal"]) / means.loc[c, "nominal"] * 100
                for c in ORDER]
    heavy_pct = [(means.loc[c, "heavy"] - means.loc[c, "nominal"]) / means.loc[c, "nominal"] * 100
                for c in ORDER]
    ax2.bar(x - width / 2, light_pct, width, label="light vs. nominal", color="#93c5fd")
    ax2.bar(x + width / 2, heavy_pct, width, label="heavy vs. nominal", color="#1e40af")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL[c] for c in ORDER])
    ax2.set_ylabel("% increase in RMSE_norm relative to nominal")
    ax2.set_title("(b) Relative degradation from the exact-model (nominal) cell")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Non-uniform controller degradation under parametric mismatch "
                 "(datasheet robots, proxy tier)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "figures" / "S2_uncertainty_degradation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"SAVED {out}")

    print("\nAbsolute means:")
    print(means.round(4).to_string())
    print("\nRelative to nominal (%):")
    for c in ORDER:
        print(f"  {LABEL[c]:8s} light {light_pct[ORDER.index(c)]:+6.1f}%   "
              f"heavy {heavy_pct[ORDER.index(c)]:+6.1f}%")


if __name__ == "__main__":
    main()

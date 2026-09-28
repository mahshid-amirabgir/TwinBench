"""
make_mismatch_robustness.py — TwinBench
========================================
Candidate figure for the parametric mismatch ablation (tab:mismatch, Section
"Parametric mismatch ablation"): mean RMSE_cart_norm vs. link-mass mismatch
magnitude, one line per controller, with the same 95% CI the manuscript table
reports (x_bar +/- t5*s/sqrt(6) across the six realization means, t5=2.571).

This is NOT currently one of the manuscript's numbered figures -- Table 9
(tab:mismatch) is the only place this result appears in the paper; the numbers reproduced here match that
table exactly (same computation as analysis/mismatch_summary.py).

Usage (run from the repository root):
    python analysis/figures/make_mismatch_robustness.py
Output: figures/figC_mismatch_robustness.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T5 = 2.571  # t-critical, 5 df, two-sided 95% -- matches Table 9's CI construction

ORDER = ["pdg", "ctc_pd", "ctc_smc", "pbc"]
COL = {"pdg": "#3b82f6", "ctc_pd": "#ef4444", "ctc_smc": "#a855f7", "pbc": "#f59e0b"}
LAB = {"pdg": "PD+g", "ctc_pd": "CTC-PD", "ctc_smc": "CTC-SMC", "pbc": "PBC"}

df = pd.read_csv("data/d1_mismatch_all.csv")
pcts = sorted(df.pct.unique())

# realization mean per (pct, controller) = mean over (robot, trajectory, params, seed);
# then mean +/- t5*sd/sqrt(6) across the 6 realization means -- identical to
# See mismatch_summary.py and Table 9.
means = {c: [] for c in ORDER}
los = {c: [] for c in ORDER}
his = {c: [] for c in ORDER}
for pct in pcts:
    d = df[df.pct == pct]
    rm = d.groupby(["realization", "controller"]).RMSE_cart_norm.mean().unstack()[ORDER]
    n = len(rm)
    mu = rm.mean()
    ci = T5 * rm.std(ddof=1) / np.sqrt(n)
    for c in ORDER:
        means[c].append(mu[c])
        los[c].append(mu[c] - ci[c])
        his[c].append(mu[c] + ci[c])

x = [p * 100 for p in pcts]

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for c in ORDER:
    ax.plot(x, means[c], "-o", color=COL[c], label=LAB[c], linewidth=1.8, markersize=5)
    ax.fill_between(x, los[c], his[c], color=COL[c], alpha=0.18, linewidth=0)

ax.set_xlabel("Link-mass mismatch magnitude (%)")
ax.set_ylabel(r"Mean RMSE$_{cart,norm}$")
ax.set_xticks(x)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9)
fig.suptitle("Controller robustness to parametric mismatch\n(shaded band: 95% CI, "
             r"$t_5$ over 6 realizations)", fontsize=10.5)
fig.tight_layout()

out = "figures/figC_mismatch_robustness.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("SAVED", out)

for i, pct in enumerate(pcts):
    row = "  ".join(f"{LAB[c]}={means[c][i]:.4f}[{los[c][i]:.4f},{his[c][i]:.4f}]" for c in ORDER)
    print(f"+/-{int(pct*100)}%: {row}")

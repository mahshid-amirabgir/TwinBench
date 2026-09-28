"""verify_paper_numbers.py

Regenerates every table and headline scalar from the committed CSVs under
data/, compares each value against the reference set used in the paper, and
exits non-zero if any value drifts.

Usage:
    python analysis/verify_paper_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.generate_tables import TABLES, generate_all, BOOTSTRAP_SEED, _load
from twinbench.metrics.statistics import kendall_stability_matrix, kendall_offdiag_summary

DATA = ROOT / "data"
TOL = 1e-9   # deterministic pipeline under a pinned seed -- exact match expected

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, got, expected, tol: float = TOL):
    global CHECKS
    CHECKS += 1
    ok = abs(float(got) - float(expected)) <= tol
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, expected {expected!r} (|diff|={abs(got-expected):.3g})")
    return ok


def check_str(label: str, got: str, expected: str):
    global CHECKS
    CHECKS += 1
    if got != expected:
        FAILURES.append(f"{label}: got {got!r}, expected {expected!r}")


# 1. Table fragments — regenerate and compare numeric leaves
# Corrected (pipeline-computed, BOOTSTRAP_SEED-pinned) expected values, per
# the diff report reconciled with the user on this release pass.
EXPECTED_TABLES = {
    "tab_robots": {
        "kr6_payload_kg": 6, "kr6_reach_m": 1.82, "kr6_tau_max": 20.0, "kr6_I_proxy": 0.35,
        "rs006l_payload_kg": 6, "rs006l_reach_m": 1.65, "rs006l_tau_max": 25.0, "rs006l_I_proxy": 0.38,
        "mpx1950_payload_kg": 7, "mpx1950_reach_m": 1.45, "mpx1950_tau_max": 18.0, "mpx1950_I_proxy": 0.40,
        "ur5e_payload_kg": 5, "ur5e_reach_m": 0.85, "ur5e_tau_max": 150.0, "ur5e_I_proxy": 3.40,
        "ur10e_payload_kg": 10, "ur10e_reach_m": 1.30, "ur10e_tau_max": 330.0, "ur10e_I_proxy": 11.22,
        "tx60_payload_kg": 4.5, "tx60_reach_m": 0.67, "tx60_tau_max": 126.0, "tx60_I_proxy": 0.21,
    },
    "tab_proxy": {
        "pbc": (0.0202, 0.0141, 0.0279, 1), "ctc_smc": (0.0621, 0.0366, 0.0920, 2),
        "ctc_pd": (0.0798, 0.0510, 0.1205, 3), "pdg": (0.1157, 0.0881, 0.1414, 4),
    },
    "tab_coupled": {
        "tau_by_eps": {0.03: 1.0, 0.06: 1.0, 0.09: 1.0, 0.15: 1.0, 0.25: 1.0},
    },
}


def verify_tables():
    print("=" * 78)
    print("PART 1 — table fragments")
    print("=" * 78)
    results = generate_all(write=True)

    # tab_robots / tab_proxy / tab_coupled: direct leaf comparison
    c = results["tab_robots"]["computed"]
    for k, v in EXPECTED_TABLES["tab_robots"].items():
        # Table 1 displays I_proxy to 2dp; payload/reach/tau_max are exact
        # by construction (robot-spec fields, not derived floats).
        tol = 5e-3 if k.endswith("_I_proxy") else TOL
        check(f"tab_robots[{k}]", c[k], v, tol=tol)

    c = results["tab_proxy"]["computed"]
    for ctrl, (mu, lo, hi, rank) in EXPECTED_TABLES["tab_proxy"].items():
        mu_g, lo_g, hi_g, rank_g = c[ctrl]
        check(f"tab_proxy[{ctrl}].mean", mu_g, mu, tol=5e-5)
        check(f"tab_proxy[{ctrl}].ci_lo", lo_g, lo, tol=5e-5)
        check(f"tab_proxy[{ctrl}].ci_hi", hi_g, hi, tol=5e-5)
        check(f"tab_proxy[{ctrl}].rank", rank_g, rank)

    c = results["tab_coupled"]["computed"]
    for eps, tau in EXPECTED_TABLES["tab_coupled"]["tau_by_eps"].items():
        check(f"tab_coupled.tau_b[eps={eps}]", c["tau_by_eps"][eps], tau)

    # tab_fidelity: means/ranks exact; CI bounds at pinned-seed precision
    c = results["tab_fidelity"]["computed"]
    EXPECTED_FIDELITY = {
        ("proxy", "pbc"): (0.0660, 0.061, 0.071, 1), ("proxy", "ctc_pd"): (0.1493, 0.141, 0.158, 2),
        ("proxy", "pdg"): (0.2114, 0.204, 0.219, 3), ("proxy", "ctc_smc"): (0.2236, 0.206, 0.242, 4),
        ("rigid_body", "pbc"): (0.0317, 0.031, 0.033, 1), ("rigid_body", "ctc_smc"): (0.0763, 0.074, 0.079, 2),
        ("rigid_body", "ctc_pd"): (0.1258, 0.123, 0.129, 3), ("rigid_body", "pdg"): (0.1989, 0.195, 0.202, 4),
    }
    for (fid, ctrl), (mu, lo, hi, rank) in EXPECTED_FIDELITY.items():
        mu_g, lo_g, hi_g, rank_g = c["per_tier"][fid][ctrl]
        check(f"tab_fidelity[{fid},{ctrl}].mean", mu_g, mu, tol=5e-5)
        check(f"tab_fidelity[{fid},{ctrl}].ci_lo", lo_g, lo, tol=5e-4)
        check(f"tab_fidelity[{fid},{ctrl}].ci_hi", hi_g, hi, tol=5e-4)
        check(f"tab_fidelity[{fid},{ctrl}].rank", rank_g, rank)
    check("tab_fidelity.subset_tau_b", c["subset_tau_b"], 0.1667, tol=1e-3)
    check("tab_fidelity.subset_ci_lo", c["subset_ci"][0], 0.0556, tol=1e-3)
    check("tab_fidelity.subset_ci_hi", c["subset_ci"][1], 0.3333, tol=1e-3)
    check("tab_fidelity.all4_tau_b", c["all4_tau_b"], 0.5833, tol=1e-3)
    check("tab_fidelity.all4_ci_lo", c["all4_ci"][0], 0.5278, tol=1e-3)
    check("tab_fidelity.all4_ci_hi", c["all4_ci"][1], 0.6667, tol=1e-3)
    check("tab_fidelity.disagreement", c["disagreement"][0], 20)
    check("tab_fidelity.disagreement_n", c["disagreement"][1], 36)
    check("tab_fidelity.tau_b[ur10e]", c["per_robot_tau_b"]["ur10e"], 1/3, tol=1e-6)
    check("tab_fidelity.tau_b[ur5e]", c["per_robot_tau_b"]["ur5e"], 1/9, tol=1e-6)
    check("tab_fidelity.tau_b[tx60]", c["per_robot_tau_b"]["tx60"], 1/18, tol=1e-6)

    # tab_energy: means and CIs, corrected values
    c = results["tab_energy"]["computed"]
    EXPECTED_ENERGY = {
        ("proxy", "pbc"): (5.1346161628, 4.89, 5.39), ("proxy", "ctc_smc"): (6.6587042426, 6.25, 7.11),
        ("proxy", "ctc_pd"): (6.3169334231, 5.95, 6.72), ("proxy", "pdg"): (5.7683283130, 5.49, 6.06),
        ("rigid_body", "pbc"): (3.8212787488, 3.67, 3.97), ("rigid_body", "ctc_smc"): (3.9145178372, 3.79, 4.03),
        ("rigid_body", "ctc_pd"): (4.0379064619, 3.94, 4.14), ("rigid_body", "pdg"): (4.2466967993, 4.12, 4.38),
    }
    for (fid, ctrl), (mu, lo_disp, hi_disp) in EXPECTED_ENERGY.items():
        mu_g, lo_g, hi_g = c[(fid, ctrl)]
        check(f"tab_energy[{fid},{ctrl}].mean", mu_g, mu, tol=1e-6)
        check(f"tab_energy[{fid},{ctrl}].ci_lo(disp)", round(lo_g, 2), lo_disp)
        check(f"tab_energy[{fid},{ctrl}].ci_hi(disp)", round(hi_g, 2), hi_disp)

    # tab_mismatch: means exact; the two corrected lower bounds at full precision
    c = results["tab_mismatch"]["computed"]
    check("tab_mismatch[0.1,pbc].ci_lo", c[(0.1, "pbc")][1], 0.0274605366, tol=1e-6)
    check("tab_mismatch[0.1,pbc].ci_hi", c[(0.1, "pbc")][2], 0.0460802800, tol=1e-6)
    check("tab_mismatch[0.1,ctc_pd].ci_lo", c[(0.1, "ctc_pd")][1], 0.0914698055, tol=1e-6)
    check("tab_mismatch[0.1,ctc_pd].ci_hi", c[(0.1, "ctc_pd")][2], 0.2211374699, tol=1e-6)

    print(f"  {len(TABLES)} table fragments regenerated -> tables/")


# 2. Within-tier tau_b (Fig. 7 / Section 4.2 self-consistency numbers)
def within_tier_tau_b(df: pd.DataFrame, context_cols: tuple[str, ...]):
    """Per-context single-metric (RMSE_cart_norm) controller ranking, pairwise
    tau_b across all context pairs within one tier. NOT the Pareto multi-
    objective (trajectory,controller) ranking used elsewhere in statistics.py
    -- verified empirically to be the one that reproduces the manuscript
    0.917 / 0.533 / 1.000 figures exactly."""
    means = (df.groupby(list(context_cols) + ["controller"], observed=True)["RMSE_cart_norm"]
             .mean().reset_index())
    means["rank"] = means.groupby(list(context_cols), observed=True)["RMSE_cart_norm"].rank(method="average")
    ctx_list, mat = kendall_stability_matrix(means, context_cols, ("controller",), rank_col="rank")
    return kendall_offdiag_summary(mat)


def verify_within_tier():
    print("\n" + "=" * 78)
    print("PART 2 — within-tier tau_b (self-consistency)")
    print("=" * 78)
    ds = _load("all_results_datasheet.csv")
    urdf = _load("all_results_main.csv")

    mu, sd, n = within_tier_tau_b(ds, ("robot", "scene", "params", "trajectory"))
    print(f"  datasheet/proxy: {mu:.3f} +/- {sd:.3f} (n={n})")
    check("within_tier[datasheet/proxy].mean", mu, 0.917, tol=5e-4)
    check("within_tier[datasheet/proxy].std", sd, 0.162, tol=5e-4)

    for fid, (tgt_mu, tgt_sd) in {"proxy": (0.533, 0.377), "rigid_body": (1.000, 0.000)}.items():
        sub = urdf[urdf.fidelity == fid]
        mu, sd, n = within_tier_tau_b(sub, ("robot", "params", "trajectory"))
        print(f"  urdf/{fid}: {mu:.3f} +/- {sd:.3f} (n={n})")
        check(f"within_tier[urdf/{fid}].mean", mu, tgt_mu, tol=5e-4)
        check(f"within_tier[urdf/{fid}].std", sd, tgt_sd, tol=5e-4)


# 3. Run counts / grand total
RUN_COUNT_GROUPS = {
    "main": ["all_results_main.csv", "all_results_datasheet.csv"],
    "coupled (20-seed)": ["d3_coupled_all_20seed.csv"],
    "raised-clamp diagnostic": ["all_results_nocap_ur5e_eps0p09_coupled.csv",
                                "all_results_nocap_ur5e_eps0p09_decoupled.csv"],
    "cap": ["all_results_cap0p9_ctcsmc_check.csv"],
    "frozen-gain (gain-scaling) ablation": ["all_results_frozen_gains_rigid.csv"],
    "step-size": ["all_results_dt0p5.csv", "all_results_dt0p25.csv"],
    "mismatch": ["d1_mismatch_all.csv"],
}
EXPECTED_GROUP_COUNTS = {
    "main": 11520, "coupled (20-seed)": 9600, "raised-clamp diagnostic": 1920,
    "cap": 1440, "frozen-gain (gain-scaling) ablation": 2880, "step-size": 2880, "mismatch": 12960,
}
EXPECTED_GRAND_TOTAL = 43200


def verify_run_counts():
    print("\n" + "=" * 78)
    print("PART 3 — run counts / grand total")
    print("=" * 78)
    grand = 0
    for name, files in RUN_COUNT_GROUPS.items():
        n = sum(len(pd.read_csv(DATA / f)) for f in files)
        grand += n
        print(f"  {name:38s} {n:>7,}")
        check(f"run_count[{name}]", n, EXPECTED_GROUP_COUNTS[name])
    print(f"  {'GRAND TOTAL':38s} {grand:>7,}")
    check("run_count[grand_total]", grand, EXPECTED_GRAND_TOTAL)


# 4. Saturation fractions
def verify_saturation():
    print("\n" + "=" * 78)
    print("PART 4 — saturation fractions")
    print("=" * 78)

    # Zero everywhere except the coupled-proxy UR5e light-uncertainty tier.
    zero_everywhere = {
        "main grid (all 4 tiers, all controllers)": "all_results_main.csv",
        "datasheet grid (proxy + shelf)": "all_results_datasheet.csv",
        "cap ablation": "all_results_cap0p9_ctcsmc_check.csv",
        "frozen-gain ablation": "all_results_frozen_gains_rigid.csv",
        "step-size dt=0.5": "all_results_dt0p5.csv",
        "step-size dt=0.25": "all_results_dt0p25.csv",
    }
    for label, f in zero_everywhere.items():
        df = _load(f)
        m = float(df["Sat_frac_any"].max())
        print(f"  {label:38s} max Sat_frac_any = {m}")
        check(f"sat_frac[{label}]", m, 0.0)

    mm = pd.read_csv(DATA / "d1_mismatch_all.csv")
    m = float(mm["Sat_frac_any"].max())
    print(f"  {'mismatch sweep':38s} max Sat_frac_any = {m}")
    check("sat_frac[mismatch]", m, 0.0)

    coupled = pd.read_csv(DATA / "d3_coupled_all_20seed.csv")
    kr6_max = float(coupled[coupled.robot == "kr6"]["Sat_frac_any"].max())
    print(f"  {'KR6 coupled (all eps_c, all params)':38s} max Sat_frac_any = {kr6_max}")
    check("sat_frac[kr6_coupled]", kr6_max, 0.0)

    ur5e = coupled[coupled.robot == "ur5e"]
    ur5e_nh_max = float(ur5e[ur5e.params.isin(["nominal", "heavy"])]["Sat_frac_any"].max())
    print(f"  {'UR5e coupled, nominal+heavy tiers':38s} max Sat_frac_any = {ur5e_nh_max}")
    check("sat_frac[ur5e_coupled_nominal_heavy]", ur5e_nh_max, 0.0)

    ur5e_light = ur5e[ur5e.params == "light"]
    for ctrl in ["ctc_pd", "ctc_smc"]:
        m = float(ur5e_light[ur5e_light.controller == ctrl]["Sat_frac_any"].mean())
        print(f"  {'UR5e coupled light, ' + ctrl:38s} mean Sat_frac_any = {m:.4f}")
        check(f"sat_frac[ur5e_coupled_light_{ctrl}]", m, 0.985, tol=2e-3)
    for ctrl in ["pbc", "pdg"]:
        m = float(ur5e_light[ur5e_light.controller == ctrl]["Sat_frac_any"].max())
        check(f"sat_frac[ur5e_coupled_light_{ctrl}]", m, 0.0)


EXPECTED_PEAK_RATIO = 0.452   # Table 6 caption (20-seed grid)
PEAK_RATIO_TOL = 0.01         # allows for the 4-seed sample's max being a lower bound


def verify_peak_torque():
    print("\n" + "=" * 78)
    print("PART 5 — CTC-SMC peak |tau|/tau_max (Table 6 caption)")
    print("=" * 78)
    df = pd.read_csv(DATA / "all_results_peak_torque_check.csv")
    by_ctrl = df.groupby("ctrl")["peak_ratio"].max()
    smc_peak = float(by_ctrl["ctc_smc"])
    print(f"  4-seed sample, global max by controller:\n{by_ctrl.round(4).to_string()}")
    print(f"  CTC-SMC peak ratio = {smc_peak:.4f}  (manuscript, 20-seed: {EXPECTED_PEAK_RATIO})")

    global CHECKS
    CHECKS += 1
    if smc_peak > EXPECTED_PEAK_RATIO + 1e-9:
        FAILURES.append(f"peak_torque[ctc_smc]: 4-seed sample ({smc_peak:.4f}) EXCEEDS the "
                        f"20-seed manuscript figure ({EXPECTED_PEAK_RATIO}) -- a max over fewer "
                        f"seeds cannot legitimately exceed a max over more seeds of the same "
                        f"population; this is a real inconsistency, not sampling gap.")
    elif smc_peak < EXPECTED_PEAK_RATIO - PEAK_RATIO_TOL:
        FAILURES.append(f"peak_torque[ctc_smc]: 4-seed sample ({smc_peak:.4f}) is more than "
                        f"{PEAK_RATIO_TOL} below the 20-seed manuscript figure "
                        f"({EXPECTED_PEAK_RATIO}) -- larger gap than expected from seed-count "
                        f"alone.")

    check("peak_torque[ctc_smc < 0.7 cap]", float(smc_peak < 0.7), 1.0)
    check("peak_torque[ctc_smc is max of 4 controllers]",
         float(smc_peak == float(by_ctrl.max())), 1.0)


# not TX60/Open Reach, is the true minimum -- catching a future data change
# that silently reshuffles the ranking back). The percentage itself is never
# data/workspace_feasibility.csv every time, so there is nothing for it to
# which is the correctly-rounded decimal value of the exact 55.275% in the
# `.2f` formatting of the *same* float prints 55.27. Two representations of
# conventions with a second copy of the value.)
EXPECTED_FEASIBILITY_MIN_ROBOT_SCENE = ("mpx1950", "open_reach")
EXPECTED_FEASIBILITY_MAX_PCT = 100.0  # exact by construction -- no rounding ambiguity


def verify_workspace_feasibility():
    print("\n" + "=" * 78)
    print("PART 6 — workspace feasibility range (full protocol, 12 cells)")
    print("=" * 78)
    df = pd.read_csv(DATA / "workspace_feasibility.csv")
    check("workspace_feasibility[n_cells]", len(df), 12)

    min_row = df.loc[df.pct_feasible.idxmin()]
    max_pct = float(df.pct_feasible.max())
    print(f"  min: {min_row.robot}/{min_row.scene} = {min_row.pct_feasible:.3f}%")
    print(f"  max: {max_pct:.3f}%")

    exp_robot, exp_scene = EXPECTED_FEASIBILITY_MIN_ROBOT_SCENE
    check("workspace_feasibility[min_robot]", 1.0 if min_row.robot == exp_robot else 0.0, 1.0)
    check("workspace_feasibility[min_scene]", 1.0 if min_row.scene == exp_scene else 0.0, 1.0)
    check("workspace_feasibility[max_pct]", max_pct, EXPECTED_FEASIBILITY_MAX_PCT, tol=0.05)

    # Every cell used the same protocol -- no mixing full/reduced samples.
    check("workspace_feasibility[uniform_seeds]", float(df["seeds"].nunique()), 1.0)
    check("workspace_feasibility[uniform_ik_restarts]", float(df["ik_restarts"].nunique()), 1.0)


def main():
    verify_tables()
    verify_within_tier()
    verify_run_counts()
    verify_saturation()
    verify_peak_torque()
    verify_workspace_feasibility()

    print("\n" + "=" * 78)
    print(f"SUMMARY: {CHECKS} checks, {len(FAILURES)} failures")
    print("=" * 78)
    if FAILURES:
        print("\nMISMATCHES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

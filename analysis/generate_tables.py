"""generate_tables.py

Build every numeric table from the committed CSVs in data/ and optionally
write LaTeX fragments under tables/.

Usage:
    python analysis/generate_tables.py
    python analysis/generate_tables.py --write
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, t as student_t

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from twinbench.metrics.statistics import (
    bootstrap_ci_mean, cluster_bootstrap_tau_b, compute_per_context_cross_fidelity_tau,
)
from twinbench.dynamics.robot_loader import (
    load_robot_spec, robot_spec_path_from_id, get_dynamics_cfg_for_params,
)

DATA = ROOT / "data"
OUT = ROOT / "tables"

# Single RNG seed for every bootstrap CI in this file (cluster-bootstrap-of-mean,
# cluster-bootstrap tau_b, and plain per-run bootstrap alike). Pinned so that
# re-running this script reproduces byte-identical .tex fragments; this is the
# Bootstrap seed for all reported confidence intervals.
BOOTSTRAP_SEED = 20260406

CTRL_ORDER_RANK = ["pbc", "ctc_smc", "ctc_pd", "pdg"]   # display order used by most tables
SUBSET = ["ctc_pd", "ctc_smc", "pdg"]                    # model-based subset (excludes PBC)
ABBR = {"ctc_pd": "PD", "ctc_smc": "SM", "pdg": "Pg"}


# Small shared helpers
def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name)
    if "ik_ok" in df.columns:
        df = df[df["ik_ok"].astype(bool)].copy()
    return df


def cluster_bootstrap_mean(df: pd.DataFrame, value_col: str, cluster_col: str = "robot",
                            B: int = 2000, seed: int = BOOTSTRAP_SEED):
    """Hierarchical bootstrap: resample clusters (robots) with replacement, pool
    all rows of the chosen clusters, take the mean. Same resampling scheme as
    twinbench.metrics.statistics.cluster_bootstrap_tau_b, generalized from tau_b
    to a raw metric mean (used by tab:proxy). With only 3 clusters the result is
    the percentile of a small (<=10-point) discrete distribution and is not
    sensitive to the seed; a fixed seed is kept for exact reproducibility."""
    rng = np.random.default_rng(seed)
    clusters = sorted(df[cluster_col].unique())
    vals = {c: df.loc[df[cluster_col] == c, value_col].dropna().to_numpy() for c in clusters}
    mu = float(df[value_col].mean())
    boot = np.empty(B)
    for b in range(B):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        boot[b] = np.concatenate([vals[c] for c in chosen]).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mu, float(lo), float(hi)


def top_controller_disagreement(per_ctx_stable: pd.DataFrame, ctrls=SUBSET) -> tuple[int, int]:
    """Count contexts where the top-ranked (min-RMSE) controller among `ctrls`
    differs between the proxy_* and rb_* columns of a per-context frame built
    by compute_per_context_cross_fidelity_tau."""
    n_dis = 0
    for _, row in per_ctx_stable.iterrows():
        proxy_top = min(ctrls, key=lambda c: row[f"proxy_{c}"])
        rb_top = min(ctrls, key=lambda c: row[f"rb_{c}"])
        if proxy_top != rb_top:
            n_dis += 1
    return n_dis, len(per_ctx_stable)


def subset_rank_string(df: pd.DataFrame, fidelity: str, ctrls=SUBSET) -> str:
    means = (df[df.fidelity == fidelity].groupby("controller")["RMSE_cart_norm"].mean()
             .reindex(ctrls).sort_values())
    return " $>$ ".join(ABBR[c] for c in means.index)


def cross_fidelity_block(urdf: pd.DataFrame, B: int = 2000):
    """The full 36-context cross-fidelity τ_b protocol, shared by tab:fidelity,
    tab:cap, and tab:frozen. `urdf` must have columns robot/params/trajectory/
    controller/fidelity/RMSE_cart_norm with exactly two fidelity levels."""
    per_ctx_all = compute_per_context_cross_fidelity_tau(urdf, metric="RMSE_cart_norm")
    per_ctx_sub = compute_per_context_cross_fidelity_tau(
        urdf, metric="RMSE_cart_norm", exclude_controllers=["pbc"])
    cr_all = cluster_bootstrap_tau_b(per_ctx_all, B=B, rng=np.random.default_rng(BOOTSTRAP_SEED))
    cr_sub = cluster_bootstrap_tau_b(per_ctx_sub, B=B, rng=np.random.default_rng(BOOTSTRAP_SEED))
    n_dis, n_ctx = top_controller_disagreement(per_ctx_sub)
    per_robot = {
        robot: float(per_ctx_sub[per_ctx_sub.robot == robot]["tau_b"].mean())
        for robot in sorted(per_ctx_sub.robot.unique())
    }
    return {
        "subset_tau_b": cr_sub["point_estimate"], "subset_ci": (cr_sub["ci95_lo"], cr_sub["ci95_hi"]),
        "all4_tau_b": cr_all["point_estimate"], "all4_ci": (cr_all["ci95_lo"], cr_all["ci95_hi"]),
        "disagreement": (n_dis, n_ctx),
        "per_robot_tau_b": per_robot,
        "proxy_rank": subset_rank_string(urdf, "proxy"),
        "rigid_rank": subset_rank_string(urdf, "rigid_body"),
    }


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def fmt_ci(mu, lo, hi, nd=4):
    return f"{fmt(mu, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


# Table 1 — tab:robots
MANUFACTURER = {
    "kr6": "KUKA", "rs006l": "Kawasaki", "mpx1950": "Yaskawa",
    "ur5e": "UR", "ur10e": "UR", "tx60": "St\\\"aubli",
}
DISPLAY_NAME = {
    "kr6": "KR6 R1820", "rs006l": "RS006L", "mpx1950": "MPX1950",
    "ur5e": "UR5e", "ur10e": "UR10e", "tx60": "TX60",
}
SOURCE_TAG = {  # DS = datasheet-derived proxy robot, URDF = derived from public URDF
    "kr6": "DS", "rs006l": "DS", "mpx1950": "DS",
    "ur5e": "URDF", "ur10e": "URDF", "tx60": "URDF",
}
ROBOT_ORDER = ["kr6", "rs006l", "mpx1950", "ur5e", "ur10e", "tx60"]


def table_robots():
    import yaml
    rows = []
    for rid in ROBOT_ORDER:
        spec = load_robot_spec(robot_spec_path_from_id(rid, repo_root=ROOT))
        dyn = get_dynamics_cfg_for_params(spec, "nominal")
        with open(ROOT / "robot_specs" / f"{rid}.yaml") as f:
            raw = yaml.safe_load(f)
        rated_payload = raw["dynamics"]["rated_payload"]
        rows.append({
            "robot": rid, "name": DISPLAY_NAME[rid], "manufacturer": MANUFACTURER[rid],
            "payload_kg": float(rated_payload), "reach_m": float(dyn["Rmax"]),
            "tau_max": float(dyn["tau_max"][0]), "I_proxy": float(dyn["I_proxy"][0]),
            "source": SOURCE_TAG[rid],
        })
    df = pd.DataFrame(rows)

    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Robot & Manufacturer & Payload & Reach & $\tau_{\max}$ & $I_{\mathrm{proxy}}$ & Source \\",
             r"      &              & (kg)    & (m)   & (N\,m)        & (kg\,m$^2$)          &        \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r['name']} & {r['manufacturer']} & {r['payload_kg']:g} & "
                     f"{r['reach_m']:.2f} & {r['tau_max']:g} & {r['I_proxy']:.2f} & {r['source']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    computed = {f"{rid}_{col}": df.set_index('robot').loc[rid, col]
                for rid in ROBOT_ORDER for col in ["payload_kg", "reach_m", "tau_max", "I_proxy"]}
    return computed, "\n".join(lines)


# Table 3 — tab:proxy (datasheet robots, proxy fidelity)
def table_proxy():
    df = _load("all_results_datasheet.csv")
    computed = {}
    rows_out = []
    means = df.groupby("controller")["RMSE_cart_norm"].mean().sort_values()
    for rank, (ctrl, mu) in enumerate(means.items(), start=1):
        _, lo, hi = cluster_bootstrap_mean(df[df.controller == ctrl], "RMSE_cart_norm", seed=BOOTSTRAP_SEED)
        computed[ctrl] = (mu, lo, hi, rank)
        rows_out.append((ctrl, mu, lo, hi, rank))

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Controller & $\rmsen$ & 95\% CI & Rank \\", r"\midrule"]
    label = {"pbc": "PBC", "ctc_smc": "CTC-SMC", "ctc_pd": "CTC-PD", "pdg": "PD+g"}
    for ctrl, mu, lo, hi, rank in rows_out:
        lines.append(f"{label[ctrl]:7s} & {fmt(mu)} & [{fmt(lo)}, {fmt(hi)}] & {rank} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 4 — tab:fidelity
def table_fidelity():
    df = _load("all_results_main.csv")
    urdf = df[df.robot_group == "urdf"]

    per_tier = {}
    for fid in ["proxy", "rigid_body"]:
        sub = urdf[urdf.fidelity == fid]
        means = sub.groupby("controller")["RMSE_cart_norm"].mean().sort_values()
        tier_rows = {}
        for rank, (ctrl, mu) in enumerate(means.items(), start=1):
            # Plain (non-cluster) per-run bootstrap here -- unlike tab:proxy's
            # RMSE CI and every tau_b CI in this file, tab:fidelity's RMSE CI
            # Bootstrap is not robot-clustered.
            _, lo, hi = bootstrap_ci_mean(sub[sub.controller == ctrl]["RMSE_cart_norm"].to_numpy(),
                                          B=2000, rng=np.random.default_rng(BOOTSTRAP_SEED))
            tier_rows[ctrl] = (mu, lo, hi, rank)
        per_tier[fid] = tier_rows

    xf = cross_fidelity_block(urdf)

    computed = {"per_tier": per_tier, **xf}

    label = {"pbc": "PBC", "ctc_pd": "CTC-PD", "pdg": "PD+g", "ctc_smc": "CTC-SMC"}
    display_order = ["pbc", "ctc_pd", "pdg", "ctc_smc"]  # row order by proxy rank
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Proxy tier} & \multicolumn{2}{c}{Rigid-body tier} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Controller & $\rmsen$ & 95\% CI & $\rmsen$ & 95\% CI \\", r"\midrule"]
    for ctrl in display_order:
        mu_p, lo_p, hi_p, rk_p = per_tier["proxy"][ctrl]
        mu_r, lo_r, hi_r, rk_r = per_tier["rigid_body"][ctrl]
        lines.append(f"{label[ctrl]:7s} & ${fmt(mu_p)}^{{({rk_p})}}$ & [{fmt(mu_p*0+lo_p,3)}, {fmt(hi_p,3)}] & "
                     f"${fmt(mu_r)}^{{({rk_r})}}$ & [{fmt(lo_r,3)}, {fmt(hi_r,3)}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 5 — tab:coupled (KR6, 20-seed)
def table_coupled():
    coupled = pd.read_csv(DATA / "d3_coupled_all_20seed.csv")
    coupled = coupled[coupled.robot == "kr6"]
    decoupled = _load("all_results_d3_kr6_decoupled_20seed.csv")

    eps_list = sorted(coupled.eps_c.unique())
    label = {"pbc": "PBC", "ctc_smc": "CTC-SMC", "ctc_pd": "CTC-PD", "pdg": "PD+g"}
    display_order = ["pbc", "ctc_smc", "ctc_pd", "pdg"]

    dec_means = decoupled.groupby("controller")["RMSE_cart_norm"].mean()
    dec_rank = dec_means.rank()

    means_by_eps = {}
    tau_by_eps = {}
    for eps in eps_list:
        sub = coupled[coupled.eps_c == eps]
        m = sub.groupby("controller")["RMSE_cart_norm"].mean()
        means_by_eps[eps] = m
        common = sorted(set(m.index) & set(dec_rank.index))
        tau, _ = kendalltau(dec_rank[common].to_numpy(), m.rank()[common].to_numpy())
        tau_by_eps[eps] = float(tau)

    computed = {"means_by_eps": means_by_eps, "tau_by_eps": tau_by_eps}

    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Controller & $\varepsilon_{c} = 0.03$ & 0.06 & 0.09 & 0.15 & 0.25 \\", r"\midrule"]
    for ctrl in display_order:
        vals = " & ".join(fmt(means_by_eps[eps][ctrl]) for eps in eps_list)
        lines.append(f"{label[ctrl]:7s} & {vals} \\\\")
    lines.append(r"\midrule")
    taus = " & ".join(fmt(tau_by_eps[eps], 3) for eps in eps_list)
    lines.append(rf"$\taub$ vs.\ decoupled & {taus} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 6 — tab:cap (0.9 vs 0.7 tau_max for CTC-SMC)
def _swap_controller(main_df: pd.DataFrame, replacement: pd.DataFrame, ctrl: str) -> pd.DataFrame:
    """Return a copy of main_df with all rows for `ctrl` replaced by `replacement`."""
    keep = main_df[main_df.controller != ctrl]
    cols = keep.columns
    return pd.concat([keep, replacement[cols]], ignore_index=True)


def table_cap():
    main_df = _load("all_results_main.csv")
    urdf_main = main_df[main_df.robot_group == "urdf"]           # cap 0.7 arm (shipped)
    cap09 = _load("all_results_cap0p9_ctcsmc_check.csv")          # ctc_smc @ 0.9 cap
    urdf_cap09 = _swap_controller(urdf_main, cap09, "ctc_smc")    # cap 0.9 arm

    xf_07 = cross_fidelity_block(urdf_main)
    xf_09 = cross_fidelity_block(urdf_cap09)

    computed = {"cap07": xf_07, "cap09": xf_09}

    def row(label_, key, fmt_fn):
        v07 = fmt_fn(xf_07[key]); v09 = fmt_fn(xf_09[key])
        return rf"{label_:38s} & {v09} & {v07} \\"

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Statistic & Cap $0.9\tau_{\max}$ & Cap $0.7\tau_{\max}$ \\", r"\midrule",
             row("Subset $\\taub$", "subset_tau_b", lambda v: fmt(v, 3)),
             row("Subset CI", "subset_ci", lambda v: f"[{fmt(v[0],3)}, {fmt(v[1],3)}]"),
             row("Four-controller $\\taub$", "all4_tau_b", lambda v: fmt(v, 3)),
             row("Four-controller CI", "all4_ci", lambda v: f"[{fmt(v[0],3)}, {fmt(v[1],3)}]"),
             row("Top-controller disagreement (subset)", "disagreement", lambda v: f"{v[0]}/{v[1]}"),
             row("Proxy rank (subset)", "proxy_rank", lambda v: v),
             row("Rigid-body rank (subset)", "rigid_rank", lambda v: v),
             r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 7 — tab:frozen (gain-scaling ablation)
def table_frozen():
    main_df = _load("all_results_main.csv")
    urdf_main = main_df[main_df.robot_group == "urdf"]                 # tier-scaled (live) arm
    frozen_rigid = _load("all_results_frozen_gains_rigid.csv")          # frozen rigid-body tier
    needed = ["robot", "scene", "trajectory", "controller", "params", "seed",
              "fidelity", "RMSE_cart_norm"]
    urdf_frozen = pd.concat(
        [urdf_main[urdf_main.fidelity == "proxy"][needed], frozen_rigid[needed]],
        ignore_index=True)

    xf_live = cross_fidelity_block(urdf_main)
    xf_frozen = cross_fidelity_block(urdf_frozen)

    def rigid_rmse(urdf_df, ctrl):
        return float(urdf_df[(urdf_df.fidelity == "rigid_body") & (urdf_df.controller == ctrl)]
                     ["RMSE_cart_norm"].mean())

    computed = {
        "live": xf_live, "frozen": xf_frozen,
        "ctc_smc_rigid": (rigid_rmse(urdf_main, "ctc_smc"), rigid_rmse(urdf_frozen, "ctc_smc")),
        "pbc_rigid": (rigid_rmse(urdf_main, "pbc"), rigid_rmse(urdf_frozen, "pbc")),
    }

    def band(v):
        return f"$+{v:.3f}$" if v >= 0 else f"$-{abs(v):.3f}$"

    per_robot_str = lambda xf: "/".join(
        band(xf["per_robot_tau_b"][r]) for r in ["ur10e", "ur5e", "tx60"])

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Statistic & Tier-scaled gains & Frozen at proxy value \\", r"\midrule",
             rf"Subset $\taub$ & {fmt(xf_live['subset_tau_b'],3)} & {fmt(xf_frozen['subset_tau_b'],3)} \\",
             rf"Subset CI & [{fmt(xf_live['subset_ci'][0],3)}, {fmt(xf_live['subset_ci'][1],3)}] & "
             rf"[{fmt(xf_frozen['subset_ci'][0],3)}, {fmt(xf_frozen['subset_ci'][1],3)}] \\",
             rf"Four-controller $\taub$ & {fmt(xf_live['all4_tau_b'],3)} & {fmt(xf_frozen['all4_tau_b'],3)} \\",
             rf"Top-controller disagreement (subset) & {xf_live['disagreement'][0]}/{xf_live['disagreement'][1]} & "
             rf"{xf_frozen['disagreement'][0]}/{xf_frozen['disagreement'][1]} \\",
             rf"Per-robot $\taub$ (UR10e/UR5e/TX60) & {per_robot_str(xf_live)} & {per_robot_str(xf_frozen)} \\",
             r"\midrule",
             rf"CTC-SMC rigid-body $\rmsen$ & {fmt(computed['ctc_smc_rigid'][0])} & {fmt(computed['ctc_smc_rigid'][1])} \\",
             rf"PBC rigid-body $\rmsen$ & {fmt(computed['pbc_rigid'][0])} & {fmt(computed['pbc_rigid'][1])} \\",
             r"Rigid-body ranking & \multicolumn{2}{c}{PBC $<$ CTC-SMC $<$ CTC-PD $<$ PD+g (both)} \\",
             r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 8 — tab:dt (matched 5-seed subset across step sizes)
def table_dt():
    main_df = _load("all_results_main.csv")
    urdf_main = main_df[main_df.robot_group == "urdf"]
    matched_seeds = [0, 1, 2, 3, 4]
    dt10 = urdf_main[urdf_main.seed.isin(matched_seeds)]
    dt05 = _load("all_results_dt0p5.csv")
    dt025 = _load("all_results_dt0p25.csv")

    arms = {"1.0": dt10, "0.5": dt05, "0.25": dt025}
    computed = {}
    for tag, arm in arms.items():
        per_ctx_sub = compute_per_context_cross_fidelity_tau(arm, metric="RMSE_cart_norm",
                                                              exclude_controllers=["pbc"])
        per_ctx_all = compute_per_context_cross_fidelity_tau(arm, metric="RMSE_cart_norm")
        computed[tag] = {
            "subset_tau_b": float(per_ctx_sub["tau_b"].mean()),
            "all4_tau_b": float(per_ctx_all["tau_b"].mean()),
            "proxy_rank": subset_rank_string(arm, "proxy"),
            "rigid_rank": subset_rank_string(arm, "rigid_body"),
        }

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Statistic & $\Delta t = 1.0$\,ms & $\Delta t = 0.5$\,ms & $\Delta t = 0.25$\,ms \\", r"\midrule",
             r"Subset $\taub$ (matched seeds) & " +
             " & ".join(fmt(computed[t]["subset_tau_b"], 3) for t in ["1.0", "0.5", "0.25"]) + r" \\",
             r"Four-controller $\taub$ (matched seeds) & " +
             " & ".join(fmt(computed[t]["all4_tau_b"], 3) for t in ["1.0", "0.5", "0.25"]) + r" \\",
             rf"Proxy rank (subset) & \multicolumn{{3}}{{c}}{{{computed['1.0']['proxy_rank']} (all $\Delta t$)}} \\",
             rf"Rigid-body rank (subset) & \multicolumn{{3}}{{c}}{{{computed['1.0']['rigid_rank']} (all $\Delta t$)}} \\",
             r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 9 — tab:mismatch
def table_mismatch():
    df = pd.read_csv(DATA / "d1_mismatch_all.csv")
    label = {"pbc": "PBC", "ctc_smc": "CTC-SMC", "ctc_pd": "CTC-PD", "pdg": "PD+g"}
    display_order = ["pbc", "ctc_smc", "ctc_pd", "pdg"]
    pcts = sorted(df.pct.unique())

    computed = {}
    for pct in pcts:
        sub = df[df.pct == pct]
        for ctrl in display_order:
            csub = sub[sub.controller == ctrl]
            per_real = csub.groupby("realization")["RMSE_cart_norm"].mean()
            n = len(per_real)
            mu = float(per_real.mean())
            se = float(per_real.std(ddof=1) / np.sqrt(n))
            tcrit = float(student_t.ppf(0.975, df=n - 1))
            lo, hi = mu - tcrit * se, mu + tcrit * se
            computed[(pct, ctrl)] = (mu, lo, hi, n, tcrit)

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Controller & $\pm 10\%$ & $\pm 20\%$ & $\pm 30\%$ \\", r"\midrule"]
    for ctrl in display_order:
        cells = []
        for pct in pcts:
            mu, lo, hi, n, tcrit = computed[(pct, ctrl)]
            cells.append(f"{fmt(mu)} [{fmt(lo,3)}, {fmt(hi,3)}]")
        lines.append(f"{label[ctrl]:7s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


# Table 10 — tab:energy
def table_energy():
    df = _load("all_results_main.csv")
    urdf = df[df.robot_group == "urdf"]
    label = {"pbc": "PBC", "ctc_smc": "CTC-SMC", "ctc_pd": "CTC-PD", "pdg": "PD+g"}
    display_order = ["pbc", "ctc_smc", "ctc_pd", "pdg"]

    computed = {}
    for fid in ["proxy", "rigid_body"]:
        for ctrl in display_order:
            sub = urdf[(urdf.fidelity == fid) & (urdf.controller == ctrl)]
            mu, lo, hi = bootstrap_ci_mean(sub["E_star"].to_numpy(), B=2000,
                                           rng=np.random.default_rng(BOOTSTRAP_SEED))
            computed[(fid, ctrl)] = (mu, lo, hi)

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Controller & Proxy tier $E^{*}$ (95\% CI) & Rigid-body tier $E^{*}$ (95\% CI) \\", r"\midrule"]
    for ctrl in display_order:
        mu_p, lo_p, hi_p = computed[("proxy", ctrl)]
        mu_r, lo_r, hi_r = computed[("rigid_body", ctrl)]
        lines.append(f"{label[ctrl]:7s} & {mu_p:.2f} [{lo_p:.2f}, {hi_p:.2f}] & "
                     f"{mu_r:.2f} [{lo_r:.2f}, {hi_r:.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return computed, "\n".join(lines)


TABLES = {
    "tab_robots": table_robots,
    "tab_proxy": table_proxy,
    "tab_fidelity": table_fidelity,
    "tab_coupled": table_coupled,
    "tab_cap": table_cap,
    "tab_frozen": table_frozen,
    "tab_dt": table_dt,
    "tab_mismatch": table_mismatch,
    "tab_energy": table_energy,
}


def generate_all(write: bool = False):
    results = {}
    for name, fn in TABLES.items():
        computed, tex = fn()
        results[name] = {"computed": computed, "tex": tex}
        if write:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{name}.tex").write_text(tex + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    write = "--write" in sys.argv
    out = generate_all(write=write)
    for name in TABLES:
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        print(out[name]["tex"])
    if write:
        print(f"\nWrote {len(TABLES)} fragments -> {OUT}")

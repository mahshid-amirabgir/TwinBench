"""Ranking statistics, Kendall tau, and bootstrap utilities."""

from __future__ import annotations
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import twinbench.constants as C

MET_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RAW_CSV = MET_DIR / "all_results.csv"
MET_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CTRL_ORDER = list(C.CONTROLLERS)
TRAJ_ORDER = list(C.TRAJECTORIES)
PARAM_ORDER = list(C.PARAMS)
BOOTSTRAP_B = 2000

METRIC_COLS = [
    "RMSE_cart_norm", "Max_cart_err_norm", "E_star",
    "Tau_rms_norm", "Jerk_rms_norm", "Sat_frac_any",
    "Vel_violation_frac", "Acc_violation_frac", "Jerk_violation_frac",
]


# Helpers
def _kendall_tau_b(a, b):
    try: tau, _ = kendalltau(a, b, variant="b")
    except TypeError: tau, _ = kendalltau(a, b)
    return float(tau) if np.isfinite(tau) else np.nan

def _stable_seed(key: str) -> int:
    """Process-independent seed. Python's builtin hash() is salted per process
    (PYTHONHASHSEED), which made per-robot bootstrap CIs drift by ~1 ULP between
    runs; sha256 keeps them bit-reproducible."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def bootstrap_ci_mean(x, B=BOOTSTRAP_B, alpha=0.05, rng=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan, np.nan, np.nan
    if rng is None: rng = np.random.default_rng(12345)
    mu = float(np.mean(x))
    if x.size == 1: return mu, mu, mu
    idx = rng.integers(0, x.size, size=(B, x.size))
    boots = np.mean(x[idx], axis=1)
    return mu, float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))

def latex_mean_ci(mu, lo, hi, fmt="{:.4f}"):
    return f"{fmt.format(mu)}\\,[{fmt.format(lo)},\\,{fmt.format(hi)}]"


# T3 LaTeX table writer
def write_T3_metrics_tex(agg_T3: pd.DataFrame, out_path: Path):
    """
    Write T3_metrics.tex from agg_T3 DataFrame.

    Columns: Robot, Scene, Fidelity, Traj, Ctrl, Params, RMSE_norm, E*
    Format: mean\\,[ci_lo,\\,ci_hi]  (matches latex_mean_ci helper)

    Includes ALL robots (proxy + URDF) and adds a fidelity column so that
    URDF robots appear with both proxy and rigid_body rows.
    """
    sort_cols = ["robot", "scene", "fidelity", "trajectory", "controller", "params"]
    df = agg_T3.sort_values(sort_cols).reset_index(drop=True)

    def esc(s):
        return str(s).replace("_", r"\_")

    lines = []
    lines.append(r"\begin{table*}[t]\centering\scriptsize")
    lines.append(r"\caption{Primary metrics with 95\% bootstrap CI across all six robots "
                 r"and both fidelity tiers. URDF robots (ur5e, ur10e, tx60) appear twice: "
                 r"once under the proxy plant (Tier B-proxy) and once under the Pinocchio "
                 r"rigid-body plant (Tier B-full).}")
    lines.append(r"\label{tab:T3_metrics}")
    lines.append(r"\begin{tabular}{llllllcc}")
    lines.append(r"\toprule")
    lines.append(r"Robot & Scene & Fidelity & Traj & Ctrl & Params & "
                 r"RMSE$_{\text{norm}}$ & $E^{*}$ \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        rmse_cell = latex_mean_ci(
            r["RMSE_cart_norm_mean"],
            r["RMSE_cart_norm_ci95_lo"],
            r["RMSE_cart_norm_ci95_hi"],
            fmt="{:.4f}",
        )
        estar_cell = latex_mean_ci(
            r["E_star_mean"],
            r["E_star_ci95_lo"],
            r["E_star_ci95_hi"],
            fmt="{:.4f}",
        )
        lines.append(
            f"{esc(r['robot'])} & {esc(r['scene'])} & {esc(r['fidelity'])} & "
            f"{esc(r['trajectory'])} & {esc(r['controller'])} & {esc(r['params'])} & "
            f"{rmse_cell} & {estar_cell} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved T3 LaTeX table -> {out_path} ({len(df)} rows)")


# Pareto ranking
def pareto_front_mask(vals):
    vals = np.asarray(vals, float); n = vals.shape[0]
    ok = np.isfinite(vals).all(axis=1); idx = np.where(ok)[0]
    if idx.size == 0: return np.zeros(n, dtype=bool)
    V = vals[idx]; nd = np.ones(idx.size, dtype=bool)
    for i in range(idx.size):
        if not nd[i]: continue
        if np.where(np.all(V <= V[i], axis=1) & np.any(V < V[i], axis=1))[0].size > 0:
            nd[i] = False; continue
        dom = np.where(np.all(V[i] <= V, axis=1) & np.any(V[i] < V, axis=1))[0]
        nd[dom[dom != i]] = False
    mask = np.zeros(n, dtype=bool); mask[idx[nd]] = True
    return mask

def pareto_layer_ranking(df_block, obj_cols, id_cols):
    df = df_block.copy().reset_index(drop=True)
    finite = np.isfinite(df[obj_cols]).all(axis=1)
    df["pareto_layer"] = 999999
    remaining = df[finite].index.tolist(); layer = 1
    while remaining:
        V = df.loc[remaining, obj_cols].to_numpy(float)
        nd = pareto_front_mask(V)
        front = [remaining[i] for i in range(len(remaining)) if nd[i]]
        df.loc[front, "pareto_layer"] = layer
        remaining = [remaining[i] for i in range(len(remaining)) if not nd[i]]
        layer += 1
    df = df.sort_values(["pareto_layer"] + obj_cols + id_cols).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1, dtype=float)
    return df


# Kendall stability
def kendall_stability_matrix(ranked_df, group_cols, item_cols, rank_col="rank"):
    contexts = ranked_df[list(group_cols)].drop_duplicates().sort_values(list(group_cols))
    ctx_list = [tuple(r) for r in contexts.itertuples(index=False, name=None)]
    ctx_to_rank = {}
    for ctx in ctx_list:
        sub = ranked_df.copy()
        for c, v in zip(group_cols, ctx): sub = sub[sub[c] == v]
        keys = list(zip(*[sub[c].tolist() for c in item_cols])) if len(item_cols) > 1 else sub[item_cols[0]].astype(str).tolist()
        ctx_to_rank[ctx] = {k: float(r) for k, r in zip(keys, sub[rank_col].to_numpy())}
    n = len(ctx_list); mat = np.full((n, n), np.nan)
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i+1, n):
            common = sorted(set(ctx_to_rank[ctx_list[i]]) & set(ctx_to_rank[ctx_list[j]]))
            if len(common) < 2: continue
            a = np.array([ctx_to_rank[ctx_list[i]][k] for k in common])
            b = np.array([ctx_to_rank[ctx_list[j]][k] for k in common])
            mat[i, j] = mat[j, i] = _kendall_tau_b(a, b)
    return ctx_list, mat

def kendall_offdiag_summary(mat):
    off = mat[~np.eye(mat.shape[0], dtype=bool)]; off = off[np.isfinite(off)]
    if off.size == 0: return np.nan, np.nan, 0
    return float(off.mean()), float(off.std(ddof=1)) if off.size > 1 else 0.0, int(off.size)

def plot_kendall_heatmap(ctx_list, mat, out_path, title):
    labels = ["/".join(map(str, c)) for c in ctx_list]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.55), max(6, len(labels)*0.45)))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="Kendall $\\tau_b$")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "-", ha="center", va="center", fontsize=4)
    fig.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)
    print(f"  Saved heatmap -> {out_path}")


# Holm-Bonferroni + pairwise tests
def holm_bonferroni(pvalues):
    n = len(pvalues); order = np.argsort(pvalues); adj = np.zeros(n)
    for i, idx in enumerate(order): adj[idx] = min(1.0, pvalues[idx] * (n - i))
    for i in range(1, n): adj[order[i]] = max(adj[order[i]], adj[order[i-1]])
    return adj

def pairwise_comparisons(norm, out_dir, groupby="controller", metric="RMSE_cart_norm", suffix=""):
    from scipy.stats import mannwhitneyu
    levels = sorted(norm[groupby].dropna().unique())
    rows, pvals = [], []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            x = norm[norm[groupby] == levels[i]][metric].dropna().to_numpy()
            y = norm[norm[groupby] == levels[j]][metric].dropna().to_numpy()
            if len(x) < 2 or len(y) < 2: continue
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            pooled = np.sqrt((np.var(x, ddof=1)*(len(x)-1) + np.var(y, ddof=1)*(len(y)-1)) / (len(x)+len(y)-2))
            d = (np.mean(x) - np.mean(y)) / max(pooled, 1e-12)
            rows.append({"g1": levels[i], "g2": levels[j], "U": float(stat), "p_raw": float(p),
                         "cohens_d": float(d), "n1": len(x), "n2": len(y),
                         "mean1": float(np.mean(x)), "mean2": float(np.mean(y))})
            pvals.append(p)
    if rows:
        adj = holm_bonferroni(np.array(pvals))
        for r, pa in zip(rows, adj): r["p_holm"] = float(pa); r["sig"] = bool(pa < 0.05)
    df = pd.DataFrame(rows)
    fname = f"pairwise_{groupby}{suffix}_holm.csv"
    df.to_csv(out_dir / fname, index=False)
    print(f"  Saved -> {fname}")
    return df


# Per-context cross-fidelity τ_b (ITEM 9 + 10)
def compute_per_context_cross_fidelity_tau(
    urdf: pd.DataFrame,
    metric: str = "RMSE_cart_norm",
    *,
    exclude_controllers: list[str] | None = None,
):
    """
    Compute cross-fidelity Kendall τ_b per (robot, params, trajectory) context.

    Within each context:
      - average `metric` over seeds for each controller, per fidelity
      - rank controllers under proxy and under rigid_body
      - compute τ_b between the two rankings

    Returns a DataFrame with one row per context.
    """
    df = urdf.copy()
    if exclude_controllers:
        df = df[~df["controller"].isin(list(exclude_controllers))].copy()

    context_cols = ["robot", "params", "trajectory"]
    rows = []

    for ctx_key, block in df.groupby(context_cols, observed=True):
        ctx_dict = dict(zip(context_cols, ctx_key))

        # Mean metric per (fidelity, controller), averaged over seeds
        ctx_means = (
            block.groupby(["fidelity", "controller"], observed=True)[metric]
                 .mean()
                 .reset_index()
        )
        proxy_sub = ctx_means[ctx_means["fidelity"] == "proxy"].set_index("controller")[metric]
        rb_sub = ctx_means[ctx_means["fidelity"] == "rigid_body"].set_index("controller")[metric]

        common = sorted(set(proxy_sub.index) & set(rb_sub.index))
        if len(common) < 2:
            continue

        proxy_vals = np.array([proxy_sub[c] for c in common])
        rb_vals = np.array([rb_sub[c] for c in common])

        # Rank ascending (lower metric = better rank)
        proxy_ranks = pd.Series(proxy_vals).rank(method="average").to_numpy()
        rb_ranks = pd.Series(rb_vals).rank(method="average").to_numpy()

        tau = _kendall_tau_b(proxy_ranks, rb_ranks)

        row = dict(ctx_dict)
        row["n_controllers"] = len(common)
        row["tau_b"] = tau
        # Per-context mean RMSE
        for c in common:
            row[f"proxy_{c}"] = float(proxy_sub[c])
            row[f"rb_{c}"] = float(rb_sub[c])
        rows.append(row)

    return pd.DataFrame(rows)


# Cluster bootstrap of cross-fidelity τ_b
# Resampling unit: ROBOT (not context)
def cluster_bootstrap_tau_b(per_ctx: pd.DataFrame, B: int = 2000, rng=None):
    """
    Hierarchical (cluster) bootstrap of cross-fidelity Kendall τ_b.

    Resampling unit is ROBOT, not context. Procedure per bootstrap iteration:
      1. Sample n_robots robots WITH REPLACEMENT from the available robots.
      2. For each chosen robot, take ALL of its contexts (i.e., all rows
         of per_ctx where robot == that robot).
      3. Compute the mean τ_b across the resulting context pool.
      4. Repeat B times.
    Report mean and 2.5/97.5 percentiles of the bootstrap distribution.

    This correctly accounts for the fact that contexts are nested within
    robots: 12 contexts from the same robot share dynamics, gain
    normalization, target distribution, and IK feasibility envelope, so
    they are not independent samples. The naive context-level i.i.d.
    bootstrap underestimates uncertainty when n_robots is small.

    Args:
        per_ctx: DataFrame from compute_per_context_cross_fidelity_tau.
                 Must contain columns 'robot' and 'tau_b'.
        B: number of bootstrap iterations (default 2000).
        rng: optional numpy Generator. If None, uses a fixed seed.

    Returns:
        dict with keys:
            mean, ci95_lo, ci95_hi   - aggregate over all bootstrap means
            point_estimate           - mean τ_b on the original (unresampled)
                                        per_ctx, for direct comparison
            n_robots, n_contexts     - sample sizes
            B                        - number of bootstrap iterations
            interpretation_band      - "STRONG" / "MODERATE" / "WEAK" /
                                        "spans WEAK-MODERATE" / etc., based
                                        on where the CI lies relative to
                                        Schober's 0.4 / 0.8 cut points
    """
    if rng is None:
        rng = np.random.default_rng(20260406)

    df = per_ctx.dropna(subset=["tau_b"]).copy()
    if df.empty:
        return None

    robots = sorted(df["robot"].unique())
    n_robots = len(robots)
    n_contexts = len(df)

    # Pre-bin contexts by robot for fast resampling
    robot_to_taus = {r: df.loc[df["robot"] == r, "tau_b"].to_numpy()
                     for r in robots}

    point_estimate = float(df["tau_b"].mean())

    # Bootstrap loop
    boot_means = np.empty(B, dtype=float)
    for b in range(B):
        # Resample robots with replacement
        chosen_robots = rng.choice(robots, size=n_robots, replace=True)
        # Pool ALL contexts from each chosen robot
        pooled = np.concatenate([robot_to_taus[r] for r in chosen_robots])
        boot_means[b] = float(pooled.mean())

    mu = float(boot_means.mean())
    lo = float(np.quantile(boot_means, 0.025))
    hi = float(np.quantile(boot_means, 0.975))

    # Interpretation: where does the CI lie relative to Schober bands?
    def _band(v):
        if v >= 0.8: return "strong"
        if v >= 0.4: return "moderate"
        return "weak"

    lo_band = _band(lo)
    hi_band = _band(hi)
    if lo_band == hi_band:
        interp = f"{lo_band.upper()} agreement"
    else:
        interp = f"spans {lo_band.upper()}-{hi_band.upper()} agreement"

    return {
        "point_estimate": point_estimate,
        "mean": mu,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n_robots": n_robots,
        "n_contexts": n_contexts,
        "B": B,
        "interpretation_band": interp,
    }


# Fidelity comparison (Tier B analysis — items 9, 10 integrated)
def fidelity_comparison(norm, agg, out_dir, fig_dir):
    """Compare proxy vs rigid_body rankings on URDF robots."""
    if "fidelity" not in norm.columns:
        print("  [SKIP] No fidelity column")
        return

    urdf = norm[norm["robot_group"] == "urdf"].copy()
    if urdf.empty or urdf["fidelity"].nunique() < 2:
        print("  [SKIP] Need both proxy and rigid_body for URDF robots")
        return

    print("\n-- Fidelity Comparison (URDF robots) --")

    # Mean RMSE per controller per fidelity
    fid_summary = urdf.groupby(["fidelity", "controller"], observed=True).agg(
        RMSE_mean=("RMSE_cart_norm", "mean"),
        RMSE_std=("RMSE_cart_norm", "std"),
        n=("RMSE_cart_norm", "count"),
    ).reset_index()
    fid_summary.to_csv(out_dir / "fidelity_controller_summary.csv", index=False)
    print(fid_summary.to_string(index=False))

    # Ranking per fidelity (for Pareto outputs)
    OBJ = ["RMSE_cart_norm_mean", "E_star_mean", "Jerk_rms_norm_mean"]
    urdf_agg = agg[agg["robot"].isin(list(C.ROBOTS_URDF))].copy()

    if "fidelity" not in urdf_agg.columns:
        print("  [SKIP] fidelity not in agg_T3")
        return

    for fid in ["proxy", "rigid_body"]:
        sub = urdf_agg[urdf_agg["fidelity"] == fid]
        if sub.empty: continue
        ctx_cols = ["robot", "params"]
        ranked_blocks = []
        for key, block in sub.groupby(ctx_cols, sort=True, observed=True):
            if len(block) < 2: continue
            rb = pareto_layer_ranking(block, OBJ, ["trajectory", "controller"])
            for c, v in zip(ctx_cols, key): rb[c] = v
            rb["fidelity"] = fid
            ranked_blocks.append(rb)
        if ranked_blocks:
            ranked = pd.concat(ranked_blocks, ignore_index=True)
            ranked.to_csv(out_dir / f"ranks_urdf_{fid}.csv", index=False)
            print(f"  Saved URDF {fid} ranking")

    # Global (aggregated) cross-fidelity τ_b  — OLD FRAGILE NUMBER
    print("\n  [Global (aggregated) tau_b - legacy, fragile: single 4-point rank]")
    for metric in ["RMSE_cart_norm"]:
        ranks_by_fid = {}
        for fid in ["proxy", "rigid_body"]:
            sub = urdf.groupby(["fidelity", "controller"], observed=True)[metric].mean().reset_index()
            sub = sub[sub["fidelity"] == fid].sort_values(metric)
            sub["rank"] = np.arange(1, len(sub)+1)
            ranks_by_fid[fid] = sub.set_index("controller")["rank"]

        if "proxy" in ranks_by_fid and "rigid_body" in ranks_by_fid:
            common = sorted(set(ranks_by_fid["proxy"].index) & set(ranks_by_fid["rigid_body"].index))
            if len(common) >= 2:
                a = np.array([ranks_by_fid["proxy"][c] for c in common])
                b = np.array([ranks_by_fid["rigid_body"][c] for c in common])
                tau_global = _kendall_tau_b(a, b)
                print(f"    Global tau_b (controller ranking, n={len(common)} points): {tau_global:.3f}")
                print(f"      Proxy ranking:      {dict(ranks_by_fid['proxy'])}")
                print(f"      Rigid-body ranking: {dict(ranks_by_fid['rigid_body'])}")
    print("  ------------------------------------------------------------")

    # ITEM 9: Per-context cross-fidelity τ_b with bootstrap CI
    print("\n  [Per-context tau_b with bootstrap 95% CI (robust headline)]")
    per_ctx = compute_per_context_cross_fidelity_tau(urdf, metric="RMSE_cart_norm")

    if per_ctx.empty:
        print("    [SKIP] No valid contexts found")
        print("  ------------------------------------------------------------")
        return

    per_ctx.to_csv(out_dir / "kendall_tau_contexts_all4.csv", index=False)
    print(f"    Saved per-context tau_b (all controllers) -> kendall_tau_contexts_all4.csv "
          f"({len(per_ctx)} contexts)")

    taus_all = per_ctx["tau_b"].dropna().to_numpy()
    rng_boot = np.random.default_rng(20260406)
    mu, lo, hi = bootstrap_ci_mean(taus_all, B=BOOTSTRAP_B, rng=rng_boot)

    print("")
    print(f"    HEADLINE: Cross-fidelity Kendall tau_b")
    print(f"      mean tau_b = {mu:.3f}   95% CI = [{lo:.3f}, {hi:.3f}]")
    print(f"      n = {len(taus_all)} contexts   "
          f"(std = {np.nanstd(taus_all, ddof=1):.3f})")
    print(f"      median = {np.nanmedian(taus_all):.3f}   "
          f"IQR = [{np.nanpercentile(taus_all, 25):.3f}, "
          f"{np.nanpercentile(taus_all, 75):.3f}]")

    # Interpretation band
    if mu >= 0.8:
        band = "STRONG agreement"
    elif mu >= 0.4:
        band = "MODERATE agreement"
    elif mu >= 0.0:
        band = "WEAK agreement"
    else:
        band = "NEGATIVE agreement (rankings inverted on average)"
    print(f"      Interpretation: {band}")

    # Distribution of tau values
    n_pos = int(np.sum(taus_all > 0))
    n_neg = int(np.sum(taus_all < 0))
    n_zero = int(np.sum(taus_all == 0))
    n_perfect = int(np.sum(taus_all == 1.0))
    n_inverted = int(np.sum(taus_all == -1.0))
    print(f"      Distribution: positive={n_pos}, zero={n_zero}, negative={n_neg}")
    print(f"      Extremes:     perfect(+1)={n_perfect}, inverted(-1)={n_inverted}")
    print("  ------------------------------------------------------------")

    # Model-based subset: exclude PBC, ranking only {pdg, ctc_pd, ctc_smc}.
    # This is the paper's headline cross-fidelity number (tau_b = 0.167). PBC is
    # rank-1 in every context on both tiers, so leaving it in adds a rank-stable
    # anchor that mechanically inflates tau_b to 0.583 — the same artifact the
    # paper criticises for a controller pinned at rank 4. Report the subset value.
    # (Before the 4-controller migration this block excluded "adaptive"; that
    # controller is no longer in the default set, so the exclusion was a no-op
    # and this block silently reprinted the all-4 number.)
    print("\n  [Per-context tau_b excluding PBC (model-based subset)]")
    per_ctx_stable = compute_per_context_cross_fidelity_tau(
        urdf, metric="RMSE_cart_norm", exclude_controllers=["pbc"]
    )
    if per_ctx_stable.empty:
        print("    [SKIP] No valid contexts after excluding PBC")
        print("  ------------------------------------------------------------")
    else:
        per_ctx_stable.to_csv(out_dir / "kendall_tau_contexts_stable3.csv", index=False)
        taus_s = per_ctx_stable["tau_b"].dropna().to_numpy()
        rng_boot_s = np.random.default_rng(20260407)
        mu_s, lo_s, hi_s = bootstrap_ci_mean(taus_s, B=BOOTSTRAP_B, rng=rng_boot_s)
        print(f"    mean tau_b = {mu_s:.3f}   95% CI = [{lo_s:.3f}, {hi_s:.3f}]")
        print(f"    n = {len(taus_s)} contexts (controllers={int(per_ctx_stable['n_controllers'].median())})")
        print(f"    Saved -> kendall_tau_contexts_stable3.csv")
        print("  ------------------------------------------------------------")

        # Model-based-subset per-robot breakdown
        print("\n  [Model-based-subset per-robot cross-fidelity tau_b breakdown]")
        per_robot_rows_s = []
        for robot in sorted(per_ctx_stable["robot"].unique()):
            sub = per_ctx_stable[per_ctx_stable["robot"] == robot]
            taus_r = sub["tau_b"].dropna().to_numpy()
            if taus_r.size == 0:
                continue
            rng_r = np.random.default_rng(_stable_seed(f"subset_robot_{robot}"))
            mu_r, lo_r, hi_r = bootstrap_ci_mean(taus_r, B=BOOTSTRAP_B, rng=rng_r)
            print(
                f"    {robot:<8}  tau_b = {mu_r:+.3f}  "
                f"[{lo_r:+.3f}, {hi_r:+.3f}]   "
                f"(n={len(taus_r)} ctx, median={np.nanmedian(taus_r):+.3f})"
            )
            per_robot_rows_s.append({
                "robot": robot,
                "tau_b_mean": mu_r,
                "tau_b_ci95_lo": lo_r,
                "tau_b_ci95_hi": hi_r,
                "tau_b_median": float(np.nanmedian(taus_r)),
                "tau_b_std": float(np.nanstd(taus_r, ddof=1)) if taus_r.size > 1 else 0.0,
                "n_contexts": int(taus_r.size),
            })
        if per_robot_rows_s:
            pd.DataFrame(per_robot_rows_s).to_csv(
                out_dir / "kendall_tau_per_robot_stable3.csv", index=False
            )
            print("")
            print("  Saved -> kendall_tau_per_robot_stable3.csv")
        print("  ------------------------------------------------------------")

        # Model-based-subset cluster bootstrap (robot as cluster)
        print("\n  [Model-based-subset cluster bootstrap (resampling unit: ROBOT)]")
        cluster_result_s = cluster_bootstrap_tau_b(
            per_ctx_stable, B=BOOTSTRAP_B,
            rng=np.random.default_rng(20260407),
        )
        if cluster_result_s is not None:
            crs = cluster_result_s
            print("    HEADLINE (model-based subset, cluster-bootstrapped):")
            print(f"      point estimate tau_b  = {crs['point_estimate']:.3f}")
            print(f"      bootstrap mean        = {crs['mean']:.3f}")
            print(f"      bootstrap 95% CI      = [{crs['ci95_lo']:.3f}, {crs['ci95_hi']:.3f}]")
            print(f"      n_robots              = {crs['n_robots']}")
            print(f"      n_contexts            = {crs['n_contexts']}")
            print(f"      B                     = {crs['B']}")
            print(f"      Interpretation        : {crs['interpretation_band']}")
            pd.DataFrame([crs]).to_csv(
                out_dir / "kendall_tau_cluster_bootstrap_stable3.csv", index=False
            )
            print("")
            print("    Saved -> kendall_tau_cluster_bootstrap_stable3.csv")
        print("  ------------------------------------------------------------")

    # CLUSTER BOOTSTRAP: resample robots, not contexts
    # Honest CI given that contexts are nested within only 3 robots
    print("\n  [Cluster bootstrap (resampling unit: ROBOT)]")
    cluster_result = cluster_bootstrap_tau_b(
        per_ctx, B=BOOTSTRAP_B,
        rng=np.random.default_rng(20260407),
    )
    if cluster_result is not None:
        cr = cluster_result
        print(f"    HEADLINE (cluster-bootstrapped):")
        print(f"      point estimate tau_b  = {cr['point_estimate']:.3f}")
        print(f"      bootstrap mean        = {cr['mean']:.3f}")
        print(f"      bootstrap 95% CI      = [{cr['ci95_lo']:.3f}, "
              f"{cr['ci95_hi']:.3f}]")
        print(f"      n_robots              = {cr['n_robots']}")
        print(f"      n_contexts            = {cr['n_contexts']}")
        print(f"      B                     = {cr['B']}")
        print(f"      Interpretation        : {cr['interpretation_band']}")
        print("")
        print("    Note: cluster bootstrap accounts for nesting of contexts within robots.")

        # Persist to disk for the paper
        pd.DataFrame([cr]).to_csv(
            out_dir / "kendall_tau_cluster_bootstrap.csv", index=False
        )
        print("")
        print(f"    Saved -> kendall_tau_cluster_bootstrap.csv")
    print("  ------------------------------------------------------------")

    # ITEM 10: Per-robot cross-fidelity τ_b breakdown
    print("\n  [Per-robot cross-fidelity tau_b breakdown]")
    per_robot_rows = []
    for robot in sorted(per_ctx["robot"].unique()):
        sub = per_ctx[per_ctx["robot"] == robot]
        taus_r = sub["tau_b"].dropna().to_numpy()
        if taus_r.size == 0:
            continue

        rng_r = np.random.default_rng(_stable_seed(f"robot_{robot}"))
        mu_r, lo_r, hi_r = bootstrap_ci_mean(taus_r, B=BOOTSTRAP_B, rng=rng_r)

        print(f"    {robot:<8}  tau_b = {mu_r:+.3f}  "
              f"[{lo_r:+.3f}, {hi_r:+.3f}]   "
              f"(n={len(taus_r)} ctx, median={np.nanmedian(taus_r):+.3f})")

        per_robot_rows.append({
            "robot": robot,
            "tau_b_mean": mu_r,
            "tau_b_ci95_lo": lo_r,
            "tau_b_ci95_hi": hi_r,
            "tau_b_median": float(np.nanmedian(taus_r)),
            "tau_b_std": float(np.nanstd(taus_r, ddof=1)) if taus_r.size > 1 else 0.0,
            "n_contexts": int(taus_r.size),
        })

    if per_robot_rows:
        per_robot_df = pd.DataFrame(per_robot_rows)
        per_robot_df.to_csv(out_dir / "kendall_tau_per_robot.csv", index=False)
        print("")
        print(f"  Saved -> kendall_tau_per_robot.csv")
    print("-----------------------------------------------")

    # Box plot: proxy vs rigid-body RMSE by controller
    fig, ax = plt.subplots(figsize=(10, 5))
    fids = sorted(urdf["fidelity"].unique())
    ctrls = CTRL_ORDER
    labels = []
    for ci, ctrl in enumerate(ctrls):
        for fi, fid in enumerate(fids):
            sub = urdf[(urdf["controller"] == ctrl) & (urdf["fidelity"] == fid)]["RMSE_cart_norm"].dropna()
            pos = ci * (len(fids) + 1) + fi
            bp = ax.boxplot(sub, positions=[pos], widths=0.6, patch_artist=True)
            color = "#4477AA" if fid == "proxy" else "#CC6677"
            bp["boxes"][0].set_facecolor(color)
            if ci == 0:
                labels.append(fid)
    tick_pos = [ci * (len(fids) + 1) + 0.5 for ci in range(len(ctrls))]
    ax.set_xticks(tick_pos); ax.set_xticklabels(ctrls)
    ax.set_ylabel("RMSE_cart_norm"); ax.set_title("Proxy vs Rigid-Body (URDF robots)")
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#4477AA"), Patch(facecolor="#CC6677")],
              ["Proxy", "Rigid-body"], loc="upper left")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "fidelity_boxplot.png", dpi=300); plt.close(fig)
    print(f"\n  Saved fidelity boxplot -> {fig_dir / 'fidelity_boxplot.png'}")


# MAIN
def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing: {RAW_CSV}")

    raw = pd.read_csv(RAW_CSV)
    print(f"\nLoaded {len(raw)} rows from {RAW_CSV}")

    required = ["robot", "scene", "trajectory", "controller", "params", "seed",
                "ik_ok", "fidelity", "robot_group"] + METRIC_COLS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # IK report
    print("\n-- IK Failure Report --")
    ik_rep = raw.groupby(["robot_group", "robot", "scene", "fidelity", "params"], observed=True).agg(
        total=("ik_ok", "size"), ok=("ik_ok", lambda x: x.astype(bool).sum())).reset_index()
    ik_rep["fail_%"] = 100 * (1 - ik_rep["ok"] / np.maximum(ik_rep["total"], 1))
    ik_rep.to_csv(MET_DIR / "ik_fail_report.csv", index=False)
    fails = ik_rep[ik_rep["fail_%"] > 0]
    if len(fails) > 0:
        print(fails.to_string(index=False))
    else:
        print("  No IK failures")

    # Filter
    norm = raw[raw["ik_ok"].astype(bool)].copy().dropna(subset=METRIC_COLS)
    print(f"\nAfter filtering: {len(norm)} rows")
    norm.to_csv(MET_DIR / "normalized_metrics.csv", index=False)

    print("\n-- Quick Summary --")
    for (grp, fid, ctrl), sub in norm.groupby(["robot_group", "fidelity", "controller"], observed=True):
        print(f"  {grp}/{fid}/{ctrl}: RMSE={sub['RMSE_cart_norm'].mean():.4f} "
              f"E*={sub['E_star'].mean():.4f} (n={len(sub)})")

    print("\n-- Pairwise Controller Comparisons --")
    pairwise_comparisons(norm, MET_DIR, groupby="controller", suffix="_all")

    # Per robot_group
    for grp in ["proxy", "urdf"]:
        sub = norm[norm["robot_group"] == grp]
        if len(sub) > 0:
            print(f"\n  [{grp} group]")
            pairwise_comparisons(sub, MET_DIR, groupby="controller", suffix=f"_{grp}")

    print("\n-- Bootstrap CI Aggregation --")
    grp_cols = ["robot", "robot_group", "scene", "fidelity", "trajectory", "controller", "params"]
    rng_ci = np.random.default_rng(20250101)
    agg_rows = []
    for key, blk in norm.groupby(grp_cols, sort=True, observed=True):
        row = dict(zip(grp_cols, key))
        row["n_seeds"] = int(blk["seed"].nunique())
        for m in METRIC_COLS:
            mu, lo, hi = bootstrap_ci_mean(blk[m].to_numpy(), B=BOOTSTRAP_B, rng=rng_ci)
            row[f"{m}_mean"] = mu; row[f"{m}_ci95_lo"] = lo; row[f"{m}_ci95_hi"] = hi
        agg_rows.append(row)
    agg_T3 = pd.DataFrame(agg_rows)
    agg_T3.to_csv(MET_DIR / "agg_T3.csv", index=False)
    print(f"  Saved agg_T3 ({len(agg_T3)} rows)")
    write_T3_metrics_tex(agg_T3, MET_DIR / "T3_metrics.tex")

    print("\n-- Pareto Ranking --")
    OBJ = ["RMSE_cart_norm_mean", "E_star_mean", "Jerk_rms_norm_mean"]

    all_ranked = []
    for (grp, fid), agg_sub in agg_T3.groupby(["robot_group", "fidelity"], observed=True):
        ctx_cols = ["robot", "scene", "params"]
        ranked_blocks = []
        for key, block in agg_sub.groupby(ctx_cols, sort=True, observed=True):
            if len(block) < 2: continue
            rb = pareto_layer_ranking(block, OBJ, ["trajectory", "controller"])
            for c, v in zip(ctx_cols, key): rb[c] = v
            rb["robot_group"] = grp; rb["fidelity"] = fid
            ranked_blocks.append(rb)
        if ranked_blocks:
            ranked = pd.concat(ranked_blocks, ignore_index=True)
            ranked.to_csv(MET_DIR / f"ranks_pareto_{grp}_{fid}.csv", index=False)
            all_ranked.append(ranked)

            ctx_list, tau_mat = kendall_stability_matrix(
                ranked, tuple(ctx_cols), ("trajectory", "controller"))
            mu, sd, n = kendall_offdiag_summary(tau_mat)
            print(f"  {grp}/{fid}: tau_b = {mu:.3f} +/- {sd:.3f} (n={n})")
            plot_kendall_heatmap(ctx_list, tau_mat,
                                FIG_DIR / f"F7_kendall_{grp}_{fid}.png",
                                f"Kendall tau_b - {grp}/{fid}")

    fidelity_comparison(norm, agg_T3, MET_DIR, FIG_DIR)

    print("\n-- Cross-Group Controller Rankings --")
    for fid in ["proxy"]:
        for grp in ["proxy", "urdf"]:
            sub = norm[(norm["robot_group"] == grp) & (norm["fidelity"] == fid)]
            if sub.empty: continue
            means = sub.groupby("controller", observed=True)["RMSE_cart_norm"].mean().sort_values()
            ranking = " > ".join([f"{c}({v:.4f})" for c, v in means.items()])
            print(f"  {grp}/{fid}: {ranking}")

    print(f"\nOK: Section 3.7 complete. Outputs in {MET_DIR} and {FIG_DIR}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None,
                    help="Path to input CSV. Default: data/all_results.csv")
    args = ap.parse_args()
    if args.input is not None:
        RAW_CSV = Path(args.input)
    main()

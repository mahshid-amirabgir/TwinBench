"""Coupled-proxy sweep for datasheet robots."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from twinbench.dynamics.coupled_proxy_plant import simulate_coupled_proxy
from twinbench.dynamics.robot_loader import (
    load_robot_spec, robot_spec_path_from_id,
    build_dh_from_spec, get_limits_from_spec,
    get_dynamics_cfg_for_params,
)
from twinbench.ids import canon_robot_id, canon_scene_id, canon_traj_id, canon_ctrl_id, canon_params
from twinbench.scene_specs import SceneCatalog
import twinbench.constants as C

# Import from experiment runner
from experiments.run_main_grid import (
    control_torque, compute_metrics, gen_trajectory_for_run,
    scene_to_targets, seed_target, seed_noise,
    DOF, DT, T_BASE, VERSION,
    QD_CAP_FRAC, TAU_CAP_FRAC,
    DEFAULT_NOISE, DEFAULT_DISTURB,
)

ROBOT = "kr6"
SCENE = "open_reach"
COUPLING_STRENGTH = 0.3
TRAJECTORIES = list(C.TRAJECTORIES)
CONTROLLERS = list(C.CONTROLLERS)
PARAMS = list(C.PARAMS)
SEEDS = list(C.SEEDS)

METRIC_DIR = ROOT / "data"
METRIC_DIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = METRIC_DIR / "coupled_ablation_results.csv"
SUMMARY_CSV = METRIC_DIR / "coupled_ablation_summary.csv"
REPORT_TXT = METRIC_DIR / "coupled_ablation_report.txt"
TIER_A_CSV = METRIC_DIR / "all_results.csv"

CATALOG = SceneCatalog.from_repo_root(ROOT)


def run_coupled_single(robot_id, scene_id, traj_id, ctrl_id, params_id, seed,
                       coupling_strength=COUPLING_STRENGTH):
    """Run one simulation on the coupled proxy plant. Returns result dict."""
    rid = canon_robot_id(robot_id)
    sid = canon_scene_id(scene_id)
    traj = canon_traj_id(traj_id)
    ctrl = canon_ctrl_id(ctrl_id)
    prm = canon_params(params_id)

    spec = load_robot_spec(robot_spec_path_from_id(rid, repo_root=ROOT))

    # Effective (plant) dynamics — scaled by params
    dyn = get_dynamics_cfg_for_params(spec, prm)
    I_eff = np.array(dyn["I_proxy"])
    B_eff = np.array(dyn["B_proxy"])
    G_eff = np.array(dyn["G_nom"])
    tau_max = np.array(dyn["tau_max"])
    qd_max_rad = np.array(dyn["qd_max_rad"])
    R_max = float(dyn["Rmax"])
    payload_used = float(dyn["payload_used"])

    # Nominal (controller) dynamics — always nominal
    dyn_nom = get_dynamics_cfg_for_params(spec, "nominal")
    I_nom = np.array(dyn_nom["I_proxy"])
    B_nom = np.array(dyn_nom["B_proxy"])
    G_nom = np.array(dyn_nom["G_nom"])

    # Caps — same as run_experiments.py
    dq_cap = QD_CAP_FRAC * qd_max_rad
    ddq_cap_raw = TAU_CAP_FRAC * tau_max / (I_eff + 1e-12)
    ddq_cap = np.clip(ddq_cap_raw, 10.0, 1000.0)

    # Noise / disturbance
    ncfg = DEFAULT_NOISE
    sigma_q = float(ncfg["sigma_q"])
    sigma_qdot = float(ncfg["sigma_qdot"])

    dcfg = DEFAULT_DISTURB
    dj = int(dcfg["joint_index0"])
    d_frac = float(dcfg["tau_frac_of_max"])
    d_onset = float(dcfg["onset"])
    d_dur = float(dcfg["duration"])
    d_end = d_onset + d_dur
    d_tau = d_frac * float(tau_max[dj])

    # Scene targets (same seeds as Tier A for fair comparison)
    rng_target = np.random.default_rng(seed_target(rid, sid, seed))
    q_start, q_goal, dh, qmin, qmax, scene_meta = scene_to_targets(
        spec, rid, sid, rng_target)

    base_out = {
        "robot": rid, "scene": sid, "trajectory": traj, "controller": ctrl,
        "params": prm, "seed": int(seed),
        "fidelity": "coupled_proxy",
        "coupling_strength": coupling_strength,
        "R_max_m": R_max, "payload_used_kg": payload_used,
        "robot_group": "proxy",
        "version": VERSION,
    }

    if not scene_meta.get("ik_ok", False):
        base_out["ik_ok"] = False
        base_out["rejected_reason"] = scene_meta.get("rejected_reason", "ik_fail")
        for col in ["RMSE_cart_norm", "Max_cart_err_norm", "E_star",
                     "Tau_rms_norm", "Jerk_rms_norm", "Sat_frac_any",
                     "Vel_violation_frac", "Acc_violation_frac", "Jerk_violation_frac",
                     "RMSE_cart_m", "Max_cart_err_m", "Energy", "T_eff"]:
            base_out[col] = float("nan")
        base_out["cart_err_ok"] = False
        return base_out

    # Trajectory
    T = T_BASE
    t, qd, dqd, ddqd, T_eff, traj_info = gen_trajectory_for_run(
        traj, q_start, q_goal, T, DT, dyn_nom)

    rng_noise = np.random.default_rng(seed_noise(rid, sid, traj, prm, seed))

    # Simulate on COUPLED proxy plant
    q_sim, dq_sim, tau_hist, ddq_hist, sat_hist = simulate_coupled_proxy(
        qd=qd, dqd=dqd, ddqd=ddqd, t=t, T_eff=T_eff, dh=dh,
        q_init=qd[0].copy(), qmin=qmin, qmax=qmax,
        I_nom=I_nom, B_nom=B_nom, G_nom=G_nom, tau_max=tau_max,
        I_eff=I_eff, B_eff=B_eff, G_eff=G_eff,
        dq_cap=dq_cap, ddq_cap=ddq_cap, ctrl=ctrl,
        rng_noise=rng_noise, sigma_q=sigma_q, sigma_qdot=sigma_qdot,
        disturb_joint=dj, disturb_tau=d_tau,
        disturb_onset=d_onset, disturb_end=d_end,
        control_fn=control_torque,
        coupling_strength=coupling_strength,
    )

    # Metrics (same as Tier A)
    metrics = compute_metrics(
        q=q_sim, dq_sim=dq_sim, tau_hist=tau_hist,
        ddq_hist=ddq_hist, sat_hist=sat_hist,
        qd=qd, dqd=dqd, t=t, T_eff=T_eff, dh=dh,
        R_max=R_max, payload_used=payload_used,
        tau_max=tau_max, dq_cap=dq_cap, ddq_cap=ddq_cap,
        urdf_robot=None)

    out = dict(base_out)
    out["ik_ok"] = True
    out["T_eff"] = T_eff
    out["rejected_reason"] = ""
    out.update(metrics)
    out.update({f"traj_{k}": v for k, v in traj_info.items()})
    return out


def compute_rankings(df, metric="RMSE_cart_norm"):
    """Compute mean metric per controller, return sorted ranking dict."""
    ok = df[df["ik_ok"].astype(bool) & df["cart_err_ok"].astype(bool)]
    if len(ok) == 0:
        return {}
    grp = ok.groupby("controller")[metric].mean().sort_values()
    return dict(grp)


def kendall_tau_from_rankings(rank_a, rank_b):
    """Kendall tau-b between two ranking dicts (controller -> mean metric)."""
    from scipy.stats import kendalltau
    controllers = sorted(set(rank_a.keys()) & set(rank_b.keys()))
    if len(controllers) < 2:
        return float("nan"), float("nan")
    vals_a = [rank_a[c] for c in controllers]
    vals_b = [rank_b[c] for c in controllers]
    tau, pval = kendalltau(vals_a, vals_b)
    return float(tau), float(pval)


def main():
    total = len(TRAJECTORIES) * len(CONTROLLERS) * len(PARAMS) * len(SEEDS)
    print(f"\n{'='*70}")
    print(f"  TwinBench — Coupled Proxy Ablation")
    print(f"{'='*70}")
    print(f"  Robot:             {ROBOT}")
    print(f"  Scene:             {SCENE}")
    print(f"  Coupling strength: {COUPLING_STRENGTH}")
    print(f"  Trajectories:      {TRAJECTORIES}")
    print(f"  Controllers:       {CONTROLLERS}")
    print(f"  Params:            {PARAMS}")
    print(f"  Seeds:             {len(SEEDS)}")
    print(f"  Total runs:        {total}")
    print(f"{'='*70}\n")

    t0 = time.time()
    results = []
    i = 0

    for traj in TRAJECTORIES:
        for ctrl in CONTROLLERS:
            for prm in PARAMS:
                for sd in SEEDS:
                    i += 1
                    res = run_coupled_single(ROBOT, SCENE, traj, ctrl, prm, sd,
                                             coupling_strength=COUPLING_STRENGTH)
                    results.append(res)
                    if i % 100 == 0 or i == total:
                        rmse = res.get("RMSE_cart_norm", np.nan)
                        ok = res.get("ik_ok", False)
                        elapsed = time.time() - t0
                        rate = i / max(elapsed, 1e-6)
                        eta = (total - i) / max(rate, 1e-6)
                        print(f"  [{i:>4}/{total}] {traj}/{ctrl}/{prm}/s{sd:>2} "
                              f"RMSE_norm={rmse:.4f} ik_ok={ok} "
                              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\n  Completed {total} runs in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save raw results
    df_coupled = pd.DataFrame(results)
    df_coupled.to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV}")

    print(f"\n{'='*70}")
    print(f"  Ranking Comparison: Decoupled (Tier A) vs Coupled Proxy")
    print(f"{'='*70}")

    rank_coupled = compute_rankings(df_coupled, "RMSE_cart_norm")

    # Load Tier A results (KR6, open_reach, proxy)
    rank_decoupled = {}
    tau_rmse = float("nan")
    tau_estar = float("nan")

    if TIER_A_CSV.exists():
        df_all = pd.read_csv(TIER_A_CSV)
        df_tierA_kr6 = df_all[
            (df_all["robot"] == ROBOT) &
            (df_all["scene"] == SCENE) &
            (df_all["fidelity"] == "proxy")
        ]
        if len(df_tierA_kr6) > 0:
            rank_decoupled = compute_rankings(df_tierA_kr6, "RMSE_cart_norm")
            tau_rmse, pval_rmse = kendall_tau_from_rankings(rank_decoupled, rank_coupled)

            rank_decoupled_e = compute_rankings(df_tierA_kr6, "E_star")
            rank_coupled_e = compute_rankings(df_coupled, "E_star")
            tau_estar, pval_estar = kendall_tau_from_rankings(rank_decoupled_e, rank_coupled_e)
        else:
            print("  ⚠️  No KR6/open_reach/proxy data in all_results.csv")
    else:
        print(f"  ⚠️  {TIER_A_CSV} not found — cannot compare to Tier A")

    # Print comparison
    print(f"\n  {'Controller':<12} {'Decoupled':>12} {'Coupled':>12} {'Delta':>10} {'Delta%':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    controllers_sorted = sorted(rank_coupled.keys(),
                                key=lambda c: rank_coupled.get(c, 999))
    for ctrl in controllers_sorted:
        v_c = rank_coupled.get(ctrl, float("nan"))
        v_d = rank_decoupled.get(ctrl, float("nan"))
        if np.isfinite(v_c) and np.isfinite(v_d):
            delta = v_c - v_d
            pct = 100 * delta / max(abs(v_d), 1e-12)
            print(f"  {ctrl:<12} {v_d:>12.4f} {v_c:>12.4f} {delta:>+10.4f} {pct:>+9.1f}%")
        else:
            print(f"  {ctrl:<12} {v_d:>12.4f} {v_c:>12.4f} {'N/A':>10} {'N/A':>10}")

    print(f"\n  Decoupled ranking: {' < '.join(rank_decoupled.keys()) if rank_decoupled else 'N/A'}")
    print(f"  Coupled ranking:   {' < '.join(controllers_sorted)}")
    print(f"\n  Kendall τ_b (RMSE_norm):  {tau_rmse:+.3f}")
    if np.isfinite(tau_estar):
        print(f"  Kendall τ_b (E_star):     {tau_estar:+.3f}")

    if np.isfinite(tau_rmse) and tau_rmse >= 0.8:
        print(f"\n  ✅ Rankings highly stable under coupling (τ_b ≥ 0.8)")
        print(f"     → CTC-PD recommendation is robust to structural mismatch")
    elif np.isfinite(tau_rmse) and tau_rmse >= 0.3:
        print(f"\n  ⚠️  Rankings partially shift under coupling (0.3 ≤ τ_b < 0.8)")
        print(f"     → Fidelity-dependent recommendation warranted")
    elif np.isfinite(tau_rmse):
        print(f"\n  ❌ Rankings invert under coupling (τ_b < 0.3)")
        print(f"     → Decoupled proxy rankings are unreliable for controller selection")

    # Save summary
    summary_rows = []
    for ctrl in CONTROLLERS:
        summary_rows.append({
            "controller": ctrl,
            "RMSE_norm_decoupled": rank_decoupled.get(ctrl, float("nan")),
            "RMSE_norm_coupled": rank_coupled.get(ctrl, float("nan")),
            "kendall_tau_rmse": tau_rmse,
            "kendall_tau_estar": tau_estar,
            "coupling_strength": COUPLING_STRENGTH,
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\n  Summary → {SUMMARY_CSV}")

    rep_lines = [
        "=" * 70,
        f"COUPLED-PROXY ABLATION — KR6, eps_c = {COUPLING_STRENGTH}",
        "=" * 70, "",
        f"  Controller         Decoupled       Coupled       Delta      Delta%",
        f"  {'-'*12}       {'-'*12}  {'-'*12}  {'-'*10} {'-'*10}",
    ]
    for ctrl in controllers_sorted:
        v_c = rank_coupled.get(ctrl, float("nan"))
        v_d = rank_decoupled.get(ctrl, float("nan"))
        if np.isfinite(v_c) and np.isfinite(v_d):
            delta = v_c - v_d
            pct = 100 * delta / max(abs(v_d), 1e-12)
            rep_lines.append(
                f"  {ctrl:<12}      {v_d:>12.4f}  {v_c:>12.4f}  "
                f"{delta:>+10.4f}  {pct:>+9.1f}%"
            )
    rep_lines += [
        "",
        f"  Decoupled ranking: {' < '.join(rank_decoupled.keys()) if rank_decoupled else 'N/A'}",
        f"  Coupled ranking:   {' < '.join(controllers_sorted)}",
        f"  Kendall tau_b (RMSE_norm): {tau_rmse:+.3f}",
    ]
    if np.isfinite(tau_estar):
        rep_lines.append(f"  Kendall tau_b (E_star):    {tau_estar:+.3f}")
    rep_lines += ["", "INTERPRETATION"]
    if np.isfinite(tau_rmse) and tau_rmse >= 0.8:
        rep_lines += [
            f"  Rankings highly stable under coupling (tau_b = {tau_rmse:+.3f} >= 0.8).",
            "  -> CTC-PD recommendation is robust to structural mismatch at this eps_c.",
        ]
    elif np.isfinite(tau_rmse) and tau_rmse >= 0.3:
        rep_lines += [
            f"  Rankings partially shift under coupling "
            f"(0.3 <= tau_b = {tau_rmse:+.3f} < 0.8).",
            "  -> Fidelity-dependent recommendation warranted at this eps_c.",
        ]
    elif np.isfinite(tau_rmse):
        rep_lines += [
            f"  Rankings invert under coupling (tau_b = {tau_rmse:+.3f} < 0.3).",
            "  -> Decoupled proxy rankings are unreliable for controller selection",
            "     at this eps_c.",
        ]
    else:
        rep_lines += [
            "  Decoupled baseline (Tier A) data unavailable; cannot compute tau_b.",
            "  Run analysis/run_experiments.py first to populate all_results.csv.",
        ]
    REPORT_TXT.write_text("\n".join(rep_lines))
    print(f"  Report  → {REPORT_TXT}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling-strength", type=float, default=None,
                    help="Override coupling strength (default: 0.3)")
    ap.add_argument("--output-tag", type=str, default=None,
                    help="Tag for output filenames, e.g. 'eps0p15'")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="Smoke-test option: restrict to the first N seeds (default: all 20).")
    args = ap.parse_args()

    if args.coupling_strength is not None:
        COUPLING_STRENGTH = args.coupling_strength
        print(f"  ** Coupling strength overridden to {COUPLING_STRENGTH}")

    if args.output_tag is not None:
        OUT_CSV = METRIC_DIR / f"coupled_ablation_results_{args.output_tag}.csv"
        SUMMARY_CSV = METRIC_DIR / f"coupled_ablation_summary_{args.output_tag}.csv"
        REPORT_TXT = METRIC_DIR / f"coupled_ablation_report_{args.output_tag}.txt"
        print(f"  ** Output tag: {args.output_tag}")

    if args.n_seeds is not None:
        SEEDS = SEEDS[:args.n_seeds]
        print(f"  ** Restricting to {len(SEEDS)} seeds (smoke test)")

    main()
"""Coupled-proxy sweep for UR5e."""

from __future__ import annotations

import sys
import time
from pathlib import Path
import argparse
import glob

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from twinbench.dynamics.coupled_proxy_plant import simulate_coupled_proxy
from twinbench.dynamics.robot_loader import (
    load_robot_spec, robot_spec_path_from_id,
    get_limits_from_spec, get_dynamics_cfg_for_params,
)
from twinbench.ids import (
    canon_robot_id, canon_scene_id, canon_traj_id,
    canon_ctrl_id, canon_params, URDF_ROBOTS,
)
from twinbench.scene_specs import SceneCatalog
import twinbench.constants as C

from experiments.run_main_grid import (
    control_torque, compute_metrics, gen_trajectory_for_run,
    scene_to_targets, seed_target, seed_noise,
    DOF, DT, T_BASE, VERSION,
    QD_CAP_FRAC, TAU_CAP_FRAC,
    DEFAULT_NOISE, DEFAULT_DISTURB,
)


ROBOT = "ur5e"
SCENE = "open_reach"
COUPLING_STRENGTH = 0.3   # overridable via --coupling-strength
TRAJECTORIES = list(C.TRAJECTORIES)
CONTROLLERS = list(C.CONTROLLERS)
PARAMS = list(C.PARAMS)
SEEDS = list(C.SEEDS)

METRIC_DIR = ROOT / "data"
METRIC_DIR.mkdir(exist_ok=True, parents=True)
TIER_B_PROXY_CSV = METRIC_DIR / "all_results.csv"
CATALOG = SceneCatalog.from_repo_root(ROOT)


def run_coupled_single(robot_id, scene_id, traj_id, ctrl_id, params_id, seed,
                       coupling_strength=COUPLING_STRENGTH):
    """One coupled-proxy run on UR5e (or any URDF robot). Uses URDF-derived
    I_proxy = diag(M(0)) from the robot spec, same as Tier B-proxy."""
    rid = canon_robot_id(robot_id)
    sid = canon_scene_id(scene_id)
    traj = canon_traj_id(traj_id)
    ctrl = canon_ctrl_id(ctrl_id)
    prm = canon_params(params_id)

    spec = load_robot_spec(robot_spec_path_from_id(rid, repo_root=ROOT))

    dyn = get_dynamics_cfg_for_params(spec, prm)
    I_eff = np.array(dyn["I_proxy"])
    B_eff = np.array(dyn["B_proxy"])
    G_eff = np.array(dyn["G_nom"])
    tau_max = np.array(dyn["tau_max"])
    qd_max_rad = np.array(dyn["qd_max_rad"])
    R_max = float(dyn["Rmax"])
    payload_used = float(dyn["payload_used"])

    dyn_nom = get_dynamics_cfg_for_params(spec, "nominal")
    I_nom = np.array(dyn_nom["I_proxy"])
    B_nom = np.array(dyn_nom["B_proxy"])
    G_nom = np.array(dyn_nom["G_nom"])

    dq_cap = QD_CAP_FRAC * qd_max_rad
    ddq_cap_raw = TAU_CAP_FRAC * tau_max / (I_eff + 1e-12)
    ddq_cap = np.clip(ddq_cap_raw, 10.0, 1000.0)

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

    # URDF robot for Pinocchio-based IK / FK
    urdf_robot = None
    if rid in URDF_ROBOTS:
        from twinbench.dynamics.urdf_plant import get_urdf_robot
        urdf_path = ROOT / C.URDF_PATHS[rid]
        urdf_robot = get_urdf_robot(urdf_path)

    rng_target = np.random.default_rng(seed_target(rid, sid, seed))
    q_start, q_goal, dh, qmin, qmax, scene_meta = scene_to_targets(
        spec, rid, sid, rng_target, urdf_robot=urdf_robot)

    base_out = {
        "robot": rid, "scene": sid, "trajectory": traj, "controller": ctrl,
        "params": prm, "seed": int(seed),
        "fidelity": "coupled_proxy",
        "coupling_strength": coupling_strength,
        "R_max_m": R_max, "payload_used_kg": payload_used,
        "robot_group": "urdf",
        "version": VERSION,
    }

    if not scene_meta.get("ik_ok", False):
        base_out["ik_ok"] = False
        base_out["rejected_reason"] = scene_meta.get("rejected_reason", "ik_fail")
        for col in ["RMSE_cart_norm", "Max_cart_err_norm", "E_star",
                    "Tau_rms_norm", "Jerk_rms_norm", "Sat_frac_any",
                    "Vel_violation_frac", "Acc_violation_frac",
                    "Jerk_violation_frac",
                    "RMSE_cart_m", "Max_cart_err_m", "Energy", "T_eff"]:
            base_out[col] = float("nan")
        base_out["cart_err_ok"] = False
        return base_out

    T = T_BASE
    t, qd, dqd, ddqd, T_eff, traj_info = gen_trajectory_for_run(
        traj, q_start, q_goal, T, DT, dyn_nom)

    rng_noise = np.random.default_rng(seed_noise(rid, sid, traj, prm, seed))

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

    metrics = compute_metrics(
        q=q_sim, dq_sim=dq_sim, tau_hist=tau_hist,
        ddq_hist=ddq_hist, sat_hist=sat_hist,
        qd=qd, dqd=dqd, t=t, T_eff=T_eff, dh=dh,
        R_max=R_max, payload_used=payload_used,
        tau_max=tau_max, dq_cap=dq_cap, ddq_cap=ddq_cap,
        urdf_robot=urdf_robot)

    out = dict(base_out)
    out["ik_ok"] = True
    out["T_eff"] = T_eff
    out["rejected_reason"] = ""
    out.update(metrics)
    out.update({f"traj_{k}": v for k, v in traj_info.items()})
    return out


def compute_rankings(df, metric="RMSE_cart_norm"):
    ok = df[df["ik_ok"].astype(bool) & df["cart_err_ok"].astype(bool)]
    if len(ok) == 0:
        return {}
    grp = ok.groupby("controller")[metric].mean().sort_values()
    return dict(grp)


def kendall_tau_from_rankings(rank_a, rank_b):
    from scipy.stats import kendalltau
    ctrls = sorted(set(rank_a.keys()) & set(rank_b.keys()))
    if len(ctrls) < 2:
        return float("nan"), float("nan")
    tau, p = kendalltau([rank_a[c] for c in ctrls],
                        [rank_b[c] for c in ctrls])
    return float(tau), float(p)


def run_one_eps(coupling_strength, output_tag, seeds=None):
    seeds = list(SEEDS) if seeds is None else list(seeds)
    total = len(TRAJECTORIES) * len(CONTROLLERS) * len(PARAMS) * len(seeds)
    print(f"\n{'='*70}")
    print(f"  UR5e Coupled Proxy Ablation (§IV-C replicate)")
    print(f"{'='*70}")
    print(f"  Robot: {ROBOT}   Coupling: {coupling_strength}   Runs: {total}")
    print(f"{'='*70}\n")

    t0 = time.time()
    results = []
    i = 0
    for traj in TRAJECTORIES:
        for ctrl in CONTROLLERS:
            for prm in PARAMS:
                for sd in seeds:
                    i += 1
                    res = run_coupled_single(
                        ROBOT, SCENE, traj, ctrl, prm, sd,
                        coupling_strength=coupling_strength)
                    results.append(res)
                    if i % 100 == 0 or i == total:
                        rmse = res.get("RMSE_cart_norm", np.nan)
                        ok = res.get("ik_ok", False)
                        el = time.time() - t0
                        eta = (total - i) / max(i / max(el, 1e-6), 1e-6)
                        print(f"  [{i:>4}/{total}] {traj}/{ctrl}/{prm}/s{sd:>2} "
                              f"RMSE_norm={rmse:.4f} ik_ok={ok} "
                              f"({el:.0f}s / ~{eta:.0f}s left)")

    df = pd.DataFrame(results)
    tag = f"_{output_tag}" if output_tag else ""
    out_csv = METRIC_DIR / f"coupled_ablation_ur5e{tag}.csv"
    sum_csv = METRIC_DIR / f"coupled_ablation_ur5e_summary{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv}")

    rank_coupled = compute_rankings(df)
    rank_decoupled = {}
    tau_rmse = float("nan")
    if TIER_B_PROXY_CSV.exists():
        df_all = pd.read_csv(TIER_B_PROXY_CSV)
        df_dec = df_all[
            (df_all["robot"] == ROBOT) &
            (df_all["scene"] == SCENE) &
            (df_all["fidelity"] == "proxy")
        ]
        if len(df_dec) > 0:
            rank_decoupled = compute_rankings(df_dec)
            tau_rmse, _ = kendall_tau_from_rankings(rank_decoupled, rank_coupled)

    print(f"\n  Decoupled ranking: "
          f"{' < '.join(rank_decoupled.keys()) if rank_decoupled else 'N/A'}")
    print(f"  Coupled ranking:   {' < '.join(rank_coupled.keys())}")
    print(f"  Kendall tau_b:     {tau_rmse:+.3f}")

    summary_rows = [{
        "controller": c,
        "RMSE_norm_decoupled": rank_decoupled.get(c, float("nan")),
        "RMSE_norm_coupled": rank_coupled.get(c, float("nan")),
        "kendall_tau_rmse": tau_rmse,
        "coupling_strength": coupling_strength,
    } for c in CONTROLLERS]
    pd.DataFrame(summary_rows).to_csv(sum_csv, index=False)
    print(f"  Summary -> {sum_csv}")


def summarize_sweep():
    """Aggregate all coupled_ablation_ur5e_summary_*.csv into a final report."""
    pattern = str(METRIC_DIR / "coupled_ablation_ur5e_summary_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No summary files found matching {pattern}")
        print("Run the sweep first (see USAGE in file header).")
        return
    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_csv = METRIC_DIR / "coupled_ur5e_sweep_summary.csv"
    combined.to_csv(out_csv, index=False)

    lines = [
        "=" * 70,
        "UR5e COUPLED-PROXY SWEEP — §IV-C mechanism replicate",
        "=" * 70, "",
        "Per-eps_c stable-3 ranking and tau_b vs decoupled URDF-proxy:",
        "-" * 70,
    ]
    stable3 = [c for c in CONTROLLERS if c != "adaptive"]
    for eps in sorted(combined.coupling_strength.unique()):
        sub = combined[combined.coupling_strength == eps]
        sub3 = sub[sub.controller.isin(stable3)].sort_values("RMSE_norm_coupled")
        order = sub3.controller.tolist()
        tau = sub.kendall_tau_rmse.iloc[0] if len(sub) > 0 else float("nan")
        lines.append(f"  eps_c = {eps:>5.2f}   "
                     f"order: {' > '.join(order):35s}   tau_b = {tau:+.3f}")

    taus = combined.groupby("coupling_strength")["kendall_tau_rmse"].first()
    tau_min = float(taus.min()) if len(taus) else float("nan")
    lines += ["", "INTERPRETATION"]
    if np.isfinite(tau_min) and tau_min >= 0.999:
        lines += [
            "  All tau_b = 1.000 -> KR6 mechanism result REPLICATES on UR5e.",
            "  Strengthen §V-A: two-robot evidence for configuration-dependence.",
        ]
    else:
        lines += [
            f"  tau_b falls to {tau_min:+.3f} on UR5e -> mechanism does NOT",
            "  fully replicate. Coupling alone shifts some rankings on UR5e.",
            "  Scope §V-A mechanism claim to KR6; update abstract accordingly.",
        ]

    report_path = METRIC_DIR / "coupled_ur5e_sweep_report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nCombined -> {out_csv}")
    print(f"Report   -> {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling-strength", type=float, default=None)
    ap.add_argument("--output-tag", type=str, default="")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--num-seeds", "--n-seeds", type=int, default=None,
                    dest="num_seeds",
                    help="Smoke-test option: restrict to the first N seeds (default: all). "
                         "Both --num-seeds and --n-seeds are accepted.")
    args = ap.parse_args()

    if args.summarize:
        summarize_sweep()
    else:
        eps = args.coupling_strength if args.coupling_strength is not None else COUPLING_STRENGTH
        seeds = list(SEEDS) if args.num_seeds is None else list(SEEDS)[:args.num_seeds]
        run_one_eps(eps, args.output_tag, seeds=seeds)
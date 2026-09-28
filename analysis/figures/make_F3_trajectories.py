"""Generate trajectory comparison figure."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from twinbench.ids import canon_robot_id, canon_scene_id, URDF_ROBOTS
from twinbench.dynamics.robot_loader import (
    load_robot_spec, robot_spec_path_from_id,
    get_limits_from_spec, get_dynamics_cfg_for_params,
)
from experiments.run_main_grid import (
    DOF, DT, T_BASE, G0, DDQ_MA_WIN,
    gen_trajectory_for_run, scene_to_targets,
    seed_target, moving_average, CATALOG,
    TAU_CAP_FRAC,
)
import twinbench.constants as C

FIG_DIR = ROOT / "figures"
MET_DIR = ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

DEMO_ROBOTS  = ["kr6", "ur5e"]
DEMO_SCENE   = "open_reach"
DEMO_PARAMS  = "nominal"
DEMO_SEEDS   = {"kr6": 0, "ur5e": 2}
PLOT_JOINT   = 1          # 0-based -> Joint 2 (shoulder)
TRAJECTORIES = ["cubic", "minjerk", "fsmj"]

# Match original colors/labels exactly
TRAJ_COLORS = {"cubic": "#1f77b4", "minjerk": "#ff7f0e", "fsmj": "#2ca02c"}
TRAJ_LABELS = {"cubic": "Cubic", "minjerk": "Min-Jerk", "fsmj": "FSMJ"}


# Proxy torque helper
def torque_proxy(Q, DQ, DDQ, I_nom, B_nom, G_nom):
    I = I_nom.reshape(1, -1)
    B = B_nom.reshape(1, -1)
    G = G_nom.reshape(1, -1)
    return I * DDQ + B * DQ + G * np.sin(Q)


# Run one robot — return traj_data dict + dyn_nom + meta
def run_one_robot(robot_id: str) -> Dict:
    rid = canon_robot_id(robot_id)
    sid = canon_scene_id(DEMO_SCENE)

    spec    = load_robot_spec(robot_spec_path_from_id(rid, repo_root=ROOT))
    dyn_nom = get_dynamics_cfg_for_params(spec, DEMO_PARAMS)

    urdf_robot = None
    if rid in URDF_ROBOTS:
        from twinbench.dynamics.urdf_plant import get_urdf_robot
        urdf_path = ROOT / C.URDF_PATHS[rid]
        urdf_robot = get_urdf_robot(urdf_path)

    seed = DEMO_SEEDS.get(rid, 0)
    rng  = np.random.default_rng(seed_target(rid, sid, seed))
    q_start, q_goal, dh, qmin, qmax, meta = scene_to_targets(
        spec, rid, sid, rng, urdf_robot=urdf_robot
    )

    if not meta.get("ik_ok", False):
        raise RuntimeError(
            f"IK failed: robot={rid}, scene={sid}. "
            f"reason={meta.get('rejected_reason', '?')}"
        )

    print(f"\n  {rid}/{sid}: target at "
          f"({meta.get('x_m',0):.3f}, "
          f"{meta.get('y_m',0):.3f}, "
          f"{meta.get('z_m',0):.3f})")

    traj_data = {}
    for traj_id in TRAJECTORIES:
        t, qd, dqd, ddqd, T_eff, info = gen_trajectory_for_run(
            traj_id, q_start, q_goal, T_BASE, DT, dyn_nom
        )
        traj_data[traj_id] = dict(
            t=t, qd=qd, dqd=dqd, ddqd=ddqd, T_eff=T_eff, info=info
        )
        print(f"    {traj_id}: T_eff={T_eff:.3f}s, N={len(t)}")

    return dict(
        rid=rid, sid=sid,
        traj_data=traj_data,
        dyn_nom=dyn_nom,
        meta=meta,
    )


# Combined 4×2 PDF figure
def plot_combined(results: List[Dict], joint: int, out_path: Path) -> None:
    """
    4 rows × 2 columns, PDF output.
    Rows   : Position | Velocity | Acceleration | Proxy Torque
    Columns: kr6 (left) | ur5e (right)

    Style identical to original per-robot plot_trajectory_comparison:
      - same colors (#1f77b4 / #ff7f0e / #2ca02c)
      - per-panel legend with T_eff in label
      - τ_cap dashed red lines in torque panel
      - bold suptitle per column as top-panel title
      - grid alpha=0.3
    """
    fig, axes = plt.subplots(
        4, 2,
        figsize=(14.0, 10.0),
        sharex="col",
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )

    row_ylabels = [
        f"q [rad]",
        "dq [rad/s]",
        "ddq [rad/s²]",
        "τ [Nm]",
    ]
    row_titles = [
        f"Position — Joint {joint+1}",
        "Velocity",
        "Acceleration",
        "Proxy Torque (nominal)",
    ]

    for col, res in enumerate(results):
        rid      = res["rid"]
        sid      = res["sid"]
        dyn_nom  = res["dyn_nom"]
        meta     = res["meta"]
        traj_data = res["traj_data"]

        I_nom = np.array(dyn_nom["I_proxy"])
        B_nom = np.array(dyn_nom["B_proxy"])
        G_nom = np.array(dyn_nom["G_nom"])
        tau_max = float(np.array(dyn_nom["tau_max"])[joint])
        cap     = TAU_CAP_FRAC * tau_max

        axes[0, col].set_title(
            f"Trajectory Comparison — {rid} / {sid} / Joint {joint+1}\n"
            f"target=({meta.get('x_m',float('nan')):.3f}, "
            f"{meta.get('y_m',float('nan')):.3f}, "
            f"{meta.get('z_m',float('nan')):.3f}) m",
            fontsize=10, fontweight="bold",
        )

        for traj_id, data in traj_data.items():
            t    = data["t"]
            qd   = data["qd"]
            dqd  = data["dqd"]
            ddqd = data["ddqd"]
            T_eff = data["T_eff"]
            c    = TRAJ_COLORS.get(traj_id, "gray")
            lab  = f"{TRAJ_LABELS.get(traj_id, traj_id)} (T={T_eff:.2f}s)"

            tau = torque_proxy(qd, dqd, ddqd, I_nom, B_nom, G_nom)

            axes[0, col].plot(t, qd[:, joint],   color=c, lw=1.5, label=lab)
            axes[1, col].plot(t, dqd[:, joint],  color=c, lw=1.5, label=lab)
            axes[2, col].plot(t, ddqd[:, joint], color=c, lw=1.5, label=lab)
            axes[3, col].plot(t, tau[:, joint],  color=c, lw=1.5, label=lab)

        axes[3, col].axhline(
            cap,  color="r", linestyle="--", alpha=0.5, lw=0.8,
            label=f"τ_cap={cap:.1f}",
        )
        axes[3, col].axhline(
            -cap, color="r", linestyle="--", alpha=0.5, lw=0.8,
        )
        axes[3, col].set_xlabel("Time [s]")

        for row in range(4):
            ax = axes[row, col]
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_ylabels[row])
            # Row title only on left column to avoid duplication;
            # right column gets same title for symmetry
            ax.set_title(row_titles[row] if row > 0 else ax.get_title(),
                         fontsize=9 if row > 0 else ax.title.get_fontsize())
            ax.legend(fontsize=8 if row == 3 else 9)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved -> {out_path}")


# Metrics helper
def collect_metrics(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        rid     = res["rid"]
        sid     = res["sid"]
        dyn_nom = res["dyn_nom"]
        I_nom   = np.array(dyn_nom["I_proxy"])
        B_nom   = np.array(dyn_nom["B_proxy"])
        G_nom   = np.array(dyn_nom["G_nom"])
        R_max   = float(dyn_nom["Rmax"])
        payload = float(dyn_nom["payload_used"])

        for traj_id, data in res["traj_data"].items():
            t, qd, dqd, ddqd, T_eff = (
                data["t"], data["qd"], data["dqd"], data["ddqd"], data["T_eff"]
            )
            tau   = torque_proxy(qd, dqd, ddqd, I_nom, B_nom, G_nom)
            power = np.sum(np.abs(tau * dqd), axis=1)
            energy = float(np.trapezoid(power, t))
            E_star = energy / (payload * G0 * max(R_max, 1e-9))
            rows.append({
                "robot": rid, "scene": sid, "trajectory": traj_id,
                "T_eff": T_eff, "Energy": energy, "E_star": E_star,
                "max_ddq": float(np.max(np.abs(ddqd))),
                "max_tau": float(np.max(np.abs(tau))),
            })
    return pd.DataFrame(rows)


# Main
def main():
    print(f"\n{'='*60}")
    print(f"  Trajectory Figures — Combined PDF (F3)")
    print(f"{'='*60}")

    results = []
    for robot_id in DEMO_ROBOTS:
        print(f"\n[RUN] {robot_id} / {DEMO_SCENE} ...")
        res = run_one_robot(robot_id)
        results.append(res)
        print(f"[OK]  {robot_id} done.")

    out_pdf = FIG_DIR / "F3_combined_open_reach.pdf"
    plot_combined(results, joint=PLOT_JOINT, out_path=out_pdf)

    df = collect_metrics(results)
    csv_path = MET_DIR / "trajectory_comparison_F3_combined.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK]  Metrics saved -> {csv_path}")
    print(f"[OK]  Figure  saved -> {out_pdf}  (PDF, vector)")
    print(df[["robot", "trajectory", "T_eff", "E_star", "max_ddq"]].to_string(index=False))
    print(f"\n[OK] F3 combined figure complete.\n")


if __name__ == "__main__":
    main()
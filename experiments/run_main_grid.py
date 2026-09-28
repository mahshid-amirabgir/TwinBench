"""Main multi-fidelity experiment grid and related ablations.

Runs proxy and rigid-body tiers across robots, scenes, trajectories, and controllers.
Results are written to data/all_results_<tag>.csv.
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from twinbench.ids import (
    canon_robot_id, canon_scene_id, canon_traj_id,
    canon_ctrl_id, canon_params, canon_fidelity_id,
    URDF_ROBOTS,
)
from twinbench.scene_specs import SceneCatalog
from twinbench.dynamics.robot_loader import (
    load_robot_spec, robot_spec_path_from_id,
    build_dh_from_spec, get_limits_from_spec,
    get_dynamics_cfg_for_params,
)
from twinbench.dynamics.robot_model import fk_mdh, ik_dls
from twinbench.trajectories.generation import (
    cubic_traj, minjerk_traj, fsmj_trajectory,
)
import twinbench.constants as C
import argparse

G0 = C.G0
T_BASE = C.T_NOM
DT = C.SIM_DT
DDQ_MA_WIN = C.DDQ_MA_WIN
WN_FRAC = C.WN_FRAC
ZETA = C.ZETA
QD_CAP_FRAC = C.QD_CAP_FRAC
TAU_CAP_FRAC = C.TAU_CAP_FRAC
TAU_CAP_FRAC_SMC = C.TAU_CAP_FRAC_SMC
SMC_PHI_FRAC = C.SMC_PHI_FRAC
SMC_KS_FRAC = C.SMC_KS_FRAC
ADAPTIVE_INIT_FRAC = C.ADAPTIVE_INIT_FRAC
ADAPTIVE_GAMMA_FRAC = C.ADAPTIVE_GAMMA_FRAC
DEFAULT_NOISE = C.DEFAULT_NOISE
DEFAULT_DISTURB = C.DEFAULT_DISTURB

METRIC_DIR = ROOT / "data"
METRIC_DIR.mkdir(exist_ok=True, parents=True)
RAW_OUT_CSV = METRIC_DIR / "all_results.csv"

DOF = 6
EPS = 1e-9
IK_MAX_ITERS = 200
IK_TOL_M = 1e-3
IK_ERR_GUARD_M = 5e-3
IK_STEP_CLIP_DEG = 10.0
IK_LAM = 1e-2
MAX_TRIES = 60

CATALOG = SceneCatalog.from_repo_root(ROOT)
VERSION = "twinbench"

# Module-level mismatch config — overridden by __main__ via CLI
PARAM_MISMATCH_FRAC = None
PARAM_MISMATCH_SEED = 42


# Seeds
def stable_int_seed(*parts, base=0):
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode()).hexdigest()
    return int((base + int(h[:8], 16)) % (2**31 - 1))

def seed_target(rid, sid, seed):
    return stable_int_seed("target", rid, sid, int(seed), base=11111)

def seed_noise(rid, sid, traj, prm, seed):
    return stable_int_seed("noise", rid, sid, traj, prm, int(seed), base=22222)


# FK / IK — works for both MDH robots and URDF robots
def ee_pos(q, dh=None, urdf_robot=None):
    if urdf_robot is not None:
        return urdf_robot.fk(q)
    return fk_mdh(q, dh)[:3, 3].copy()

def ee_pos_batch(Q, dh=None, urdf_robot=None):
    if urdf_robot is not None:
        return urdf_robot.fk_all_positions(Q)
    return np.array([fk_mdh(Q[i], dh)[:3, 3] for i in range(len(Q))])

def ik_pos(p_des, q0, dh, qmin, qmax):
    Td = np.eye(4)
    Td[:3, 3] = p_des
    q, ok, err = ik_dls(
        Td, q0, dh, iters=IK_MAX_ITERS, lam=IK_LAM,
        pos_only=True, pos_tol=IK_TOL_M,
        step_clip_deg=IK_STEP_CLIP_DEG, qmin=qmin, qmax=qmax,
    )
    ok = bool(ok) and (float(err) <= IK_ERR_GUARD_M)
    return q, ok, float(err)

def ik_pos_urdf(p_des, q0, urdf_robot, qmin, qmax):
    """IK for URDF robots using Pinocchio analytical Jacobian."""
    import pinocchio as pin
    q = q0.copy()
    model = urdf_robot.model
    data = urdf_robot.data
    frame_id = model.nframes - 1

    for _ in range(IK_MAX_ITERS):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = data.oMf[frame_id].translation.copy()
        e = p_des - p
        err = float(np.linalg.norm(e))
        if err <= IK_TOL_M:
            return q, True, err

        # Pinocchio analytical Jacobian (position rows only)
        J_full = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos = J_full[:3, :]  # top 3 rows = position Jacobian

        # DLS step
        H = J_pos.T @ J_pos + (IK_LAM**2) * np.eye(DOF)
        dq = np.linalg.solve(H, J_pos.T @ e)
        step_clip = np.deg2rad(IK_STEP_CLIP_DEG)
        dq = np.clip(dq, -step_clip, step_clip)
        q = np.clip(q + dq, qmin, qmax)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p = data.oMf[frame_id].translation.copy()
    err = float(np.linalg.norm(p_des - p))
    return q, (err <= IK_ERR_GUARD_M), err


# Scene sampling
def sample_open_reach(scene_spec, rng):
    R = float(scene_spec["Rmax_m"])
    sampling = scene_spec["sampling"]
    radii = np.array(sampling["radii_m"], float)
    fracs = np.array(sampling["radius_fractions"], float)
    az_range = sampling["azimuth_range_deg"]
    el_range = sampling["elevation_range_deg"]
    idx = int(rng.integers(0, len(radii)))
    r, shell = float(radii[idx]), float(fracs[idx])
    az = float(np.deg2rad(rng.uniform(az_range[0], az_range[1])))
    el = float(np.deg2rad(rng.uniform(el_range[0], el_range[1])))
    x, y, z = r*np.cos(el)*np.cos(az), r*np.cos(el)*np.sin(az), r*np.sin(el)
    return {"scene_type": "open_reach", "r_over_Rmax": shell,
            "x_m": float(x), "y_m": float(y), "z_m": float(z),
            "x_over_Rmax": x/R, "y_over_Rmax": y/R, "z_over_Rmax": z/R}

def sample_shelf_reach(scene_spec, rng):
    R = float(scene_spec["Rmax_m"])
    geom = scene_spec["geometry"]
    box = geom["geometry_m"]
    w, h, d = float(box["width_m"]), float(box["height_m"]), float(box["depth_m"])
    z0 = float(geom["vertical_offset_m"])
    cone_deg = float(geom["approach_cone_deg"])
    x_lo, x_hi = 0.4 * R, max(0.4 * R + 1e-9, d)
    for _ in range(500):
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(-w/2, w/2))
        z = float(rng.uniform(z0, z0 + h))
        v = np.array([x, y, z])
        nv = float(np.linalg.norm(v))
        angle = float(np.degrees(np.arccos(np.clip(v[0]/max(nv, 1e-12), -1, 1))))
        if angle <= cone_deg:
            return {"scene_type": "shelf_reach", "r_over_Rmax": nv/R,
                    "x_m": x, "y_m": y, "z_m": z,
                    "x_over_Rmax": x/R, "y_over_Rmax": y/R, "z_over_Rmax": z/R}
    return {"scene_type": "shelf_reach", "rejected_reason": "cone_sampling_failed"}

def scene_to_targets(spec, robot_id, scene_id, rng, urdf_robot=None):
    """Generate IK-solved targets. Uses MDH IK for proxy robots, Pinocchio IK for URDF robots."""
    rid = canon_robot_id(robot_id)
    sid = canon_scene_id(scene_id)

    if rid in URDF_ROBOTS and urdf_robot is not None:
        # URDF robot: use Pinocchio FK/IK
        dyn = get_dynamics_cfg_for_params(spec, "nominal")
        qmin = np.deg2rad(np.array(get_limits_from_spec(spec)[0], float))
        qmax = np.deg2rad(np.array(get_limits_from_spec(spec)[1], float))
        q_home = 0.5 * (qmin + qmax)
        dh = None  # not used for URDF robots
    else:
        # Proxy robot: use MDH FK/IK
        dh = build_dh_from_spec(spec)
        qmin = np.deg2rad(np.array(get_limits_from_spec(spec)[0], float))
        qmax = np.deg2rad(np.array(get_limits_from_spec(spec)[1], float))
        q_home = 0.5 * (qmin + qmax)

    scene_spec = CATALOG.build_scaled_scene(rid, sid)
    meta = {"scene_type": sid, "ik_ok": False}

    sampler = sample_open_reach if sid == "open_reach" else sample_shelf_reach

    for attempt in range(1, MAX_TRIES + 1):
        pt = sampler(scene_spec, rng)
        if pt.get("rejected_reason"):
            continue

        p_des = np.array([pt["x_m"], pt["y_m"], pt["z_m"]])

        if rid in URDF_ROBOTS and urdf_robot is not None:
            q_sol, ok, err = ik_pos_urdf(p_des, q_home, urdf_robot, qmin, qmax)
        else:
            q_sol, ok, err = ik_pos(p_des, q_home, dh, qmin, qmax)

        if ok:
            meta.update(pt)
            meta["ik_ok"] = True
            meta["ik_final_err_m"] = err
            meta["attempts"] = attempt
            return q_home, q_sol, dh, qmin, qmax, meta

    meta["rejected_reason"] = "ik_fail"
    meta["attempts"] = MAX_TRIES
    return q_home, q_home.copy(), dh, qmin, qmax, meta


# Trajectory generation
def gen_trajectory_for_run(traj_id, q0, qf, T, dt, dyn_cfg):
    N = max(int(round(T / dt)), 10)
    info = {}
    q0 = np.asarray(q0, float).ravel()
    qf = np.asarray(qf, float).ravel()

    if traj_id == "cubic":
        t, qd, dqd, ddqd, _ = cubic_traj(q0, qf, T, N)
        return t, qd, dqd, ddqd, T, info
    elif traj_id == "minjerk":
        t, qd, dqd, ddqd, _ = minjerk_traj(q0, qf, T, N)
        return t, qd, dqd, ddqd, T, info
    elif traj_id in ("fsmj", "fsmj_novia"):
        I_p = np.array(dyn_cfg["I_proxy"])
        B_p = np.array(dyn_cfg["B_proxy"])
        G_n = np.array(dyn_cfg["G_nom"])
        tau_m = np.array(dyn_cfg["tau_max"])
        qd_m = np.array(dyn_cfg["qd_max_rad"])
        t, qd, dqd, ddqd, _, s_star = fsmj_trajectory(
            q0, qf, T, I_p, B_p, G_n, tau_m, qd_m, n=N)
        T_eff = s_star * T
        info["s_star"] = s_star
        info["T_eff"] = T_eff
        return t, qd, dqd, ddqd, T_eff, info
    else:
        raise ValueError(f"Unknown trajectory: {traj_id}")


# Controllers
def compute_gains(ddq_cap):
    wn = WN_FRAC * np.sqrt(np.maximum(ddq_cap, 1e-6))
    return wn**2, 2.0 * ZETA * wn

def control_torque(q, dq, qd_k, dqd_k, ddqd_k, *,
                   I_nom, B_nom, G_nom, tau_max, ddq_cap, dq_cap,
                   ctrl, adaptive_state=None,
                   tau_cap_frac: float = TAU_CAP_FRAC,
                   tau_cap_frac_smc: float = TAU_CAP_FRAC_SMC):
    Kp, Kd = compute_gains(ddq_cap)
    e, de = qd_k - q, dqd_k - dq
    gq = G_nom * np.sin(q)

    if ctrl == "pdg":
        tau = I_nom * (Kp * e + Kd * de) + gq
        cap = tau_cap_frac * tau_max
    elif ctrl == "ctc_pd":
        tau = I_nom * (ddqd_k + Kp * e + Kd * de) + B_nom * dq + gq
        cap = tau_cap_frac * tau_max
    elif ctrl == "ctc_smc":
        wn_smc = 2.0 * np.pi / max(T_BASE, 1e-6)
        Lambda = 2.0 * ZETA * wn_smc
        s = de + Lambda * e
        tau_eq = I_nom * (ddqd_k + Lambda * de) + B_nom * dq + gq
        # Normalized Ks: scales with robot's natural PD torque level
        Ks = SMC_KS_FRAC * I_nom * np.maximum(Kp, 1.0)
        phi = SMC_PHI_FRAC * dq_cap
        tau = tau_eq + Ks * np.tanh(s / (phi + EPS))
        cap = tau_cap_frac_smc * tau_max
    elif ctrl == "pbc":
        # PBC (passivity-based). Fully independent, STATELESS branch (Decision 1.4):
        # it reads/writes no adaptive_state and allocates nothing any other
        # controller consumes.
        # Lambda must stay in sync with CTC-SMC Lambda. Duplicated inline on
        # purpose — do NOT refactor CTC-SMC's inline Lambda into a shared helper
        # (refactoring risks perturbing CTC-SMC's bit-identical output).
        wn_pbc = 2.0 * np.pi / max(T_BASE, 1e-6)
        Lambda = 2.0 * ZETA * wn_pbc
        # +Lambda reference velocity (qd_r = qd_d + Lambda*e, qdd_r = qdd_d + Lambda*de),
        # consistent with s = de + Lambda*e and CTC-SMC's v_eq. This is DELIBERATELY
        # the opposite sign from the adaptive branch's -Lambda (see below); do not
        # "harmonize" the two.
        qd_r = dqd_k + Lambda * e
        qdd_r = ddqd_k + Lambda * de
        s = de + Lambda * e
        # K_D derived from the Eq. (3) normalization protocol, not tuned:
        # K_D = I_nom (elementwise) * K_d, K_d = 2*ZETA*wn from the scheduled-gain
        # path (compute_gains -> Kd).
        K_D = I_nom * Kd
        tau = I_nom * qdd_r + B_nom * qd_r + gq + K_D * s
        cap = tau_cap_frac * tau_max   # 0.9 cap (pdg/ctc_pd branch), NOT the SMC cap
    elif ctrl == "adaptive":
        if adaptive_state is None:
            adaptive_state = {
                "I_hat": ADAPTIVE_INIT_FRAC * I_nom.copy(),
                "B_hat": ADAPTIVE_INIT_FRAC * B_nom.copy(),
                "G_hat": ADAPTIVE_INIT_FRAC * G_nom.copy(),
                "Gamma_I": ADAPTIVE_GAMMA_FRAC * np.maximum(np.abs(I_nom), 1e-6),
                "Gamma_B": ADAPTIVE_GAMMA_FRAC * np.maximum(np.abs(B_nom), 1e-6),
                "Gamma_G": ADAPTIVE_GAMMA_FRAC * np.maximum(np.abs(G_nom), 1e-6),
            }
        wn_a = 2.0 * np.pi / max(T_BASE, 1e-6)
        Lambda = 2.0 * ZETA * wn_a
        s = de + Lambda * e
        dq_r = dqd_k - Lambda * e
        ddq_r = ddqd_k - Lambda * de
        tau = (adaptive_state["I_hat"] * ddq_r + adaptive_state["B_hat"] * dq_r +
               adaptive_state["G_hat"] * np.sin(q) + Kp * e + Kd * de)
        adaptive_state["I_hat"] -= adaptive_state["Gamma_I"] * ddq_r * s * DT
        adaptive_state["B_hat"] -= adaptive_state["Gamma_B"] * dq_r * s * DT
        adaptive_state["G_hat"] -= adaptive_state["Gamma_G"] * np.sin(q) * s * DT
        cap = tau_cap_frac * tau_max
    else:
        raise ValueError(f"Unknown controller: {ctrl}")
    return np.asarray(tau, float), np.asarray(cap, float), adaptive_state


# Simulation — proxy plant
def moving_average(x, win):
    if win <= 1: return x.copy()
    win = int(win) | 1
    pad = win // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    return np.vstack([np.convolve(xpad[:, j], kernel, mode="valid")
                      for j in range(x.shape[1])]).T

def simulate_proxy(*, qd, dqd, ddqd, t, dh,
                   q_init, qmin, qmax,
                   I_nom, B_nom, G_nom, tau_max,
                   I_eff, B_eff, G_eff,
                   dq_cap, ddq_cap, ctrl,
                   tau_cap_frac: float = TAU_CAP_FRAC,
                   tau_cap_frac_smc: float = TAU_CAP_FRAC_SMC,
                   rng_noise, sigma_q, sigma_qdot,
                   disturb_joint, disturb_tau, disturb_onset, disturb_end):
    N, dt = len(t), float(t[1] - t[0]) if len(t) > 1 else DT
    q = np.zeros((N, DOF)); dq_sim = np.zeros((N, DOF))
    tau_hist = np.zeros((N, DOF)); ddq_hist = np.zeros((N, DOF))
    sat_hist = np.zeros((N, DOF), dtype=bool)
    q[0] = q_init; dq_sim[0] = dqd[0]
    adaptive_state = None
    # Safety clamps (prevent numerical overflow on poorly-conditioned proxy models)
    dq_clamp = 3.0 * dq_cap   # allow 3x velocity cap before clamping
    ddq_clamp = 500.0          # rad/s^2 hard limit
    for k in range(1, N):
        q_m = q[k-1] + rng_noise.normal(0, sigma_q, DOF)
        dq_m = dq_sim[k-1] + rng_noise.normal(0, sigma_qdot, DOF)
        tau_cmd, tau_cap, adaptive_state = control_torque(
            q_m, dq_m, qd[k], dqd[k], ddqd[k],
            I_nom=I_nom, B_nom=B_nom, G_nom=G_nom, tau_max=tau_max,
            ddq_cap=ddq_cap, dq_cap=dq_cap, ctrl=ctrl, adaptive_state=adaptive_state,
            tau_cap_frac=tau_cap_frac, tau_cap_frac_smc=tau_cap_frac_smc)
        if disturb_onset <= t[k] <= disturb_end:
            tau_cmd = tau_cmd.copy(); tau_cmd[disturb_joint] += disturb_tau
        sat_hist[k] = np.abs(tau_cmd) > (tau_cap + 1e-12)
        tau = np.clip(tau_cmd, -tau_cap, tau_cap); tau_hist[k] = tau
        ddq_k = (tau - B_eff * dq_sim[k-1] - G_eff * np.sin(q[k-1])) / (I_eff + 1e-12)
        ddq_k = np.clip(ddq_k, -ddq_clamp, ddq_clamp)
        ddq_hist[k] = ddq_k
        dq_sim[k] = np.clip(dq_sim[k-1] + ddq_k * dt, -dq_clamp, dq_clamp)
        q[k] = np.clip(q[k-1] + dq_sim[k] * dt, qmin, qmax)
    return q, dq_sim, tau_hist, ddq_hist, sat_hist


# Metrics
def compute_metrics(*, q, dq_sim, tau_hist, ddq_hist, sat_hist,
                    qd, dqd, t, T_eff, dh,
                    R_max, payload_used, tau_max, dq_cap, ddq_cap,
                    urdf_robot=None):
    N = len(t)
    dt = float(t[1] - t[0]) if N > 1 else DT
    try:
        p_des = ee_pos_batch(qd, dh=dh, urdf_robot=urdf_robot)
        p_act = ee_pos_batch(q, dh=dh, urdf_robot=urdf_robot)
        cart_err = np.linalg.norm(p_des - p_act, axis=1)
        RMSE_cart = float(np.sqrt(np.mean(cart_err**2)))
        Max_cart = float(np.max(cart_err))
        cart_ok = True
    except Exception:
        RMSE_cart = Max_cart = float("nan"); cart_ok = False

    RMSE_norm = RMSE_cart / max(R_max, 1e-12) if np.isfinite(RMSE_cart) else float("nan")
    Max_norm = Max_cart / max(R_max, 1e-12) if np.isfinite(Max_cart) else float("nan")
    power = np.sum(np.abs(tau_hist * dq_sim), axis=1)
    Energy = float(np.trapezoid(power, t))
    E_star = Energy / (payload_used * G0 * max(R_max, 1e-9))
    Tau_rms = float(np.sqrt(np.mean(tau_hist**2)))
    Tau_rms_norm = Tau_rms / max(float(np.max(tau_max)), 1e-12)
    ddq_s = moving_average(ddq_hist, DDQ_MA_WIN)
    jerk = np.gradient(ddq_s, dt, axis=0)
    Jerk_rms = float(np.sqrt(np.mean(jerk**2)))
    jerk_cap = ddq_cap / max(T_BASE, 0.1)  # jerk limit = ddq_cap / trajectory time
    Jerk_rms_norm = Jerk_rms / max(float(np.max(jerk_cap)), 1e-12)
    Sat = float(np.mean(np.any(sat_hist, axis=1)))
    Vel_v = float(np.mean(np.any(np.abs(dq_sim) > dq_cap.reshape(1,-1), axis=1)))
    Acc_v = float(np.mean(np.any(np.abs(ddq_hist) > ddq_cap.reshape(1,-1), axis=1)))
    Jerk_v = float(np.mean(np.any(np.abs(jerk) > jerk_cap.reshape(1,-1), axis=1)))
    return {"RMSE_cart_m": RMSE_cart, "Max_cart_err_m": Max_cart,
            "RMSE_cart_norm": RMSE_norm, "Max_cart_err_norm": Max_norm,
            "Energy": Energy, "E_star": E_star,
            "Tau_rms": Tau_rms, "Tau_rms_norm": Tau_rms_norm,
            "Jerk_rms": Jerk_rms, "Jerk_rms_norm": Jerk_rms_norm,
            "Sat_frac_any": Sat, "Vel_violation_frac": Vel_v,
            "Acc_violation_frac": Acc_v, "Jerk_violation_frac": Jerk_v,
            "cart_err_ok": cart_ok}


# Single run
def run_single(robot, scene, trajectory, controller, params, seed,
               fidelity="proxy", noise_cfg=None, disturb_cfg=None,
               tau_cap_frac_smc: float | None = None,
               symmetric_rigid_plant: bool = False,
               gain_inertia_source: str = "I_eff",
               custom_IBG_factors: tuple[float, float, float] | None = None,
               include_B_feedforward: bool = False,
               common_random_numbers: bool = False):
    """
    include_B_feedforward: default False preserves the original shipped
        behavior (rigid-tier ctc_pd/ctc_smc/pbc have NO B_nom feedforward term
        -- a confirmed mismatch against Table 2's caption). When True, adds
        Table 2's B_nom*qd (ctc_pd/ctc_smc) / B_nom*qd_r (pbc) term at the
        rigid tier too, using nominal B_nom at both tiers, so both tiers
        implement Table 2's laws exactly and differ only in the
        inertia/Coriolis/gravity terms. No effect on the proxy tier (already
        has this term) or on pdg (no B term at either tier per Table 2).
    common_random_numbers: default False preserves the original shipped
        behavior (seed_noise hashes on the uncertainty label `params`). When
        True, seed_noise is always called with "nominal" as the label
        regardless of the actual params level, so light/nominal/heavy share
        the same noise realization (common random numbers) for a cleaner
        controlled comparison across uncertainty levels.
    symmetric_rigid_plant: if True, at fidelity="rigid_body" the plant model is
        a ScaledURDFRobot with (I_factor, G_factor) taken from C.PARAM_SCALES[params]
        instead of the nominal urdf_robot -- i.e. the rigid-tier plant sees the same
        (I, G) uncertainty scaling the proxy tier already applies. B is unaffected
        (already scaled via B_eff regardless of this flag). Default False preserves
        exactly the original behavior (plant = nominal urdf_robot at every
        uncertainty level). Symmetric ablation only -- see
        experiments/run_symmetric_uncertainty_ablation.py.
    gain_inertia_source: "I_eff" (default, original behavior) or "I_nom" -- which
        inertia array feeds ddq_cap (hence omega_n, Kp, Kd, Phi) in the gain
        -normalization protocol. Default "I_eff" preserves exactly the original
        behavior.
    custom_IBG_factors: (I_factor, B_factor, G_factor) or None (default). When
        set, OVERRIDES the params-keyed (I,B,G) triple from C.PARAM_SCALES[params]
        with these three independent factors -- for one-factor-at-a-time ablations
        (e.g. I_factor=0.8 with B_factor=G_factor=1.0). Applies to I_eff/B_eff/G_eff
        (proxy tier physics) and, when symmetric_rigid_plant=True, to the
        ScaledURDFRobot's (I_factor, G_factor) at the rigid tier too (B_eff is
        shared). `params` is still used for seeding (seed_noise depends on it) and
        should be passed as "light" or "heavy" as usual; only the ACTUAL (I,B,G)
        values are overridden, not the seed derivation.
    """

    # Read module-level mismatch config (set by __main__)
    mismatch_frac = globals().get('PARAM_MISMATCH_FRAC', None)
    mismatch_seed = globals().get('PARAM_MISMATCH_SEED', 42)

    rid = canon_robot_id(robot)
    sid = canon_scene_id(scene)
    traj = canon_traj_id(trajectory)
    ctrl = canon_ctrl_id(controller)
    prm = canon_params(params)
    fid = canon_fidelity_id(fidelity)

    spec = load_robot_spec(robot_spec_path_from_id(rid, repo_root=ROOT))

    dyn = get_dynamics_cfg_for_params(spec, prm)
    I_eff = np.array(dyn["I_proxy"]); B_eff = np.array(dyn["B_proxy"])
    G_eff = np.array(dyn["G_nom"]); tau_max = np.array(dyn["tau_max"])
    qd_max_rad = np.array(dyn["qd_max_rad"])
    R_max = float(dyn["Rmax"]); payload_used = float(dyn["payload_used"])

    dyn_nom = get_dynamics_cfg_for_params(spec, "nominal")
    I_nom = np.array(dyn_nom["I_proxy"]); B_nom = np.array(dyn_nom["B_proxy"])
    G_nom = np.array(dyn_nom["G_nom"])

    if custom_IBG_factors is not None:
        _I_factor, _B_factor, _G_factor = custom_IBG_factors
        I_eff = I_nom * _I_factor
        B_eff = B_nom * _B_factor
        G_eff = G_nom * _G_factor

    dq_cap = QD_CAP_FRAC * qd_max_rad
    # Physical acceleration limit: ddq_max = tau_max / I_proxy
    # Clipped to [10, 1000] rad/s² to prevent extreme gains on low-inertia distal joints
    if gain_inertia_source == "I_nom":
        _I_for_gains = I_nom
    elif gain_inertia_source == "I_eff":
        _I_for_gains = I_eff
    else:
        raise ValueError(f"gain_inertia_source must be 'I_eff' or 'I_nom', got {gain_inertia_source!r}")
    ddq_cap_raw = TAU_CAP_FRAC * tau_max / (_I_for_gains + 1e-12)
    ddq_cap = np.clip(ddq_cap_raw, 10.0, 1000.0)

    ncfg = noise_cfg or DEFAULT_NOISE
    sigma_q = float(ncfg.get("sigma_q", 0.002))
    sigma_qdot = float(ncfg.get("sigma_qdot", 0.01))

    dcfg = disturb_cfg or DEFAULT_DISTURB
    dj = int(dcfg.get("joint_index0", 1))
    d_frac = float(dcfg.get("tau_frac_of_max", 0.05))
    d_onset = float(dcfg.get("onset", 1.0))
    d_dur = float(dcfg.get("duration", 0.2))
    d_end = d_onset + d_dur
    d_tau = d_frac * float(tau_max[dj])

    # Load URDF robot if needed
    urdf_robot = None
    if rid in URDF_ROBOTS:
        from twinbench.dynamics.urdf_plant import get_urdf_robot
        urdf_path = ROOT / C.URDF_PATHS[rid]
        urdf_robot = get_urdf_robot(urdf_path)

    # Scene targets
    rng_target = np.random.default_rng(seed_target(rid, sid, seed))
    q_start, q_goal, dh, qmin, qmax, scene_meta = scene_to_targets(
        spec, rid, sid, rng_target, urdf_robot=urdf_robot)

    base_out = {
        "robot": rid, "scene": sid, "trajectory": traj, "controller": ctrl,
        "params": prm, "seed": int(seed), "fidelity": fid,
        "R_max_m": R_max, "payload_used_kg": payload_used,
        "robot_group": "urdf" if rid in URDF_ROBOTS else "proxy",
        "version": VERSION,
        "sim_dt": DT,
        "target_x_m": float(scene_meta.get("x_m", np.nan)),
        "target_y_m": float(scene_meta.get("y_m", np.nan)),
        "target_z_m": float(scene_meta.get("z_m", np.nan)),
        "parametric_mismatch_frac": float(mismatch_frac) if mismatch_frac else 0.0,
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

    T = T_BASE
    t, qd, dqd, ddqd, T_eff, traj_info = gen_trajectory_for_run(
        traj, q_start, q_goal, T, DT, dyn_nom)

    _noise_label = "nominal" if common_random_numbers else prm
    rng_noise = np.random.default_rng(seed_noise(rid, sid, traj, _noise_label, seed))

    tau_cap_frac_smc_eff = TAU_CAP_FRAC_SMC if tau_cap_frac_smc is None else float(tau_cap_frac_smc)

    if fid == "proxy":
        q_sim, dq_sim, tau_hist, ddq_hist, sat_hist = simulate_proxy(
            qd=qd, dqd=dqd, ddqd=ddqd, t=t, dh=dh,
            q_init=qd[0].copy(), qmin=qmin, qmax=qmax,
            I_nom=I_nom, B_nom=B_nom, G_nom=G_nom, tau_max=tau_max,
            I_eff=I_eff, B_eff=B_eff, G_eff=G_eff,
            dq_cap=dq_cap, ddq_cap=ddq_cap, ctrl=ctrl,
            tau_cap_frac=TAU_CAP_FRAC, tau_cap_frac_smc=tau_cap_frac_smc_eff,
            rng_noise=rng_noise, sigma_q=sigma_q, sigma_qdot=sigma_qdot,
            disturb_joint=dj, disturb_tau=d_tau,
            disturb_onset=d_onset, disturb_end=d_end)

    elif fid == "rigid_body":
        if rid not in URDF_ROBOTS or urdf_robot is None:
            raise ValueError(f"rigid_body fidelity only for URDF robots, got {rid}")
        from twinbench.dynamics.urdf_plant import simulate_urdf_full

        # Parametric mismatch: create a perturbed plant model
        urdf_robot_plant = None
        if mismatch_frac is not None and mismatch_frac > 0:
            from twinbench.dynamics.urdf_plant import get_perturbed_urdf_robot
            urdf_path = ROOT / C.URDF_PATHS[rid]
            urdf_robot_plant = get_perturbed_urdf_robot(
                urdf_path, mass_frac=mismatch_frac, seed=mismatch_seed)
        elif symmetric_rigid_plant:
            from twinbench.dynamics.urdf_plant import get_scaled_urdf_robot
            urdf_path = ROOT / C.URDF_PATHS[rid]
            if custom_IBG_factors is not None:
                I_factor, _, G_factor = custom_IBG_factors
            else:
                I_factor = C.PARAM_SCALES[prm]["I"]
                G_factor = C.PARAM_SCALES[prm]["G"]
            urdf_robot_plant = get_scaled_urdf_robot(
                urdf_path, I_factor=I_factor, G_factor=G_factor)

        q_sim, dq_sim, tau_hist, ddq_hist, sat_hist = simulate_urdf_full(
            urdf_robot=urdf_robot,
            urdf_robot_plant=urdf_robot_plant,
            qd=qd, dqd=dqd, ddqd=ddqd, t=t,
            q_init=qd[0].copy(), qmin=qmin, qmax=qmax,
            I_nom=I_nom, B_nom=B_nom, G_nom=G_nom, tau_max=tau_max,
            I_eff=I_eff, B_eff=B_eff, G_eff=G_eff,
            dq_cap=dq_cap, ddq_cap=ddq_cap, ctrl=ctrl,
            rng_noise=rng_noise, sigma_q=sigma_q, sigma_qdot=sigma_qdot,
            disturb_joint=dj, disturb_tau=d_tau,
            disturb_onset=d_onset, disturb_end=d_end,
            control_fn=control_torque,
            WN_FRAC=WN_FRAC, ZETA=ZETA, T_BASE=T_BASE, DT=DT,
            TAU_CAP_FRAC=TAU_CAP_FRAC, TAU_CAP_FRAC_SMC=tau_cap_frac_smc_eff,
            SMC_PHI_FRAC=SMC_PHI_FRAC, SMC_KS_FRAC=SMC_KS_FRAC,
            ADAPTIVE_INIT_FRAC=ADAPTIVE_INIT_FRAC,
            ADAPTIVE_GAMMA_FRAC=ADAPTIVE_GAMMA_FRAC,
            include_B_feedforward=include_B_feedforward)
    else:
        raise ValueError(f"Unknown fidelity: {fid}")

    metrics = compute_metrics(
        q=q_sim, dq_sim=dq_sim, tau_hist=tau_hist,
        ddq_hist=ddq_hist, sat_hist=sat_hist,
        qd=qd, dqd=dqd, t=t, T_eff=T_eff, dh=dh,
        R_max=R_max, payload_used=payload_used,
        tau_max=tau_max, dq_cap=dq_cap, ddq_cap=ddq_cap,
        urdf_robot=urdf_robot)

    out = dict(base_out)
    out["ik_ok"] = True; out["T_eff"] = T_eff; out["rejected_reason"] = ""
    out.update(metrics)
    out.update({f"traj_{k}": v for k, v in traj_info.items()})
    return out


# Grid runner
def run_tier(robots, scenes, fidelities, trajectories, controllers,
             params_list, seeds, *, label="", print_every=100,
             tau_cap_frac_smc: float | None = None):
    results = []
    total = len(robots)*len(scenes)*len(fidelities)*len(trajectories)*len(controllers)*len(params_list)*len(seeds)
    i = 0
    print(f"\n> {label}: {total} runs")
    for rid in robots:
        for sid in scenes:
            for fid in fidelities:
                for traj in trajectories:
                    for ctrl in controllers:
                        for prm in params_list:
                            for sd in seeds:
                                i += 1
                                res = run_single(
                                    rid, sid, traj, ctrl, prm, sd,
                                    fidelity=fid,
                                    tau_cap_frac_smc=tau_cap_frac_smc,
                                )
                                results.append(res)
                                if i % print_every == 0 or i == total:
                                    rmse = res.get("RMSE_cart_norm", np.nan)
                                    ok = res.get("ik_ok", False)
                                    print(f"  [{i}/{total}] {rid}/{sid}/{fid}/{traj}/{ctrl}/{prm} "
                                          f"RMSE={rmse:.4f} ik_ok={ok}")
    return results


def run_all(*, tau_cap_frac_smc: float | None = None,
            robots: list[str] | None = None,
            scenes: list[str] | None = None,
            fidelities: list[str] | None = None,
            trajectories: list[str] | None = None,
            controllers: list[str] | None = None,
            params: list[str] | None = None,
            seeds: list[int] | None = None):
    all_results = []
    params = list(C.PARAMS) if params is None else params
    seeds = list(C.SEEDS) if seeds is None else seeds
    trajs = list(C.TRAJECTORIES) if trajectories is None else trajectories
    ctrls = list(C.CONTROLLERS) if controllers is None else controllers

    scenes = ["open_reach", "shelf_reach"] if scenes is None else scenes
    fidelities = ["proxy", "rigid_body"] if fidelities is None else fidelities
    robots = list(C.ROBOTS) if robots is None else robots

    # Assign robots to their tiers based on registry
    robots_proxy = [r for r in robots if canon_robot_id(r) in C.ROBOTS_PROXY]
    robots_urdf = [r for r in robots if canon_robot_id(r) in C.ROBOTS_URDF]

    # Tier A: Proxy robots, Open Reach, proxy fidelity
    if ("open_reach" in scenes) and ("proxy" in fidelities) and robots_proxy:
        all_results.extend(run_tier(
            robots_proxy, ["open_reach"], ["proxy"],
            trajs, ctrls, params, seeds,
            label="Tier A (proxy robots, Open Reach, proxy)",
            tau_cap_frac_smc=tau_cap_frac_smc))

    # Tier B-proxy: URDF robots, Open Reach, proxy fidelity
    if ("open_reach" in scenes) and ("proxy" in fidelities) and robots_urdf:
        all_results.extend(run_tier(
            robots_urdf, ["open_reach"], ["proxy"],
            trajs, ctrls, params, seeds,
            label="Tier B-proxy (URDF robots, Open Reach, proxy)",
            tau_cap_frac_smc=tau_cap_frac_smc))

    # Tier B-full: URDF robots, Open Reach, rigid-body fidelity
    if ("open_reach" in scenes) and ("rigid_body" in fidelities) and robots_urdf:
        all_results.extend(run_tier(
            robots_urdf, ["open_reach"], ["rigid_body"],
            trajs, ctrls, params, seeds,
            label="Tier B-full (URDF robots, Open Reach, rigid_body)",
            tau_cap_frac_smc=tau_cap_frac_smc))

    # Tier C: Proxy robots, Shelf Reach, proxy fidelity
    if ("shelf_reach" in scenes) and ("proxy" in fidelities) and robots_proxy:
        all_results.extend(run_tier(
            robots_proxy, ["shelf_reach"], ["proxy"],
            trajs, ctrls, params, seeds,
            label="Tier C (proxy robots, Shelf Reach, proxy)",
            tau_cap_frac_smc=tau_cap_frac_smc))

    df = pd.DataFrame(all_results)

    # Fairness check
    ok_rows = df[df["ik_ok"].astype(bool)]
    if len(ok_rows) > 0:
        grp = ok_rows.groupby(["robot", "scene", "fidelity", "seed"])[
            ["target_x_m", "target_y_m", "target_z_m"]].nunique()
        bad = grp[(grp > 1).any(axis=1)]
        if len(bad) > 0:
            print(f"\nWARNING: FAIRNESS CHECK FAILED"); print(bad.head(10))
        else:
            print(f"\nOK: Fairness check passed")

    df.to_csv(RAW_OUT_CSV, index=False)
    print(f"\nOK: Saved {len(df)} results -> {RAW_OUT_CSV}")

    for key, sub in df.groupby(["robot_group", "scene", "fidelity"]):
        ok = sub["ik_ok"].astype(bool).sum()
        print(f"  {'/'.join(key)}: {len(sub)} runs, {ok} ik_ok ({100*ok/len(sub):.1f}%)")

    return df


if __name__ == "__main__":
    print(f"\n=== TwinBench — experiment runner ===")
    print(f"VERSION: {VERSION}")
    print(f"Proxy robots: {list(C.ROBOTS_PROXY)}")
    print(f"URDF robots:  {list(C.ROBOTS_URDF)}")
    print(f"Scenes: {list(C.SCENES)}")
    print(f"Trajectories: {list(C.TRAJECTORIES)}")
    print(f"Controllers: {list(C.CONTROLLERS)}")
    print(f"Seeds: {len(C.SEEDS)}")
    n_per_tier = len(C.TRAJECTORIES) * len(C.CONTROLLERS) * len(C.PARAMS) * len(C.SEEDS)
    tier_a = 3 * n_per_tier
    tier_bp = 3 * n_per_tier
    tier_bf = 3 * n_per_tier
    tier_c = 3 * n_per_tier
    total = tier_a + tier_bp + tier_bf + tier_c
    print(f"Tier A: {tier_a}, Tier B-proxy: {tier_bp}, Tier B-full: {tier_bf}, Tier C: {tier_c}")
    print(f"Total: {total} runs\n")

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--smc-tau-cap-frac", type=float, default=None,
                    help="Override CTC-SMC torque cap fraction (default uses constants.py).")
    ap.add_argument("--robots", type=str, default=None,
                    help="Comma-separated robot IDs to run (default: all).")
    ap.add_argument("--scenes", type=str, default=None,
                    help="Comma-separated scenes (open_reach,shelf_reach). Default: both.")
    ap.add_argument("--fidelities", type=str, default=None,
                    help="Comma-separated fidelities (proxy,rigid_body). Default: both.")
    ap.add_argument("--trajectories", type=str, default=None,
                    help="Comma-separated trajectories. Default: all.")
    ap.add_argument("--controllers", type=str, default=None,
                    help="Comma-separated controllers. Default: all.")
    ap.add_argument("--params", type=str, default=None,
                    help="Comma-separated params (light,nominal,heavy). Default: all.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated integer seeds. Default: constants.py (0..19).")
    ap.add_argument("--dt", type=float, default=None,
                    help="Override SIM_DT integration timestep in seconds. "
                         "Default: 0.001 (1 ms). Example: --dt 0.0005 for 0.5 ms.")
    ap.add_argument("--output-tag", type=str, default=None,
                    help="Tag appended to output CSV filename, e.g. 'dt0p5ms' "
                         "-> all_results_dt0p5ms.csv")
    ap.add_argument("--parametric-mismatch", type=float, default=None,
                    help="Fraction for URDF link-mass perturbation (e.g., 0.10 for ±10%%). "
                         "Plant uses perturbed masses; controller uses nominal URDF. "
                         "Only affects rigid_body fidelity.")
    ap.add_argument("--mismatch-seed", type=int, default=42,
                    help="RNG seed for parametric mismatch perturbation (default: 42).")
    args = ap.parse_args()

    def _split(s: str | None) -> list[str] | None:
        if s is None or str(s).strip() == "":
            return None
        return [x.strip() for x in str(s).split(",") if x.strip() != ""]

    def _split_ints(s: str | None) -> list[int] | None:
        xs = _split(s)
        if xs is None:
            return None
        return [int(x) for x in xs]

    if args.dt is not None:
        DT = args.dt            # module-level global used by control_torque (adaptive)
        C.SIM_DT = args.dt      # update constants module for any downstream readers
        print(f"  ** DT overridden to {DT} s ({DT*1000:.3f} ms)")

    if args.output_tag is not None:
        RAW_OUT_CSV = METRIC_DIR / f"all_results_{args.output_tag}.csv"
        print(f"  ** Output CSV: {RAW_OUT_CSV}")

    PARAM_MISMATCH_FRAC = args.parametric_mismatch
    PARAM_MISMATCH_SEED = args.mismatch_seed
    if PARAM_MISMATCH_FRAC is not None:
        print(f"  ** Parametric mismatch: ±{PARAM_MISMATCH_FRAC*100:.0f}% link masses "
              f"(seed={PARAM_MISMATCH_SEED})")

    run_all(
        tau_cap_frac_smc=args.smc_tau_cap_frac,
        robots=_split(args.robots),
        scenes=_split(args.scenes),
        fidelities=_split(args.fidelities),
        trajectories=_split(args.trajectories),
        controllers=_split(args.controllers),
        params=_split(args.params),
        seeds=_split_ints(args.seeds),
    )
    print("\nOK: All experiments complete.\n")
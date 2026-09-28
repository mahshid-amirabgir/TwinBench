"""Trajectory generation: cubic, minimum-jerk, FSMJ, and FSMJ without via-point."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from twinbench.constants import (
    G0, G_NOM_FRAC, QD_CAP_FRAC, TAU_CAP_FRAC,
    FSMJ_BSEARCH_EPSILON, FSMJ_BSEARCH_N_SAMPLES, FSMJ_BSEARCH_S_MIN,
)


# Proxy torque model
def torque_proxy(
    Q: np.ndarray,
    DQ: np.ndarray,
    DDQ: np.ndarray,
    I_proxy: np.ndarray,
    B_proxy: np.ndarray,
    G_nom: np.ndarray,
) -> np.ndarray:
    """
    Decoupled per-joint proxy dynamics:
      tau_i = I_eff_i * ddq_i + B_eff_i * dq_i + G_nom_i * sin(q_i)

    Where G_nom_i = G_NOM_FRAC * tau_max_i (= 0.25 * tau_max_i).

    Args:
        Q, DQ, DDQ: (N, n_joints) arrays
        I_proxy, B_proxy, G_nom: (n_joints,) arrays

    Returns:
        tau: (N, n_joints) array
    """
    I = np.asarray(I_proxy, dtype=float).reshape(1, -1)
    B = np.asarray(B_proxy, dtype=float).reshape(1, -1)
    G = np.asarray(G_nom, dtype=float).reshape(1, -1)

    return I * DDQ + B * DQ + G * np.sin(Q)


# Baseline trajectories
def cubic_traj(
    q0: np.ndarray,
    qf: np.ndarray,
    T: float,
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cubic rest-to-rest: q(0)=q0, q(T)=qf, dq(0)=dq(T)=0.
    Returns (t, Q, DQ, DDQ, DDDQ) each shape (n,) or (n, nJ).
    """
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    qf = np.asarray(qf, dtype=float).reshape(-1)
    nJ = q0.size
    t = np.linspace(0.0, T, n)
    T2, T3 = T**2, T**3

    Q = np.zeros((n, nJ))
    DQ = np.zeros_like(Q)
    DDQ = np.zeros_like(Q)
    DDDQ = np.zeros_like(Q)

    for j in range(nJ):
        dq = qf[j] - q0[j]
        a0 = q0[j]
        a2 = 3.0 * dq / T2
        a3 = -2.0 * dq / T3

        Q[:, j] = a0 + a2 * t**2 + a3 * t**3
        DQ[:, j] = 2.0 * a2 * t + 3.0 * a3 * t**2
        DDQ[:, j] = 2.0 * a2 + 6.0 * a3 * t
        DDDQ[:, j] = np.full_like(t, 6.0 * a3)

    return t, Q, DQ, DDQ, DDDQ


def minjerk_traj(
    q0: np.ndarray,
    qf: np.ndarray,
    T: float,
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimum-jerk rest-to-rest: q(0)=q0, q(T)=qf,
    dq(0)=dq(T)=0, ddq(0)=ddq(T)=0.
    5th-order polynomial.
    """
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    qf = np.asarray(qf, dtype=float).reshape(-1)
    nJ = q0.size
    t = np.linspace(0.0, T, n)
    s = t / T

    Q = np.zeros((n, nJ))
    DQ = np.zeros_like(Q)
    DDQ = np.zeros_like(Q)
    DDDQ = np.zeros_like(Q)

    for j in range(nJ):
        dq = qf[j] - q0[j]
        Q[:, j] = q0[j] + dq * (10*s**3 - 15*s**4 + 6*s**5)
        DQ[:, j] = dq * (30*s**2 - 60*s**3 + 30*s**4) / T
        DDQ[:, j] = dq * (60*s - 180*s**2 + 120*s**3) / T**2
        DDDQ[:, j] = dq * (60 - 360*s + 360*s**2) / T**3

    return t, Q, DQ, DDQ, DDDQ


# FSMJ: Feasibility-Scaled Minimum-Jerk
def _check_feasibility(
    q0: np.ndarray,
    qf: np.ndarray,
    T_scaled: float,
    I_proxy: np.ndarray,
    B_proxy: np.ndarray,
    G_nom: np.ndarray,
    tau_max: np.ndarray,
    qd_max_rad: np.ndarray,
    tau_cap_frac: float = TAU_CAP_FRAC,
    qd_cap_frac: float = QD_CAP_FRAC,
    n_samples: int = FSMJ_BSEARCH_N_SAMPLES,
) -> bool:
    """
    Check if a minimum-jerk trajectory with duration T_scaled is feasible
    under velocity and torque caps.

    Returns True if all constraints satisfied at all sample points.
    """
    _, Q, DQ, DDQ, _ = minjerk_traj(q0, qf, T_scaled, n=n_samples)

    # Velocity check
    qd_cap = qd_cap_frac * qd_max_rad
    if np.any(np.abs(DQ) > qd_cap.reshape(1, -1)):
        return False

    # Torque check
    tau = torque_proxy(Q, DQ, DDQ, I_proxy, B_proxy, G_nom)
    tau_cap = tau_cap_frac * tau_max
    if np.any(np.abs(tau) > tau_cap.reshape(1, -1)):
        return False

    return True


def _compute_max_violation_ratio(
    q0: np.ndarray,
    qf: np.ndarray,
    T_base: float,
    I_proxy: np.ndarray,
    B_proxy: np.ndarray,
    G_nom: np.ndarray,
    tau_max: np.ndarray,
    qd_max_rad: np.ndarray,
    tau_cap_frac: float,
    qd_cap_frac: float,
    n_samples: int,
) -> float:
    """
    Compute the maximum constraint violation ratio at T_base.
    This gives an upper bound for s_max in binary search.

    For velocity: ratio = max(|dq|) / (qd_cap_frac * qd_max)
    For torque: ratio = max(|tau|) / (tau_cap_frac * tau_max)

    Since velocity scales as 1/s and torque scales as ~1/s^2,
    s_max = max(vel_ratio, sqrt(tau_ratio)) is sufficient.
    """
    _, Q, DQ, DDQ, _ = minjerk_traj(q0, qf, T_base, n=n_samples)

    qd_cap = qd_cap_frac * qd_max_rad
    vel_ratios = np.abs(DQ) / np.maximum(qd_cap.reshape(1, -1), 1e-12)
    max_vel_ratio = float(np.max(vel_ratios))

    tau = torque_proxy(Q, DQ, DDQ, I_proxy, B_proxy, G_nom)
    tau_cap = tau_cap_frac * tau_max
    tau_ratios = np.abs(tau) / np.maximum(tau_cap.reshape(1, -1), 1e-12)
    max_tau_ratio = float(np.max(tau_ratios))

    # Velocity ~ 1/s, torque ~ 1/s^2
    s_from_vel = max(1.0, max_vel_ratio)
    s_from_tau = max(1.0, np.sqrt(max_tau_ratio))

    return max(s_from_vel, s_from_tau) * 1.5  # 50% margin


def fsmj_binary_search(
    q0: np.ndarray,
    qf: np.ndarray,
    T_base: float,
    I_proxy: np.ndarray,
    B_proxy: np.ndarray,
    G_nom: np.ndarray,
    tau_max: np.ndarray,
    qd_max_rad: np.ndarray,
    tau_cap_frac: float = TAU_CAP_FRAC,
    qd_cap_frac: float = QD_CAP_FRAC,
    epsilon: float = FSMJ_BSEARCH_EPSILON,
    n_samples: int = FSMJ_BSEARCH_N_SAMPLES,
) -> float:
    """
    Find minimal feasible scale factor s* >= 1 via binary search.

    Feasibility is monotonically relaxed as s increases:
      velocity ~ 1/s, acceleration ~ 1/s^2, jerk ~ 1/s^3, torque ~ 1/s^2
    So binary search on monotone feasibility converges to unique s*.

    Returns:
        s_star: minimal feasible scale factor (>= 1.0)
    """
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    qf = np.asarray(qf, dtype=float).reshape(-1)
    I_proxy = np.asarray(I_proxy, dtype=float).reshape(-1)
    B_proxy = np.asarray(B_proxy, dtype=float).reshape(-1)
    G_nom = np.asarray(G_nom, dtype=float).reshape(-1)
    tau_max = np.asarray(tau_max, dtype=float).reshape(-1)
    qd_max_rad = np.asarray(qd_max_rad, dtype=float).reshape(-1)

    # Check if already feasible at s=1
    if _check_feasibility(
        q0, qf, T_base, I_proxy, B_proxy, G_nom,
        tau_max, qd_max_rad, tau_cap_frac, qd_cap_frac, n_samples
    ):
        return 1.0

    # Compute s_max from violation ratios
    s_max = _compute_max_violation_ratio(
        q0, qf, T_base, I_proxy, B_proxy, G_nom,
        tau_max, qd_max_rad, tau_cap_frac, qd_cap_frac, n_samples
    )

    # Ensure s_max is actually feasible (expand if needed)
    max_expansion = 10
    for _ in range(max_expansion):
        if _check_feasibility(
            q0, qf, s_max * T_base, I_proxy, B_proxy, G_nom,
            tau_max, qd_max_rad, tau_cap_frac, qd_cap_frac, n_samples
        ):
            break
        s_max *= 2.0
    else:
        # If still infeasible after expansion, return s_max as best effort
        return s_max

    # Binary search
    s_lo = FSMJ_BSEARCH_S_MIN
    s_hi = s_max

    while (s_hi - s_lo) > epsilon:
        s_mid = 0.5 * (s_lo + s_hi)
        if _check_feasibility(
            q0, qf, s_mid * T_base, I_proxy, B_proxy, G_nom,
            tau_max, qd_max_rad, tau_cap_frac, qd_cap_frac, n_samples
        ):
            s_hi = s_mid
        else:
            s_lo = s_mid

    return s_hi


def fsmj_trajectory(
    q0: np.ndarray,
    qf: np.ndarray,
    T_base: float,
    I_proxy: np.ndarray,
    B_proxy: np.ndarray,
    G_nom: np.ndarray,
    tau_max: np.ndarray,
    qd_max_rad: np.ndarray,
    tau_cap_frac: float = TAU_CAP_FRAC,
    qd_cap_frac: float = QD_CAP_FRAC,
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    FSMJ trajectory: minimum-jerk profile with feasibility time-scaling.

    Returns (t, Q, DQ, DDQ, DDDQ, s_star).
    T_eff = s_star * T_base.
    """
    s_star = fsmj_binary_search(
        q0, qf, T_base,
        I_proxy, B_proxy, G_nom,
        tau_max, qd_max_rad,
        tau_cap_frac, qd_cap_frac,
    )

    T_eff = s_star * T_base
    t, Q, DQ, DDQ, DDDQ = minjerk_traj(q0, qf, T_eff, n=n)

    return t, Q, DQ, DDQ, DDDQ, s_star


# Alias for ablation (no via-point — just the binary search)
fsmj_novia_trajectory = fsmj_trajectory


# Effort metrics (reusable)
def effort_metrics(
    t: np.ndarray,
    DQ: np.ndarray,
    DDQ: np.ndarray,
    DDDQ: np.ndarray,
) -> Dict[str, float]:
    """Compute integrated squared velocity, acceleration, jerk."""
    from scipy.integrate import trapezoid
    return {
        "Vel2": float(trapezoid(np.sum(DQ**2, axis=1), t)),
        "Acc2": float(trapezoid(np.sum(DDQ**2, axis=1), t)),
        "Jerk2": float(trapezoid(np.sum(DDDQ**2, axis=1), t)),
    }

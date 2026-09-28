"""Coupled-proxy plant: diagonal proxy dynamics with tunable inter-joint coupling."""

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

DOF = 6


def coupled_proxy_dynamics(
    q: np.ndarray,
    dq: np.ndarray,
    tau: np.ndarray,
    I_eff: np.ndarray,
    B_eff: np.ndarray,
    G_eff: np.ndarray,
    coupling_strength: float = 0.3,
) -> np.ndarray:
    """
    Forward dynamics with coupling perturbations.

    M_coupled(q) * ddq + C_coupled(q,dq) * dq + B_eff * dq + G_eff * sin(q) = tau

    where:
      M_coupled(q) = diag(I_eff) + coupling_strength * M_off(q)
      C_coupled(q,dq) = coupling_strength * C_approx(q,dq)

    M_off(q) models configuration-dependent coupling between adjacent joints:
      M_off[i,j] = alpha * sqrt(I_i * I_j) * cos(q_i - q_j) for |i-j|=1

    C_approx(q,dq) models Coriolis-like velocity coupling:
      C_approx[i] = sum_j beta * sqrt(I_i * I_j) * sin(q_i - q_j) * dq_j for |i-j|<=2

    Args:
        q: joint positions (DOF,)
        dq: joint velocities (DOF,)
        tau: applied torques (DOF,)
        I_eff: effective inertia (DOF,) — from proxy model
        B_eff: viscous damping (DOF,)
        G_eff: gravity amplitude (DOF,)
        coupling_strength: 0.0 (proxy) to 1.0 (strong coupling)

    Returns:
        ddq: joint accelerations (DOF,)
    """
    q = np.asarray(q, float).ravel()
    dq = np.asarray(dq, float).ravel()
    tau = np.asarray(tau, float).ravel()
    I_eff = np.asarray(I_eff, float).ravel()
    B_eff = np.asarray(B_eff, float).ravel()
    G_eff = np.asarray(G_eff, float).ravel()
    cs = float(coupling_strength)

    # Base: diagonal mass matrix
    M = np.diag(I_eff.copy())

    # Add off-diagonal coupling (adjacent joints)
    if cs > 0:
        alpha = 0.3  # coupling coefficient (fraction of geometric mean inertia)
        for i in range(DOF):
            for j in range(DOF):
                if i == j:
                    continue
                if abs(i - j) == 1:
                    # Adjacent joint coupling — configuration dependent
                    m_ij = alpha * cs * np.sqrt(I_eff[i] * I_eff[j]) * np.cos(q[i] - q[j])
                    M[i, j] = m_ij
                elif abs(i - j) == 2:
                    # Second-neighbor coupling (weaker)
                    m_ij = 0.1 * alpha * cs * np.sqrt(I_eff[i] * I_eff[j]) * np.cos(q[i] - q[j])
                    M[i, j] = m_ij

    # Ensure M is positive definite (add small diagonal if needed)
    eigvals = np.linalg.eigvalsh(M)
    if np.min(eigvals) < 1e-6:
        M += (1e-6 - np.min(eigvals) + 1e-8) * np.eye(DOF)

    # Coriolis-like forces
    C_dq = np.zeros(DOF)
    if cs > 0:
        beta = 0.2  # Coriolis coefficient
        for i in range(DOF):
            for j in range(DOF):
                if i == j:
                    continue
                if abs(i - j) <= 2:
                    weight = 1.0 if abs(i - j) == 1 else 0.3
                    C_dq[i] += (weight * beta * cs *
                                np.sqrt(I_eff[i] * I_eff[j]) *
                                np.sin(q[i] - q[j]) * dq[j])

    # Total: M * ddq = tau - B*dq - G*sin(q) - C*dq
    rhs = tau - B_eff * dq - G_eff * np.sin(q) - C_dq

    # Solve M * ddq = rhs
    try:
        ddq = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        ddq = np.linalg.lstsq(M, rhs, rcond=None)[0]

    return ddq


def simulate_coupled_proxy(
    *, qd, dqd, ddqd, t, T_eff, dh,
    q_init, qmin, qmax,
    I_nom, B_nom, G_nom, tau_max,
    I_eff, B_eff, G_eff,
    dq_cap, ddq_cap, ctrl,
    rng_noise, sigma_q, sigma_qdot,
    disturb_joint, disturb_tau, disturb_onset, disturb_end,
    control_fn,
    coupling_strength: float = 0.3,
):
    """
    Closed-loop simulation on coupled proxy plant.
    SAME interface as simulate_proxy plus control_fn and coupling_strength.

    Controllers use PROXY parameters (I_nom, B_nom, G_nom) — decoupled.
    Plant uses COUPLED dynamics — structural mismatch.
    """
    N = len(t)
    dt = float(t[1] - t[0]) if N > 1 else 0.001

    q = np.zeros((N, DOF))
    dq_sim = np.zeros((N, DOF))
    tau_hist = np.zeros((N, DOF))
    ddq_hist = np.zeros((N, DOF))
    sat_hist = np.zeros((N, DOF), dtype=bool)

    q[0] = q_init
    dq_sim[0] = dqd[0] if len(dqd) > 0 else np.zeros(DOF)

    adaptive_state = None

    for k in range(1, N):
        q_meas = q[k-1] + rng_noise.normal(0, sigma_q, DOF)
        dq_meas = dq_sim[k-1] + rng_noise.normal(0, sigma_qdot, DOF)

        tau_cmd, tau_cap, adaptive_state = control_fn(
            q_meas, dq_meas, qd[k], dqd[k], ddqd[k],
            I_nom=I_nom, B_nom=B_nom, G_nom=G_nom,
            tau_max=tau_max, ddq_cap=ddq_cap, dq_cap=dq_cap,
            ctrl=ctrl, adaptive_state=adaptive_state,
        )

        if disturb_onset <= t[k] <= disturb_end:
            tau_cmd = tau_cmd.copy()
            tau_cmd[disturb_joint] += disturb_tau

        sat_hist[k] = np.abs(tau_cmd) > (tau_cap + 1e-12)
        tau = np.clip(tau_cmd, -tau_cap, tau_cap)
        tau_hist[k] = tau

        # COUPLED PROXY DYNAMICS
        ddq_k = coupled_proxy_dynamics(
            q[k-1], dq_sim[k-1], tau,
            I_eff, B_eff, G_eff,
            coupling_strength=coupling_strength,
        )
        ddq_hist[k] = ddq_k

        dq_sim[k] = dq_sim[k-1] + ddq_k * dt
        q[k] = q[k-1] + dq_sim[k] * dt
        q[k] = np.clip(q[k], qmin, qmax)

    return q, dq_sim, tau_hist, ddq_hist, sat_hist

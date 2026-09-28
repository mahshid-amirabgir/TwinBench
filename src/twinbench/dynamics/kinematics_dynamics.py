"""Proxy dynamics from cylinder geometry; FK/IK re-exported from robot_model."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from twinbench.constants import G0, PAYLOAD_FRAC

DOF = 6

# Re-export FK/IK from robot_model
from twinbench.dynamics.robot_model import (  # noqa: F401
    mdh_T,
    fk_mdh,
    vee_log,
    jac_numeric,
    ik_dls,
)


# DH Validation (testing utility)
def validate_dh(
    dh: Dict[str, Any],
    qmin_deg: Any,
    qmax_deg: Any,
    seeds: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """FK-IK self-consistency test. Uses full-pose IK for validation."""
    qmin = np.deg2rad(np.asarray(qmin_deg, dtype=float).reshape(-1))
    qmax = np.deg2rad(np.asarray(qmax_deg, dtype=float).reshape(-1))

    rng = np.random.default_rng(int(seed))
    posE: List[float] = []
    angE: List[float] = []
    ok = 0

    for _ in range(int(seeds)):
        q = qmin + rng.random(DOF) * (qmax - qmin)
        Td = fk_mdh(q, dh)
        q0 = q + np.deg2rad(5.0) * rng.standard_normal(DOF)

        qh, succ, _ = ik_dls(Td, q0, dh, pos_only=False, qmin=qmin, qmax=qmax)
        Trec = fk_mdh(qh, dh)
        dp = float(np.linalg.norm(Td[:3, 3] - Trec[:3, 3]))
        dw = float(np.linalg.norm(vee_log(Trec[:3, :3].T @ Td[:3, :3])))

        posE.append(dp)
        angE.append(float(np.rad2deg(dw)))
        if succ and dp <= 1e-3 and float(np.rad2deg(dw)) <= 0.5:
            ok += 1

    posE_arr = np.array(posE, dtype=float)
    angE_arr = np.array(angE, dtype=float)

    return {
        "pos_rmse_mm":  1000.0 * float(np.sqrt(np.mean(posE_arr ** 2))),
        "pos_max_mm":   1000.0 * float(np.max(posE_arr)),
        "ang_rmse_deg": float(np.sqrt(np.mean(angE_arr ** 2))),
        "ang_max_deg":  float(np.max(angE_arr)),
        "success_pct":  100.0 * float(ok) / float(seeds),
        "seeds":        float(seeds),
    }


# Cylinder geometry -> mass/inertia
def cylinder_mass_inertia(
    L: float, r: float, rho: float = 7850.0
) -> Tuple[float, float, float, float]:
    """
    Solid cylinder aligned with local z-axis.
    Returns (m, Ixx, Iyy, Izz).

    Ixx = Iyy = (1/12)*m*(3*r^2 + L^2) — transverse (joint rotation) axis
    Izz = 0.5*m*r^2 — spin axis (NOT used for joint inertia)
    """
    L, r, rho = float(L), float(r), float(rho)
    m = rho * np.pi * r**2 * L
    Izz = 0.5 * m * r**2
    Ixx = Iyy = (1.0 / 12.0) * m * (3.0 * r**2 + L**2)
    return float(m), float(Ixx), float(Iyy), float(Izz)


def make_robot_dynamics(
    dyn_cfg: Dict[str, Any],
    *,
    rho: float = 7850.0,
    g0: float | None = None,
) -> Dict[str, Any]:
    """Build link-parameter envelopes from dynamics config dict."""
    g0_val = float(G0 if g0 is None else g0)

    Rmax = float(dyn_cfg["Rmax"])
    rated_payload = float(dyn_cfg["rated_payload"])
    qd_max_deg = np.asarray(dyn_cfg["qd_max_deg"], dtype=float).reshape(-1)
    link_lengths = np.asarray(dyn_cfg["link_lengths"], dtype=float).reshape(-1)
    link_radii = np.asarray(dyn_cfg["link_radii"], dtype=float).reshape(-1)

    qd_max = np.deg2rad(qd_max_deg)
    masses, inertias = [], []

    for L, r in zip(link_lengths, link_radii):
        m, Ixx, Iyy, Izz = cylinder_mass_inertia(float(L), float(r), rho=float(rho))
        masses.append(m)
        inertias.append({"ixx": Ixx, "iyy": Iyy, "izz": Izz,
                         "ixy": 0.0, "ixz": 0.0, "iyz": 0.0})

    # Viscous proxy rule:
    # at 40% qd_max, viscous torque = 2% of effective gravitational load
    m_eff = float(PAYLOAD_FRAC) * rated_payload
    tau_eff = m_eff * g0_val * (0.5 * Rmax)
    qd_safe = np.maximum(qd_max, 1e-9)
    b_nom = (0.02 * tau_eff) / (0.4 * qd_safe)

    return {
        "Rmax": Rmax,
        "rated_payload": rated_payload,
        "qd_max_deg": qd_max_deg.tolist(),
        "link_lengths": link_lengths.tolist(),
        "link_radii": link_radii.tolist(),
        "mass_nominal": [float(x) for x in masses],
        "inertia_nominal": inertias,
        "viscous_nominal": b_nom.tolist(),
    }


def derive_proxies_from_cylinders(
    dyn_cfg: Dict[str, Any],
    *,
    rho: float = 7850.0,
    g0: float | None = None,
) -> Tuple[List[float], List[float]]:
    """
    Derive (I_proxy, B_proxy) from cylinder geometry.
    Uses Ixx (transverse axis) for joint rotation inertia.
    """
    dyn = make_robot_dynamics(dyn_cfg, rho=rho, g0=g0)

    L = np.asarray(dyn["link_lengths"], dtype=float)
    m = np.asarray(dyn["mass_nominal"], dtype=float)
    Ixx = np.array([float(I["ixx"]) for I in dyn["inertia_nominal"]], dtype=float)

    I_proxy = Ixx + m * (0.5 * L) ** 2
    B_proxy = np.asarray(dyn["viscous_nominal"], dtype=float)

    return I_proxy.tolist(), B_proxy.tolist()

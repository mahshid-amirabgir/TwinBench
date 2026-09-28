"""Craig MDH kinematics: forward kinematics, DLS inverse kinematics, and Jacobian."""

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import numpy as np

DOF = 6


# Craig MDH Transform
def mdh_T(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Craig Modified DH transform:
      T = RotZ(theta) * TransX(a) * RotX(alpha) * TransZ(d)
    """
    ca, sa = float(np.cos(alpha)), float(np.sin(alpha))
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    return np.array([
        [ct,      -st,       0.0,  float(a)],
        [st * ca,  ct * ca,  -sa,  -float(d) * sa],
        [st * sa,  ct * sa,   ca,   float(d) * ca],
        [0.0,      0.0,      0.0,  1.0],
    ], dtype=float)


# Forward Kinematics
def fk_mdh(q: np.ndarray, dh: Dict[str, Any]) -> np.ndarray:
    """Forward kinematics for 6-DOF Craig-MDH robot."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (DOF,):
        raise ValueError(f"Expected q shape ({DOF},); got {q.shape}")

    a     = np.asarray(dh["a"], dtype=float).reshape(-1)
    alpha = np.asarray(dh["alpha"], dtype=float).reshape(-1)
    d     = np.asarray(dh["d"], dtype=float).reshape(-1)
    thoff = np.asarray(dh.get("theta_offset", [0.0] * DOF), dtype=float).reshape(-1)
    sign  = np.asarray(dh.get("joint_sign", [1.0] * DOF), dtype=float).reshape(-1)

    if not np.all(np.isin(sign, [-1.0, 1.0])):
        raise ValueError("dh.joint_sign must be +/-1")

    T = np.eye(4, dtype=float)
    for i in range(DOF):
        th = float(sign[i]) * (float(q[i]) + float(thoff[i]))
        T = T @ mdh_T(float(a[i]), float(alpha[i]), float(d[i]), th)
    return T


# SO(3) Log Map
def vee_log(R: np.ndarray) -> np.ndarray:
    """SO(3) log map -> rotation vector, stable near 0."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3,3); got {R.shape}")

    t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(t))

    if th < 1e-9:
        return np.zeros(3, dtype=float)

    s = float(np.sin(th))
    if abs(s) < 1e-9:
        s = 1e-9 if s >= 0 else -1e-9

    w = (1.0 / (2.0 * s)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    return th * w


# Numerical Jacobian
def jac_numeric(q: np.ndarray, dh: Dict[str, Any], eps: float = 1e-6) -> np.ndarray:
    """Numerical geometric Jacobian (6x6) via forward differences."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (DOF,):
        raise ValueError(f"Expected q shape ({DOF},); got {q.shape}")

    T0 = fk_mdh(q, dh)
    p0, R0 = T0[:3, 3], T0[:3, :3]

    J = np.zeros((6, DOF), dtype=float)
    for i in range(DOF):
        dq = np.zeros(DOF, dtype=float)
        dq[i] = eps
        T1 = fk_mdh(q + dq, dh)
        dp = (T1[:3, 3] - p0) / eps
        dR = R0.T @ T1[:3, :3]
        w = vee_log(dR) / eps
        J[:, i] = np.hstack([dp, w])
    return J


# Inverse Kinematics (DLS)
def ik_dls(
    Td: np.ndarray,
    q0: np.ndarray,
    dh: Dict[str, Any],
    *,
    iters: int = 200,
    lam: float = 1e-2,
    pos_only: bool = True,
    pos_tol: float = 1e-3,            # 1 mm
    ang_tol_deg: float = 0.5,         # 0.5 deg (only if pos_only=False)
    step_clip_deg: float = 10.0,      # Max 10 deg per iteration per joint
    qmin: Optional[np.ndarray] = None,
    qmax: Optional[np.ndarray] = None,
    jac_eps: float = 1e-6,
) -> Tuple[np.ndarray, bool, float]:
    """
    Damped Least Squares IK (Craig MDH).

    Returns: (q, success, final_pos_err_m)

    Benchmark defaults:
      pos_only=True, lam=0.01, pos_tol=1mm, iters=200,
      step_clip=10 deg, ang_tol=0.5 deg
    """
    Td = np.asarray(Td, dtype=float)
    if Td.shape != (4, 4):
        raise ValueError(f"Td must be shape (4,4); got {Td.shape}")

    q = np.asarray(q0, dtype=float).reshape(-1).copy()
    if q.shape != (DOF,):
        raise ValueError(f"q0 must be shape ({DOF},); got {q.shape}")

    if (qmin is None) ^ (qmax is None):
        raise ValueError("Provide both qmin and qmax, or neither.")
    if qmin is not None and qmax is not None:
        qmin = np.asarray(qmin, dtype=float).reshape(-1)
        qmax = np.asarray(qmax, dtype=float).reshape(-1)
        q = np.clip(q, qmin, qmax)

    ang_tol = float(np.deg2rad(ang_tol_deg))
    step_clip = float(np.deg2rad(step_clip_deg))
    final_pos_err = float("inf")

    for _ in range(int(iters)):
        T = fk_mdh(q, dh)
        dp = Td[:3, 3] - T[:3, 3]
        final_pos_err = float(np.linalg.norm(dp))

        dw = np.zeros(3, dtype=float) if pos_only else vee_log(T[:3, :3].T @ Td[:3, :3])

        if final_pos_err <= float(pos_tol) and (pos_only or float(np.linalg.norm(dw)) <= ang_tol):
            return q, True, final_pos_err

        J = jac_numeric(q, dh, eps=float(jac_eps))
        J_use = J[:3, :] if pos_only else J
        e = dp if pos_only else np.hstack([dp, dw])

        H = J_use.T @ J_use + (float(lam) ** 2) * np.eye(DOF, dtype=float)
        dq = np.linalg.solve(H, J_use.T @ e)
        dq = np.clip(dq, -step_clip, step_clip)
        q = q + dq

        if qmin is not None and qmax is not None:
            q = np.clip(q, qmin, qmax)

    return q, False, final_pos_err

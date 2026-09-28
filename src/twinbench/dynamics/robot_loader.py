"""Load robot YAML specifications into pipeline-ready configuration dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from twinbench.ids import canon_robot_id
from twinbench.constants import (
    PAYLOAD_FRAC, G0,
    PARAM_SCALES, PARAM_SCALES_FACTORIAL,
)
from twinbench.dynamics.kinematics_dynamics import derive_proxies_from_cylinders

DOF = 6


# YAML loading
def load_robot_spec(yaml_path: str | Path) -> Dict[str, Any]:
    """Load a robot YAML file into a Python dict (does NOT mutate)."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if spec is None or not isinstance(spec, dict):
        raise ValueError(f"Invalid/empty robot spec YAML: {yaml_path}")
    return spec


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find repo root by locating robot_specs/ and scenes/ directories."""
    p = Path(start).resolve() if start else Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "robot_specs").exists() and (parent / "scenes").exists():
            return parent
    raise FileNotFoundError("Could not find repo root.")


def robot_spec_path_from_id(robot_id: str, repo_root: str | Path | None = None) -> Path:
    """Resolve robot_specs/<canon_robot_id>.yaml."""
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    rid = canon_robot_id(robot_id)
    p = root / "robot_specs" / f"{rid}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Robot spec not found for robot_id={robot_id!r} "
            f"(canonical={rid!r}) at {p}"
        )
    return p


# Validation helpers
def _as_1d(x: Any, name: str, n: int = DOF) -> np.ndarray:
    arr = np.array(x, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"Expected {name} shape ({n},); got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _require_map(obj: Any, name: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected {name} to be dict, got {type(obj).__name__}")
    return obj


# DH parameters (Craig MDH only)
def build_dh_from_spec(spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build DH parameter dict from YAML. Craig MDH only."""
    if "dh" not in spec:
        raise KeyError("Robot spec missing 'dh'")
    dh = _require_map(spec["dh"], "dh")

    for k in ["a", "alpha", "d"]:
        if k not in dh:
            raise KeyError(f"Robot spec missing dh.{k}")

    a     = _as_1d(dh["a"], "dh.a")
    alpha = _as_1d(dh["alpha"], "dh.alpha")
    d     = _as_1d(dh["d"], "dh.d")
    thoff = _as_1d(dh.get("theta_offset", [0.0] * DOF), "dh.theta_offset")
    sign  = _as_1d(dh.get("joint_sign", [1.0] * DOF), "dh.joint_sign")

    if not np.all(np.isin(sign, [-1.0, 1.0])):
        raise ValueError("dh.joint_sign must be +/-1 for all joints")

    conv = str(dh.get("convention", "craig")).strip().lower()
    if conv == "urdf":
        # URDF robots use Pinocchio for kinematics, not MDH.
        # Return dummy DH dict so downstream code doesn't crash.
        return {
            "a": a, "alpha": alpha, "d": d,
            "theta_offset": thoff, "joint_sign": sign,
            "convention": "urdf",
        }
    if conv != "craig":
        raise ValueError("TwinBench uses Craig MDH only.")

    return {
        "a": a, "alpha": alpha, "d": d,
        "theta_offset": thoff, "joint_sign": sign,
        "convention": conv,
    }


# Joint limits
def get_limits_from_spec(spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract joint limits in DEGREES."""
    if "joint_limits_deg" not in spec:
        raise KeyError("Robot spec missing 'joint_limits_deg'")
    lims = _require_map(spec["joint_limits_deg"], "joint_limits_deg")
    qmin = _as_1d(lims["min"], "joint_limits_deg.min")
    qmax = _as_1d(lims["max"], "joint_limits_deg.max")
    if np.any(qmax <= qmin):
        bad = np.where(qmax <= qmin)[0].tolist()
        raise ValueError(f"Invalid joint limits: max <= min at joints {bad}")
    return qmin, qmax


def get_limits_rad_from_spec(spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Joint limits in radians."""
    qmin_deg, qmax_deg = get_limits_from_spec(spec)
    return np.deg2rad(qmin_deg), np.deg2rad(qmax_deg)


# Uncertainty scaling
def _resolve_param_scales(params: str) -> Dict[str, float]:
    """
    Resolve parameter set name to (I, B, G) scale factors.
    Supports both combined (light/nominal/heavy) and factorial IDs.
    """
    p = str(params).strip().lower()

    # Check combined first
    if p in PARAM_SCALES:
        return PARAM_SCALES[p]

    # Check factorial
    if p in PARAM_SCALES_FACTORIAL:
        return PARAM_SCALES_FACTORIAL[p]

    raise ValueError(
        f"Unknown params={params!r}. Expected one of "
        f"{sorted(set(PARAM_SCALES) | set(PARAM_SCALES_FACTORIAL))}"
    )


def scale_proxies_for_param(
    I_proxy: np.ndarray,
    B_proxy: np.ndarray,
    G_nom: np.ndarray,
    params: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply uncertainty scaling to (I_proxy, B_proxy, G_nom).

    Returns scaled (I, B, G) arrays.
    Supports combined AND factorial uncertainty IDs.
    """
    scales = _resolve_param_scales(params)
    sI = float(scales["I"])
    sB = float(scales["B"])
    sG = float(scales["G"])
    return sI * I_proxy, sB * B_proxy, sG * G_nom


# Dynamics configuration
def get_dynamics_cfg_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract dynamics config from YAML. Ensures proxy fields exist (nominal).

    Policy:
    - YAML must include: Rmax, rated_payload, qd_max_deg, link_lengths, link_radii
    - If I_proxy/B_proxy missing, derive from cylinder geometry.
    - payload_used = PAYLOAD_FRAC * rated_payload (enforced).
    """
    if "dynamics" not in spec:
        raise KeyError("Robot spec missing 'dynamics'")
    dyn = _require_map(spec["dynamics"], "dynamics")

    required = ["Rmax", "rated_payload", "qd_max_deg", "link_lengths", "link_radii"]
    missing = [k for k in required if k not in dyn]
    if missing:
        raise KeyError(f"dynamics missing keys: {missing}")

    Rmax = float(dyn["Rmax"])
    rated_payload = float(dyn["rated_payload"])
    if not (np.isfinite(Rmax) and Rmax > 0):
        raise ValueError(f"dynamics.Rmax must be > 0 (got {Rmax})")
    if not (np.isfinite(rated_payload) and rated_payload > 0):
        raise ValueError(f"dynamics.rated_payload must be > 0 (got {rated_payload})")

    qd_max_deg = _as_1d(dyn["qd_max_deg"], "dynamics.qd_max_deg")
    qd_max_rad = np.deg2rad(qd_max_deg)
    link_lengths = _as_1d(dyn["link_lengths"], "dynamics.link_lengths")
    link_radii = _as_1d(dyn["link_radii"], "dynamics.link_radii")

    # Benchmark rule: payload_used is fraction of rated payload
    payload_used = float(PAYLOAD_FRAC * rated_payload)

    cfg: Dict[str, Any] = {
        "Rmax": Rmax,
        "rated_payload": rated_payload,
        "payload_used": payload_used,
        "qd_max_deg": qd_max_deg.tolist(),
        "qd_max_rad": qd_max_rad.tolist(),
        "link_lengths": link_lengths.tolist(),
        "link_radii": link_radii.tolist(),
    }

    # Torque limits
    if "tau_max" in dyn:
        cfg["tau_max"] = _as_1d(dyn["tau_max"], "dynamics.tau_max").tolist()

    # Proxy parameters: manual from YAML or derived from cylinders
    has_I = "I_proxy" in dyn
    has_B = "B_proxy" in dyn

    if has_I and has_B:
        cfg["I_proxy"] = _as_1d(dyn["I_proxy"], "dynamics.I_proxy").tolist()
        cfg["B_proxy"] = _as_1d(dyn["B_proxy"], "dynamics.B_proxy").tolist()
        cfg["proxy_source"] = "manual"
    else:
        I_proxy, B_proxy = derive_proxies_from_cylinders(
            {
                "Rmax": Rmax,
                "rated_payload": rated_payload,
                "qd_max_deg": qd_max_deg.tolist(),
                "link_lengths": link_lengths.tolist(),
                "link_radii": link_radii.tolist(),
            },
            g0=G0,
        )
        cfg["I_proxy"] = I_proxy
        cfg["B_proxy"] = B_proxy
        cfg["proxy_source"] = "derived"

    # Gravity proxy: G_nom_i = G_NOM_FRAC * tau_max_i
    from twinbench.constants import G_NOM_FRAC
    if "tau_max" in cfg:
        tau_max = np.array(cfg["tau_max"], dtype=float)
        cfg["G_nom"] = (G_NOM_FRAC * tau_max).tolist()
    else:
        cfg["G_nom"] = [0.0] * DOF

    return cfg


def get_dynamics_cfg_for_params(
    spec: Dict[str, Any],
    params: str,
) -> Dict[str, Any]:
    """
    Get dynamics config with proxies scaled for the given uncertainty variant.
    Supports both combined (light/nominal/heavy) and factorial IDs.
    """
    cfg = get_dynamics_cfg_from_spec(spec)

    I0 = np.array(cfg["I_proxy"], dtype=float)
    B0 = np.array(cfg["B_proxy"], dtype=float)
    G0_arr = np.array(cfg["G_nom"], dtype=float)

    I_s, B_s, G_s = scale_proxies_for_param(I0, B0, G0_arr, params=params)

    cfg2 = dict(cfg)
    cfg2["I_proxy"] = I_s.tolist()
    cfg2["B_proxy"] = B_s.tolist()
    cfg2["G_nom"] = G_s.tolist()
    cfg2["params"] = str(params).strip().lower()
    cfg2["proxy_scaled"] = True
    return cfg2


# Backward compat alias
get_dynamics_cfg_from_spec_for_params = get_dynamics_cfg_for_params


def robot_rmax_m_from_spec(spec: Dict[str, Any]) -> float:
    dyn = get_dynamics_cfg_from_spec(spec)
    return float(dyn["Rmax"])

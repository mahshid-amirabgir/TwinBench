"""Canonical robot, scene, trajectory, and controller ID normalization."""

from __future__ import annotations
from typing import Any
import re

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")


def canon_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = _CAMEL_1.sub(r"\1_\2", s)
    s = _CAMEL_2.sub(r"\1_\2", s)
    s = _NON_ALNUM.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def _prenorm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = _NON_ALNUM.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Alias maps
_ROBOT = {
    # Proxy robots
    "kr6": "kr6", "kuka_kr6": "kr6", "kuka_kr6_r1820": "kr6",
    "kuka_kr_6": "kr6", "kuka_kr_6_r1820": "kr6", "kr6_r1820": "kr6",
    "rs006l": "rs006l", "rs006_l": "rs006l", "kawasaki_rs006l": "rs006l",
    "kawasaki_rs006l_a": "rs006l", "kawasaki_rs006_l": "rs006l",
    "kawasaki_rs006_l_a": "rs006l",
    "mpx1950": "mpx1950", "yaskawa_mpx1950": "mpx1950",
    # URDF robots
    "ur5e": "ur5e", "ur_5e": "ur5e", "universal_robots_ur5e": "ur5e",
    "ur5": "ur5e",
    "ur10e": "ur10e", "ur_10e": "ur10e", "universal_robots_ur10e": "ur10e",
    "ur10": "ur10e",
    "tx60": "tx60", "staubli_tx60": "tx60", "staubli_tx_60": "tx60",
    "tx_60": "tx60", "tx60l": "tx60",
}

_SCENE = {
    "open_reach": "open_reach", "openreach": "open_reach", "open": "open_reach",
    "shelf_reach": "shelf_reach", "shelfreach": "shelf_reach", "shelf": "shelf_reach",
}

_TRAJ = {
    "cubic": "cubic",
    "minjerk": "minjerk", "min_jerk": "minjerk", "minimum_jerk": "minjerk",
    "fsmj": "fsmj", "optimized": "fsmj", "time_scaled_minjerk": "fsmj",
    "feasibility_scaled_minjerk": "fsmj",
    "fsmj_novia": "fsmj_novia", "fsmj_no_via": "fsmj_novia",
    "fsmj_direct": "fsmj_novia",
}

_CTRL = {
    "pdg": "pdg", "pd_g": "pdg",
    "ctc_pd": "ctc_pd", "ctcpd": "ctc_pd",
    "ctc_smc": "ctc_smc", "ctcsmc": "ctc_smc",
    "adaptive": "adaptive", "adaptive_slotine_li": "adaptive",
    "slotine_li": "adaptive",
    "pbc": "pbc", "passivity": "pbc", "passivity_based": "pbc",
    "pbc_passivity": "pbc",
}

_PARAM = {
    "light": "light", "nominal": "nominal", "heavy": "heavy",
    "i_light": "i_light", "i_heavy": "i_heavy",
    "b_light": "b_light", "b_heavy": "b_heavy",
    "g_light": "g_light", "g_heavy": "g_heavy",
}

_FIDELITY = {
    "proxy": "proxy", "decoupled": "proxy", "decoupled_proxy": "proxy",
    "rigid_body": "rigid_body", "rb": "rigid_body", "pinocchio": "rigid_body",
    "full_rb": "rigid_body", "urdf": "rigid_body",
}


def canon_robot_id(x): p = _prenorm(x); return _ROBOT.get(p, _ROBOT.get(canon_id(x), canon_id(x)))
def canon_scene_id(x): p = _prenorm(x); return _SCENE.get(p, _SCENE.get(canon_id(x), canon_id(x)))
def canon_traj_id(x): p = _prenorm(x); return _TRAJ.get(p, _TRAJ.get(canon_id(x), canon_id(x)))
def canon_ctrl_id(x): p = _prenorm(x); return _CTRL.get(p, _CTRL.get(canon_id(x), canon_id(x)))
def canon_params(x): p = _prenorm(x); return _PARAM.get(p, _PARAM.get(canon_id(x), canon_id(x)))
def canon_fidelity_id(x): p = _prenorm(x); return _FIDELITY.get(p, _FIDELITY.get(canon_id(x), canon_id(x)))

KNOWN_ROBOTS     = {"kr6", "rs006l", "mpx1950", "ur5e", "ur10e", "tx60"}
KNOWN_SCENES     = {"open_reach", "shelf_reach"}
KNOWN_TRAJS      = {"cubic", "minjerk", "fsmj", "fsmj_novia"}
KNOWN_CTRLS      = {"pdg", "ctc_pd", "ctc_smc", "adaptive", "pbc"}
KNOWN_PARAMS     = {"light", "nominal", "heavy", "i_light", "i_heavy",
                    "b_light", "b_heavy", "g_light", "g_heavy"}
KNOWN_FIDELITIES = {"proxy", "rigid_body"}

# Which robots are URDF-based (need Pinocchio for rigid_body fidelity)
URDF_ROBOTS = {"ur5e", "ur10e", "tx60"}
PROXY_ROBOTS = {"kr6", "rs006l", "mpx1950"}

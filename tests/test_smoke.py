"""
test_smoke.py — fast pipeline-correctness checks.

One call to run_single() per (robot, fidelity) pair, pinned to:
  trajectory = cubic, controller = ctc_pd, params = nominal, seed = 0.

Verifies the simulator runs to completion and produces finite Cartesian
RMSE on each robot. NOT a paper-numbers check — that's covered by the
full sweeps under analysis/.

Pinocchio gating: URDF robots (UR5e / UR10e / TX60) require Pinocchio
even at proxy fidelity, because target-pose IK uses Pinocchio's
kinematics. CI sets SKIP_PINOCCHIO=1; under that environment only the
three datasheet-tier robots (KR6 / RS006L / MPX1950) at proxy fidelity
are exercised.
"""
from __future__ import annotations

import math

import pytest

from tests.conftest import SKIP_PINOCCHIO

PROXY_TIER_ROBOTS = ["kr6", "rs006l", "mpx1950"]
URDF_TIER_ROBOTS = ["ur5e", "ur10e", "tx60"]

_pinocchio_required = pytest.mark.skipif(
    SKIP_PINOCCHIO,
    reason="Pinocchio not installed (SKIP_PINOCCHIO=1; CI default).",
)


def _smoke_run(robot: str, fidelity: str) -> dict:
    """One simulation call. Imported lazily so pytest collection works
    even when pinocchio is missing — the import only fails inside
    URDF-tier branches."""
    from experiments.run_main_grid import run_single

    return run_single(
        robot=robot,
        scene="open_reach",
        trajectory="cubic",
        controller="ctc_pd",
        params="nominal",
        seed=0,
        fidelity=fidelity,
    )


def _assert_smoke_ok(res: dict) -> None:
    assert bool(res["ik_ok"]) is True, f"IK failed: {res.get('rejected_reason', 'unknown')}"
    rmse = float(res["RMSE_cart_norm"])
    assert math.isfinite(rmse), f"RMSE_cart_norm is not finite: {rmse}"
    assert rmse >= 0.0, f"RMSE_cart_norm is negative: {rmse}"


@pytest.mark.parametrize("robot", PROXY_TIER_ROBOTS)
def test_smoke_proxy_tier(robot: str) -> None:
    """Datasheet-tier robots at proxy fidelity. No Pinocchio dependency."""
    _assert_smoke_ok(_smoke_run(robot, "proxy"))


@_pinocchio_required
@pytest.mark.parametrize("robot", URDF_TIER_ROBOTS)
def test_smoke_urdf_proxy(robot: str) -> None:
    """URDF robots at proxy fidelity. Needs Pinocchio for IK only."""
    _assert_smoke_ok(_smoke_run(robot, "proxy"))


@_pinocchio_required
@pytest.mark.parametrize("robot", URDF_TIER_ROBOTS)
def test_smoke_urdf_rigid_body(robot: str) -> None:
    """URDF robots at rigid-body fidelity. Pinocchio for IK + dynamics."""
    _assert_smoke_ok(_smoke_run(robot, "rigid_body"))

"""
test_scaled_urdf_robot.py — ScaledURDFRobot physics checks.

Verifies the deterministic, uniform I/G-scaling plant used by the symmetric
rigid-tier uncertainty ablation:
  - I_factor=G_factor=1.0 reproduces the unperturbed model's M(q), g(q) exactly.
  - At several random configurations, M(q) = I_factor * M_nominal(q) and
    g(q) = G_factor * g_nominal(q), for both the light (0.8, 0.9) and heavy
    (1.2, 1.1) uncertainty factors used in the ablation.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SKIP_PINOCCHIO

pytestmark = pytest.mark.skipif(
    SKIP_PINOCCHIO, reason="Pinocchio not installed (SKIP_PINOCCHIO=1; CI default)."
)

import twinbench.constants as C
from twinbench.dynamics.urdf_plant import URDFRobot, ScaledURDFRobot

RNG = np.random.default_rng(20260922)
N_CONFIGS = 5


def _random_configs(model, n=N_CONFIGS):
    lo, hi = model.lowerPositionLimit, model.upperPositionLimit
    lo = np.where(np.isfinite(lo), lo, -np.pi)
    hi = np.where(np.isfinite(hi), hi, np.pi)
    return [RNG.uniform(lo, hi) for _ in range(n)]


@pytest.mark.parametrize("robot_id", ["ur5e", "ur10e", "tx60"])
def test_identity_scale_matches_nominal(robot_id):
    urdf_path = C.URDF_PATHS[robot_id]
    nominal = URDFRobot(urdf_path)
    scaled = ScaledURDFRobot(urdf_path, I_factor=1.0, G_factor=1.0)
    for q in _random_configs(nominal.model):
        np.testing.assert_allclose(scaled.mass_matrix(q), nominal.mass_matrix(q), atol=1e-12)
        np.testing.assert_allclose(scaled.gravity(q), nominal.gravity(q), atol=1e-12)


@pytest.mark.parametrize("robot_id", ["ur5e", "ur10e", "tx60"])
@pytest.mark.parametrize("I_factor,G_factor", [(0.8, 0.9), (1.2, 1.1)])
def test_mass_and_gravity_scale_as_specified(robot_id, I_factor, G_factor):
    urdf_path = C.URDF_PATHS[robot_id]
    nominal = URDFRobot(urdf_path)
    scaled = ScaledURDFRobot(urdf_path, I_factor=I_factor, G_factor=G_factor)
    for q in _random_configs(nominal.model):
        M_nom = nominal.mass_matrix(q)
        M_scl = scaled.mass_matrix(q)
        np.testing.assert_allclose(M_scl, I_factor * M_nom, rtol=1e-10, atol=1e-12)

        g_nom = nominal.gravity(q)
        g_scl = scaled.gravity(q)
        np.testing.assert_allclose(g_scl, G_factor * g_nom, rtol=1e-10, atol=1e-12)

"""URDF-based rigid-body dynamics via Pinocchio (Articulated-Body Algorithm)."""

from __future__ import annotations
from typing import Any, Dict, Tuple
from pathlib import Path
import numpy as np

try:
    import pinocchio as pin
except ImportError:
    raise ImportError("Pinocchio required: conda install -c conda-forge pinocchio")

DOF = 6


class URDFRobot:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = str(urdf_path)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if self.model.nv != DOF:
            raise ValueError(f"Expected 6-DOF, got nv={self.model.nv}")
        q0 = np.zeros(self.model.nq)
        M0 = pin.crba(self.model, self.data, q0).copy()
        self.I_proxy = np.maximum(np.diag(M0), 1e-4)
        self.g_q0 = np.array(pin.computeGeneralizedGravity(self.model, self.data, q0)).copy()

    def mass_matrix(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def nonlinear_effects(self, q, dq):
        return np.array(pin.rnea(self.model, self.data, q, dq, np.zeros(DOF))).ravel()

    def coriolis_matrix(self, q, dq):
        # Explicit Coriolis matrix C(q, dq) for the Slotine-Li passivity term
        # C(q, dq) @ qd_r used by PBC. Consistent with rnea() used elsewhere:
        # C(q,v) @ v + gravity(q) == rnea(q, v, 0) (verified to ~1e-14).
        return np.array(pin.computeCoriolisMatrix(self.model, self.data, q, dq))

    def gravity(self, q):
        return np.array(pin.computeGeneralizedGravity(self.model, self.data, q)).ravel()

    def forward_dynamics_full(self, q, dq, tau, B_eff):
        tau_net = tau - B_eff * dq
        return np.asarray(pin.aba(self.model, self.data, q, dq, tau_net), float).ravel()

    def fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.model.nframes - 1].translation.copy()

    def fk_all_positions(self, Q):
        return np.array([self.fk(Q[i]) for i in range(len(Q))])


class PerturbedURDFRobot:
    """URDFRobot with perturbed link masses for parametric mismatch studies.

    Loads a fresh Pinocchio model from the same URDF, then scales each link's
    mass and rotational inertia by a per-link random factor drawn from
    U[1-frac, 1+frac]. Center-of-mass positions are preserved (geometry unchanged).

    This simulates the realistic scenario where the URDF's mass parameters are
    approximate (e.g., from CAD without measured payloads/cables/fixtures).
    """

    def __init__(self, urdf_path: str | Path, mass_frac: float = 0.10, seed: int = 42):
        self.urdf_path = str(urdf_path)
        self.mass_frac = mass_frac
        self.seed = seed

        # Load a fresh model (independent of any cached nominal model)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        if self.model.nv != DOF:
            raise ValueError(f"Expected 6-DOF, got nv={self.model.nv}")

        # Perturb link masses deterministically
        rng = np.random.default_rng(seed)
        self.mass_scales = {}

        for i in range(1, self.model.njoints):  # skip universe joint (index 0)
            scale = float(rng.uniform(1.0 - mass_frac, 1.0 + mass_frac))
            self.mass_scales[i] = scale

            original = self.model.inertias[i]
            new_mass = original.mass * scale
            new_lever = original.lever.copy()  # CoM unchanged (same geometry)
            new_inertia = original.inertia * scale  # I ∝ mass for fixed geometry

            self.model.inertias[i] = pin.Inertia(new_mass, new_lever, new_inertia)

        # Recreate data after modifying model
        self.data = self.model.createData()

        # Recompute proxy quantities on perturbed model
        q0 = np.zeros(self.model.nq)
        M0 = pin.crba(self.model, self.data, q0).copy()
        self.I_proxy = np.maximum(np.diag(M0), 1e-4)

    def mass_matrix(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def nonlinear_effects(self, q, dq):
        return np.array(pin.rnea(self.model, self.data, q, dq, np.zeros(DOF))).ravel()

    def coriolis_matrix(self, q, dq):
        # Explicit Coriolis matrix C(q, dq) for the Slotine-Li passivity term
        # C(q, dq) @ qd_r used by PBC. Consistent with rnea() used elsewhere:
        # C(q,v) @ v + gravity(q) == rnea(q, v, 0) (verified to ~1e-14).
        return np.array(pin.computeCoriolisMatrix(self.model, self.data, q, dq))

    def gravity(self, q):
        return np.array(pin.computeGeneralizedGravity(self.model, self.data, q)).ravel()

    def forward_dynamics_full(self, q, dq, tau, B_eff):
        tau_net = tau - B_eff * dq
        return np.asarray(pin.aba(self.model, self.data, q, dq, tau_net), float).ravel()

    def fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.model.nframes - 1].translation.copy()

    def fk_all_positions(self, Q):
        return np.array([self.fk(Q[i]) for i in range(len(Q))])


def get_perturbed_urdf_robot(urdf_path, mass_frac=0.10, seed=42) -> PerturbedURDFRobot:
    """Create a perturbed URDF robot (not cached — each call creates fresh perturbation)."""
    return PerturbedURDFRobot(str(urdf_path), mass_frac=mass_frac, seed=seed)


class ScaledURDFRobot:
    """URDFRobot with every link's mass and rotational inertia uniformly scaled by
    I_factor, and gravity separately adjusted so the net gravity torque g(q) scales
    by G_factor -- independently of I_factor.

    Unlike PerturbedURDFRobot (per-link random mismatch, for the mismatch ablation),
    this applies ONE deterministic scale factor to every link at once, mirroring the
    proxy tier's (I, B, G) uncertainty scaling applied directly to the rigid-body
    model. Used for the symmetric rigid-tier uncertainty ablation (light/heavy).

    Because CRBA/RNEA are linear in each link's spatial inertia, uniformly scaling
    every link's mass and rotational inertia by I_factor scales the whole M(q) by
    I_factor exactly. That same mass scaling also scales g(q) by I_factor (gravity
    torque is linear in mass), so to get a NET g(q) scaled by G_factor instead, the
    gravitational acceleration itself is set to nominal_gravity * (G_factor /
    I_factor). See tests/test_scaled_urdf_robot.py, which verifies both M(q) and
    g(q) scale exactly as intended at several random configurations, and that
    I_factor=G_factor=1.0 reproduces the unperturbed model exactly.
    """

    def __init__(self, urdf_path: str | Path, I_factor: float = 1.0, G_factor: float = 1.0):
        self.urdf_path = str(urdf_path)
        self.I_factor = float(I_factor)
        self.G_factor = float(G_factor)

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if self.model.nv != DOF:
            raise ValueError(f"Expected 6-DOF, got nv={self.model.nv}")

        for i in range(1, self.model.njoints):
            original = self.model.inertias[i]
            new_mass = original.mass * self.I_factor
            new_lever = original.lever.copy()  # CoM unchanged (same geometry)
            new_inertia = original.inertia * self.I_factor  # I ∝ mass for fixed geometry
            self.model.inertias[i] = pin.Inertia(new_mass, new_lever, new_inertia)

        # Net gravity torque g(q) is already scaled by I_factor from the mass
        # scaling above; rescale the gravitational acceleration itself by
        # (G_factor / I_factor) so the NET g(q) scales by G_factor.
        g = self.model.gravity
        gain = self.G_factor / self.I_factor
        self.model.gravity = pin.Motion(np.array(g.linear) * gain, np.array(g.angular) * gain)

        # Recreate data after modifying model
        self.data = self.model.createData()

        q0 = np.zeros(self.model.nq)
        M0 = pin.crba(self.model, self.data, q0).copy()
        self.I_proxy = np.maximum(np.diag(M0), 1e-4)

    def mass_matrix(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def nonlinear_effects(self, q, dq):
        return np.array(pin.rnea(self.model, self.data, q, dq, np.zeros(DOF))).ravel()

    def coriolis_matrix(self, q, dq):
        return np.array(pin.computeCoriolisMatrix(self.model, self.data, q, dq))

    def gravity(self, q):
        return np.array(pin.computeGeneralizedGravity(self.model, self.data, q)).ravel()

    def forward_dynamics_full(self, q, dq, tau, B_eff):
        tau_net = tau - B_eff * dq
        return np.asarray(pin.aba(self.model, self.data, q, dq, tau_net), float).ravel()

    def fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.model.nframes - 1].translation.copy()

    def fk_all_positions(self, Q):
        return np.array([self.fk(Q[i]) for i in range(len(Q))])


_scaled_urdf_cache: Dict[Tuple[str, float, float], ScaledURDFRobot] = {}
def get_scaled_urdf_robot(urdf_path, I_factor: float = 1.0, G_factor: float = 1.0) -> ScaledURDFRobot:
    """Create (or reuse) a deterministically I/G-scaled URDF robot.

    Cached by (urdf_path, I_factor, G_factor): for a given robot and uncertainty
    level, the scaled model is identical across every trajectory/controller/seed,
    so rebuilding it from the URDF on every call (as the first implementation did)
    was pure waste -- only 3 distinct models per robot are ever needed. This also
    avoids the memory growth that killed the first full-scale run (2,880 fresh
    Pinocchio models built and discarded in one process).
    """
    key = (str(urdf_path), float(I_factor), float(G_factor))
    if key not in _scaled_urdf_cache:
        _scaled_urdf_cache[key] = ScaledURDFRobot(key[0], I_factor=key[1], G_factor=key[2])
    return _scaled_urdf_cache[key]


def control_torque_pinocchio(
    q, dq, qd_k, dqd_k, ddqd_k, *,
    urdf_robot, tau_max, ddq_cap, dq_cap, B_nom,
    ctrl, adaptive_state=None,
    WN_FRAC, ZETA, T_BASE, DT,
    TAU_CAP_FRAC, TAU_CAP_FRAC_SMC,
    SMC_PHI_FRAC, SMC_KS_FRAC,
    ADAPTIVE_INIT_FRAC, ADAPTIVE_GAMMA_FRAC,
    I_nom, G_nom,
    include_B_feedforward: bool = False,
):
    """Controller with Pinocchio-based feedforward for Tier B-full.

    include_B_feedforward: default False preserves the original shipped
        behavior exactly (ctc_pd/ctc_smc/pbc have NO B_nom*qd feedforward term
        at the rigid tier -- B_nom is otherwise unused here except in the
        adaptive branch's state init, which is a real, confirmed mismatch
        against Table 2's caption). When True, adds Table 2's B_nom*qd term
        to ctc_pd and ctc_smc, and B_nom*qd_r to pbc, using NOMINAL B_nom at
        both tiers (never the uncertainty-scaled B_eff) -- matching what the
        proxy tier already does and what Table 2 actually specifies. pdg has
        no B term at either tier per Table 2, so it is unaffected either way.
    """
    EPS = 1e-9
    wn = WN_FRAC * np.sqrt(np.maximum(ddq_cap, 1e-6))
    Kp, Kd = wn**2, 2.0 * ZETA * wn
    e, de = qd_k - q, dqd_k - dq

    M = urdf_robot.mass_matrix(q)
    nle = urdf_robot.nonlinear_effects(q, dq)
    g_q = urdf_robot.gravity(q)

    if ctrl == "pdg":
        tau = M @ (Kp * e + Kd * de) + g_q
        cap = TAU_CAP_FRAC * tau_max
    elif ctrl == "ctc_pd":
        v = ddqd_k + Kp * e + Kd * de
        tau = M @ v + nle
        if include_B_feedforward:
            tau = tau + B_nom * dq
        cap = TAU_CAP_FRAC * tau_max
    elif ctrl == "ctc_smc":
        wn_smc = 2.0 * np.pi / max(T_BASE, 1e-6)
        Lambda = 2.0 * ZETA * wn_smc
        s = de + Lambda * e
        v_eq = ddqd_k + Lambda * de
        tau_eq = M @ v_eq + nle
        if include_B_feedforward:
            tau_eq = tau_eq + B_nom * dq
        # Normalized Ks: scales with robot's natural torque level
        M_diag = np.diag(M)
        Ks = SMC_KS_FRAC * M_diag * np.maximum(Kp, 1.0)
        phi = SMC_PHI_FRAC * dq_cap
        tau = tau_eq + Ks * np.tanh(s / (phi + EPS))
        cap = TAU_CAP_FRAC_SMC * tau_max
    elif ctrl == "pbc":
        # PBC (passivity-based), rigid-body tier. Fully independent, STATELESS
        # branch (Decision 1.4): it reads/writes no adaptive_state and shares
        # nothing any other controller consumes.
        # Lambda must stay in sync with CTC-SMC Lambda. Duplicated inline on
        # purpose — do NOT refactor CTC-SMC's inline Lambda into a shared helper
        # (refactoring risks perturbing CTC-SMC's bit-identical output).
        wn_pbc = 2.0 * np.pi / max(T_BASE, 1e-6)
        Lambda = 2.0 * ZETA * wn_pbc
        # Reference velocity: qd_r = qd_d + Lambda*e, qdd_r = qdd_d + Lambda*de.
        # consistent with s = de + Lambda*e and CTC-SMC's v_eq. This is DELIBERATELY
        # the opposite sign from the adaptive branch's -Lambda (see below); do not
        # "harmonize" the two.
        qd_r = dqd_k + Lambda * e
        qdd_r = ddqd_k + Lambda * de
        s = de + Lambda * e
        # Slotine-Li passivity Coriolis term: C(q, dq) @ qd_r — Coriolis matrix at
        # the ACTUAL velocity dq, multiplied by the REFERENCE velocity qd_r. Uses
        # the explicit Coriolis matrix, NOT rnea(q, qd_r, 0), which would evaluate
        # C at qd_r in both slots (wrong passivity form; verified to differ by ~12).
        C = urdf_robot.coriolis_matrix(q, dq)
        # K_D derived from the Eq. (3) normalization protocol, not tuned. Uses the
        # DIAGONAL of the LIVE mass matrix diag(M(q)) * K_d (K_d = 2*ZETA*wn),
        # matching ctc_smc/adaptive's np.diag(M) convention. This is deliberately
        # the diagonal, NOT the full live M(q) that pdg/ctc_pd use: here K_D is a
        # dissipation gain (conventionally diagonal), a distinct role from the
        # M(q) @ qdd_r feedforward term (which does use the full live matrix).
        # (Previously used the static, floored urdf_robot.I_proxy = max(diag(M(0)),
        # 1e-4); harmonized to live diag(M(q)) — numerically interchangeable, ~5e-4.)
        K_D = np.diag(M) * Kd
        tau = M @ qdd_r + C @ qd_r + g_q + K_D * s
        if include_B_feedforward:
            tau = tau + B_nom * qd_r
        # 0.9 cap (pdg/ctc_pd branch). Reads the same TAU_CAP_FRAC on both tiers,
        # so PBC does not inherit the SMC cap-flag asymmetry.
        cap = TAU_CAP_FRAC * tau_max
    elif ctrl == "adaptive":
        if adaptive_state is None:
            M_diag = np.diag(M)
            adaptive_state = {
                "I_hat": ADAPTIVE_INIT_FRAC * M_diag.copy(),
                "B_hat": ADAPTIVE_INIT_FRAC * B_nom.copy(),
                "G_hat": ADAPTIVE_INIT_FRAC * np.maximum(np.abs(g_q), 1e-6),
                "Gamma_I": ADAPTIVE_GAMMA_FRAC * np.maximum(M_diag, 1e-6),
                "Gamma_B": ADAPTIVE_GAMMA_FRAC * np.maximum(np.abs(B_nom), 1e-6),
                "Gamma_G": ADAPTIVE_GAMMA_FRAC * np.maximum(np.abs(g_q), 1e-6),
            }
        wn_a = 2.0 * np.pi / max(T_BASE, 1e-6)
        Lambda = 2.0 * ZETA * wn_a
        s = de + Lambda * e
        dq_r = dqd_k - Lambda * e
        ddq_r = ddqd_k - Lambda * de
        tau = (adaptive_state["I_hat"] * ddq_r + adaptive_state["B_hat"] * dq_r +
               adaptive_state["G_hat"] * np.sign(g_q) * np.sin(q) + Kp * e + Kd * de)
        adaptive_state["I_hat"] -= adaptive_state["Gamma_I"] * ddq_r * s * DT
        adaptive_state["B_hat"] -= adaptive_state["Gamma_B"] * dq_r * s * DT
        adaptive_state["G_hat"] -= adaptive_state["Gamma_G"] * np.sin(q) * s * DT
        cap = TAU_CAP_FRAC * tau_max
    else:
        raise ValueError(f"Unknown controller: {ctrl}")
    return np.asarray(tau, float), np.asarray(cap, float), adaptive_state


def simulate_urdf_full(
    *, urdf_robot,
    urdf_robot_plant=None,
    qd, dqd, ddqd, t,
    q_init, qmin, qmax,
    I_nom, B_nom, G_nom, tau_max,
    I_eff, B_eff, G_eff,
    dq_cap, ddq_cap, ctrl,
    rng_noise, sigma_q, sigma_qdot,
    disturb_joint, disturb_tau, disturb_onset, disturb_end,
    control_fn,
    WN_FRAC=0.25, ZETA=0.9, T_BASE=2.0, DT=0.001,
    TAU_CAP_FRAC=0.90, TAU_CAP_FRAC_SMC=0.70,
    SMC_PHI_FRAC=0.60, SMC_KS_FRAC=0.05,
    ADAPTIVE_INIT_FRAC=0.50, ADAPTIVE_GAMMA_FRAC=0.10,
    include_B_feedforward: bool = False,
):
    """Closed-loop on Pinocchio plant with Pinocchio-based controllers."""
    N = len(t)
    dt = float(t[1] - t[0]) if N > 1 else 0.001
    q = np.zeros((N, DOF)); dq_sim = np.zeros((N, DOF))
    tau_hist = np.zeros((N, DOF)); ddq_hist = np.zeros((N, DOF))
    sat_hist = np.zeros((N, DOF), dtype=bool)
    q[0] = q_init
    dq_sim[0] = dqd[0] if len(dqd) > 0 else np.zeros(DOF)
    adaptive_state = None

    # If a separate plant robot is provided, use it for dynamics
    # Controller feedforward still uses urdf_robot (nominal)
    plant = urdf_robot_plant if urdf_robot_plant is not None else urdf_robot

    for k in range(1, N):
        q_meas = q[k-1] + rng_noise.normal(0, sigma_q, DOF)
        dq_meas = dq_sim[k-1] + rng_noise.normal(0, sigma_qdot, DOF)

        tau_cmd, tau_cap, adaptive_state = control_torque_pinocchio(
            q_meas, dq_meas, qd[k], dqd[k], ddqd[k],
            urdf_robot=urdf_robot, tau_max=tau_max,
            ddq_cap=ddq_cap, dq_cap=dq_cap, B_nom=B_nom,
            ctrl=ctrl, adaptive_state=adaptive_state,
            WN_FRAC=WN_FRAC, ZETA=ZETA, T_BASE=T_BASE, DT=DT,
            TAU_CAP_FRAC=TAU_CAP_FRAC, TAU_CAP_FRAC_SMC=TAU_CAP_FRAC_SMC,
            SMC_PHI_FRAC=SMC_PHI_FRAC, SMC_KS_FRAC=SMC_KS_FRAC,
            ADAPTIVE_INIT_FRAC=ADAPTIVE_INIT_FRAC,
            ADAPTIVE_GAMMA_FRAC=ADAPTIVE_GAMMA_FRAC,
            I_nom=I_nom, G_nom=G_nom,
            include_B_feedforward=include_B_feedforward,
        )

        if disturb_onset <= t[k] <= disturb_end:
            tau_cmd = tau_cmd.copy(); tau_cmd[disturb_joint] += disturb_tau

        sat_hist[k] = np.abs(tau_cmd) > (tau_cap + 1e-12)
        tau = np.clip(tau_cmd, -tau_cap, tau_cap); tau_hist[k] = tau

        ddq_k = plant.forward_dynamics_full(q[k-1], dq_sim[k-1], tau, B_eff)
        ddq_k = np.clip(ddq_k, -500.0, 500.0); ddq_hist[k] = ddq_k

        dq_sim[k] = dq_sim[k-1] + ddq_k * dt
        q[k] = np.clip(q[k-1] + dq_sim[k] * dt, qmin, qmax)

    return q, dq_sim, tau_hist, ddq_hist, sat_hist


_urdf_cache: Dict[str, URDFRobot] = {}
def get_urdf_robot(urdf_path) -> URDFRobot:
    key = str(urdf_path)
    if key not in _urdf_cache:
        _urdf_cache[key] = URDFRobot(key)
    return _urdf_cache[key]
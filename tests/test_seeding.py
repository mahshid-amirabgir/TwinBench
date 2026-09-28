"""
test_seeding.py — verifies the determinism + fairness invariants of the
SHA-256 seed scheme described in README "Determinism and fairness".

The scheme has two seeds, both controller-independent:

  target seed = stable_int_seed("target", robot, scene,           seed)
  noise  seed = stable_int_seed("noise",  robot, scene, traj, prm, seed)

Both seeds are deterministic in their inputs. Neither seed depends on
the controller, so all four controllers see *identical* targets and
*identical* noise sequences for a given (robot, scene, traj, params,
seed) tuple. This is the load-bearing fairness property for §IV-A's
cross-controller rank comparisons.
"""
from __future__ import annotations

import tests.conftest  # noqa: F401  — bootstraps sys.path

from experiments.run_main_grid import seed_noise, seed_target, stable_int_seed


# stable_int_seed: pure deterministic function over its inputs

def test_stable_int_seed_is_deterministic() -> None:
    a = stable_int_seed("target", "kr6", "open_reach", 0, base=11111)
    b = stable_int_seed("target", "kr6", "open_reach", 0, base=11111)
    assert a == b


def test_stable_int_seed_seed_axis_is_significant() -> None:
    a = stable_int_seed("target", "kr6", "open_reach", 0, base=11111)
    b = stable_int_seed("target", "kr6", "open_reach", 1, base=11111)
    assert a != b


def test_stable_int_seed_robot_axis_is_significant() -> None:
    a = stable_int_seed("target", "kr6", "open_reach", 0, base=11111)
    b = stable_int_seed("target", "rs006l", "open_reach", 0, base=11111)
    assert a != b


def test_stable_int_seed_base_axis_is_significant() -> None:
    # Different bases (11111 vs 22222) must give different seeds even
    # for otherwise-identical inputs — this is what keeps target and
    # noise streams disjoint.
    a = stable_int_seed("target", "kr6", "open_reach", 0, base=11111)
    b = stable_int_seed("target", "kr6", "open_reach", 0, base=22222)
    assert a != b


# seed_target: bit-identical across calls; no controller dependence

def test_seed_target_is_deterministic() -> None:
    a = seed_target("kr6", "open_reach", 5)
    b = seed_target("kr6", "open_reach", 5)
    assert a == b


def test_seed_target_signature_excludes_controller() -> None:
    # If seed_target accepted a controller, this signature check would
    # have to change — and the §IV-A fairness argument would break.
    # Belt-and-braces: confirm the function only takes 3 positional args.
    import inspect

    sig = inspect.signature(seed_target)
    params = list(sig.parameters.keys())
    assert params == ["rid", "sid", "seed"], (
        f"seed_target signature drifted: {params}. Controller MUST NOT "
        "appear in the seeding inputs (see README 'Determinism and fairness')."
    )


# seed_noise: bit-identical across calls; no controller dependence

def test_seed_noise_is_deterministic() -> None:
    a = seed_noise("ur5e", "open_reach", "cubic", "nominal", 3)
    b = seed_noise("ur5e", "open_reach", "cubic", "nominal", 3)
    assert a == b


def test_seed_noise_signature_excludes_controller() -> None:
    import inspect

    sig = inspect.signature(seed_noise)
    params = list(sig.parameters.keys())
    assert params == ["rid", "sid", "traj", "prm", "seed"], (
        f"seed_noise signature drifted: {params}. Controller MUST NOT "
        "appear in the seeding inputs (see README 'Determinism and fairness')."
    )


# Disjointness between target and noise streams

def test_target_and_noise_seeds_are_disjoint() -> None:
    # Different prefixes ("target" vs "noise") + different bases (11111
    # vs 22222) ensure target sampling and per-step measurement noise
    # come from independent RNG streams for the same (robot, scene, seed).
    s_t = seed_target("kr6", "open_reach", 0)
    s_n = seed_noise("kr6", "open_reach", "cubic", "nominal", 0)
    assert s_t != s_n

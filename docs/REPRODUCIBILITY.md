# Reproducibility

Exact environment, seeding, and per-experiment commands and runtimes for
reproducing every number and figure in the manuscript from this repository.

---

## 1. Environment

```bash
conda env create -f environment.yml
conda activate twinbench
```

`environment.yml` pins:

| package | pinned | verified installed (this environment) |
|---|---|---|
| python | 3.11 | 3.11.15 |
| pinocchio | 3.9.0 | 3.9.0 |
| numpy | 2.4.3 | 2.4.3 |
| pandas | 3.0.1 | 3.0.1 |
| scipy | 1.17.1 | 1.17.1 |
| matplotlib | 3.10.8 | 3.10.8 |
| pyyaml | 6.0.3 | 6.0.3 |
| pytest | 9.0.3 | — |

Pinocchio is pinned exactly because the rigid-body tier's numbers depend on
its ABA/CRBA implementation; drift there is not numerically immaterial.
Everything else is pinned to what actually produced the committed CSVs — versions used for the runtime estimates below.

A pip fallback (`pip install -r requirements.txt`) works for the proxy tier
only; without Pinocchio, any `fidelity=rigid_body` run fails at import time.

**Hardware used for every runtime figure below:** Intel Core i5-4210U @
1.70 GHz, 2 physical / 4 logical cores, 8 GB RAM — a deliberately modest
machine, so treat every runtime below as an upper bound on typical hardware,
not a lower bound.

---

## 2. Deterministic seeding and how to verify bit-exact reproduction

Two SHA-256-derived seeds drive every stochastic element of a simulation
run (`experiments/run_main_grid.py:stable_int_seed`):

| seed | inputs | used for |
|---|---|---|
| target seed | (`robot`, `scene`, `seed`) | sampling start/goal joint configurations |
| noise seed | (`robot`, `scene`, `trajectory`, `params`, `seed`) | per-step measurement noise on q and q̇ |

Neither seed depends on the controller, so all controllers on a given
(robot, scene, trajectory, params, seed) tuple see identical targets and
identical noise — eliminating target-luck and noise-luck as confounds.

**To verify bit-exact reproduction of a simulation run:** re-run any
`experiments/run_*.py` command below and `diff` the resulting CSV against
the committed one.

**To verify bit-exact reproduction of every *table*:** the bootstrap and
cluster-bootstrap statistics in `analysis/generate_tables.py` use a single
pinned seed (`BOOTSTRAP_SEED`, in that file's header) independent of the
simulation seeds above. Running `python analysis/generate_tables.py --write`
twice produces byte-identical `.tex` fragments under `tables/` — this
is checked, not assumed; see `analysis/verify_paper_numbers.py` below.

---

## 3. The one command that matters most

```bash
python analysis/generate_tables.py --write     # regenerate every table (~1 min)
python analysis/verify_paper_numbers.py        # regenerate + check everything (~2 min)
```

`verify_paper_numbers.py` regenerates every manuscript table, every
cross-fidelity τ_b and its bootstrap CI, the within-tier self-consistency
figures, every ablation's run count and the 43,200 grand total, and every
saturation fraction — all from committed CSVs, no new simulation — and
exits non-zero if anything drifts from the expected (pipeline-computed, not
manuscript-transcribed) value. This is the regression test that did not
exist before this release; run it after touching any committed CSV or any
function in `generate_tables.py`.

Both commands run in well under a minute of simulation time each; they read
`data/*.csv` and do arithmetic, nothing more.

---

## 4. Per-experiment commands, inputs, outputs, and measured/estimated runtime

Runtimes marked **measured** were timed on the hardware above during this
release pass; runtimes marked **estimated** are extrapolated from measured
per-run cost on the same hardware and were not timed end-to-end — treat them
as order-of-magnitude, not exact.

| experiment | command | reads | writes | runtime |
|---|---|---|---|---|
| Main grid, URDF robots (5,760 runs) | `python experiments/run_main_grid.py --robots ur5e,ur10e,tx60 --scenes open_reach --output-tag main_rerun` | robot specs, URDF models | `data/all_results_main_rerun.csv` | **estimated** ~1.5–2 h (rigid-body tier is the expensive half; proxy tier is fast) |
| Main grid, datasheet robots (5,760 runs) | `python experiments/run_main_grid.py --robots kr6,rs006l,mpx1950 --scenes open_reach,shelf_reach --fidelities proxy --output-tag datasheet_rerun` | robot specs | `data/all_results_datasheet_rerun.csv` | **estimated** ~30–45 min (proxy-only, no Pinocchio) |
| Coupled-proxy sweep, one (robot, ε_c) cell, 20 seeds (960 runs) | `python experiments/run_coupled_sweep_ur5e.py --coupling-strength 0.09 --output-tag eps0p09` | robot specs, URDF (UR5e) | `data/coupled_ablation_ur5e_eps0p09.csv` | **measured**, comparable cell: ~342 s (~6 min) |
| Full coupled-proxy sweep, both robots, 5×ε_c (9,600 runs) | 10 invocations per the README's loop | — | 10 chunk CSVs, merge to reproduce `d3_coupled_all_20seed.csv` | **estimated** ~50–60 min total |
| Raised-clamp diagnostic (1,920 runs) | `python experiments/run_raised_clamp_diagnostic.py` | robot specs, URDF (UR5e) | `data/all_results_nocap_ur5e_eps0p09_{coupled,decoupled}.csv` | **measured** 447 s (~7.5 min), 3 parallel workers |
| Torque-cap robustness, CTC-SMC only (1,440 incremental runs) | `python experiments/run_main_grid.py --robots ur5e,ur10e,tx60 --scenes open_reach --controllers ctc_smc --smc-tau-cap-frac 0.9 --output-tag cap0p9_ctcsmc_rerun` | robot specs, URDF | `data/all_results_cap0p9_ctcsmc_rerun.csv` | **estimated** ~15–20 min |
| Gain-scaling (frozen vs. live) ablation (2,880 runs) | `python experiments/run_frozen_gain_ablation.py` | robot specs, URDF | `data/all_results_frozen_gains_rigid.csv` | **measured** 1,095 s (~18 min), single-threaded |
| Integration step-size, Δt=0.5ms (1,440 incremental runs) | `python experiments/run_main_grid.py --robots ur5e,ur10e,tx60 --scenes open_reach --seeds 0,1,2,3,4 --dt 0.0005 --output-tag dt0p5_rerun` | robot specs, URDF | `data/all_results_dt0p5_rerun.csv` | **estimated** ~10–15 min |
| Integration step-size, Δt=0.25ms (1,440 incremental runs) | as above with `--dt 0.00025 --output-tag dt0p25_rerun` | robot specs, URDF | `data/all_results_dt0p25_rerun.csv` | **estimated** ~15–20 min (more steps/run) |
| Parametric mismatch, one (robot, pct) cell (6 realizations × 5 seeds = 480 runs) | `python experiments/drivers/mismatch_driver.py ur5e 0.10 0,1,2,3,4,5 0,1,2,3,4 ur5e_pct10` | robot specs, URDF | `data/d1_mismatch_ur5e_pct10.csv` | **estimated** ~3–5 min |
| Full mismatch sweep, 3 robots × 3 magnitudes (12,960 runs) | 9 invocations per the README's loop | — | 9 chunk CSVs, merge to reproduce `d1_mismatch_all.csv` | **estimated** ~30–45 min total |
| Peak-torque check (1,152 runs, 4 seeds/cell) | `python diagnostics/smc_torque_peak_full_grid.py` | robot specs, URDF | `data/all_results_peak_torque_check.csv` | **measured** 371 s (~6 min) |

Every command above is deterministic (Section 2): re-running it and diffing
the output CSV against the committed one is the correctness check, not just
a claim.

---

## 5. Recompute every published number and figure from the shipped data (no simulation)

```bash
./scripts/reproduce.sh tables        # or  .\scripts\reproduce.ps1 tables
./scripts/reproduce.sh figures
./scripts/reproduce.sh diagnostics
./scripts/reproduce.sh tests
```

Individual results, if you want one specific number rather than everything:

| result | command | reads | writes |
|---|---|---|---|
| every manuscript table | `python analysis/generate_tables.py --write` | see README's table/figure→source mapping | `tables/*.tex` |
| everything above, plus every headline scalar, checked | `python analysis/verify_paper_numbers.py` | same, plus run-count/saturation checks | `tables/*.tex`, exit code |
| datasheet proxy benchmark + effect sizes | `python analysis/datasheet_benchmark.py` | `data/all_results_datasheet.csv` | stdout |
| cross-fidelity τ_b, rankings, Pareto, T3 LaTeX | `python src/twinbench/metrics/statistics.py --input data/all_results_main.csv` | main grid | `data/T3_metrics.tex`, `data/kendall_*.csv`, `figures/fidelity_boxplot.png` |
| main-grid rankings / PBC rank-1 retention | `python analysis/main_grid_rankings.py` | main grid | stdout |
| coupled-proxy sweep τ_b | `python analysis/coupled_summary.py` and `python analysis/coupled_rank_check.py` | `data/d3_coupled_all_20seed.csv`, `data/all_results_d3_kr6_decoupled_20seed.csv` | stdout |
| mismatch sweep (t(5) CIs, rank-1 retention) | `python analysis/mismatch_summary.py` | `data/d1_mismatch_all.csv` | stdout |
| torque-cap ablation | `python analysis/cap_ablation.py` | main grid + `data/all_results_cap0p9_ctcsmc_check.csv` | stdout |
| integration step-size ablation | `python analysis/dt_ablation.py` | main grid + `data/all_results_dt0p{5,25}_corrected.csv` | stdout |
| controller-vs-trajectory effect size | `python analysis/effect_size.py` | datasheet grid | stdout |
| effective-gain / bandwidth analysis (the closed-form gain-scaling table cited in Limitations) | `python diagnostics/effective_gains.py` | constants + robot specs | `data/effective_gains.csv` |
| E* summary | `python diagnostics/estar_table.py` | main + datasheet grids | `data/estar_summary.csv` |
| CTC-SMC peak torque ratio (Table 6 caption) | `python diagnostics/smc_torque_peak_full_grid.py` | robot specs, URDF | `data/all_results_peak_torque_check.csv` |

### Figures

| figure | command |
|---|---|
| F1 pipeline | `python analysis/figures/make_F1_pipeline.py` |
| F3 trajectory profiles | `python analysis/figures/make_F3_trajectories.py` |
| F7 Kendall heatmaps | `python analysis/figures/make_F7_kendall.py` |
| ranking bars (figA) | `python analysis/figures/make_figA_ranking_bars.py` |
| tracking with impulse (figB) | `python analysis/figures/make_figB_tracking.py` |
| methodology threshold | `python analysis/figures/make_methodology_threshold.py` |
| S2 controller degradation by uncertainty | `python analysis/figures/make_S2_uncertainty_degradation.py` |
| fidelity boxplot | `python src/twinbench/metrics/statistics.py --input data/all_results_main.csv` |
| Supplementary Fig. S1 (workspace feasibility) | `python analysis/figures/make_S1_workspace_feasibility.py` |
| **all of the above, one command** | `python analysis/figures/make_all.py` |

Supplementary Fig. S1's data (`data/workspace_feasibility.csv`) was generated
one (robot, scene) cell at a time — full protocol (12,000/15,000 candidate
targets per robot, 10 seeds, 5 IK restarts) — as
`data/workspace_feasibility_fullproto_<robot>_<scene>.csv`, one committed CSV
per cell, since running the full 6-robot sweep as a single process exhausted
this machine's memory. Some cells took hours (MPX1950/Open Reach: ~3h). See
the README's workspace-feasibility table for the full 12-cell result and the
true range (55.28% MPX1950/Open Reach to 100% UR5e/UR10e).

---

## 6. Re-running the full simulation suite (hours)

```bash
./scripts/reproduce.sh grid          # main URDF grid, 5,760 runs, both fidelity tiers
./scripts/reproduce.sh datasheet     # datasheet grid, 5,760 runs, proxy only
./scripts/reproduce.sh ablations     # cap 0.9 arm + frozen-gain + raised-clamp + dt 0.5/0.25 ms
./scripts/reproduce.sh sweeps        # coupled-proxy sweeps (KR6 and UR5e); mismatch driver invocation printed
```

The full default grid (`python experiments/run_main_grid.py` with no flags)
is all four tiers = **11,520 runs**, the main grid (5,760) and the
datasheet grid (5,760) together.

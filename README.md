# TwinBench

Multi-fidelity benchmarking of trajectory controllers for 6-DoF industrial manipulators.

Code and results accompanying the paper  
*TwinBench: Multi-Fidelity Benchmarking of Controllers for 6-DOF Industrial Manipulators*.

| Resource | Link |
|----------|------|
| **GitHub** | https://github.com/mahshid-amirabgir/TwinBench |
| **Zenodo (archive)** | https://doi.org/10.5281/zenodo.23012381 |

## What this repository contains

| Path | Role |
|------|------|
| `src/twinbench/` | Dynamics (proxy + Pinocchio), controllers, trajectories, metrics |
| `experiments/` | Main grid and ablation drivers |
| `analysis/` | Table generation, numerical checks, figure scripts |
| `data/` | Committed experimental results (source of all reported numbers) |
| `figures/` | Published figures |
| `tables/` | LaTeX table fragments |
| `robot_specs/`, `urdf_models/`, `scenes/` | Robot and task definitions |
| `docs/` | Controller laws and reproducibility notes |
| `tests/` | Unit tests |

## Setup

```bash
conda env create -f environment.yml
conda activate twinbench
```

Proxy-only alternative: `pip install -r requirements.txt`  
(Rigid-body tier needs Pinocchio: `conda install -c conda-forge pinocchio=3.9.0`.)

On Windows, use a single-line command (no `\` line continuations). Proxy-only runs on datasheet robots (`kr6`, `rs006l`, `mpx1950`) do not require Pinocchio.

## Check results

```bash
# Match every paper number to the committed CSVs (~1 min, no simulation)
python analysis/verify_paper_numbers.py

# Unit tests
pytest tests -q

# Optional: regenerate figures from data/
python analysis/figures/make_all.py
```

`verify_paper_numbers.py` exits 0 only if all checks pass (152 checks).

## Optional: re-run simulation

```bash
# Spot check without Pinocchio (datasheet robot)
python experiments/run_main_grid.py --robots kr6 --scenes open_reach --fidelities proxy --trajectories cubic --params nominal --seeds 0 --controllers pdg,ctc_pd,ctc_smc,pbc --output-tag spotcheck

# URDF / rigid-body runs require Pinocchio
python experiments/run_main_grid.py --robots ur5e --scenes open_reach --fidelities proxy --trajectories cubic --params nominal --seeds 0 --controllers pdg,ctc_pd,ctc_smc,pbc --output-tag spotcheck
```

See `docs/REPRODUCIBILITY.md` for ablation commands and environment pins.

## Controllers

PD+g (`pdg`), CTC-PD (`ctc_pd`), CTC-SMC (`ctc_smc`), PBC (`pbc`).  
Laws and conventions: `docs/CONTROLLERS.md`.

## Citation

If you use TwinBench, please cite the paper and this software archive:

```
DOI: 10.5281/zenodo.23012381
```

See `CITATION.cff` for machine-readable citation metadata.

## License

MIT — see `LICENSE`.

# Where the control laws live

TwinBench evaluates four deployable controllers. Each is implemented **twice** — once
for the proxy (decoupled second-order) plant and once for the rigid-body (Pinocchio ABA)
plant — because the two plants expose different model information to the controller.
Both implementations are a single flat `if/elif` inside the plant's own integration loop.

| controller | id | proxy tier | rigid-body tier |
|---|---|---|---|
| PD + gravity | `pdg` | `experiments/run_main_grid.py` → `control_torque()` | `src/twinbench/dynamics/urdf_plant.py` → `control_torque_pinocchio()` |
| Computed torque + PD | `ctc_pd` | same | same |
| Computed torque + sliding mode | `ctc_smc` | same | same |
| Passivity-based | `pbc` | same | same |
| naive adaptive (baseline only) | `adaptive` | same | same |

**Why they are not factored into a `controllers/` package.** Both functions read
module-level constants bound at definition time (`TAU_CAP_FRAC`, `TAU_CAP_FRAC_SMC`, `DT`,
`T_BASE`) and are called once per integration step in the hot loop; the CLI's `--dt`
override works by rebinding one of those module globals. Lifting them into a separate
module would change which globals they close over, and the repository's whole claim is
that the shipped CSVs reproduce **bit-for-bit**. The gain in tidiness is not worth
perturbing outputs that the paper's numbers rest on. The laws are documented here and
each `elif` branch is commented in place instead.

## Sign conventions (read this before editing)

Error convention throughout is `e = q_d − q`, so the sliding variable
`s = ė + Λe` equals `−s_std` (the textbook `s` under `e = q − q_d`).

- **CTC-SMC switching term is `+Ks·tanh(s/φ)`.** Under this repo's error convention the
  plus sign is the stabilizing one; implemented at
  `experiments/run_main_grid.py:323` and `src/twinbench/dynamics/urdf_plant.py:181`.
- **PBC uses `+Λ` reference velocity** (`q̇_r = q̇_d + Λe`), matching CTC-SMC's `s`.
  The `adaptive` baseline branch uses `−Λ`. This opposition is deliberate — do **not**
  "harmonize" the two, and do not refactor CTC-SMC's inline `Λ` into a shared helper
  with PBC's (both are duplicated inline on purpose, with comments saying so).
- **PBC damping gain** is `K_D = I_nom · K_d` on the proxy tier and
  `K_D = diag(M(q)) · K_d` (live inertia) on the rigid-body tier.
- **PBC's rigid Coriolis term** is the explicit `pin.computeCoriolisMatrix(q, q̇) @ q̇_r` —
  *not* `rnea(q, q̇_r, 0)`, which is the wrong form for a passivity-based law.
- **Torque caps.** `pdg`, `ctc_pd` and `pbc` clamp at `TAU_CAP_FRAC = 0.90 · τ_max`;
  only `ctc_smc` reads `TAU_CAP_FRAC_SMC = 0.70 · τ_max`. On every shipped grid the caps
  never bind (peak |τ|/τ_max ≈ 0.45), which is why the 0.7-vs-0.9 cap ablation comes out
  identical to round-trip noise — see `analysis/cap_ablation.py`.

## Gain normalization

No controller is hand-tuned per robot. All gains derive from one protocol in
`src/twinbench/constants.py`:

```
ddq_cap = clip(TAU_CAP_FRAC · τ_max / I_proxy, 10, 1000)
ω_n     = WN_FRAC · sqrt(ddq_cap)
K_p     = ω_n²          K_d = 2 · ZETA · ω_n
Λ       = 2 · ZETA · (2π / T_NOM)
```

This shared protocol does not give the four controllers equal *effective* bandwidth.
`diagnostics/effective_gains.py` reconstructs what each law actually applies at the
acceleration level and writes `data/effective_gains.csv`: PBC's effective position
stiffness is 1.8–8.1× CTC-PD's (5.75× mean over the six robots, 5.93× on KR6). That
asymmetry is a stated limitation, not a tuning artifact — it follows from applying one
normalization protocol to laws with different structures.

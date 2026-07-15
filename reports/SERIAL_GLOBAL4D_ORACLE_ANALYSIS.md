# Serial Global4D Residual Oracle Analysis

## Decision

The validation-only Oracle supports continuing the Serial Global4D residual
direction.  A Global Coupled 4D Jacobian can explain a substantial part of the
residual left by the frozen Cartesian Adapter, especially for high-flexibility
molecules.  This is a theoretical ceiling, not evidence that a learned Stage 2
model will attain the Oracle result.

No test data was read and no Serial Global4D training was started.

## Frozen inputs

- split: validation only
- cohort: formal-large Confirm30, 30 molecules and 60 records
- Cartesian teacher: validation-selected step 100000 checkpoint
- Cartesian refinement: 10 steps, update scale 0.5
- Oracle states: `t = 0, 0.125, 0.25`
- ridge: `1e-5`
- rank tolerance: `1e-6`
- lambda scan: `0.1, 0.25, 0.5, 0.75, 1.0`
- manifest raw SHA256:
  `087b674007c415d773e602af06ff3e5ca9d98e4b44bc62410aba6cac4c84e556`
- manifest canonical SHA256:
  `5a7da0b3fdbdf88aafe565c45728d65ff112151dd75162cb3b4b0022924162c2`
- teacher checkpoint SHA256:
  `600d312328b31ab85ba13183f4db0f37951054c753dfacc024b6aeed334f973e`
- teacher config SHA256:
  `2e72151e3f6a149526f31050c4eaef3a99653ab97d0a21a08d1525557b1c9714`

All 60 manifest sample IDs were present in the formal validation cache.  All
60 manifest, persisted-cache, and recomputed `x_init_hash` values matched.
All Cartesian rollouts were stable and finite, and no `x_cart` was an exact
copy of its reference.

## Overall result

The Cartesian baseline RMSD is `1.394668`.  This is within `1.5e-5` of the
frozen Confirm30 selection result (`1.394654`); the Oracle analysis uses the
fixed Stage 2 aligned target.

| lambda | Oracle RMSD | improved fraction | degraded fraction |
|---:|---:|---:|---:|
| 0.10 | 1.325777 | 1.000 | 0.000 |
| 0.25 | 1.227364 | 1.000 | 0.000 |
| 0.50 | 1.081639 | 1.000 | 0.000 |
| 0.75 | 0.971416 | 1.000 | 0.000 |
| 1.00 | **0.917641** | **1.000** | **0.000** |

The best scanned value is `lambda = 1.0`.

## Flexibility strata at lambda 1.0

| tier | Cartesian RMSD | Oracle RMSD | projection energy ratio | improved | degraded |
|---|---:|---:|---:|---:|---:|
| low (0-2) | 0.677048 | 0.613392 | 0.126025 | 1.000 | 0.000 |
| medium (3-5) | 1.032191 | 0.724827 | 0.430792 | 1.000 | 0.000 |
| high (6+) | 1.875526 | 1.147601 | 0.522635 | 1.000 | 0.000 |
| overall | 1.394668 | 0.917641 | 0.425919 | 1.000 | 0.000 |

The high-flexibility tier has the largest structured-residual ceiling: about
52.3% of residual energy is expressible by the damped Global4D Jacobian.

## Component contribution

Fractions below normalize the Cartesian energies produced by the separately
mapped stretch, bending, and torsion coefficient components.  Cross terms mean
they are a component-energy diagnostic, not a unique orthogonal decomposition
of the final Cartesian residual.

| tier | stretch | bending | torsion |
|---|---:|---:|---:|
| low | 0.413785 | 0.192519 | 0.393696 |
| medium | 0.131009 | 0.335506 | 0.533485 |
| high | 0.082092 | 0.483588 | 0.434320 |
| overall | 0.153680 | 0.385716 | 0.460604 |

## State consistency

`lambda = 1.0` is optimal at each analyzed state:

| t | projection energy ratio | Oracle RMSD |
|---:|---:|---:|
| 0.000 | 0.404746 | 0.940638 |
| 0.125 | 0.423682 | 0.920705 |
| 0.250 | 0.449328 | 0.891581 |

## Interpretation and remaining risks

The result shows clear theoretical residual-correction capacity for a Serial Global4D refiner; it
does not imply that Cartesian coordinates cannot represent these motions.
Global4D is explaining internally structured prediction residuals left by the
frozen Cartesian teacher.

The learned model still has to predict the Oracle coefficients from label-free
inputs.  The 100% Oracle improvement rate follows a target-aware least-squares
direction and must not be attributed to a learned gate.  Phase A coefficient
learning, Phase B benefit-aware gate calibration, trust-region clipping, and
geometry backtracking remain necessary before a pilot can be considered safe.

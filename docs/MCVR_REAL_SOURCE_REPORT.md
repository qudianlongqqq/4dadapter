# MCVR real-source report

## Decision and identities

Stage C real-source construction is complete. It read only formal `train` and
`val` caches; no test path or test record was opened. The train and validation
molecule sets are disjoint.

The frozen Cartesian teacher is checkpoint
`600d312328b31ab85ba13183f4db0f37951054c753dfacc024b6aeed334f973e`
with resolved-config SHA256
`2e72151e3f6a149526f31050c4eaef3a99653ab97d0a21a08d1525557b1c9714`.
Every Cartesian rollout uses `train_range`, has `t_max <= 0.25`, applies its
update scale once, and persists the exact time array. The historical 10-step
rollout is absent from the default dataset and remains OOD-only.

The chemical-validity statistics file is byte-for-byte unchanged from Stage B
(file SHA256 `ae5afaa8d3fce1b5418295309bf2c3197997180298e1781b4efc5c265258852e`,
canonical identity
`66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3`).
It contains 250 train molecules and 1,982 train reference conformers. Its
fallback order is detailed, coarse, basic, then global for bond/angle;
detailed then global for ring planarity; and detailed, coarse, then global for
the auxiliary torsion prior. The minimum count is 20. No validation or test
statistics were fitted.

## Composition

| Split | Records | Molecules | ETFlow normal | Cartesian | Severity distribution |
|---|---:|---:|---:|---:|---|
| train | 750 | 500 | 250 | 500 | normal 300, mild 200, medium 175, severe 75 |
| val | 130 | 100 | 70 | 60 | normal 101, mild 22, medium 7 |

Each selected Cartesian molecule contributes a one-step and a two-step view.
Severity is determined jointly from molecule RMS displacement, maximum atom
displacement, bond/angle/ring outliers, clash, and torsion change. Fixed
train-only score quantiles 0.10/0.50/0.85 define normal/mild/medium/severe;
only an out-of-range time schedule can receive `out_of_domain_extreme`.

The validation-only Cartesian update scale `0.35` is frozen as the unseen
condition; training uses `0.50`. This holdout was declared before any MCVR
training and is not selected after results.

Every row records generator, checkpoint/config identities, seed, NFE, solver,
rollout steps, time schedule, `t_min/t_max`, update scale, severity, full
source validity, molecule/sample IDs, coordinate SHA256, reference
availability, and provenance.

# Serial Global4D Confirm30 Validation

The full validation-only pipeline passed. Cartesian RMSD was `1.395221`.
One-step full safety reached `1.394399`; two-step full safety reached
`1.393855` and is recommended. High-flex RMSD improved from `1.875878` to
`1.872742` with two steps.

One-step learned gating improved 56.7% and degraded 43.3% of molecules. Full
safety had zero rejects/failures; one step clipped 16.7% and backtracked 8.3%,
while two steps clipped 20% and backtracked 10%. Every step recomputed the
backbone, q, Jacobian, gate, trust region, and geometry guard.

Coverage at threshold 1.25 remained `COV-P=0.45`, `COV-R=0.2692`. Two-step
`MAT-P=1.39256` and `MAT-R=1.80521` both improved over Cartesian
(`1.39465`, `1.80965`). Stretch-only was the strongest learned component
ablation (`1.393194`); angular-only and torsion-only did not beat Cartesian.

The learned result remains far from the Oracle RMSD `0.917641`; this is a
small but correctly directed pilot gain, not Oracle saturation.

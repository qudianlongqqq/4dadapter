# MCVR V5 Constraint-Space Hybrid Summary

Decision: `V5_CONSTRAINT_HYBRID_NOT_YET_SUPPORTED`.

Prototype A is a strong engineering control: Bond, weighted BAC, Ring, RMSD,
MAT, acceptance, and displacement improve over D1. Its active-Angle difference
versus D1 is `-0.000102` with 95% CI `[-0.000543, +0.000283]`; component
diagnostics show a Bond-dominated field, so multi-head specialization is not
established.

Prototype B is the recommended paper mechanism. Its active-Angle difference is
`-0.001220` with 95% CI `[-0.001690, -0.000710]`, while Bond, weighted BAC, and
Ring also improve and acceptance/public metrics remain stable. Solver failures
are zero. Movement is `1.1032x` D1 and narrowly fails the unchanged `1.1x`
gate, so B is not ready for 10k, formal-large, test, or frozen holdout.

Records: 1024. Molecules: 512. Seed: 43018. Test records read: 0. Test assets
opened: false. Frozen holdout records opened: 0. Target rematerialization:
false. Hidden/layer change: false.

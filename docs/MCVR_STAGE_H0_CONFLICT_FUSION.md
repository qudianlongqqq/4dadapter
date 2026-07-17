# MCVR Stage H0 Local Conflict-Aware Branch Fusion

Stage H0 is a validation-only structural diagnostic. It trains no model and reads no test records.

For each active sign-safe bond, LCBF compares the first-order bond-length changes from the Cartesian and safe Bond branches. A conflict exists only when the two non-negligible axial changes have opposite signs. Pairwise removal accumulates equal-and-opposite endpoint corrections. Minimum-norm removal solves all conflict constraints in a molecule simultaneously in float64.

LCBF is not the Global4D Jacobian orthogonal-complement projection. It removes only Cartesian axial motion opposing an active sign-safe Bond correction. It preserves same-direction Cartesian motion, inactive bonds, and all unconstrained Cartesian components.

All eleven preregistered variants are retained. Stage H0 cannot authorize training, 100k execution, test evaluation, or additional seeds.

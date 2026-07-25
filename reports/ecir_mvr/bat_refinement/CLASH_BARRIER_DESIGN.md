# Steric soft barrier and hard guard design

For each eligible heavy-atom pair, `d_safe = f_topology (r_i^vdW + r_j^vdW)`, with frozen factors 0.60 for graph-distance 3 (1–4) and 0.75 otherwise. The catastrophic boundary is `0.50 (r_i+r_j)`. Topological distances 1 and 2 are excluded. `tau=0.05 Å`.

The soft energy is the mean of `tau^2 softplus((d_safe-d)/tau)^2`, but is returned as exact zero when no pair has `d < d_safe`. This explicit gate prevents exponentially small safe-region gradients from moving normal molecules.

Squared hinge was rejected as primary because its derivative is discontinuous at the boundary. Softplus provides smooth autograd and backtracking. Deep overlap produces a near-quadratic penalty; far outside the boundary both value and gradient approach zero.

No arbitrary external-tuned lambda is used. When steric is active, BA and steric Cartesian gradients are separately rigid-mode projected and RMS-normalized, then averaged. BAT adds an independently normalized active torsion group. This is equal standardized group aggregation, a disclosed inductive bias.

After the trust projection, fractions 1, 1/2, 1/4 and 1/8 are tried. A candidate must not worsen catastrophic penetration, ring guard, topology or chirality. Failure falls back to the already-safe frozen BA candidate (or Source for BA failure). Primary graph RMS remains 0.003 Å and atom cap 0.03 Å.

These constants are pre-external design choices based on chemical radii, topological relation and the existing safety convention. They cannot be changed after preregistration.


# Global Coupled 4D post-implementation review

Conclusion: PASS_WITH_WARNINGS

## Review

- The Jacobian block is `[e, -[x_i-p]_x]`, matching `s e + omega x (x_i-p)`.
- Pivot is the parent-side bond atom; axis points from parent to child.
- The deterministic rooted fragment topology supplies complete downstream subtrees.
- Ancestor contributions use addition only. There is no contribution-count division.
- The dense matrix has shape `[3N,4M]`; the complete `J^T W J`, including off-diagonal joint blocks, is used.
- `svd_oracle` is undamped and rank-aware. `gram_solve` uses Cholesky, solve, least squares, then SVD fallback.
- The Cartesian raw head is projected with the same full Jacobian and zero damping before residual addition.
- Stretch is invariant. The global angular vector is assembled from invariant coefficients and geometry-derived equivariant vectors; it rotates with the molecule.
- Training coefficient supervision is column-norm weighted, while the principal losses live in mapped Cartesian velocity space.
- Training and sampling call the same forward mapping. Sampling uses no reference coordinates.
- Coordinate-independent topology templates are cached; axes, pivots, Jacobians, and projections are recomputed from current rollout coordinates.
- Invalid rings, disconnected graphs, missing rotatable edges, and degenerate axes fail closed to residual-only behavior and are reported.

## Warnings

- This checkout contains no formal data cache, manifest, or `logs_flexbond_formal_small/flexbond4d_hybrid_5k` runtime directory. Oracle, Smoke, and rollout therefore cannot be executed locally before server synchronization.
- The small-run protocol is fixed at 5k steps, batch 4, accumulation 2, and learning rate 0.0002. Missing optional old-config fields use documented Global4D fair-config fallbacks and do not block the new experiment.
- All CPU unit tests pass. CUDA numerical agreement, GPU memory, and throughput remain to be measured in the project GPU environment.

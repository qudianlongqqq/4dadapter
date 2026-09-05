# Architecture comparison

Both v1 and v2 use the same 3-layer, 128-dimensional invariant graph backbone,
Reliability head, Adaptive-BA head, unrestricted Softplus tau head, analytic
primitive Jacobian-transpose, rigid-mode projection, and graph-RMS normalized
direction. The v2 Delta-Q and sigma heads append one current Source primitive
scalar to the unchanged graph primitive representation. No hidden width,
message-passing depth, or action capacity is increased.

Delta-Q heads are signed linear outputs and start at exact zero. Sigma heads
retain the v1 `sigma_stat * exp(6*tanh(raw/6))` parameterization and exact
statistical anchor. Reliability remains separate and receives `q_source`,
`mu_eff`, sigma, and the existing primitive representation.

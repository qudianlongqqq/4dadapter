# ECIR losses and labels

The total objective is:

`L = L_flow + lambda_mode L_mode + lambda_error L_error + lambda_identity L_identity + lambda_trust L_trust`.

- `L_flow`: Huber loss between complete Cartesian velocity and
  `x_target - x_input` along the straight flow path.
- `L_mode`: Huber loss between directional internal operators
  `B_mode(x_t)v_theta` and `B_mode(x_t)u_t`. Implemented modes are bond length,
  bond angle and rotatable torsion velocity.
- `L_error`: six-mode heteroscedastic Gaussian NLL with clipped log variance.
- `L_identity`: squared update on clean identity records.
- `L_trust`: differentiable penalty beyond atom and molecule RMS limits.

Every term plus gate mean is logged separately to `history.csv`. Degenerate or
absent torsions contribute an empty/constant-safe term and cannot introduce
NaN. The trust loss uses an epsilon-stabilized graph RMS at zero update.

No main label is produced by solving
`pseudoinverse(J)(x_reference - x_upstream)`. Four-dimensional rates are limited
to diagnostics and ablations.

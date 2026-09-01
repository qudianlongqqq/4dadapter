# Frozen scientific definition

## Shared model

- Scope: ETFlow-source, fixed-topology, one-to-one local conformer refinement.
- Backbone: 3 invariant message-passing layers, hidden dimension 128.
- Primitive means: positive Bond-length mean `mu_B=softplus(raw_B)` in Å and Angle-cosine mean `mu_A=tanh(raw_A)`.
- J1 belief objective: Gaussian beta-NLL with `beta=0.5`; the beta weight is stop-gradient. Bond and Angle primitive losses are averaged per molecule with equal family aggregation.
- Sigma: learned direct heteroscedastic/predictive scale, `sigma=sigma_stat*exp(6*tanh(raw/6))`. It is not established as calibrated uncertainty.
- R1 Reliability: a trainable source-conditioned sigmoid gate per primitive using local features, source primitive value, predicted mean and scale, signed/absolute/standardized source defect, and primitive family.
- Adaptive BA: graph-conditioned `[128,64,2]` SiLU head with softmax output `(w_B,w_A)`, initialized at `(0.5,0.5)`.
- Cartesian action: analytic first-order primitive Jacobian-transpose contributions. It is not an explicit autograd VJP and is not a second-order coordinate method.
- Action coefficients use detached source primitive values, predicted means and predicted scales; primitive derivative tensors are detached.
- The combined Cartesian vector has translation/rotation modes removed, then is normalized to unit graph-RMS direction. Nonfinite or numerically degenerate directions become zero.
- Reference coordinates and labels are used in TRAIN losses only. No Reference coordinate or upstream ID is an inference feature.

## Restricted operating point

```text
TAU_PARAMETERIZATION = 0.010 * sigmoid(raw)
TAU_MAX_ANGSTROM = 0.010
PER_ATOM_CAP_ANGSTROM = 0.030
MOVEMENT_REGULARIZER = mean((tau / 0.010)^2)
MOVEMENT_REGULARIZER_COEFFICIENT = 0.40793421960700144
PROPOSAL = scaled_proposal(source, rigid_projected_graph_rms_normalized_direction, tau, atom_cap=0.030)
SCIENTIFIC_ROLLBACK = NO
```

The runner computes a `safety_accept`/fallback flag for diagnostics, but the frozen proposal coordinates are not replaced by fallback coordinates. `rollback` is therefore not a scientific transformation, endpoint, loss, selection gate or safety penalty.

The exact Restricted training loss is:

```text
L_total = L_J1_beta_NLL + L_post + 0.40793421960700144 * L_move
L_move = mean((tau / 0.010)^2)
```

## Unrestricted operating point

```text
TAU_PARAMETERIZATION = softplus(raw)
INITIAL_TAU_ANGSTROM = 0.003 via exact inverse-softplus bias
FINITE_TAU_UPPER_BOUND = NONE
PER_ATOM_CAP = NONE
MOVEMENT_REGULARIZER = NONE
PROPOSAL = source + tau * rigid_projected_graph_rms_normalized_direction
SCIENTIFIC_ROLLBACK = NONE
```

The exact Unrestricted training loss is:

```text
L_total = L_J1_beta_NLL + L_post
```

## Frozen language and claim boundary

Allowed scoped language:

- local conformer refinement;
- source-preserving refinement;
- unseen-molecule evaluation, when the evaluated molecule identities were unseen in TRAIN/DEV;
- random-seed robustness;
- structural validity;
- GFN2-xTB single-point energy.

Do not use without independent future evidence:

- universal refinement;
- global conformer recovery;
- cross-upstream generalization;
- calibrated uncertainty;
- guaranteed improvement;
- monotonic energy improvement;
- SOTA.

Frozen sigma wording: **predicted heteroscedastic scale** (or, when context requires, **learned predictive scale**). Do not call it calibrated uncertainty.

```text
SCIENTIFIC_DEFINITION_FROZEN = YES
CLAIM_VOCABULARY_FROZEN = YES
SIGMA_LANGUAGE_FROZEN = PREDICTED_HETEROSCEDASTIC_SCALE__NOT_CALIBRATED_UNCERTAINTY
```

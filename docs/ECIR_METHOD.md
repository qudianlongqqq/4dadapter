# ECIR-Flow method

ECIR-Flow (Error-Calibrated Internal-Validity Refinement Flow) is a
generator-agnostic, few-step Cartesian conformer repair model. It does not
replace or delete the existing Cartesian Adapter or Global4D implementations.

## Model

The error encoder consumes graph features, current coordinates and optional
four-value upstream metadata. Its six normalized outputs are ordered as:

`bond, angle, torsion, ring, clash, chirality`.

It predicts a nonnegative mean, clipped log variance, graph repair gate, atom
gate and directed-bond gate. Metadata dropout is 0.5. Missing metadata is an
all-zero masked vector, so source identity is never required for a forward pass.

The refiner reuses the E(n)-equivariant Cartesian backbone. It returns only
`v_theta: [N,3]`; there is no default `q` head and no Cartesian residual
pseudoinverse target. The four-step teacher recomputes error estimates and
velocity at every step:

`x_next = x_current + step_size * gate * trust_clip(v_theta)`.

The gate combines learned repair benefit, predicted uncertainty and an identity
factor. `gate_override=0` is an exact no-op. Atom and molecule RMS trust limits
are label-free.

## Role of 4D

Global4D/Jacobian code remains available for structured corruption, internal
mode diagnostics, stretch/bending/torsion labels and ablation baselines. It is
not the ECIR prediction head.

## Current Stage 2 result

The heterogeneous 500/100, 5k pilot learned meaningful internal repair but did
not pass the configured RMSD noninferiority CI and has no unseen checkpoint/NFE
evidence. The scientific decision is therefore `NO_GO`; Stage 3 is blocked.

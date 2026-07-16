# MCVR chemical-validity definition

## Scope and identity

The Stage B validity score is label-free at inference time. It does not use the
selected reference, RMSD, MAT, COV, the restrained target, or any training
label to accept a candidate. Its thresholds are fitted only from reference
conformers in the formal train split.

The frozen statistics are stored in
`data/ecir_mvr/validity_reference_stats.json` with schema
`mcvr-chemical-validity-v1`. They contain 250 train molecules and 1,982 train
reference conformers. The train-split SHA256 is
`328b661f325e1b07aa46dd24316546a65ec915c3b6d6420efa309fa3adbcc49e`;
`validation_used=false` and `test_used=false`. The canonical statistics
identity recorded in the payload is
`66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3`.

## Robust envelopes and fallback

For bond lengths, angles, and ring planarity the train distribution stores the
median, MAD, 0.5/99.5 percentiles, sample count, and the envelope

`[min(q0.005, median - 4.5 * (1.4826 MAD + eps)),
  max(q0.995, median + 4.5 * (1.4826 MAD + eps))]`.

An observation outside this interval is an outlier. Its magnitude is the
distance beyond the nearest boundary divided by the robust scale. Environments
with fewer than 20 observations fall back from detailed chemistry to coarser
atom/bond classes and finally the global train distribution. The selected
statistics and fallback hierarchy are deterministic; no validation or test
coordinates are consulted.

Bond environments condition on atom identity, hybridization, aromaticity,
ring membership, and bond type. Angle environments condition on the center and
neighbor atom identities, center hybridization, aromaticity, and ring
membership. Ring planarity conditions on ring size and aromaticity before a
global fallback.

## Metrics

- `bond_outlier_rate`: fraction of unique bonds outside their train-derived
  envelope.
- `bond_outlier_magnitude`: mean robust-scale excess beyond the bond envelope.
- `angle_outlier_rate`: fraction of bonded angles outside their envelope.
- `angle_outlier_magnitude`: mean robust-scale excess beyond the angle envelope.
- `severe_clash_rate`: nonbonded severe-clash diagnostic at 0.6 Å.
- `clash_penetration`: nonbonded penetration below 1.0 Å.
- `ring_bond_outlier_rate`: bond outlier rate restricted to ring bonds.
- `ring_planarity_outlier_rate`: fraction of applicable rings outside the
  train-derived planarity envelope.
- `chirality_preserved`: one minus the stereocenter sign-mismatch fraction
  relative to the inference input, not a reference conformer.
- `stereocenter_degenerate_rate`: fraction of stereocenters whose signed local
  volume has magnitude at most `1e-5`.
- `torsion_prior_outlier_score`: negative log probability from a periodic
  36-bin train histogram conditioned on the rotatable-bond environment. It is
  auxiliary and has zero weight in the Stage B gate.
- `total_thresholded_validity_score`: weighted sum of thresholded bond, angle,
  clash, ring, and stereocenter diagnostics. Lower is better.

The historical target-relative names are retained only as diagnostics:
`bond_target_mae`, `angle_target_mae_rad`, `ring_bond_target_mae`, and
`torsion_reference_error`. They are not called chemical-validity losses and do
not participate in inference acceptance.

## Deterministic acceptance

A final-step or best-of-trajectory candidate is accepted only when it has a
minimum positive validity gain, improves at least one thresholded local mode,
does not add a severe clash, chirality flip, stereocenter degeneracy, or ring
outlier, and stays inside atom, molecule, torsion, and uncertainty trust limits.
If no candidate passes, the exact input coordinates are returned. Every
decision records the reject reasons, score breakdown, selected step,
displacement, torsion change, uncertainty, and input/candidate validity.

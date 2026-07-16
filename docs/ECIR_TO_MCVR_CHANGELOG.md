# ECIR to MCVR changelog

## Stage C

- Preserved the old ECIR model and frozen 5k checkpoint path.
- Kept the Stage B train-only chemical-validity statistics byte-identical.
- Replaced historical Cartesian 10-step normal-error use with in-range
  one/two-step sources and explicit provenance.
- Added minimal-displacement, thresholded-excess validity targets with exact
  identity fallback and no soft-reference/MMFF fallback.
- Added a 45/30/25 real/synthetic/identity dataset with source/severity
  balancing and six explicit synthetic corruption types.
- Added the Cartesian MCVR rigid/flexible model, conservative torsion gate,
  uncertainty/safety gate, and nine-term loss.
- Strengthened deterministic acceptance for ring planarity and machine-
  readable rejection/score breakdowns.
- Added Stage C tests while retaining all prior ECIR and 4D baselines.

Stage 2b training has not started. No 20k or 100k command was run or emitted.

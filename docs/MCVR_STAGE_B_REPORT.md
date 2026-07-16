# ECIR MCVR Stage B report

## Decision

`EXISTING_CKPT_RESCUED`

This decision applies only to conservative inference with the frozen 5k ECIR
checkpoint. It makes that checkpoint an eligible Stage 2b baseline; it does not
authorize 20k, Stage 2b training, or any other training. The next permitted
stage is Stage C.

Frozen checkpoint SHA256:
`232e47865d01a71543cf2cd16ede577764fd3d94ac843d78dcdcf8c9789fa98d`.
No test split was used and no training was started.

Frozen identities used by Stage B:

- ECIR 5k resolved config:
  `2060d765031fc2bdb4f73cf7008b40906e90aef0d24912354c529536ee1ed79d`
- Cartesian 100k teacher checkpoint:
  `600d312328b31ab85ba13183f4db0f37951054c753dfacc024b6aeed334f973e`
- Cartesian teacher resolved config:
  `2e72151e3f6a149526f31050c4eaef3a99653ab97d0a21a08d1525557b1c9714`
- validation atlas:
  `8501185f916cf6f048bd56fc4343e5c2b2f38b9ca96523f2f6b6351628654820`
- Stage B view manifest:
  `09eb2d4b0aa8c48e251098154c2f6c967e2fd87af353fc40494b347d967abdf8`

## Time scheduling

The common interface implements `legacy_full`, `train_range`, `fixed`, and
`explicit`. New inference defaults to `train_range`; out-of-range schedules
warn or fail in strict mode, and one-step behavior is explicit.

The Cartesian teacher and the frozen ECIR refiner have different histories:

- Cartesian 100k was trained on `[0, 0.25]`; its Stage B 1/2/4-step severity
  rollouts are constrained to this interval. The historical 10-step rollout is
  retained only as `extrapolated_extreme`.
- The frozen ECIR 5k model was trained by uniform sampling over `[0, 1]`. Its
  old checkpoint predates explicit range metadata, so this range is recovered
  from the frozen training implementation. New checkpoints materialize
  `training_t_min` and `training_t_max` in their resolved config.

Consequently, ECIR `train_range` and `legacy_full` are mathematically identical
for this checkpoint. Across 90 matched coarse configurations, the maximum
absolute validity-delta drift was `1.353e-7`; paired tolerance
`atol=1e-6, rtol=1e-6` classifies `train_range` as numerically non-worse. The
earlier global-minimum comparison at `1e-8` was an asymmetric floating-point
false negative.

## Search and best setting

The validation-only sweep evaluated 630 coarse configurations, one historical
Stage A setting, and 20 Pareto-neighborhood fine configurations. The three
views contain frozen mixed validation, formal ETFlow-normal inputs, and
Cartesian mild/medium/severe inputs plus a separately labelled extrapolated
extreme diagnostic.

Best conservative setting:

- schedule: `train_range` (`[0, 1]` for this ECIR checkpoint)
- teacher steps: 2
- update scale: 1.0
- trust-radius scale: 0.5
- gate threshold: 0.0
- acceptance: deterministic final-step

On frozen mixed validation, acceptance was 0.88, rejection 0.12, and validity
worsening 0.00. Acceptance reduced RMSD-worsened fraction from 0.98 before
acceptance to 0.92 after acceptance. Mean aligned molecule displacement was
0.02996 Å and mean maximum-atom displacement was 0.04845 Å.

## Paired mixed-validation results

Accuracy deltas (candidate minus input; lower RMSD/MAT is better):

| Metric | Delta |
|---|---:|
| aligned RMSD | +0.00625 Å |
| MAT-P | +0.00871 Å |
| MAT-R | +0.00713 Å |
| COV-P | +0.00000 |
| COV-R | -0.00067 |
| diversity | +0.02572 Å |

The molecule-level paired bootstrap RMSD delta was +0.00585 Å with 95% CI
[+0.00410, +0.00771], inside the Stage B noninferiority margin.

True chemical-validity deltas and molecule-level paired 95% intervals:

| Metric | Mean delta | 95% CI |
|---|---:|---:|
| bond outlier rate | -0.07610 | [-0.09344, -0.05928] |
| angle outlier rate | -0.03384 | [-0.04407, -0.02385] |
| ring-bond outlier rate | -0.06027 | [-0.07849, -0.04303] |
| severe-clash rate | 0.00000 | [0.00000, 0.00000] |
| total thresholded validity | -0.80633 | [-1.03036, -0.58060] |

Three true outlier metrics therefore have CIs entirely below zero.

## Source, severity, and flexibility

- ETFlow normal: RMSD +0.00089 Å, MAT-P +0.00089 Å, MAT-R +0.00106 Å,
  total validity -0.04761; accuracy is effectively neutral.
- Cartesian mild: RMSD +0.00977 Å, total validity -0.93279.
- Cartesian medium: RMSD +0.00962 Å, total validity -1.24253.
- Cartesian severe: RMSD +0.01226 Å, total validity -1.79443.
- Extrapolated extreme: RMSD +0.01160 Å, total validity -2.29050. This row is
  diagnostic only; the rescue does not depend on it because normal, mild, and
  medium inputs all improve validity.
- High flexibility (`rotatable >= 6`): RMSD +0.01251 Å and total validity
  -2.11832. The directional RMSD change remains within the Stage B 0.015 Å
  criterion, though MAT-P is +0.01553 Å and remains a Stage C/D risk to monitor.

## Gate audit

All eight Stage B rescue checks pass: three validity CIs improve, ETFlow-normal
is noninferior, mild/medium inputs benefit, improvement is not extreme-only,
overall RMSD/MAT/COV margins pass, acceptance reduces RMSD worsening,
high-flex RMSD passes its directional margin, and paired `train_range` is
numerically non-worse than `legacy_full`.

The result does not establish a GO decision for 20k. Stage C must rebuild the
real-error sources and pass the Minimal-Validity Target gate before any Stage
2b training can be considered.

## Provenance note

The original Stage A SHA manifest has 122 entries. After the required state
transition, 121 still match exactly; the sole mismatch is
`reports/ecir_mvr/progressive_state.json`, whose purpose is to advance from
`STAGE_A_COMPLETE` to `STAGE_B_COMPLETE`. The Stage A SHA manifest itself and
all immutable Stage A diagnostics remain unchanged. The new dynamic state is
bound by the separate Stage B SHA manifest.

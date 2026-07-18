# MCVR V2-BAC Phase-1 Recovery Report

## Decision

`PHASE1_FIXED`, limited to Bond and active-Angle Cartesian recovery. Clash is
`INCONCLUSIVE_DATA_SUPPORT`, not fixed or validated. No 10k/formal-large run and
no model expansion are recommended. The next permitted step is an independent
offline Jacobian comparison on the same development cohort.

Formal test and frozen validation holdout were never opened. Every Phase-1
artifact records `test_records_read=0`, `test_assets_opened=false`, and
`frozen_holdout_records_opened=0`.

## Root cause

The original D0 failure was not caused by absent Angle gradients, severe target
direction conflict, numerical epsilon, or model capacity. The demonstrated
causes are proposal attenuation and safety/training mismatch:

- only 1.47% of angle constraints are active on average, but the original
  per-atom scatter denominator counts inactive constraints whose numerator is
  zero;
- Angle/Clash fusion changes the base proposal norm by only 0.46% on average;
- the median global safety-gate scale is 0.0116;
- legacy BAC loss supervises Angle+Clash-only movement at `t=0`, step scale 1,
  while inference uses the complete field at teacher time 1 and step scale 0.25;
- 51/116 rejected diagnostic records have positive BAC gain, 70 hit both trust
  limits, and 33 are trust-only failures, but the existing backtracking helper
  was not used by inference;
- epsilon probes from zero through `1e-6` do not change acceptance.

Clash has a separate data-support failure: zero active sparse clashes in the
256-record diagnostic cohort and only one active record in the 1024-record
development cohort. No Clash claim has adequate statistical power.

## Repair

All new behavior is opt-in. Legacy defaults and old checkpoints are unchanged.

1. Model scatter normalization

   Before, each atom used `sum(weight * direction) / incident_constraint_count`,
   including inactive constraints. D1 divides by active incident constraint
   count. Constraint directions, weights, and unified Cartesian output remain
   unchanged.

2. BAC proposal loss

   Before, `x_proposal = x_input + v_angle_fused + v_clash_fused` from a second
   `t=0` forward. D1 uses
   `x_proposal = x_input + 0.25 * v_final(t=1)`, matching the first inference
   step through base field, fusion, clipping, and global gate.

3. Inference backtracking

   Before, each integrated trajectory state was either accepted at full scale
   or rejected. D1 tests the fixed scales 1, 0.5, and 0.25 and accepts the first
   state satisfying every existing hard Bond/Angle/Clash/Ring, chirality,
   identity, finite, trust, and minimum-gain condition. No safety condition is
   removed and epsilon remains zero.

These changes add no parameter tensors, model-dependent cache entries, dataset
fields, or target schema. D1-B and V2 strict state loading remain compatible.
Existing target assets do not require rematerialization.

## Development experiment

The recovery manifest identity is
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.
It contains 512 development molecules/1024 records, with a nested 128
molecule/256 record diagnostic cohort, seed 43017, and zero frozen-holdout
molecule overlap. Training uses the same frozen 4096-record train-only V2
target asset for all candidates. A0, D0, and D1 use seed 43018, identical sample
order, batch 64, the seed43 D1-B initialization, width 64, and four layers.

| Candidate | Bond delta | Angle delta | Active-Angle delta | Acceptance | Ring delta | RMSD delta | Mean displacement |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | -0.001238 | 0 | 0 | 5.18% | 0 | +4.61e-7 | 2.89e-6 |
| D0 | -0.000601 | 0 | 0 | 2.44% | 0 | +2.26e-8 | 1.37e-6 |
| D1 | -0.091570 | -0.002333 | -0.004498 | 97.27% | -0.008831 | +0.000411 | 0.006701 |

D1 active-Angle evaluation covers 576 records/339 molecules. Its paired
molecule bootstrap 95% CI is `[-0.005235, -0.003873]`, strictly below zero.
Bond and Ring improve, chirality delta is zero, failure rate is zero, and
MAT/COV remain noninferior. The single Clash-active record is reported but is
not inferential evidence.

## Success and remaining risk

Phase 1 meets the preregistered criterion that at least one non-Bond component
shows independent active-subset improvement without Bond, Ring, chirality, or
public noninferiority failure. D1 is therefore the repaired Cartesian baseline
for Phase 2.

The result is not a 10k recommendation. D1 moves coordinates materially more
than A0/D0, though mean 0.0067 Angstrom and maximum observed 0.0411 Angstrom are
inside hard trust limits. RMSD worsens on many records but by only 0.000411 on
average. Logged pre-clip gradient norm reaches 141.7 and is repeatedly clipped
to 1.0. The three linked fixes were evaluated as one D1 candidate, so their
individual ablation is unresolved and cannot be pursued under the exhausted
Phase-1 budget. No fresh holdout confirmation is allowed.

## Compute accounting

The maximum of five runs was exhausted exactly: one two-batch read-only
diagnostic, one D1 200-step smoke, and one 1k pilot each for A0, D0, and D1.
There were no retries, background processes, 10k runs, formal-large runs, or
capacity changes.

## Recommendation

Protect Phase 1 in a local commit, create `feat/mcvr-v3-bac-jacobian`, and use
D1 as the fixed Cartesian comparator. The Jacobian experiment must use the same
development inputs, detector, objective thresholds, safety, and movement
reporting. It must explicitly test whether any gain persists after controlling
for D1's larger displacement. Do not enter learned Jacobian, 10k, or
formal-large training unless the independent solver is supported by that
comparison.

# MCVR V7 Formal Validation Acceptance Collapse Audit

## Current decision

`FORMAL_RUNNER_SEMANTICS_BUG`

The first Seed43 formal-large validation result is invalid for scientific
interpretation. It is preserved at
`diagnostics/ecir_mvr/v7_formal_validation/seed43` and was not overwritten.
Neither that result nor this audit is a formal-test or frozen-holdout run.

The collapse had two evaluator causes:

1. the formal BAC loop did not supply the dynamic deterministic error features
   and torsion trust input used by the completed D1-B checkpoint, and used the
   reverse BAC time schedule instead of the checkpoint-native schedule;
2. its `no_bac_gain` check used an unweighted sum of Bond/Angle/Clash outlier
   rates, although the frozen design requires reduction of the weighted BAC
   objective (`total_thresholded_validity_score`). The formal checkpoint mainly
   improves outlier magnitudes without crossing a rate threshold, so valid
   proposals were classified as zero gain and rolled back.

No D1, V5-B, or V7 scientific parameter, trust radius, safety epsilon,
backtracking scale, hidden size, layer count, checkpoint, or data identity was
changed.

## Actual failed-run identity

- Seed: `43`.
- Checkpoint SHA256:
  `c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca`.
- Training config SHA256:
  `fd1f5b6780c781d8e7681b31fd93b1459f6b30ebf0e6bf4a564ecab5c16e41db`.
- V7 config SHA256:
  `5737ce5aa3bad729a6748a3fb9f0eea515bd96765df15e99bba6bd70297b8b4b`.
- Validation source SHA256:
  `e7d29f971124f51bd385ec987372ab85181b152250ec0789407a867ff81e3c1a`.
- Validation target SHA256:
  `4b4ef42c9905c3bbe2dbe911c57827ce594583c66a52f94d7c4d9b5ca70de4c7`.
- Recorded process: Windows PID `59428`, completed normally.
- Cohort: 5,000 molecules / 10,000 records.

The checkpoint's own step-25000 validation on the same frozen cohort reports
D1 acceptance `97.82%`, Bond delta `-0.007994`, and displacement
`0.000243 A`. This rules out the initial 5.05% result as the native checkpoint
behavior.

The development D1 checkpoint and the formal D1-B checkpoint are different
trained priors. Therefore the development Bond delta near `-0.094` is not the
direct expected scale for formal validation; the formal checkpoint's own
`-0.007994` is the correct same-checkpoint control. The failed runner's
`-0.001157` was still materially attenuated relative to that control and is
explained by the missing trajectory inputs plus rollback.

## Static execution comparison

| Item | Frozen 10K BAC runner | Failed formal runner | Corrected formal runner | Impact |
|---|---|---|---|---|
| Checkpoint load | strict | strict | strict | consistent |
| `model.eval()` / inference mode | yes | yes | yes | consistent |
| dtype / device | float32 / CUDA | float32 / CUDA | float32 / CUDA | consistent |
| current-coordinate update | `x <- x + 0.25 v` | same | same | consistent |
| teacher steps | 4 | 4 | 4 | consistent |
| time schedule | `1,.75,.5,.25` | `1,.75,.5,.25` | `0,1/3,2/3,1` | formal checkpoint required native schedule |
| deterministic error features | absent by legacy design | absent | recomputed each step | missing input collapsed formal gates/velocity |
| torsion trust remaining | absent | absent | recomputed each step | restores native call contract |
| proposal step-size multiplication | once | once | once | no double/missing multiplication |
| candidate selection | BAC hard safety/backtracking | same | same | retained |
| objective used by `no_bac_gain` | legacy rate sum | legacy rate sum | weighted thresholded validity | fixes design/implementation mismatch |
| hard Bond/Angle/Clash/Ring checks | enabled | enabled | enabled | unchanged |
| chirality/stereocenter checks | enabled | enabled | enabled | unchanged |
| graph/atom trust checks | 0.06 / 0.12 A | same | same | unchanged |
| backtracking scales | 1,.5,.25 | same | same | unchanged |
| final metrics | accepted/rollback coordinate | same | same | no aggregation scaling bug |
| molecule aggregation | per molecule after per-record evaluation | same | same | no batch/record double division |

The native D1-B evaluator uses best-of-trajectory acceptance based on
`total_thresholded_validity_score`, a local threshold/magnitude improvement,
and its existing hard safeguards. The corrected formal BAC path retains the
stricter BAC per-component safeguards and backtracking, but now evaluates its
minimum-gain requirement against the same weighted objective.

The apparent all-zero RMSD/MAT/COV row was not a missing evaluator path.
Unrounded failed-run RMSD and MAT deltas are about `3.3e-7 A` and were printed
as `0.000000`; COV is a discrete threshold statistic and did not change under
microscopic accepted movement. Per-record aligned RMSD and reference candidates
were present. The microscopic movement itself came from proposal attenuation
and rollback, not an extra division during reporting.

## Fixed 100-molecule cross reproduction

The fixed formal sample identity is
`d96c10f95946dccc40263b7dd679328e44f652438c09aefa3429fb9acf851943`
(100 molecules / 200 records).

### Before weighted-objective correction

| Semantics / method | Acceptance | Proposal RMS A | Accepted RMS A | Bond delta | Angle delta |
|---|---:|---:|---:|---:|---:|
| legacy BAC / D1 | 2.0% | 0.0000395 | 0.0000015 | -0.000515 | 0.000000 |
| corrected trajectory, legacy objective / D1 | 27.5% | 0.0005673 | 0.0001943 | -0.008980 | -0.000144 |
| native D1-B / D1 | 93.5% | 0.0005673 | 0.0004078 | -0.009119 | -0.000144 |

Corrected D1 raw proposals and native D1 raw proposals have exact per-record
Bond/Angle/Clash/Ring metrics. Maximum proposal displacement differences are
`6.4e-8 A` graph RMS and `6.5e-7 A` atom max.

Of the 145/200 corrected-trajectory D1 rollbacks, 143 had primary reason
`no_bac_gain`; the raw proposal nevertheless matched native D1. This isolates
the remaining failure to objective semantics, not model inference or solver
numerics.

### After weighted-objective correction

| Semantics / method | Acceptance | Proposal RMS A | Accepted RMS A | Bond delta | Angle delta |
|---|---:|---:|---:|---:|---:|
| corrected formal / D1 | 95.5% | 0.0005673 | 0.0004053 | -0.008980 | -0.000144 |
| corrected formal / V5-B | 99.5% | 0.0009294 | 0.0007385 | -0.008334 | -0.000144 |
| corrected formal / V7 | 96.5% | 0.0008092 | 0.0006491 | -0.009254 | -0.000144 |
| native D1-B / D1 | 93.5% | 0.0005673 | 0.0004078 | -0.009119 | -0.000144 |

Remaining formal D1 rejections are explicit weighted-objective, no-gain, or
hard Clash failures; they are not silent rollbacks.

## Frozen development cross reproduction

The fixed development sample identity is
`dd91e0b1f8f7381a8985c4caae4a2087e7b893bb74857f602892d2c697816033`
(100 molecules / 300 records).

The legacy path reproduced the persisted historical chunk on exactly the same
sample IDs: Bond and Angle metrics and accepted flags have maximum absolute
difference `0`; displacement differences are at most `4.1e-7 A`.

| Legacy method | Acceptance | Accepted RMS A | Bond delta | Angle delta |
|---|---:|---:|---:|---:|
| D1 | 92.67% | 0.005544 | -0.088429 | -0.001717 |
| V5-B | 94.67% | 0.006278 | -0.091314 | -0.001807 |
| V7 | 92.67% | 0.006347 | -0.089281 | -0.002506 |

This proves the backward-compatible default did not alter the frozen 10K
evaluator.

## Data binding audit

Formal and development use identical target builder code/config identities.
Full-manifest checks found no duplicate, missing, mismatched, train, test, or
holdout records. Formal is exactly 5,000 molecules / 10,000 records with two
records per molecule.

| Distribution | Development | Formal validation |
|---|---:|---:|
| Mean atoms | 44.473 | 44.645 |
| Mean rotatable bonds | 7.623 | 7.723 |
| Mean source Bond rate | 0.117965 | 0.117676 |
| Mean source Angle rate | 0.015656 | 0.015477 |
| Active-Angle fraction | 57.263% | 57.240% |
| Mean source-target RMS A | 0.005622 | 0.005563 |
| Mean target max displacement A | 0.015550 | 0.015437 |

For 20 deterministic records from each cohort, target payload `x_input`, atom
order, coordinate shape, source-file SHA, and source-coordinate SHA all match
the source cache. Target coordinates are not swapped with references and are
not source copies except explicitly recorded identity targets. No unit or
topology mismatch was found.

In the deterministic formal sample, mean source-target RMS is `0.00479 A`,
while mean source-reference and target-reference RMS are both about `1.308 A`.
The corresponding development values are `0.00590 A` and about `1.303 A`.
This directly excludes source/target/reference exchange.

## Environment

- Python: 3.11.15 (conda-forge)
- Environment: `etflow-5080-v2`
- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- torch-geometric: 2.8.0
- NumPy: 1.26.4
- pandas: 3.0.3
- RDKit: 2026.03.4
- GPU: NVIDIA GeForce RTX 5080
- Audit inference batch size: 64

The Triton message is a FLOP-counter warning only. No OOM or solver failure
occurred.

## Full reruns

### Frozen 10K development reproduction

The complete D1/V5-B/V7 reproduction ran from clean detached commit
`52ae6a89d3a3c8d038058ba4d52ed8c377931de0` and completed all 120 method
chunks. Its output is preserved at
`diagnostics/ecir_mvr/v7_formal_acceptance_audit/development10k_reproduction_20260720_r4`.

| Method | Acceptance | Bond delta | Active Angle delta | Displacement A | Difference from history |
|---|---:|---:|---:|---:|---:|
| D1 | 96.9833% | -0.093574 | -0.004572 | 0.006691272 | core metrics exactly equal |
| V5-B | 97.3867% | -0.095549 | -0.005860 | 0.007348858 | core metrics exactly equal |
| V7 | 96.8633% | -0.092312 | -0.007362 | 0.007062350 | core metrics exactly equal |

The only displacement differences are `1.2e-10` to `4.4e-10 A`, attributable
to serialization/aggregation precision. V7 again passed every frozen support
check. Its solver completed 120,000 calls with zero failures, 62,172 solved
calls, 57,828 inactive calls, and 251 truncated directions. Wall-clock time was
1 h 56 min. This rules out `HISTORICAL_10K_NOT_REPRODUCIBLE`.

### Corrected Seed43 formal validation

The corrected run completed in a new directory without overwriting the invalid
run:
`diagnostics/ecir_mvr/v7_formal_validation/seed43_corrected_20260720`.
It used the same Seed43 checkpoint, configs, validation cohort, method set,
batch size 64, and scientific thresholds. Only the identified evaluator
semantics changed.

| Method | Invalid acceptance | Corrected acceptance | Invalid Bond | Corrected Bond | Invalid Active Angle | Corrected Active Angle | Corrected displacement A |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 5.05% | 98.98% | -0.001157 | -0.007937 | -0.000012 | -0.000189 | 0.000299 |
| V5-B | 5.13% | 99.38% | -0.001149 | -0.007573 | -0.000012 | -0.000161 | 0.000497 |
| V7 | 5.24% | 99.18% | -0.001171 | -0.007982 | -0.000012 | -0.000220 | 0.000395 |

Corrected weighted BAC deltas were `-0.018951` (D1), `-0.025686` (V5-B),
and `-0.020692` (V7). Rollback fell to 1.02%, 0.62%, and 0.82%, respectively.
This jointly explains all four original symptoms: proposals regain their native
scale, the weighted objective no longer rolls almost all of them back, method
specific angle corrections survive to the final coordinates, and the already
healthy solver remains healthy.

On the active-angle cohort, V7 minus D1 Active Angle was `-3.031e-5` with 95%
CI `[-5.858e-5, -6.656e-6]`. V7 minus D1 weighted BAC was `-0.002642`
with 95% CI `[-0.003506, -0.001907]`; Bond was `-0.000069` with CI
`[-0.000243, 0.000113]`. V7 therefore shows a statistically supported formal
active-angle gain without evidence of Bond regression. Against V5-B, V7 had
better Active Angle and Bond but a weaker weighted BAC total, so the corrected
run supports the constraint-specific trade-off rather than universal dominance.

The V7 angle solver completed 40,000 calls with zero failures, 22,872 solved
calls, 17,128 inactive calls, mean/max condition number `4.6186 / 7943.5638`,
effective rank `2.1880`, and 56 truncated directions. Evaluation took
2,346.2 seconds (39.1 min). All 50 chunks, the final report, per-record and
per-molecule tables, environment manifest, and SHA256 manifest completed.

### Reproduction commands

The frozen 10K reproduction used:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_ecir_mvr_v7_10k_audit_reproduction.ps1 `
  -CleanWorktree .audit_worktrees/v7_10k_repro_lf `
  -Workspace . `
  -OutputRoot diagnostics/ecir_mvr/v7_formal_acceptance_audit/development10k_reproduction_20260720_r4 `
  -Python E:/miniconda/envs/etflow-5080-v2/python.exe `
  -FormalRoot E:/3dconformergenerationcode/dataset/data/4dadapter/ecir_mvr/formal_large `
  -SourceRoot E:/3dconformergenerationcode/dataset/flexbond_cache_formal_large `
  -Device cuda:0
```

The corrected formal run used `scripts/run_ecir_mvr_v7_formal_validation.py`
with Seed43, the frozen Seed43 binding/checkpoint/configs, the frozen validation
source and target manifests, `--molecules-per-chunk 100`, `--batch-size 64`,
`--device cuda:0`, and the new output directory above. The exact resolved
command is stored in its `launch.json`.

### Key SHA256 identities

| Artifact | SHA256 |
|---|---|
| Invalid formal method summary | `da9ae54e06bcece3eafe51322264a566ffd58c70a3b033dc9e1223f8881289a2` |
| Formal-100 corrected summary | `44851cac0ab259fec6fb65b509941847ef0c49c070e41edb7525e0ef31d94d92` |
| Development-100 summary | `ceb1d34e2d6888b256bd1f5f54e7346cdc06c7653c3c4da7e2957e0c02752842` |
| Data-binding summary | `05be36445d06c7387e4ab7ec4a91d8bf02020969ba898d3c82169e7c9f91f4d0` |
| Frozen 10K reproduction summary | `1dbebc9e0642f5fe60cf545292b044001e4c89cdbc206f24dcd03f8fb8e5013f` |
| Corrected formal method summary | `d9d3c5ed91aacf550f0a72d6f0501572c2abc0b6c75b0782de9ed05ebc70dd67` |
| Corrected formal run metadata | `4ea8959d853adc16dceb6fe0aef8c7f80f06de0618cb7a439d60a89db11e152a` |

The final decision is `FORMAL_RUNNER_SEMANTICS_BUG`. The bug is fixed and
cross-reproduced; the original Seed43 formal result remains invalid, while the
corrected Seed43 validation is suitable for scientific interpretation. No
formal test or frozen holdout was opened.

## Isolation proof

```text
test_records_read=0
test_assets_opened=false
frozen_holdout_records_opened=0
formal_test_run=false
training_performed=false
target_rematerialization=false
```

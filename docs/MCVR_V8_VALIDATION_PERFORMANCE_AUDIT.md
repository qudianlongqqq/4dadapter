# MCVR V8 Validation Performance Audit

## Scope and frozen baseline

This audit was completed before validation implementation changes on branch
`perf/mcvr-v8-validation`. The scientific baseline is the frozen Full Seed43
step-5000 checkpoint with freeze identity
`eb1c36837cb0be29fe0740c34fc7f09237a44060b7b935c9acc5d2c5ca90322b`.
Only the formal-large validation split was read. Formal test, Minimal Validity
Target test, and frozen holdout access remained zero.

The legacy evaluator is `scripts/evaluate_ecir_mvr_v8_validation.py` at SHA256
`09404502525d3b7121a3a880ae9b68e3cc37eccad9c095a5c844b80129954849`.
It evaluates records serially with an effective prediction batch size of one.

## Measurements

The completed step-5000 run provides the authoritative wall-clock baseline:

| Phase | Start/end evidence | Time | Throughput |
|---|---|---:|---:|
| Batched validation loss, 10K | progress update 21:42:16 to final checkpoint 21:54:40 | 744 s | 13.44 records/s |
| Legacy deployment validation, 10K | final checkpoint 21:54:40 to deployment report 22:29:20 | 2,080 s | 4.81 records/s |
| Combined final validation | 21:42:16 to final status 22:29:21 | 2,825 s | 3.54 validation records/s per 10K report |

A separate unmodified cProfile run on the first 1,000 records took 154.342 s
inside `main` (160.745 s including interpreter/import startup), or 6.48
records/s. It executed 191,400,912 Python calls. The runtime artifact is kept
outside Git under
`diagnostics/ecir_mvr/validation_performance_audit/legacy_1000`.

Resource observations during the profile were 15–18% instantaneous GPU
utilization, about 2.3 GiB peak Python working set, and about 5.4 GiB total GPU
memory in use by the desktop plus evaluator. The V8 run itself previously
reported only about 0.16 GiB peak PyTorch allocation. These observations show
a CPU/Python scheduling workload rather than VRAM or GPU-compute saturation.

## Stage attribution

Times below are cumulative cProfile times for 1,000 records. Nested rows must
not be added together. A dash means the stage does not exist in the legacy
evaluator and therefore could not be timed without changing its semantics or
scope.

| Required stage | Legacy measurement | Finding |
|---|---:|---|
| Parquet/manifest loading | included in 4.24 s dataset construction | manifests are read more than once during setup |
| Source/target deserialization | 73.15 s in 2,000 `_load_record_and_coordinates` calls | every one of 1,000 records is loaded twice |
| Topology/cache loading | 70.04 s in 2,000 formal adapter calls | dominant loading cost; no evaluator LRU is enabled |
| Graph batching | batch-of-one construction, included in loop remainder | no multi-record model batching |
| V8 model forward | 62.40 s | 40.4% of profiled main time |
| Two-step EGNN | 39.01 s prior forward; 18.99 s encoder calls | two unrolled prior evaluations are retained |
| Bond/Angle residual | 3.00 s bond rows; 4.19 s angle rows | per-graph construction |
| Jacobian | included in bond/angle row times | not independently instrumented by legacy code |
| Float64 solver | 9.48 s unified solve | small per-graph dense solves |
| Clash graph | included in model/loss and validity calls | not independently instrumented |
| Ring/Chirality | included in validity calls | not independently instrumented |
| Deployment backtracking | 3.91 s `select_safe_bac_proposal` | 1,000 calls |
| Rollback | included in deployment safety | deterministic V7 safety semantics |
| D1 baseline inference | — | not executed by legacy V8 evaluator |
| V5-B baseline inference | — | not executed by legacy V8 evaluator |
| V7 baseline inference | — | not executed by legacy V8 evaluator |
| Coordinate/result serialization | negligible single JSON report | no reusable coordinate cache is written |
| Evaluator record joining | — | no paired baseline join exists |
| BAC/Ring/Chirality metrics | 4.57 s in 3,498 validity evaluations | repeated before/after evaluation |
| Kabsch/RMSD | — | not computed |
| MAT/COV | — | not computed |
| Paired bootstrap | — | not computed |
| Report writing | below profiler top costs | one small JSON write |
| CPU/GPU synchronization | distributed through tensor `bool`, `float`, `.cpu()` and solver diagnostics | frequent per-record and per-graph synchronization |

The legacy report's `rmsd_mat_cov_status` explicitly says
`not_available_without_reference-ensemble binding`. The formal validation
records do contain validation reference candidates, so the optimized FULL
protocol must bind them explicitly rather than silently continue calling the
legacy report complete.

## Five largest bottlenecks

1. Duplicate source loading and formal RDKit/topology adaptation: 73.15 s.
2. V8 two-step forward: 62.40 s.
3. NetworkX isomorphism used by formal atom mapping: 29.63 s, nested in adapter time.
4. Per-graph constraint layer: 19.97 s, including 9.48 s solver time.
5. Diagnostic `torch.linalg.eigvalsh`: 6.17 s despite not affecting predictions.

The first optimization target is therefore immutable loading/topology cache and
prediction/evaluation separation. The second is safe multi-record prediction
batching. Diagnostic frequency can be reduced only if cached outputs and parity
prove that coordinates, decisions, and metrics are unchanged.

## Audit conclusion

The existing 10K output is scientifically valid under the frozen V7 BAC safety
semantics, but it is not the requested new FULL protocol: it has no immutable
prediction cache, paired D1/V5-B/V7 join, RMSD/MAT/COV, cohort report, or paired
bootstrap. Optimization must first preserve the existing raw/safe coordinates
and BAC result exactly, then add the missing evaluation-only products without
rerunning a checkpoint prediction.

Isolation at audit completion:

- `formal_test_records_read=0`
- `formal_test_assets_opened=false`
- `minimal_validity_target_test_used=false`
- `frozen_holdout_records_read=0`
- `parameter_selection_from_formal_test=false`

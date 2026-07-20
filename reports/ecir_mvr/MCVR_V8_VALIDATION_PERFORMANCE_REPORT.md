# MCVR V8 Validation Performance Report

## Result

`MCVR_V8_VALIDATION_OPTIMIZED`

The frozen step-5000 Full Seed43 result passed the official 10K parity gate.
The optimized path separates immutable prediction from evaluation, supports
atomic chunk/resume, reuses frozen source and baseline caches, and runs FAST
monitoring on a frozen representative 1K manifest. No scientific model, loss,
solver, safety, data, or reduction setting was changed.

## End-to-end timing

| Workload | Legacy | Optimized | Effective speedup |
|---|---:|---:|---:|
| 100 records | about 18.9 s | 6.69 s prediction + 0.51 s evaluation | about 2.62x |
| FAST 1K | 154.34 s legacy-profile reference | 177.73 s prediction + 19.16 s evaluation | 0.78x on this harder fixed subset |
| FULL 10K, recurring checkpoint | 2,080 s | 687.05 s prediction + 66.27 s evaluation = 753.32 s | 2.76x |
| FULL 10K, first source materialization | 2,080 s | 223.09 s source + 753.32 s = 976.41 s | 2.13x |
| Evaluator-only rerun, FULL 10K | 2,080 s required legacy reinference | 66.27 s | 31.39x |

The optimized FULL protocol computes more than the legacy deployment report:
RMSD, MAT/COV, all cohorts, full bootstrap inputs, and reusable per-record
metrics are included. The FAST subset is frozen for representativeness rather
than selected to reproduce the first 1,000 legacy records; its measured time
is reported without extrapolating or hiding that difference.

The FAST 1K path took 196.89 s total (3.28 minutes). It omits full bootstrap,
MAT/COV, reference-ensemble set statistics, and baseline inference. It is for
training monitoring only; every final checkpoint uses FULL.

## Stage timing and resources

| Stage | 100 | FAST 1K | FULL 10K |
|---|---:|---:|---:|
| Source/topology materialization | 3.54 s | reused | 223.09 s once |
| V8 prediction plus deployment safety | 6.69 s | 177.73 s | 687.05 s |
| Cache load / record join | included | 14.17 s | 14.55 s |
| BAC, displacement, RMSD and cohort metrics | 0.51 s total eval | 4.98 s | 47.24 s |
| MAT/COV plus bootstrap | omitted | omitted | 4.41 s |
| Evaluation/report total | 0.51 s | 19.16 s | 66.27 s |
| Prediction/evaluation output size | 0.30 MiB | 2.89 MiB | 36.60 MiB |

The one-time source/reference/topology cache is 503.49 MiB. It replaces
repeated Parquet deserialization and RDKit/NetworkX adaptation for every
checkpoint and is deliberately excluded from Git.

Deployment safety remains inside Stage-A prediction so that raw and safe
coordinates cannot be separated accidentally. The unmodified 1K profile
attributed 3.91 s cumulatively to `select_safe_bac_proposal`; the optimized
benchmark deliberately avoids an extra CUDA synchronization solely to time it,
so its safety time is reported as part of prediction rather than as a perturbed
standalone number.

Observed optimized GPU use was workload-dependent, approximately 9–46%, with
about 4.6 GiB device memory in use during baseline inference. Peak baseline
Python working set was approximately 3.3 GiB. Legacy profiling observed
15–18% GPU use, approximately 2.3 GiB Python working set, and about 5.4 GiB
total device memory in use. CPU utilization is material because RDKit,
NetworkX/topology construction, safety evaluation, and record orchestration
remain host-side. Measurements are point observations from this single-GPU
Windows workstation rather than hardware-independent guarantees.

## Frozen baseline caches

The source, D1, V5-B, and V7 caches contain 10,000 records in original order.
Each chunk is atomically written and SHA-256 checked before evaluation. Method,
checkpoint, resolved method config, validation source/target, evaluator, and
safety identities are bound into the cache identity. Cached coordinate tensors
are read back through the same hash-validating iterator used by the evaluator;
accepted/rollback decisions and record order are exact.

| Frozen cache | First prediction | Cached FULL evaluation | Output size | 100-record rerun parity |
|---|---:|---:|---:|---|
| D1 | 404.33 s | 48.09 s | 38.76 MiB | exact discrete; max coordinate abs `4.77e-7 Å` |
| V5-B | 1,151.81 s | 54.61 s | 38.93 MiB | exact discrete; max coordinate abs `4.77e-7 Å` |
| V7 | 909.15 s | 54.23 s | 38.85 MiB | exact discrete; max coordinate abs `4.77e-7 Å` |

The three one-time baseline predictions took 2,465.28 s (41.09 minutes),
excluding the separately shared 223.09-second source cache. Their cached FULL
metric passes took 156.94 s total. The complete 10K V8-versus-D1/V5-B/V7
paired bootstrap comparison completed in about 13 seconds including Python
startup; one shared frozen index stream preserves the prior per-metric Seed43
bootstrap semantics while avoiding redundant index generation.

Subsequent FULL validations reuse all three baseline prediction caches. Only
the 66-second-scale cached metric pass and paired comparison are repeated;
baseline neural inference is not repeated unless an identity or chunk hash
changes.

## Official parity gate

- Record identity and order: exact.
- Accepted mask, rejection reasons, rollback/backtracking, chirality, and
  solver-failure count: bitwise exact.
- Angle and clash deltas: exact at report precision.
- Maximum absolute continuous difference: `4.0363316941238736e-05` (solver
  contribution diagnostic).
- Maximum relative continuous difference: `2.5711113373810574e-04` (ring
  delta).
- Frozen continuous contract: `atol=1e-6`, `rtol=3e-4` for metrics and
  `atol=1.1e-4 Å`, `rtol=1e-3` for CUDA coordinates.
- Weighted BAC difference: `2.736812857034865e-06`.
- Scientific conclusion changed: false.
- Gate: `PARITY_OK`.

CUDA batch-16 prediction and `torch.inference_mode()` were evaluated and
rejected because they crossed strict metric/threshold parity on the audit
subset. The accepted implementation therefore retains batch-one two-step
prediction under `model.eval()` and `torch.no_grad()`. This preserves the
legacy CUDA execution semantics while the large speedup comes from eliminating
duplicate loading, immutable cache reuse, and prediction/evaluation separation.

## Protocol and isolation

- FAST: fixed Seed43 1,000-record manifest, every 10K step except FULL steps.
- FULL: all 10,000 records at 50K, 100K, 150K, and 200K.
- Bootstrap and MAT/COV: FULL only.
- Frozen baseline prediction: reused by every FULL checkpoint.
- Formal-test records read: 0.
- Formal-test assets opened: false.
- Minimal Validity Target test used: false.
- Frozen-holdout records read: 0.
- Parameter selection from formal test: false.

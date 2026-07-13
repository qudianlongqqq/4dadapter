# Global Coupled 4D real-bundle CPU performance audit

## Result

The current formal sampler performs a complete accumulated-prefix rewrite after every successful record. The partial tensor payload and both state JSON snapshots grow with progress, so cumulative persistence is `O(N^2)` in serialized bytes and `fsync` work when record sizes are bounded and saving occurs every record.

The shared provenance builder adds a separate scaling hazard: each partial save hashes and embeds the full manifest, recomputes the inference-cohort hash for the completed prefix, and `_ordered_manifest_rows()` calls `manifest_order.index()` once per completed ID. For source-manifest size `M` and completed count `k`, that ordered-row check is `O(M*k)` per save; over `N` saves it is `O(M*N^2)`, or worst-case `O(N^3)` when `M=N`. This is in addition to, not a replacement for, the proven `O(N^2)` write amplification.

No production sampler behavior or model mathematics was changed. The diagnostic profiler computes the selected real records once and replays those in-memory results through bounded persistence simulations.

## Static persistence confirmation

The formal sampler's successful-record sequence is:

1. Build and atomically write a pre-record `sampling_state.json` containing all completed sample IDs.
2. Run one record through ten refinement steps.
3. Build `completed_manifest` from the complete prefix.
4. Call the shared `build_manifest_aware_sample_payload()` with the complete accumulated `records` list.
5. Atomically overwrite `partial_samples.pt` with the full prefix.
6. Build and atomically write a post-record state containing the new complete ID prefix.
7. After all records, build and write the complete final evaluator payload once more.

There is no save-frequency control in the formal sampler. Consequently, record 1 is serialized `N` times, record 2 is serialized `N-1` times, and so on. The idealized equal-size tensor write amplification is `(N+1)/2` final-file equivalents.

## Measurement scope

- Bundle: `artifacts/global4d_profile_bundle`
- Bundle validation: `VALID`; all exported SHA-256 values passed before this audit.
- Runtime: Python 3.11, PyTorch `2.11.0+cpu`, four PyTorch CPU threads.
- Cohort: two real molecules, 30 real cache records; 4 records have 31 atoms/3 rotatable bonds and 26 have 47 atoms/11 rotatable bonds.
- Model: real `step1000.ckpt`, 10 refinement steps, update scale 0.2.
- Model execution: exactly once per matrix run; 30 records and 300 refinement steps. Each persistence row replays the same real CPU tensors and does not rerun inference.
- Timing: `time.perf_counter()` wall time. Atomic writes include temporary-file creation, flush, `fsync`, and rename.

This Windows CPU/filesystem result is not an estimate of RTX 5090 throughput. The formal-protocol replay uses the bundle's reduced 30-row manifest, so its provenance-build and payload-size measurements understate the cost of embedding and scanning the full formal manifest.

## 30-record CPU measurement

Pure rollout compute was 0.907018 s for 30 records (33.075 records/s). Complete no-persistence record processing, including CPU transfers, diagnostic record construction and loop overhead, was 0.919369 s.

| Protocol | Pure compute s | Payload/provenance build s | Tensor serialization s | State JSON s | Protocol overhead s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Partial disabled | 0.907018 | 0 | 0 | 0 | 0 | 0.919369 |
| Full rewrite every 1 | 0.907018 | 0 | 0.153048 | 0.208758 | 0.363053 | 1.282422 |
| Full rewrite every 10 | 0.907018 | 0 | 0.016640 | 0.017825 | 0.034598 | 0.953967 |
| Full rewrite every 30 | 0.907018 | 0 | 0.006660 | 0.005938 | 0.012632 | 0.932001 |
| Current formal full rewrite every 1 | 0.907018 | 0.018281 | 0.159051 | 0.212166 | 0.390831 | 1.310199 |
| Chunk/shard every 10 | 0.907018 | 0 | 0.013789 | 0.014835 | 0.028734 | 0.948103 |

For the exact current-formal replay, persistence was 29.83% of the combined total. Within persistence overhead, state JSON was 54.28%, tensor serialization 40.70%, provenance/payload construction 4.68%, and other loop/filesystem bookkeeping 0.34%. It serialized 44,550,669 cumulative bytes for a 3,268,629-byte final partial file, a 13.63x observed byte amplification on this nonuniform 4+26-record cohort.

Relative to the current-formal 0.390831 s persistence overhead, full rewrite every 10 reduced persistence overhead by 91.15%, full rewrite every 30 by 96.77%, and chunk/shard every 10 by 92.65%. These are local 30-record ratios, not full-run runtime promises.

## Compute-stage measurement

The stage rows are inclusive timing regions and must not be summed.

| Stage | CPU wall s | Share of no-save record processing |
| --- | ---: | ---: |
| Rollout total | 0.907018 | 98.66% |
| EGNN backbone | 0.550296 | 59.86% |
| Rank check and SVD | 0.071937 | 7.82% |
| Jacobian assembly | 0.050745 | 5.52% |
| Local frame | 0.041489 | 4.51% |
| Cartesian projection/diagnostics | 0.036777 | 4.00% |
| Joint head | 0.034881 | 3.79% |
| Topology preparation | 0.010161 | 1.11% |
| Gram matrix | 0.008724 | 0.95% |

The 31-atom records averaged about 0.026 s rollout time in this final run; the 47-atom records were slower and account for 26/30 records. The earlier clean repetitions placed total pure rollout between roughly 0.91 and 1.07 seconds, so small absolute differences should not be overinterpreted.

## Solver fallback and repeated SVD

- Preferred full-rank backend: Cholesky after the rank check.
- Observed backend: `svd_fallback` for all 300 graph-step solves.
- 31-atom molecule: Jacobian `93 x 12`, effective rank 11 for all sampled steps.
- 47-atom molecule: Jacobian `141 x 44`, effective rank 41 for all sampled steps.
- Cholesky, dense solve and `lstsq` calls on these records: zero.

`solver_fallback_rate=1.0` therefore does not mean Cholesky crashed 300 times. The rank-aware code first computes one SVD, detects an undamped rank-deficient Jacobian, and directly uses that same decomposition for the exact minimum-norm orthogonal projector. The stable, repeatable deficiencies of one and three columns across real records are consistent with structural linear dependence in the global coupled joint basis; this sample provides no evidence of an implementation failure.

Correctness is preserved by the SVD minimum-norm path. The effect is performance: one full SVD is required per graph per step, and the current code has already built `J^T J` and `J^T u` before learning that the rank-deficient path will not use them. There is no second SVD in the optimized path. The legacy path contains the old redundant `svdvals` check, but formal refinement calls the optimized path by default. SVD factors cannot simply be cached across refinement steps because coordinates and the Jacobian change at every step.

## CPU/GPU synchronization audit

Normal non-profile CUDA rollout has no explicit per-component `torch.cuda.synchronize()`, but it contains host scalar materializations that synchronize the CUDA stream:

| Site | Frequency | Finding |
| --- | --- | --- |
| `int(atom_batch.max())` in `forward()` | once per refinement step | Recomputes graph count and synchronizes; graph count is static for a record. |
| Effective-rank `.item()` after SVD | once per graph per step | Required by the current Python branch between rank-deficient SVD and full-rank solve. |
| Condition-number `float(cuda_tensor)` | once per graph per step | Diagnostic scalar materialization. |
| Coordinate finite `bool(cuda_tensor)` | once per step | Preserves fail-fast semantics but synchronizes. |
| Coordinate bound `bool(cuda_tensor)` | once per finite step | A second fail-fast synchronization. |
| Fallback-rate `float(cuda_tensor)` | once per step | Diagnostic scalar can potentially be deferred without changing coordinates. |
| Topology key `.cpu()` plus graph/start scalar reads | once per record preparation | Repeats CPU tuple construction even on prepared-cache hits. |
| Final coordinate `.cpu()` | once per record | Necessary output transfer and durable-payload boundary. |
| Sampler `_sync()` around transfers | twice per record | Used for timing; it is not required by the numerical dependency itself. |

For the current single-graph records, the main refinement loop therefore has about six host scalar synchronization sites per step before considering library-internal CUDA solver behavior. Profile mode deliberately adds many explicit synchronizations around component regions; those measurements are diagnostic and should not be treated as normal-rollout behavior.

## Repeated tensor construction and Python loops

Static coordinate-independent topology, fragment membership, masks, ancestor incidence and flattened Jacobian indices are cached. The following objects are still constructed each refinement step:

- scalar time tensor and, for an unbatched record, an atom-batch zero tensor;
- `v_internal`, `v_projection`, downstream sums, fragment pools and joint feature concatenations;
- current local frame, dense Jacobian storage and internal velocity;
- all-ones weights, weighted Jacobian/basis/vector, Gram matrix and right-hand side;
- SVD outputs and projection diagnostic tensor products;
- Python lists/dictionaries for graph details, timings and backend counts.

The Python nesting is record loop -> refinement-step loop -> graph loop -> backbone-layer loop. The sampler also rebuilds growing ID lists, provenance lists and topology-cache key tuples. Coordinate-dependent frames, Jacobians and SVDs are real work and cannot be reused unchanged; graph count, batch vector, some dtype/device conversions, diagnostic materialization and prefix metadata are the safer optimization targets.

## Linux RTX 5090 bounded command

Run this from the repository root after placing the verified bundle at the same relative path. It executes at most 20 records (2 warmup + 18 measured) and does not launch training or formal sampling:

```bash
python scripts/profile_global4d_sampling.py --checkpoint artifacts/global4d_profile_bundle/checkpoint/step1000.ckpt --config artifacts/global4d_profile_bundle/config/config.resolved.yaml --cache_dir artifacts/global4d_profile_bundle/cache --manifest artifacts/global4d_profile_bundle/manifest/profile_manifest.json --split test --refinement_steps 10 --max_molecules 2 --max_records 20 --warmup_records 2 --profile_records 18 --device cuda --cuda_sync_timing --disable_partial_save --skip_batch_benchmark --output_dir reports/profile_linux_rtx5090_max20
```


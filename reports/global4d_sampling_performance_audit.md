# Global Coupled 4D sampling performance audit

## Executive result

The first bottleneck is the resume-save protocol, not a hidden CUDA-to-CPU solver fallback.

After every completed record, the current sampler constructs a partial payload containing **all records completed so far**, writes the complete payload to a temporary file, flushes and `fsync`s it, then renames it over `partial_samples.pt`. It also atomically rewrites a growing `sampling_state.json` before and after each record. At 23,882 records this is an `O(N²)` cumulative-write algorithm.

For `N` equal-size records, saving every record writes approximately:

```text
1 + 2 + ... + N = N(N+1)/2 records
```

At `N=23,882`, that is **11,941.5 final-file equivalents**. A final partial file of 100 MiB would therefore imply roughly 1.17 TiB of cumulative partial serialization, before JSON state writes and the final `samples.pt`.

No formal training, large sampling, server profiling, or model-semantic change was performed by this audit.

## Evidence boundaries

Three evidence classes are kept separate:

1. Source-code facts: call counts, synchronization sites and asymptotic behavior.
2. Local measured I/O benchmark: real `torch.save`, temporary file, flush, `fsync` and rename on 500 synthetic records.
3. Server facts supplied with the request: CUDA enabled, 23,882 records, about 11 hours elapsed and about 0.233 records/s near the end.

The audit machine has PyTorch `2.13.0+cpu`, no CUDA, no `torch_cluster`, and none of the target checkpoint/cache artifacts. Consequently, this report does **not** invent a server GPU compute/SVD percentage or pure-compute records/s. The added bounded profiler produces those fields on the target server without launching a large job.

## Current call chain

The current sampler first scans the cache and validates every selected manifest row into an in-memory `by_id` mapping. “Per-record data read” is therefore mostly paid during initialization rather than immediately before each rollout.

For each manifest record:

```text
by_id lookup
  → one MoleculeData.to(device)
  → one coordinate-independent topology preparation/cache lookup
  → 10 serial refinement steps
      → EGNN backbone
      → fragment pooling and joint head
      → local frame
      → Global 4D Jacobian J
      → Jq internal velocity
      → JᵀJ and Jᵀu
      → full SVD of weighted J, full_matrices=False
      → rank-aware minimum-norm solve or Cholesky path
      → Cartesian orthogonal projection
      → residual + internal velocity
      → clipping, finite/bounds check and coordinate update
  → one device-to-CPU coordinate transfer
  → Python record/dictionary construction
  → complete accumulated partial payload construction
  → complete partial_samples.pt atomic rewrite
  → sampling_state.json atomic rewrite
```

After all records, the sampler performs one additional full write to the final `samples.pt` and deletes the partial file.

For the reported 23,882-record job, assuming one graph per record and a joint-bearing molecule:

| Operation | Calls |
| --- | ---: |
| CPU→GPU record transfer | 23,882 |
| GPU→CPU output transfer | 23,882 |
| Refinement steps | 238,820 |
| EGNN backbone forwards | 238,820 |
| Joint-head forwards | 238,820 |
| Jacobian assemblies | 238,820 |
| Full SVD calls | approximately 238,820 |
| Coordinate updates | 238,820 |
| Complete partial rewrites | 23,882 |
| Normal state JSON atomic writes | 47,764 |
| Final sample write | 1 |

Records with no valid joint skip the solver. Batched records would multiply graph-level solver calls inside each step unless a later batched solver is implemented.

## Partial/resume audit

All seven high-priority suspicions are confirmed.

| Question | Result |
| --- | --- |
| Does each record save all accumulated samples? | Yes. The full `records` list is passed into the partial payload after every record. |
| Does the file grow with completed count? | Yes, approximately linearly with record count. |
| Are previous records serialized repeatedly? | Yes. Record 1 is serialized N times, record 2 N-1 times, and so on. |
| Does atomic save write a complete temporary file? | Yes, followed by flush, `fsync`, and rename. |
| Is cumulative I/O approximately `O(N²)`? | Yes. |
| Is state JSON also expensive? | Yes. It is written twice per successful record and contains a growing ordered-ID list. |
| Does save time increase with progress? | Yes in the local measurement and necessarily in serialized bytes. |

### Local 500-record measurement

Each synthetic record contained 40 atoms, `x_init`, `x_refined`, atomic numbers and representative metadata. Times include real serialization and crash-safe filesystem operations.

| Policy | Save frequency | Total I/O-only s | Serialized MiB |
| --- | ---: | ---: | ---: |
| Full accumulated rewrite | 1 | 10.1988 | 257.20 |
| Full accumulated rewrite | 10 | 0.9553 | 26.18 |
| Full accumulated rewrite | 50 | 0.2245 | 5.65 |
| Full accumulated rewrite | 100 | 0.1105 | 3.08 |
| Append-only chunks | 1 | 3.9782 | 1.72 |
| Append-only chunks | 10 | 0.4032 | 1.06 |
| Append-only chunks | 50 | 0.1005 | 1.03 |
| Append-only chunks | 100 | 0.0604 | 1.03 |
| Single final write only | 500 | 0.0247 | 1.03 |

Within the current-policy I/O-only benchmark:

- partial `torch.save`: 6.9707 s, 68.35%;
- two growing state JSON writes per event: 3.2082 s, 31.46%;
- other loop/filesystem overhead: about 0.20%.

The cumulative partial bytes were 250.49 times the final partial size, matching the theoretical `(500+1)/2 = 250.5` multiplier. File size versus record index correlation was `0.9999998`; save-event time versus record index correlation was `0.5351`. Wall-time correlation is noisier than the exact byte curve because every event includes two latency-dominated `fsync` calls.

| Segment | Mean save event s |
| --- | ---: |
| First 10% | 0.01143 |
| Middle 50% | 0.01972 |
| Last 10% | 0.02936 |

The last 10% cost 2.57 times the first 10% per save event. Absolute times depend on disk, filesystem, record size, antivirus, page cache and `fsync` behavior; the complexity and byte amplification do not.

## Where the 11 hours went

The available evidence supports the following conclusion:

- A substantial and progress-dependent part is accumulated partial/state serialization.
- There are also 238,820 real EGNN/Jacobian/SVD/projection steps, so compute is not free.
- The existing `--profile` implementation measures `molecule_time` **before** the partial save. Its component report cannot explain end-to-end throughput or assign the 11 hours between compute and save.

Exact target-server percentages for compute, SVD, Python, synchronization and disk are therefore currently unknown. Reporting precise values from the supplied facts alone would be false precision. The new profiler measures them with partial saving both enabled and disabled:

```bash
python scripts/profile_global4d_sampling.py \
  --checkpoint logs_global_coupled_4d/global4d_local025_seed42_5000step/checkpoints/step1000.ckpt \
  --config logs_global_coupled_4d/global4d_local025_seed42_5000step/config.resolved.yaml \
  --cache_dir data/flexbond_inference_formal_small \
  --manifest eval_manifest_formal_small.json \
  --split test --max_molecules 1 --max_records 20 \
  --warmup_records 2 --profile_records 18 \
  --refinement_steps 10 --device cuda \
  --disable_partial_save --cuda_sync_timing \
  --output_dir reports/global4d_profile_compute_only
```

Then run the same bounded prefix with current-style saving:

```bash
python scripts/profile_global4d_sampling.py \
  --checkpoint logs_global_coupled_4d/global4d_local025_seed42_5000step/checkpoints/step1000.ckpt \
  --config logs_global_coupled_4d/global4d_local025_seed42_5000step/config.resolved.yaml \
  --cache_dir data/flexbond_inference_formal_small \
  --manifest eval_manifest_formal_small.json \
  --split test --max_molecules 1 --max_records 20 \
  --warmup_records 2 --profile_records 18 \
  --refinement_steps 10 --device cuda \
  --save_every_records 1 --cuda_sync_timing \
  --output_dir reports/global4d_profile_end_to_end
```

The first run gives bounded pure-compute records/s. The difference between the runs gives resume-write overhead on the server filesystem. Profiling synchronization overhead is separately reported.

## CPU/GPU synchronization audit

| Site | Frequency | Synchronizes? | Impact | Low-risk direction |
| --- | --- | --- | --- | --- |
| Topology cache key copies `edge_index` and rotatable indices to CPU | once per uncached record topology | Yes | Small device transfer and Python tuple construction | Keep molecule-level cache; derive stable cache identity from CPU metadata before transfer |
| `int(atom_batch.max())` in every forward | once per refinement step | Yes | 238,820 scalar GPU synchronizations | Cache/pass graph count with prepared topology |
| SVD effective-rank `.item()` | once per graph per step | Yes | Required host branch between rank-deficient and full-rank solve | Measure; batching or device-side strategy requires regression |
| Cholesky/solve `info` and finite checks | only full-rank/failure paths | Yes | Branch synchronization | Current rank-deficient SVD path usually avoids later checks |
| Coordinate finite/bounds `bool(cuda_tensor)` | two checks per step | Yes | Up to 477,640 synchronization points | Consider deferred status aggregation only with failure-semantics tests |
| `float(output["solver_fallback_rate"])` | once per step | Yes | Diagnostic scalar sync | Accumulate tensor and materialize once per record |
| Output `.cpu()` | once per record | Yes/transfer | Necessary for durable CPU payload | Retain once-per-record behavior |
| `num_rotatable_bonds.item()` | once per record | Yes if still on GPU | Redundant metadata sync | Use manifest/CPU metadata |
| Explicit `torch.cuda.synchronize` in normal sampling | transfer boundaries only | Yes | Two per record | Needed for current timing, not computation; make timing optional after audit |
| Per-component synchronizes in profile mode | many per step | Yes | Intentionally distorts profile runs | Disabled in normal rollout; profiler reports this overhead |

There is no per-step `.cpu()`, `.numpy()`, `tolist()` or RDKit operation in the main numerical rollout. `torch.save` receives CPU tensors.

## Why CUDA can coexist with 98% CPU utilization

CUDA use does not imply the host process sleeps. The present workload combines:

- batch size 1;
- one Python iteration per record and per refinement step;
- many small GPU kernels rather than a few large kernels;
- small, variable-shape SVDs;
- graph-dependent Python lists and dictionaries;
- device scalar reads that synchronize the stream;
- complete partial serialization and JSON generation;
- `fsync` and rename after each record;
- irregular molecule sizes that prevent straightforward kernel batching;
- one record's disk work blocking launch of the next record.

The GPU can therefore spend substantial time waiting for Python, scalar synchronization or disk even though every EGNN and SVD operation itself is on CUDA.

## Batching audit

Current behavior is unequivocally batch size 1 at the sampler level:

- one manifest record/conformer is moved to the device;
- its ten refinement steps are serial;
- the EGNN backbone is invoked separately for every record and step;
- records from the same molecule are not grouped;
- the prepared-topology cache avoids reconstructing identical structures but does not batch coordinates;
- PyG batching is supported by the model's graph loop, but Global 4D projection remains one solver call per graph.

The new profiler attempts read-only same-molecule batch sizes 1, 2, 4 and 8 and reports records/s. This is Level 2 work: it needs coordinate equivalence, stable output ordering, per-graph failure handling and batched/shape-bucketed solver regression before use.

## SVD and linear algebra audit

Current optimized projection does the following per graph per step:

1. Build `JᵀJ` and `Jᵀu`.
2. Build weighted `J` and weighted `u`.
3. Run one full `torch.linalg.svd(weighted_J, full_matrices=False)`.
4. Determine effective rank.
5. For undamped rank-deficient systems, directly form the exact minimum-norm projection from that SVD.
6. For full-rank systems, prefer Cholesky and retain solve/lstsq/SVD fallbacks.

The earlier redundant `svdvals + full SVD` has already been removed. There is no second SVD on the normal rank-deficient path. However, `JᵀJ` is currently built even when the rank check subsequently chooses the SVD path; when `solver_fallback_rate=1.0`, that Gram construction does not contribute to the returned coefficients. Removing it conditionally is a plausible Level 1/2 optimization, but it should be driven by measured Gram time and must preserve diagnostics.

CPU versus GPU solver crossover, QR, `lstsq`, and batched SVD are analysis candidates only. No ridge, approximate projection or alternate mathematical definition was introduced. Any replacement must report maximum coordinate difference, RMSD difference, orthogonality error, reconstruction error, failures and speed on the same records.

## Dynamic allocation and Python-loop audit

Already cached per molecule/device:

- rooted topology;
- fragment membership;
- downstream/ancestor masks;
- joint-to-atom incidence;
- Jacobian flat scatter index.

Still created each step because coordinates or network features change:

- local axis and bending frame;
- fragment feature pool tensor;
- joint feature concatenation;
- Jacobian values;
- `v_internal` and `v_projection` zero tensors;
- Gram/weighted basis/SVD outputs;
- Python lists for `q`, axes, graph details and timing dictionaries.

Potential reusable objects include graph count, fixed slice metadata, dtype/device-converted fragment counts, empty tensors and some output containers. Jacobian values, local frames and SVD factors are coordinate-dependent and cannot simply be cached across steps.

## Record distribution and long tail

“100 molecules” does not mean 100 rollout calls. Manifest selection groups by molecule, chooses molecule IDs, then retains **all manifest rows belonging to those molecules**. In the reported run:

```text
100 selected source molecules
→ 23,882 generated conformer records
→ 10 refinement steps each
→ 238,820 serial rollouts
```

Reference conformer count affects evaluator pairwise RMSD work, but does not multiply sampling calls.

The added distribution tool reports per-molecule generated records, reference conformers, atom/rotatable/J-column distributions, slowest molecules/records and correlations with profile time:

```bash
python scripts/analyze_global4d_record_distribution.py \
  --manifest eval_manifest_formal_small.json \
  --cache_dir data/flexbond_inference_formal_small \
  --reference_cache data/flexbond_cache_formal_small \
  --profile_csv reports/global4d_profile_end_to_end/global4d_sampling_profile.csv \
  --split test --output_dir reports/global4d_record_distribution
```

Because the server data is unavailable locally, the requested min/median/mean/max and slowest-20/100 tables are generated by the tool rather than fabricated in this committed report.

## Screen10 and confirm30

Both remain exposed to record explosion:

- screen10 chooses 2 low + 3 medium + 5 high-flexibility molecules, then retains every record for those 10 molecules;
- confirm30 chooses 5 + 10 + 15 molecules, then retains every record for those 30 molecules;
- neither stage currently enforces a total-record cap.

Yes, screen10 should have both a deterministic independent-molecule cap and a deterministic total-record/per-molecule conformer cap. The chosen sample IDs and hashes must be frozen in the manifest so Cartesian and Global 4D see identical ordered records. Final testing can retain the complete record set.

## Optimization levels

### Level 0: protocol/cost controls

| Proposal | Speed source | Difficulty | Numerical risk | Paper protocol | Priority |
| --- | --- | --- | --- | --- | --- |
| Dual molecule + total-record cap for screen/confirm | Fewer serial rollouts; fewer save bytes | Low/medium | None for selected records | Changes selection cohort; freeze and document | High |
| Full records only for final test | Avoid repeated full expansion across 8+2 candidates | Low | None | Must be predeclared | High |

No fixed speedup is promised: compute scales roughly with retained records, while current save bytes scale with the square of retained records.

### Level 1: low-risk, exact output

| Proposal | Measured/expected source | Difficulty | Numerical risk | Priority |
| --- | --- | --- | --- | --- |
| Append-only chunks of 25–50 records, final merge once | Converts resume bytes from `O(N²)` to `O(N)`; chunk-50 local I/O-only 10.1988→0.1005 s | Medium | None if order/hash validated | 1 |
| Compact atomic index once per committed chunk | Removes growing ID-list JSON; state was 31.46% of local current-policy I/O-only time | Low | None | 2 |
| Cache graph count and avoid redundant diagnostic scalar materialization | Reduces per-step host synchronization | Low/medium | None if checks retained | 3 |
| Skip/defer Gram construction on proven SVD-only path | Avoids unused `JᵀJ` when rank deficient | Medium | Low; diagnostics regression required | 4 |

A 50-record chunk bounds crash loss at 49 records. A smaller 10–25 record chunk trades more `fsync`s for less potential lost work.

### Level 2: measured numerical regression required

- same-topology PyG batches;
- atom-count/J-column shape buckets;
- batched Jacobian assembly;
- batched SVD where shapes match;
- asynchronous prefetch/pinned transfer;
- CPU/GPU solver crossover for very small matrices.

These can reduce Python and kernel-launch overhead but complicate failure isolation and exact ordering. The new batch benchmark gathers evidence; no production batching was implemented.

### Level 3: algorithmic changes, analysis only

- fewer refinement steps;
- ridge/approximate solver;
- low-rank approximation;
- altered projection definition.

These change the experimental method or numerical result and were not implemented or recommended as the first response.

## Answers to the twelve priority questions

1. **Where did 11 hours go?** A mixture of 238,820 real refinement steps and an `O(N²)` resume protocol. Source and I/O measurements identify saving as the first critical bottleneck; exact server phase shares require the bounded profile.
2. **Compute/SVD/Python/sync/disk percentages?** Exact target-server values are not recoverable from the existing profile. Local I/O-only split is 68.35% partial `torch.save`, 31.46% state JSON, 0.19% other. Server GPU shares are intentionally left unreported until measured.
3. **Is partial save `O(N²)`?** Yes.
4. **Why is the second half slower?** Each save rewrites a larger prefix; local last-10% save events were 2.57× first-10%.
5. **Why is CPU near 100% under CUDA?** Python batch-1 scheduling, scalar synchronizations, serialization, JSON and `fsync` keep the host busy while the GPU often waits.
6. **What limits GPU utilization?** Batch 1, small irregular kernels/SVDs, serial ten-step rollouts, host branches and disk barriers.
7. **Pure compute theoretical speed?** Not measurable on this CPU-only host without the target artifacts. The new `--disable_partial_save` run produces the answer on 18 bounded records.
8. **Resume overhead?** Locally, current-policy I/O was 10.1988 s for 500 records versus 0.0247 s for one final write. This is an I/O-only comparison, not an end-to-end server percentage.
9. **Safest three optimizations?** Chunked append, compact chunk-level state, and deterministic dual caps for selection stages.
10. **Can screen10/confirm30 still expand too much?** Yes; both retain all conformers of selected molecules.
11. **Should screen10 use molecule and record limits?** Yes, with a frozen ordered manifest shared by both methods.
12. **How long will the full 100-molecule final test take?** The current observed run is about 11 hours near completion. Holding the late 0.233 records/s rate constant would be a pessimistic 28.47 hours from scratch, but that is not a valid average. An optimized forecast awaits pure-compute server profiling.

## Safe server observation commands

These are read-only examples; none was started by this audit:

```bash
nvidia-smi dmon -s pucvmet -d 2
nvidia-smi pmon -s um -d 2
pidstat -p <PID> -rud -h 2
iostat -xz 2
sudo iotop -oPa -p <PID>
nsys profile --sample=none --trace=cuda,nvtx,osrt -d 60 -o /tmp/global4d_60s -p <PID>
```

Attaching `nsys` to an existing process depends on installed permissions/version; verify locally before use. The lower-risk choice is the bounded standalone profiler after the current server job finishes.

To generate an optional PyTorch trace on a bounded run, add `--torch_profiler`. Trace directories are gitignored and must not be committed.

## Recommendation on changing the formal sampler

Do not alter the running job and do not silently change the formal protocol in this audit commit. The evidence is strong enough to prioritize a follow-up implementation of append-only chunks, but that change should have its own provenance version, crash-recovery test, ordered-ID/hash validation, final-payload equivalence test and server-side before/after benchmark.

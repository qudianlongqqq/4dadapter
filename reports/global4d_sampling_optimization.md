# Global Coupled 4D sampling persistence optimization

## Outcome

The production sampler now defaults to append-only, crash-safe chunks of 50
records. It no longer rebuilds and overwrites the complete accumulated prefix
after every record. The final `samples.pt` schema is unchanged and is still
built by `build_manifest_aware_sample_payload()`, so the shared evaluator sees
the same manifest, split, ordered IDs, molecule IDs, `x_init_hash` values,
inference-cache cohort, checkpoint identity and sample count.

No formal-large data build, training, screen10, confirm30, final test or server
job was started. All local model runs were bounded to the verified 30-record,
two-molecule profile bundle.

## Persistence protocol

The default layout is:

```text
<output_parent>/
├── samples.pt
├── sampling_state.json
└── partial_chunks/
    ├── chunk_000000.pt
    ├── chunk_000001.pt
    └── ...
```

Each chunk contains one exact contiguous interval of the selected manifest,
the ordered-ID hash, a content hash over every record, the complete run
identity, and the previous chunk's file hash. Resume scans verify continuous
chunk numbering, bounds, IDs, record content, checkpoint/config/manifest
hashes, the hash chain, duplicates and omissions. Existing valid chunks are
never overwritten. Chunks and final payloads use temporary files followed by
atomic rename; chunk paths are generated from validated integer indices and
symlink roots/files are rejected.

`sampling_state.json` is approximately constant size. It records counters,
hashes, the last durable chunk, ETA and fixed-size aggregate I/O metrics, but
not the growing completed-ID list, records, backend history or per-record
timings. It is written only at initialization, after a chunk is durable, on
failure, and on completion. With the default size, interruption can lose at
most 49 unsealed in-memory records.

Legacy resume remains explicit through `--partial_format legacy`. The
`convert_legacy_partial_to_chunks.py` tool validates the old payload's shared
manifest provenance plus checkpoint/config/manifest identities, retains the
legacy source and converts idempotently without mixing formats.

## Complexity

- Legacy save-every-1: cumulative record serialization and growing state are
  `O(N²)`. The removed `manifest_order.index()` loop made repeated provenance
  building `O(M*N²)`, worst-case `O(N³)` when `M=N`.
- Chunked: each record is serialized once into a bounded chunk, so cumulative
  persistence is `O(N)` for bounded record size. Resume/final scan is `O(N)`.
- Provenance ordering: one `order_map` costs `O(M)` and every ID lookup is
  `O(1)`. Duplicate manifest IDs and payload IDs still fail validation.

At 23,882 records, save-every-1 performs 285,186,903 prefix-record
serializations, or an idealized 11,941.5 final-file equivalents. Chunk size 50
requires at most 478 chunks and serializes each record once. These are
complexity counts only; no absolute RTX 5090 runtime is inferred.

## Real 30-record CPU measurement

Python 3.11, PyTorch `2.11.0+cpu`, four CPU threads. One set of 30 real outputs
was computed once, then replayed through every protocol so model-time noise is
shared. Pure rollout was 1.104432 s; complete record processing without partial
persistence was 1.120194 s.

| Protocol | Payload build s | Tensor save s | State JSON s | Merge/scan s | Combined total s | Bytes written | Tensor amp | Records/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. legacy / 1 | 0.020113 | 0.170180 | 0.217673 | 0 | 1.531762 | 44,602,670 | 13.630x | 19.585 |
| B. legacy / 10 | 0.002172 | 0.019407 | 0.116459 | 0 | 1.259729 | 6,094,406 | 1.856x | 23.815 |
| C. chunked / 10 | 0 | 0.014674 | 0.014959 | 0.018316 | 1.168246 | 3,266,700 | 1.000x | 25.680 |
| D. chunked / 50 | 0 | 0.007997 | 0.004982 | 0.008178 | 1.141387 | 3,262,288 | 1.000x | 26.284 |
| E. disabled | 0 | 0 | 0 | 0 | 1.120194 | 0 | 0 | 26.781 |

Relative to the old formal save-every-1 protocol, default chunk/50 reduced the
measured combined time by 0.390375 s (25.49%), persistence time by 94.85%, and
bytes written by 92.69%. Tensor write amplification fell from 13.63x to 1.00x.
The order lookup microbenchmark over the same 30 IDs and 10,000 repetitions
fell from 0.045594 s for `list.index` to 0.009280 s for the prebuilt map, a
4.91x local speedup.

These Windows CPU/filesystem numbers do not predict Linux RTX 5090 absolute
speed. They establish call behavior, complexity and local I/O ratios only.

## Numerical and evaluator regression

The formal sampler was run separately in legacy/1, chunked/10 and disabled
modes over the same real checkpoint, config, manifest and 30 cache records.

- Sample IDs and order: exact match.
- Maximum coordinate difference: `0.0`.
- RMS coordinate difference: `0.0`.
- Solver backend count: identical, `svd_fallback: 300`.
- Checkpoint inference, config, manifest and inference-cache cohort hashes:
  exact match.
- Shared `_load_method_records()` accepted all three final payloads with no
  missing or failed IDs.
- Shared evaluator summaries: six summary rows and eleven diagnostic rows,
  exact match across all three protocols.
- Partial payloads remain marked partial and are rejected by the evaluator;
  it reads only final `samples.pt`.

The end-to-end automated regression additionally exercises
`sample_global_coupled_4d_flow.py -> eval_global_coupled_4d_flow.py ->
eval_flexbond_optimizer.py`, including different sampler/evaluator path aliases
to prove identity is content-hash based rather than path-string based.

## Lazy Gram and solver fallback

The preferred backend for a full-rank system remains Cholesky. All 300 real
refinement solves used `svd_fallback` because the undamped Jacobians were
structurally rank deficient:

- small molecule: `93 x 12`, effective rank 11;
- large molecule: `141 x 44`, effective rank 41.

This is not 300 failed Cholesky attempts. The rank-aware solver computes one
SVD, detects rank deficiency, and uses that same factorization for the exact
minimum-norm orthogonal projector. No ridge was added, rank tolerance is
unchanged and no approximate solve was introduced. Correctness is unaffected;
the fallback costs performance only.

The rank-deficient branch now returns before constructing `J^T J` and `J^T u`.
The prior audit measured 0.008724 s of Gram construction over these 300 steps;
the post-change measurement is 0 because all 300 were rank deficient. A direct
SVD-oracle regression measured zero maximum and RMS coordinate difference,
identical rank, orthogonality error, reconstruction statistic and backend.

## CUDA synchronization audit

Static inspection reduced normal single-graph host scalar materializations from
about six per step to about two:

- graph count and atom batch are prepared once per record;
- fallback counts remain Python integers/floats;
- condition number is materialized only for profile or collected diagnostics;
- finite and coordinate-bound checks form one device flag and one success-path
  materialization rather than two;
- effective rank still materializes because it controls the exact solver
  branch, and the combined coordinate guard still materializes because it
  protects correctness.

NaN/Inf, coordinate-bound and solver checks remain present. Final coordinate
transfer remains once per record. Profile mode intentionally adds
synchronization for timing. This is a static reduction in explicit sites, not
an RTX 5090 timing claim; CUDA libraries may synchronize internally.

## Formal selection caps

The deterministic validation selectors now limit both molecules and records:

- screen10: 10 stratified molecules, at most 200 records;
- confirm30: 30 stratified molecules, at most 600 records.

Records remain a subsequence of original manifest order, every selected
molecule retains at least one record, and a selection report lists counts per
molecule and all truncations. Cartesian and Global4D commands consume the same
single manifest, hence the same IDs and manifest hash. Test is not used for
screen/confirm selection; the frozen 100-molecule test protocol is unchanged.

## Bounded Linux RTX 5090 commands (not executed)

Legacy control, at most 20 records:

```bash
python scripts/profile_global4d_sampling.py --checkpoint artifacts/global4d_profile_bundle/checkpoint/step1000.ckpt --config artifacts/global4d_profile_bundle/config/config.resolved.yaml --cache_dir artifacts/global4d_profile_bundle/cache --manifest artifacts/global4d_profile_bundle/manifest/profile_manifest.json --split test --refinement_steps 10 --max_molecules 2 --max_records 20 --warmup_records 2 --profile_records 18 --device cuda --cuda_sync_timing --partial_format legacy --save_every_records 1 --skip_batch_benchmark --output_dir reports/profile_linux_rtx5090_legacy_max20
```

Chunked implementation, at most 20 records:

```bash
python scripts/profile_global4d_sampling.py --checkpoint artifacts/global4d_profile_bundle/checkpoint/step1000.ckpt --config artifacts/global4d_profile_bundle/config/config.resolved.yaml --cache_dir artifacts/global4d_profile_bundle/cache --manifest artifacts/global4d_profile_bundle/manifest/profile_manifest.json --split test --refinement_steps 10 --max_molecules 2 --max_records 20 --warmup_records 2 --profile_records 18 --device cuda --cuda_sync_timing --partial_format chunked --save_every_records 10 --skip_batch_benchmark --output_dir reports/profile_linux_rtx5090_chunked_max20
```

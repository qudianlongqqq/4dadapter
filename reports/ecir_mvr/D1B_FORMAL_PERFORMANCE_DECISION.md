# D1-B Formal-Large Performance Decision

## Final Decision

The formal main run remains **micro 64, accumulation 1, effective batch 64, 25,000 optimizer steps, 1,600,000 sample exposures, LR 2e-4, seed 42**.

Micro 256 passed Linux capacity testing, but is not promoted. Its 276.486 records/s is only 0.035% above micro 128 and about 1.03% above micro 64. Memory is demonstrably not the bottleneck. Micro 64 preserves the largest optimizer-update count under the original matched-exposure budget and introduces no new scientific variable.

No formal training was started. No test data was read. No formal target or checkpoint was modified.

## Model and Capacity

The model is a lightweight EGNN, not a Transformer:

- 384,678 total and trainable parameters;
- hidden and edge hidden dimensions 64;
- four main backbone layers and three encoder layers;
- approximately 1.47 MiB FP32 parameters or 5.87 MiB for parameters, gradients, and two Adam moments;
- structurally identical to Stage D 5k D1-B, with all 176 checkpoint keys strict-load compatible.

The small model explains why GPU compute is short relative to the input pipeline. It does not prove scientific under-capacity. That still requires train/validation curves and a controlled model-size ablation.

## Linux Capacity Evidence

| Micro / accum | Records/s | Torch peak | Card peak | Decision |
|---|---:|---:|---:|---|
| 64 / 1 | 273.666 | 197.714 MiB | 23,109 MiB | formal choice |
| 128 / 1 | 276.389 | 357.555 MiB | 23,274 MiB | capacity evidence |
| 256 / 1 | 276.486 | 679.985 MiB | 23,686 MiB | capacity pass only |

The card baseline was approximately 22,341 MiB. Micro 256 was memory-safe and had no OOM, NaN, or Inf. These are user-supplied completed Linux results; this Windows task did not rerun or fabricate them.

## Profiler Closure

`scripts/profile_ecir_mvr_formal_step.py` reuses the real `MCVRMixedDataset`, PyG DataLoader, D1-B model construction, `MCVRLoss`, forward, backward, gradient clipping, AdamW step, and manual LR schedule.

It reports:

- dataset item time from real train items;
- official PyG `Collater` time in a controlled, separate real-microbatch probe;
- actual DataLoader wait, which includes worker scheduling and worker-side collate;
- H2D, model forward, remaining loss work, backward, optimizer, scheduler, and total step time;
- records/s, CPU RSS, GPU utilization, and CUDA memory.

The separate collate measurement is deliberate: worker-side collate duration cannot be isolated from queue wait using the unmodified official DataLoader. The report does not mislabel DataLoader wait as pure collate time.

The profiler rejects test paths and output directories overlapping formal assets, writes no checkpoint, never starts the training entry point, supports shared-GPU blocking/override, and resolves physical versus logical CUDA indices through `CUDA_VISIBLE_DEVICES`.

Every profiler invocation now runs a paired comparison for each loader setting. Baseline is LRU `0` with topology precomputation disabled; optimized is LRU `512` with topology precomputation enabled. Both variants reset seed, dataset, model, and optimizer and use the same sample order, micro batch, workers, prefetch, warmup, and measured-step count. The report computes speedup only from these same-run results; historical preflight throughput is never substituted for baseline.

Cache statistics travel through a separate multiprocessing statistics channel, not through the canonical batch. The report contains per-worker and aggregate cache hits, misses, hit rate, RDKit adapter builds, and topology builds. Prefetched worker items are included because they are real work performed by the official DataLoader.

## Round-One Safe Optimizations

All new optimization switches default off, so the formal behavior remains unchanged.

### Per-Worker Formal Adapter LRU

The optional LRU prevents repeated full formal RDKit mapping when a source reappears in the same persistent worker. Its key is a canonical SHA over:

- cache schema and current `formal_rdkit_adapter.py` file SHA;
- sample ID and source record ID;
- source/coordinate identity SHA;
- ordered SMILES;
- ordered atomic-number sequence;
- topology signature.

Only `_formal_*` runtime adapter fields are cached. The source PT is still loaded and the key is recomputed from its immutable content before reuse. Identity mismatch is a cache miss. The cache is process-local, bounded, disposable, and does not touch formal targets. No disk cache was added in this round.

Each LRU entry carries schema `ecir-mvr-formal-adapter-worker-lru-v1`, feature version `formal-rdkit-static-v1`, and its canonical identity SHA. It stores no hidden dimensions, layer-dependent tensors, weights, embeddings, or network intermediates.

### Precomputed Training Topology

The opt-in dataset path materializes only static tensors:

- `canonical_bond_index`;
- `canonical_angle_index`;
- `canonical_torsion_index`;
- `canonical_ring_bond_index`.

PyG batches these with normal node-index increments. When present, the error encoder and loss reuse them instead of moving CUDA topology to CPU and rebuilding Python lists. Coordinates, bond lengths, angles, torsions, geometry errors, model outputs, and targets remain dynamic and are recomputed every step.

The dataset declares canonical batch schema `ecir-mvr-canonical-batch-v1`. Static topology uses schema `ecir-mvr-static-topology-cache-v1`, feature version `molecular-static-topology-v1`, and a per-sample identity SHA. These fields contain molecular/sample facts only. Any future EGNN, Transformer, or SO(3)-specific representation must be produced by an independent model adapter or feature builder and may add a parallel feature version without rewriting this canonical topology.

### Explicit-Bond Projection

The projection implementation was not changed. Grouping graphs with identical `(num_atoms, num_bonds)` and using batched `solve_ex` is technically possible, but variable shapes require grouping or padding. Padding can change conditioning, failure reporting, and floating-point order. It is not safe to enable until CUDA forward, gradient, and solver-failure equivalence are established with the same damping and objective.

The existing per-graph Python loop and small solve therefore remain a known second-round optimization candidate, not a silent change.

## Numerical Equivalence

A real eight-record medium-train batch was evaluated through both paths:

| Quantity | Maximum absolute difference |
|---|---:|
| Model forward outputs | 0.0 |
| Total loss | 0.0 |
| Every sub-loss | 0.0 |
| Parameter gradients | 1.1641532182693481e-10 |
| Parameters after one AdamW step | 9.313225746154785e-10 |

Forward and loss are bitwise identical. The fixed acceptance threshold for gradients and updated parameters is `1e-8`. The non-bitwise difference is CPU parallel-reduction noise: two independent unoptimized instances produce the same `1.1641532182693481e-10` gradient variation. All values were finite, and fixed-seed repeated optimized forward was bitwise reproducible.

## Performance Boundary

Current bottleneck order is:

1. random source PT I/O and per-item RDKit/topology adaptation;
2. loss/error-encoder CPU topology construction and CUDA synchronization;
3. per-graph Python explicit-bond `solve_ex`;
4. PyG collate, H2D, and shared-storage contention;
5. GPU model compute.

One source PT is still opened for every item. Round one does not change the formal data asset format, so random small-file I/O remains an expected bottleneck. No packed dataset, LMDB, or new shard format was introduced in this task.

Safe candidates are identity-bound runtime caching, static topology indices, loader tuning, and identity-preserving file packing. Architecture size, Transformer replacement, loss/target semantics, projection objective/damping, optimizer, LR, effective batch, update count, and exposure budget are scientific changes and remain prohibited by default.

## Formal 64 Configuration Gate

The requested `reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml` is not present in this Windows workspace. The Linux formal64 preflight report and formal source/target metadata needed to bind real identities are also absent. A placeholder config would violate the identity requirements, so none was fabricated.

`scripts/finalize_ecir_mvr_formal64_config.py` now performs the strict finalization on Linux. It accepts only:

- `D1B_FORMAL_TARGETS_READY`, 150,000 train and 10,000 val targets, test reads zero;
- current source, target, builder, config, and adapter identities;
- a non-capacity `D1B_FORMAL_PREFLIGHT_PASS` for 64 x 1;
- matching base config SHA and formal asset identities;
- no formal checkpoint or training started by preflight.

It atomically writes the recommendation with runtime optimizations explicitly disabled, pins the preflight report SHA, rejects capacity256, and re-runs both formal identity gates before returning `D1B_FORMAL64_CONFIG_READY`.

The historical `reports/ecir_mvr/preflight_effective64/D1B_FORMAL_PREFLIGHT.json` is retained only as evidence. The optimization changes code identity, so it is rejected by the finalizer. On the optimization commit, formal64 preflight must be rerun; its fixed output is `reports/ecir_mvr/formal64_preflight/D1B_FORMAL_PREFLIGHT.json`. The finalizer requires that report's commit SHA to equal the current HEAD and also rejects capacity reports.

Linux formal64 preflight command on the optimization commit:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/preflight_ecir_mvr_formal_large.py \
  --config configs/ecir_mvr_formal_large_d1b_base.yaml \
  --gpu-index 1 \
  --allow-shared-gpu \
  --target-effective-batch 64
```

Linux finalization command:

```bash
python scripts/finalize_ecir_mvr_formal64_config.py \
  --base-config configs/ecir_mvr_formal_large_d1b_base.yaml \
  --preflight-report reports/ecir_mvr/formal64_preflight/D1B_FORMAL_PREFLIGHT.json \
  --output reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml
```

## Unique Linux Profiler Command

This profiles the opt-in round-one candidates without changing the finalized formal config:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/profile_ecir_mvr_formal_step.py \
  --config reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml \
  --gpu-index 1 \
  --allow-shared-gpu \
  --micro-batch 64 \
  --num-workers 0,2,4,8,12 \
  --prefetch-factors 2 \
  --persistent-workers true \
  --pin-memory true \
  --output-dir reports/ecir_mvr/formal_step_profile_round1
```

## Unique Formal Training Command

Run only after the finalizer prints `D1B_FORMAL64_CONFIG_READY` and after manual confirmation:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_ecir_mvr_medium_rescue_v2.py \
  --config reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml \
  --data_audit /home/aidd4090v2/Experiment/qdl/data/4dadapter/ecir_mvr/formal_large/statistics/validation.json \
  --device cuda:0
```

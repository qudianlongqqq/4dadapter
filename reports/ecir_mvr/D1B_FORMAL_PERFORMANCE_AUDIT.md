# D1-B Formal-Large Performance and Capacity Audit

## Decision

`AUDIT_COMPLETE_PROFILE_BEFORE_FORMAL_TRAINING`

The current D1-B is a computationally small GNN: **384,678 parameters**, all trainable. The formal model is exactly the Stage D 5k D1-B model, not a reduced formal variant. The 64 to 128 micro-batch throughput gain of only 0.995% is best explained by CPU item preparation and small-file I/O first, repeated CPU topology work and synchronization second, and per-graph small bond solves third. GPU memory is not the limiting resource.

This is a systems conclusion only. There is no evidence yet that 384,678 parameters are scientifically insufficient. That requires training and validation curves or a controlled capacity ablation.

No formal GPU experiment or training was run. No formal target, checkpoint, or recommended configuration was changed. Test records read: **0**.

## Exact Model Size

The model was instantiated from `configs/ecir_mvr_formal_large_d1b_base.yaml`; the same count was independently obtained from the Stage D resolved config.

| Module | Parameters | Share |
|---|---:|---:|
| `error_encoder` | 140,849 | 36.615% |
| `backbone` | 136,780 | 35.557% |
| `deterministic_embedding` | 2,264 | 0.589% |
| `rigid_base` | 11,777 | 3.062% |
| `rigid_edge` | 15,937 | 4.143% |
| `torsion_base` | 11,777 | 3.062% |
| `torsion_edge` | 15,937 | 4.143% |
| `rigid_gate` | 7,681 | 1.997% |
| `torsion_gate` | 7,681 | 1.997% |
| `global_safety_gate` | 7,681 | 1.997% |
| `uncertainty_head` | 7,681 | 1.997% |
| `error_auxiliary_head` | 8,006 | 2.081% |
| `bond_explicit_head` | 10,627 | 2.763% |
| **Total** | **384,678** | **100%** |

Grouped totals:

| Group | Parameters | Share |
|---|---:|---:|
| Error encoder + LightEGNN backbone | 277,629 | 72.172% |
| Four Cartesian base/edge heads | 55,428 | 14.409% |
| Explicit bond head | 10,627 | 2.763% |
| Context embedding, gates, uncertainty and auxiliary head | 40,994 | 10.657% |

FP32 parameters occupy 1.467 MiB. Parameters + gradients + two FP32 Adam moments are approximately 5.870 MiB, excluding activations and optimizer implementation overhead. The model construction is at `etflow/ecir/mvr_model.py:67`; the two sequential encoders are created at `etflow/ecir/mvr_model.py:119` and `etflow/ecir/mvr_model.py:131`.

## Real-Batch Compute Estimate

A read-only CPU pass used eight existing **medium train** records through the real `MCVRMixedDataset`, official PyG `Batch`, `MCVRModel`, and `MCVRLoss`. It did not access test data. The batch contained 294 atoms and 622 directed edges, or 36.75 atoms and 77.75 directed edges per graph.

| Estimate | Batch 8 measured shape | Batch 64 linear scale | Batch 128 linear scale |
|---|---:|---:|---:|
| Executed Linear forward FLOPs lower bound | 0.296 GFLOPs | 2.367 GFLOPs | 4.733 GFLOPs |
| Approx. Linear training FLOPs (forward + 2x backward) | 0.887 GFLOPs | 7.100 GFLOPs | 14.199 GFLOPs |
| Autograd saved-tensor footprint proxy | 18.16 MiB | 145.25 MiB | 290.50 MiB |

These are estimates, not fabricated GPU profiler results. Linear hooks count the actual executed tensor shapes, but exclude `index_add_`, geometry kernels, nonlinearities, and `solve_ex`. Backward is approximated as twice Linear forward. Saved-tensor bytes are a footprint proxy, not peak allocator memory. The proxy is nevertheless consistent with Linux peaks of 197.714 MiB at micro 64 and 357.555 MiB at micro 128.

## Stage D 5k Compatibility

| Field | Stage D D1-B 5k | Formal base | Result |
|---|---:|---:|---|
| `hidden_dim` | 64 | 64 | identical |
| `edge_hidden_dim` | 64 | 64 | identical |
| `time_embedding_dim` | 32 | 32 | identical |
| `num_layers` | 4 | 4 | identical |
| `encoder_num_layers` | 3 | 3 | identical |
| atom / edge / error / deterministic dims | 10 / 1 / 24 / 10 | same | identical |
| dropout | 0.0 | 0.0 | identical |
| explicit bond head / alpha | true / 1.0 | true / 1.0 | identical |
| atom / graph clipping | 0.12 / 0.06 | same | identical |
| bond residual / damping | 0.05 / 1e-4 | same | identical |
| all 15 loss weights | same | same | identical |
| optimizer | AdamW | AdamW | identical |
| peak LR / weight decay | 2e-4 / 1e-6 | same | identical |
| scheduler / warmup | warmup-cosine / 500 | same | identical |
| gradient clipping | 1.0 | 1.0 | identical |
| micro / accumulation / effective batch | 8 / 1 / 8 | 64 / 1 / 64 base | budget change |
| optimizer steps | 5,000 | 25,000 base | budget change |
| DataLoader | workers 0 | workers 4, persistent, prefetch 2 | systems change |

The formal fields are at `configs/ecir_mvr_formal_large_d1b_base.yaml:31`; Stage D fields are at `logs_ecir_mvr/stage_d/d1_b_explicit_bond_seed42_5k/config.resolved.yaml:40`. AdamW and the shared loss construction come from `scripts/train_ecir_mvr_medium_rescue_v2.py:336`; the same manual schedule is at `scripts/train_ecir_mvr_medium_rescue_v2.py:251`.

The Stage D checkpoint `step005000.ckpt` has SHA-256 `0900b786bb9e7994085f0ff44ae5dbba4a25f490f890c7658c401ad735e20f35`. All 176 keys match: zero missing, zero unexpected, zero shape mismatches, and strict loading passes.

The generated formal recommended YAML is absent from this Windows checkout, so it could not be directly hashed here. Its generator deep-copies the base config and changes only batch/budget/checkpoint schedule, frozen identities, and preflight provenance (`scripts/preflight_ecir_mvr_formal_large.py:736`). Therefore the checked-in code preserves model, loss, optimizer, and target semantics. The Linux file should still be identity-checked before training.

Answers: the structures are identical; formal creation did not shrink the model; the old checkpoint is shape-compatible; this is **same model, more data**.

## One Optimizer Step

| Stage | Real path | Performance observation |
|---:|---|---|
| 1 | `MCVRMixedDataset.__init__`, `etflow/ecir/mvr_dataset.py:207` | source and target parquet are loaded once into pandas |
| 2 | `_dataset`, `scripts/train_ecir_mvr_run_a.py:118`; `__getitem__`, `mvr_dataset.py:237` | deterministic balanced mixture plan |
| 3 | `_load_record_and_coordinates`, `mvr_dataset.py:134` | every item opens one source PT via `torch.load` |
| 4 | `mvr_dataset.py:252` | real-error only (45%) opens one target PT; synthetic/clean do not |
| 5 | `mvr_dataset.py:140` | formal source is adapted on every item |
| 6 | `formal_rdkit_adapter.py:418`, `:375`, `:464` | `MolFromSmiles`, `AddHs`, NetworkX graph isomorphism, hydrogen mapping and topology proof |
| 7 | `mvr_dataset.py:252`, `:278`, `:292` | real/synthetic/clean paths; dynamic corruption and validity evaluation |
| 8 | `mvr_dataset.py:307` | creates a PyG `Data` object |
| 9 | `train_ecir_mvr_medium_rescue_v2.py:532` | official PyG DataLoader workers |
| 10 | PyG DataLoader collator | batches variable-size graph tensors; no repository custom collate |
| 11 | `train_ecir_mvr_medium_rescue_v2.py:715` | `batch.to(device, non_blocking=pin_memory)` moves all PyG tensors |
| 12 | `mvr_loss.py:54`, model call inside loss | loss constructs interpolation then invokes real model forward |
| 13 | `mvr_loss.py:70` through `:190` | geometry modes, auxiliary and explicit-bond losses |
| 14 | `_forward_loss_backward`, training script `:366` / loop `:720` | scaled real loss backward |
| 15 | training script `:741` and `:745` | gradient clip and AdamW step |
| 16 | `_learning_rate_at_step`, training script `:251` | manual warmup-cosine LR is assigned before the step |

Tensor placement is correct at the main boundary: model parameters, PyG batch, coordinates, targets, masks, `edge_index`, `edge_attr`, and scalar losses are placed on CUDA. LightEGNN message aggregation uses native tensor `index_add_` (`etflow/models/components/light_egnn_refiner.py:85`) and does not call a CPU torch-cluster kernel.

The loss does deliberately move topology back to CPU: `angle_triplets(edge_index.cpu())` and `torsion_quads(edge_index.cpu(), rotatable.cpu())` at `etflow/ecir/mvr_loss.py:77`. `internal_mode_velocities` repeats that construction (`etflow/ecir/geometry.py:190`), and angle/torsion builders call `.tolist()` (`geometry.py:29`, `:43`). This occurs for predicted modes, target modes, torsion contribution, and bond consistency. It is a concrete performance defect, not a model-device placement failure.

## CPU and I/O Audit

- `adapt_formal_cache_record` runs for every formal `__getitem__`. There is no cross-item adapter cache.
- Each call can parse SMILES, add hydrogens, enumerate heavy-atom graph isomorphisms, map explicit H, renumber, and hash topology (`formal_rdkit_adapter.py:375`, `:384`, `:419`, `:424`, `:525`). The mapping proven during target construction is therefore recomputed during training.
- Source parquet is an in-memory path/index table after dataset initialization, but each item opens a separate source PT.
- Target parquet is also in memory. A separate target PT is opened only for the 45% real-error component.
- Synthetic corruption is built online. Synthetic records run validity once; clean records can run it twice.
- `ChemicalValidity` has `_environment_cache` and `_prepared_cache` (`chemical_validity.py:232`). Persistent workers retain these caches, but every worker has a private process-local cache. Formal adapter output itself is not retained.
- Formal base settings do activate `persistent_workers=true` and `prefetch_factor=2` when `num_workers>0` (`train_ecir_mvr_medium_rescue_v2.py:209`). This avoids worker restart, not repeated source loading or adapter execution.
- No per-item JSON read, SHA calculation, pandas parquet read, or Python deep copy was found. Formal identity JSON is reread every 50 optimizer steps, which is minor relative to item work.
- No explicit inter-worker lock was found. Linux DataLoader workers avoid a single shared Python GIL because they are processes, but compete for storage/page cache and independently repeat RDKit/NetworkX/cache work.

## GPU and Explicit-Bond Path

The model runs a 3-layer error encoder and a 4-layer LightEGNN backbone sequentially (`mvr_model.py:179`, `:182`), then several small MLP heads. This produces many small kernels rather than large dense GEMMs.

`batched_bond_projection` loops over every graph in Python (`etflow/ecir/bond_explicit.py:156`) and calls a small `torch.linalg.solve_ex` (`:130`). `int(atom_batch.max())` and `int(atom_ids[0])` (`:152`, `:158`) force device-to-host synchronization. Similar `bool(cuda_tensor.any())` branches occur throughout `mvr_loss.py`. These patterns become launch/synchronization bound before they become compute bound.

The local environment has torch 2.11.0+cu128, PyG 2.8.0, pyg-lib 0.7.0, `WITH_PYG_LIB=true`, torch-scatter, and torch-sparse. The warning that torch-cluster is no longer necessary and is ignored does **not** explain this path: MCVR LightEGNN uses `index_add_`, not `radius_graph` or a torch-cluster operator. The Linux environment should be recorded by the profiler, but the warning itself is benign here.

## Preflight Statistics

The preflight calculation is internally correct:

- Records/s is `effective_batch_size / mean optimizer-step time` (`preflight_ecir_mvr_formal_large.py:489`), and each optimizer step includes every accumulation micro-batch.
- The first 20 optimizer steps are excluded with `rows[warmup_steps:]` (`:296`).
- Each candidate runs a two-step smoke, then recreates dataset/model/optimizer for 100 steps (`:533`, `:541`).
- `_seed(42)`, `shuffle=false`, and a recreated dataset make the candidate sequence comparable (`:391`, `:402`).
- DataLoader construction and iterator initialization precede the phase timer, so they are not included in step time.
- Physical/logical CUDA selection is explicit (`:110`, `:868`); card peak uses the physical GPU and includes the shared process baseline.

However, preflight calls `torch.cuda.synchronize` at every stage boundary (`:415`, `:436`, `:441`, `:449`, `:461`) and launches `nvidia-smi` once per optimizer step (`:464`), in addition to 0.2-second background sampling. This is appropriate for component timing and safety, but serializes normal asynchronous execution. The reported 273.666 and 276.389 records/s are therefore **synchronized profiling throughput**, not guaranteed production throughput.

## Why 64 to 128 Gives Only 1%

1. **Per-record CPU preparation and small-file I/O** scale with records and are not accelerated by a larger GPU batch. Four workers must still load PT files and redo formal RDKit/NetworkX adaptation and validity work.
2. **Loss topology rebuilds and explicit-bond projection are Python/synchronization heavy.** Doubling graphs doubles many small solves and Python-loop iterations rather than creating one large efficient GEMM.
3. **The preflight itself serializes stages and polls `nvidia-smi`.** Shared-GPU variation and this instrumentation can easily hide a one-percent hardware batching gain.

The first bottleneck is most likely the combined CPU item pipeline and random small-file I/O. Current evidence cannot cleanly split CPU chemistry from storage or collate; that is exactly what the new read-only profiler measures. Dense GPU compute is last.

## Safe Optimizations

These do not need to change scientific semantics, provided immutable identities and sample order are preserved:

- Persist the proven formal RDKit mapping/topology in the source asset or an identity-keyed sidecar; alternatively add a bounded per-worker LRU cache.
- Precompute angle triplets, torsion quads, and chiral quads, include them in PyG `Data`, and eliminate repeated `.cpu().tolist()` reconstruction.
- Pack small PT files into identity-preserving shards, LMDB, or another indexed container; retain payload SHA and strict validation.
- Batch or vectorize bond projection, and eliminate device-to-host scalar branches.
- Tune worker count, prefetch, persistent workers, and pinning with the added profiler.
- Keep stage synchronizations and per-step `nvidia-smi` in profiling only, not normal training.

Changes that are **scientific experiments**, not silent optimizations: hidden/layer sizes, bond head/projection semantics, loss weights, clipping, optimizer/LR schedule, target semantics, mixture ratios, effective batch, and optimizer update budget.

## Capacity Ablation Envelope

The following are proposed future ablations only. Counts are exact model instantiations with other dimensions unchanged. Linear compute ratios and saved-tensor proxies use the same real batch shape.

| Candidate | Hidden / layers | Parameters | Parameter ratio | Linear compute ratio | Saved-tensor proxy ratio | Param+grad+Adam |
|---|---|---:|---:|---:|---:|---:|
| current | 64 / backbone 4 / encoder 3 | 384,678 | 1.00x | 1.00x | 1.00x | 5.87 MiB |
| medium | 128 / backbone 6 / encoder 4 | 1,760,236 | 4.58x | 4.93x | 2.63x | 26.86 MiB |
| large-192 | 192 / backbone 8 / encoder 6 | 4,908,980 | 12.76x | 14.33x | 5.41x | 74.91 MiB |
| large-256 | 256 / backbone 8 / encoder 6 | 8,609,652 | 22.38x | 25.23x | 7.57x | 131.37 MiB |

Parameter-state memory excludes activations and graph workspaces. A larger model may use the GPU better, but that is not evidence it will improve chemistry. Underfitting requires train loss staying high, train and validation losses remaining high together, all sublosses failing to fit, little train/validation gap, and a larger model improving validation under a controlled budget.

## Required Answers

1. **Parameters:** 384,678 total/trainable; zero frozen.
2. **Same as 5k:** yes, structurally identical with strict checkpoint compatibility.
3. **Top three causes of 1% gain:** item CPU/I/O; loss/projection synchronization and tiny kernels; synchronized preflight/shared-GPU measurement overhead.
4. **First bottleneck:** CPU item preparation plus small-file I/O, then topology/loss CPU work; collate is unmeasured secondary; GPU compute is not first.
5. **Implementation error:** no model/config/checkpoint correctness error. Significant avoidable performance inefficiencies are present.
6. **Safe work:** mapping/topology caching, precomputed topology tensors, identity-preserving file packing, vectorized projection, loader tuning, less production instrumentation.
7. **Scientific changes:** architecture, target/loss, projection semantics, clipping, optimizer/schedule, mixture, effective batch/update budget.
8. **Batch 256/512:** little or uncertain gain is expected before fixing the feed/synchronization path. Passing VRAM capacity does not establish throughput or scientific equivalence.
9. **Run decision:** run the short read-only profile first and optimize the pipeline before the formal run. The current config is scientifically runnable, but increasing batch based on free VRAM is not justified.
10. **Separate conclusions:** model compute is small enough to contribute to low utilization; model expressive capacity remains unknown.

## Read-Only Linux Profiler

The added `scripts/profile_ecir_mvr_formal_step.py` uses the real train dataset, official PyG collate, model, loss, backward, AdamW step, and manual LR schedule. It profiles 5 warmup + 30 measured steps per loader setting, blocks shared GPUs unless explicitly allowed, never loads val/test, never saves a checkpoint, and rejects output paths overlapping formal assets.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/profile_ecir_mvr_formal_step.py --config reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml --gpu-index 1 --allow-shared-gpu --micro-batch 128 --num-workers 0,2,4,8,12 --prefetch-factors 2 --persistent-workers true --pin-memory true --output-dir reports/ecir_mvr/formal_step_profile
```

Outputs: `reports/ecir_mvr/formal_step_profile/profile.json` and `profile_steps.csv`. No Linux profiler result is claimed in this audit.

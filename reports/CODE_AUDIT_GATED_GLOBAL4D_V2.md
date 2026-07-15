# Code audit: Gated Global4D V2

## Decision

Windows CPU implementation review: **no open Blocker or High finding**.

Linux smoke is allowed after installing the declared `etflow-5090` environment.
Formal-large training is not allowed until Linux CUDA smoke, the real capacity
scan, and the 2k pilot are reviewed.

## Blocker

No open finding.

Verified items:

- `gated_additive` and `additive` do not call Cartesian projection and retain the
  complete `v_cart_raw`.
- `P_J target` is used only for the internal auxiliary target in V2.
- sampling remains structurally label-free through `FlexBondInferenceDataset`;
  forbidden reference fields fail schema validation;
- sampling, resume, and chunk identities include fusion mode, beta, gate
  override, joint mode, checkpoint/config/manifest identities, refinement steps,
  and update scale;
- existing outputs and chunks with missing or different semantics are rejected;
- the historical test split is not used by the training or pilot selection code;
- new gated configs explicitly select `gated_additive`.

## High

No open finding.

Verified items:

- additive/gated Cartesian loss is `MSE(v_cart_raw, target)`, not residual loss;
- graph gate pooling and atom broadcast are tensorized;
- backbone, Cartesian, joint, and gate heads receive finite gradients;
- old checkpoints without `fusion_mode` load as strict and have no gate head;
- gated loading with a missing gate fails unless initialization is explicit;
- CLI overrides are normalized before resolved config hashing and passed to the
  DataModule, Trainer, checkpoint hparams, and run state;
- `num_workers=0` omits prefetch and disables persistent workers;
- each benchmark condition runs in a fresh process, so CUDA OOM state cannot be
  reused by the next condition.

## Medium

- Real GPU memory, throughput, utilization, and OOM boundaries remain unmeasured
  because the current PyTorch build is CPU-only.
- `nvidia-smi` must be available on the future Windows/Linux CUDA host to record
  utilization; a failure is reported explicitly rather than replaced by zero.
- Gate regularization (`mean(gate**2)`, weight `1e-4` in new configs) and raw
  rotatable-count feature scaling require pilot validation for saturation.
- Inference-time `--initialize_missing_gate` is deterministic through
  `--missing_gate_seed`, but is intended for diagnostics/pilot use, not a fair
  formal checkpoint.
- The full repository suite cannot currently be collected on Windows because
  `torch_cluster` and `pydantic` are missing.  The complete relevant suite passes.

## Low

- PyTorch emits upstream `torch.jit.script` deprecation warnings.
- Direct unit invocation of Lightning `_shared_step` emits a harmless
  no-Trainer logging warning in one historical test.

## Fixed during review

- Restricted Cartesian projection to `strict_orthogonal` only.
- Added deterministic missing-gate initialization identity.
- Added fresh-subprocess benchmark isolation and separate allocated/reserved
  memory reporting.
- Added `drop_last=True` so a nominal capacity condition never benchmarks a
  smaller final batch.
- Deferred the optional `torch_cluster` import for cache-only CPU validation;
  radius-graph construction still fails clearly when the dependency is absent.
- Made benchmark report creation fail rather than overwrite an existing report.

## Windows status

- Formal cache counts: train 150000, val 10000, test 23882, test molecules 100.
- Read-only train/val validation and PyG collation passed.
- Formal-cache strict/additive/gated forward passed.
- Gated loss/backward and finite gate gradients passed.
- CUDA capacity benchmark: `SKIPPED_CUDA_UNAVAILABLE`.

## Linux risks

- CUDA/PyG binary compatibility for RTX 5090 must be verified in `etflow-5090`.
- Multiprocess DataLoader throughput and host-memory pressure are not measurable
  in the Windows CPU environment.
- Batch 96/128 may be compute- or host-memory-limited even if GPU memory fits.

## Gate to Linux smoke

**ALLOW**, with no formal-large launch.  Run environment checks, relevant unit
tests, 20-step CUDA smoke with checkpoint/resume, label-free sampling/evaluation,
then the isolated capacity scan and 2k pilot.

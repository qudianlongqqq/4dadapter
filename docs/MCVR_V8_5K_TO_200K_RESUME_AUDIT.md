# MCVR V8 Full v1: 5K to 200K Resume Audit

## Decision

`SAFE_TO_RESUME_5K_TO_200K`

The frozen Seed43 checkpoint at global step 5000 can continue to a total target
of 200000 optimizer steps without changing the already executed training path.
The runner uses constant learning rates and has no learning-rate scheduler,
warmup schedule, gradient scaler, or EMA whose semantics depend on the original
5000-step horizon.

## Audited parent

- Run: `diagnostics/ecir_mvr/v8_full_v1/formal_large_5k/full_seed43`
- Checkpoint: `checkpoints/last.ckpt`
- Checkpoint SHA-256: `988115d947325a3d57577a5597f485bd6c3eefbf0024ae5d83b1d8d7f6277716`
- Checkpoint schema: `mcvr-v8-full-v1-checkpoint-v1`
- Global step: 5000
- Seed: 43
- Parent result: frozen and read-only

## State inventory

| State | Audit result |
|---|---|
| Model | Present; strict state loading |
| Optimizer | Present; AdamW, all three parameter groups and moments restored |
| Learning rates | Constant: V8 heads `2e-4`, D1 correction head `5e-5`, D1 backbone `2e-5` |
| Scheduler | None; no scheduler class or state |
| Warmup / decay | None |
| Global step | Restored as 5000; next optimizer step is 5001 |
| Sampler | Deterministic Seed43 stratified schedule, indexed by global step |
| Python RNG | Stored and restored |
| NumPy RNG | Stored and restored |
| Torch CPU RNG | Stored and restored |
| Torch CUDA RNG | Stored and restored |
| Gradient scaler | Not used |
| EMA | Not used |
| Resolved config | Stored; scientific identity checked before loading mutable state |
| Atomic checkpoint | Temporary-file write followed by atomic replace; smoke passed |

## Horizon and exposure continuity

The 5K configuration and the 200K configuration have identical scientific
fields. The resume identity permits differences only in run horizon,
checkpoint/validation scheduling, output metadata, and long-run parent
metadata. Architecture, loss weights, solver settings, data identities,
sampler weights, safety settings, optimizer policy, batch size, and seed are
required to match exactly and fail closed otherwise.

The first 320000 sampled record indices (5000 steps times the effective batch
size) from the 5K schedule are bitwise identical to the prefix of the 200K
schedule. Both prefixes have SHA-256
`f0c4797a803d30856bf56419de3d36e96d8dda6d063082a95ce101cb1aeaeed6`.
Therefore step 5001 consumes the next exposure; no record is repeated or
skipped because of the horizon extension.

## Runtime evidence

- Two-step resume smoke: completed at step 5002, solver failures 0.
- Twenty-step resume smoke: completed at step 5020, solver failures 0,
  finite loss, finite gradients, and valid atomic checkpoint.
- Formal-test records read: 0.
- Formal-test assets opened: false.
- Minimal Validity Target test used: false.
- Frozen-holdout records read: 0.

## Required launch semantics

The formal 200K run must load the frozen 5K `last.ckpt`, restore model,
optimizer and every RNG state, retain global step 5000, and train until global
step 200000. A fresh output directory must be used. Any scientific identity
mismatch, non-contiguous step, corrupted checkpoint, or isolation-field
violation changes the status to `FAILED_CLOSED` and prohibits launch/resume.

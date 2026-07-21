# MCVR V8 matched D1-only 12.5K audit

Audit status: `MATCHED_D1_12P5K_READY`

## Scientific match

The resolved candidate and V8 Full configurations were compared field-by-field. Seed 43,
formal-large train/validation manifests, train scales, stratified sampler, effective batch
64, AdamW policy, weight decay, gradient clipping, D1 trainability, D1-head LR `5e-5`,
D1-backbone LR `2e-5`, 200K horizon, checkpoint/FAST/FULL protocol, evaluator, cache
identities, and applicable safety semantics are identical.

The run starts from the frozen Seed43 D1 checkpoint
`c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca`.
It does not initialize from the V8 5K or 12.5K checkpoint.

The only scientific differences from V8 Full are:

- Error-State Head disabled and frozen;
- learned confidence disabled (`fixed_confidence=1.0`);
- step embedding disabled and frozen;
- differentiable Bond/Angle solver disabled;
- V8 error/confidence/Bond/Angle/Clash/Ring/Chirality/step-consistency losses zero;
- one-step output is the matched D1 Cartesian correction.

The model parity test proves `delta_final == D1 prior v_final` exactly. There is no
`v8_new_heads` optimizer group or trainable V8 parameter. Solver status is `DISABLED` and
`solver_call_count=0`.

## Scheduler and exposure

Both runs use the runner's same constant-LR/no-scheduler policy: scheduler class and state
are `null` in checkpoints. At step 12500 the matched D1 parameter-group LRs are therefore
identical to V8 Full: D1 correction head `5e-5`, D1 backbone `2e-5`. This is a checkpoint
from the original 200K horizon, not a newly designed 12.5K schedule.

- Planned optimizer steps: `200000`
- Graceful stop step: `12500`
- Effective batch: `64`
- Total exposure: `800000`
- Equivalent old batch-8 steps: `100000`

## Preflight evidence

1. Data identities: exact; all four manifest SHA256 values match V8 Full.
2. Train/validation molecule overlap: `0`.
3. Formal-test reads/assets opened: `0 / false`.
4. Frozen-holdout reads: `0`.
5. Frozen D1 strict load: passed.
6. Forward/backward: passed; D1 backbone/head gradients are nonzero.
7. 20-step smoke: passed with finite losses and `solver_call_count=0`.
8. Atomic checkpoint/resume: step20 and `last` are byte-identical; step20→21 resume
   preserved RNG, optimizer and continuous sampler exposure `1280→1344`.
9. FAST 1000 prediction/evaluation smoke: passed.
10. Graceful-stop smoke: planned 200K, stopped exactly at step20, saved atomic checkpoint,
    ran FULL 10K, and exited normally.
11. Source/D1/V5-B/V7 frozen cache identities: passed.
12. Matched exposure calculation: `12500 × 64 = 800000` passed.

The full V8 regression suite passed: `44 passed`. No NaN/Inf was observed. The smoke
checkpoint contains optimizer state (two D1 groups), RNG states for Python/NumPy/Torch/CUDA,
explicit null scheduler state, and sampler/exposure state.

## Decision

`MATCHED_D1_12P5K_READY`

The formal matched D1-only Seed43 run may start in its dedicated output directory. No
no-Angle, no-normalization, other seed, other upstream model, formal-test, or holdout run is
authorized by this audit.

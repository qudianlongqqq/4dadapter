# MCVR V7 Formal-Large Integration Audit

## Decision

`V7_FORMAL_LARGE_PIPELINE_PREPARED`

V7 does not require, and cannot support, an independent training loop. The
formal trainable object remains the existing D1-B Cartesian prior. V7 is a
parameter-free inference wrapper around each completed seed checkpoint.
Integration therefore uses the existing D1-B formal trainer plus a new frozen
factory/config and binding runner. No V7 mathematical code was changed.

## Repository state

- Branch: `feat/mcvr-v7-formal-large`
- Frozen implementation HEAD: `ff19f8c12fe039c6906d2ea841dc6acec2c57d4a`
- Worktree clean: false
- Policy followed: no reset, stash, clean, checkout overwrite, commit, or push

The pre-existing dirty worktree was preserved. The frozen V7 implementation
identity remains:
`74bde53ee2ab1ac22137f90c66bfa3d25a5c8fe97141731b99c4f37656fc711f`.

## V7 architecture

### Backbone and Bond operator

The trainable prior is the formal D1-B `MCVRModel`: a
`LightEGNNRefinerBackbone`, error encoder, rigid Cartesian branch, global
safety gate, and explicit Bond head. The V7 factory constructs the compatible
default `MCVRBACModel`, which is a subclass of `MCVRModel` with no learned
Angle/Clash BAC modules enabled, and strict-loads the D1-B state dict.

V7 uses `base["v_raw"]` as its Bond/Cartesian component. Thus the formal Bond
operator preserves the trained D1-B graph prior and explicit Bond projection;
it does not introduce a second learned head.

### Angle operator

Only active Angle residuals enter the analytic cosine-Angle Jacobian. The
operator uses damped least squares, truncated SVD, the frozen rank threshold,
near-linear weighting, rigid-motion removal, finite checks, and graph/atom
trust caps. It has no trainable parameters and no backward path.

### Clash operator

Clash uses the existing sparse spatial repulsion field with topology-distance
exclusion, degeneracy fail-closed behavior, rigid-motion removal, and fixed
graph/atom trust caps. It is not a learned Jacobian or network branch.

### Fusion and safety

Bond, Angle, and Clash coordinate steps are added and normalized by the frozen
constraint-aware trust rule. The result is multiplied by the inherited D1-B
global safety gate. Formal evaluation must continue to use the existing BAC
safety/backtracking evaluator with the unchanged thresholds in
`configs/ecir_mvr_v7_formal_large.yaml`.

## Formal-large compatibility

No new training script is needed. The existing entry point
`scripts/train_ecir_mvr_medium_rescue_v2.py` already implements the formal
D1-B 25K schedule, 1.6M sample exposures, validation checkpoints, metadata,
and checkpoint persistence. The V7 integration adds only:

- `configs/ecir_mvr_v7_formal_large.yaml`: seed-independent frozen V7 values;
- `etflow/ecir/mvr_v7_formal.py`: fail-closed checkpoint factory;
- `scripts/prepare_ecir_mvr_v7_formal_seed.py`: binding/provenance/checksums;
- `scripts/run_v7_formal_large_seed.sh`: seed/device/config orchestration.

The formal train dataset does not contain `active_angle_constraint_index` or
allowed-range fields because they are not needed for D1-B training. V7 must
therefore remain outside the training loss. At evaluation time the existing
canonical-constraint attachment builds these fields from each source record,
the frozen formal source identity, and validity statistics.

## Checkpoint compatibility

The local seed43 checkpoint was loaded experimentally with both
`MCVRModel` and default `MCVRBACModel` under `strict=True`:

- Checkpoint schema: `ecir-mvr-medium-rescue-formal-large-d1b-checkpoint-v1`
- Model type: `MCVRModel`
- State keys: 176
- Missing keys: 0
- Unexpected keys: 0
- Default BAC modules enabled: false
- Checkpoint SHA256:
  `c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca`

The factory rejects a wrong schema, non-25K step, wrong model type, any BAC
configuration fields, non-strict state load, or any trainable V7 parameter.

The seed42 checkpoint is not present on this Windows host. Its frozen expected
SHA is
`721b4384f3a64eef48ead2fc2b4ea35bf83802b84952e8e3f3aa6c5172e33a2f`.
The Linux runner must verify that file and its resolved training config before
binding. The resolved seed42 config SHA is intentionally not inferred from the
unpopulated base template.

## Data contract

| Asset | Records | Molecules | SHA256 |
|---|---:|---:|---|
| Train sources | 150,000 | 50,000 | `fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d` |
| Train targets | 150,000 | 50,000 | `7e97c5d92529608cfcace8cd279cbd25f20e08b28e1739a191483ba3b574c242` |
| Validation sources | 10,000 | 5,000 | `e7d29f971124f51bd385ec987372ab85181b152250ec0789407a867ff81e3c1a` |
| Validation targets | 10,000 | 5,000 | `4b4ef42c9905c3bbe2dbe911c57827ce594583c66a52f94d7c4d9b5ca70de4c7` |

- Formal source identity:
  `3d86eec9ebd82ae96860330ded0fad35938be74111929ed29b9487f8b7e39a0a`
- Formal target identity:
  `4d2d45950c92894066e347a966c6d5b877afcb5fe0abe6cdb7c06e70a3148e62`
- Test is not named by the training or V7 wrapper config and remains unopened.

## Runtime

The frozen two-batch RTX 5080 benchmark measured:

- D1-B forward + loss: 0.4583 seconds/batch
- D1-B backward: 0.0784 seconds/batch
- V7 solver + fusion overhead: 0.8117 seconds/batch
- D1-B peak allocated memory: 142.54 MiB
- V7 inference peak allocated memory: 33.77 MiB
- Solver calls/failures: 128/0

The compute-only 25K prior estimate is 3.73 hours. Adding a 10K-record V7
validation estimate gives 4.07 hours. This omits validation metrics, I/O,
checkpointing, telemetry, and scheduler stalls. The completed Windows seed43
D1-B run took about 3.66 hours wall time. A 7-10 hour reservation is therefore
conservative and operationally sufficient, not a measured lower bound. See
`docs/MCVR_V7_RUNTIME_ESTIMATE.md` for the full record.

## Runner and resume semantics

Fresh prior training:

```bash
bash scripts/run_v7_formal_large_seed.sh \
  --seed 42 \
  --device cuda:0 \
  --config /path/to/frozen_seed42_config.yaml \
  --data-audit /path/to/data_audit.json
```

Bind an already completed D1-B prior without retraining:

```bash
bash scripts/run_v7_formal_large_seed.sh \
  --seed 42 \
  --device cuda:0 \
  --config /path/to/frozen_seed42_config.yaml \
  --resume /path/to/best_noninferior_validity.ckpt \
  --expected-checkpoint-sha256 721b4384f3a64eef48ead2fc2b4ea35bf83802b84952e8e3f3aa6c5172e33a2f
```

`--resume` means resume the V7 pipeline from a completed prior checkpoint. It
does not claim to restore an interrupted D1-B optimizer run: the existing
formal trainer does not authorize that checkpoint schema/boundary. Binding
creates `config.resolved.yaml`, `run_metadata.json`, `PROVENANCE.json`,
`checkpoint_identity.json`, and relative-name `SHA256SUMS.txt`.

## Actions not taken

- No formal prior training was started.
- No optimizer step was taken by the benchmark.
- No V7 structure, fusion, loss, hidden size, or layer count changed.
- No target was materialized.
- No formal test or frozen holdout was opened.

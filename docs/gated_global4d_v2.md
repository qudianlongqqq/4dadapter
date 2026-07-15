# Gated Global4D V2

## Motivation

Strict Global4D used an exact orthogonal decomposition of the Cartesian head.
Confirm30 and inference ablations showed that the internal `Jq` branch helped,
but deleting the Cartesian component inside the Jacobian subspace hurt the final
RMSD.  V2 therefore treats the Cartesian prediction as the complete primary
velocity and the Global4D branch as a gated structured increment.

## Fusion equations

The V2 main model is `gated_additive`:

```text
v_internal = J q_pred
gate = sigmoid(gate_logit)
v_final = v_cart_raw + internal_beta * gate * v_internal
```

`v_cart_raw` is never projected, clipped by `P_J`, or replaced in this mode.
The projection of the target is used only to build an auxiliary internal target.

The supported modes are deliberately separate from `joint_mode`:

```text
strict_orthogonal:
  v_final = (v_cart_raw - P_J v_cart_raw) + v_internal

additive:
  v_final = v_cart_raw + internal_beta * v_internal

gated_additive:
  v_final = v_cart_raw + internal_beta * gate * v_internal
```

Strict mode exists only for old-checkpoint compatibility, reproduction, and
ablation.  New V2 configs explicitly set `fusion_mode: gated_additive` and the
dedicated training entry point rejects any other mode.  A legacy config that has
no `fusion_mode` is interpreted as `strict_orthogonal`.

## Graph gate

The first implementation predicts one scalar per graph from:

- mean-pooled node hidden state;
- mean-pooled graph time embedding;
- number of rotatable bonds;
- rotatable-bond count divided by atom count.

Pooling uses `index_add_` and `bincount`; broadcasting uses `gate[atom_batch]`.
There is no per-atom Python loop.  The rotatable count is derived from
`rotatable_bond_index`.  If a batch also supplies `num_rotatable_bonds`, both
representations must agree or the model raises an error.  The flexibility tiers
used for metrics are low 0–2, medium 3–5, and high 6+ rotatable bonds.

The final gate layer starts with zero weights and `gate_init_bias: -2.0`, so the
initial gate is `sigmoid(-2)`.  The inference-only `gate_override` accepts a
number in `[0, 1]`; zero gives the exact Cartesian prediction and one gives the
fixed additive result when `internal_beta=1`.

## Training objectives

Strict mode retains the historical objective:

```text
L_final = MSE(v_final, target)
L_internal = MSE(v_internal, P_J target)
L_residual = MSE(v_cart_raw - P_J v_cart_raw, (I-P_J) target)
L = final_weight*L_final + internal_weight*L_internal
    + residual_weight*L_residual + coefficient_weight*L_coefficient
```

Additive and gated-additive use full Cartesian supervision:

```text
L_final = MSE(v_final, target)
L_cartesian = MSE(v_cart_raw, target)
L_internal = MSE(v_internal, P_J target)
L_gate = mean(gate**2)

L = final_weight*L_final + cartesian_weight*L_cartesian
    + internal_weight*L_internal + coefficient_weight*L_coefficient
    + gate_regularization_weight*L_gate
```

`P_J target` is never used to alter the Cartesian prediction in V2.

## Batch-size strategy

The historical fair configuration remains `batch=4`, `accumulate=2`, effective
batch 8.  The first V2 configuration is `batch=8`, `accumulate=1`, also effective
batch 8 with the same `2e-4` learning rate.  Keeping the historical strict config
unchanged avoids rewriting the old experiment contract.

The capacity benchmark tests batches 4, 8, 16, 32, 48, 64, 96, and 128.  Every
condition executes forward, loss construction, backward, optimizer step, and
zero-grad.  Low-, mixed-, and high-complexity record pools prevent a small-only
batch from being reported as generally safe.  Conditions compare a fixed target
number of records and record total atoms, edges, joints, allocated/reserved GPU
memory, throughput, finite loss, OOM, and sampled GPU utilization.  Each
condition runs in a fresh process so an OOM does not contaminate later results.

## DataLoader configuration

`batch_size`, `num_workers`, `pin_memory`, `persistent_workers`, and
`prefetch_factor` flow from YAML or CLI into the DataModule.  The resolved config,
checkpoint runtime hparams, and run state record the normalized values.  With
`num_workers=0`, persistent workers are disabled and `prefetch_factor` is omitted
from the PyTorch DataLoader call.

## Checkpoints and warm starts

Old strict checkpoints instantiate without a gate head and load strictly.  A
gated checkpoint missing gate parameters fails closed.  Explicit warm start is
available through `--warm_start_checkpoint --initialize_missing_gate` or
`scripts/warm_start_gated_global4d_v2.py`.  The load report lists loaded, missing,
unexpected, and shape-mismatched keys.  Warm start is intended only for pilot
acceleration; a formal gated comparison must include a from-scratch run.

## Sampling identity and label isolation

Sampling identities and chunk identities contain fusion mode, internal beta,
gate override, joint mode, checkpoint/config/manifest identities, refinement
steps, and update scale.  Any mismatch rejects final-sample or partial-chunk
reuse.  Sampling uses `FlexBondInferenceDataset`, whose schema rejects every
reference-coordinate and target-label field.

## Windows development and Linux training

Windows is used for code review, read-only formal-cache verification, CPU smoke,
and—after installing a CUDA-enabled PyTorch/PyG environment—capacity testing.
CUDA-unavailable runs are reported as skipped.

Linux smoke:

```bash
export GLOBAL4D_REFERENCE_CACHE=/path/to/flexbond_cache
export GLOBAL4D_INFERENCE_CACHE=/path/to/label_free_cache
export GLOBAL4D_MANIFEST=/path/to/eval_manifest.json
bash scripts/run_gated_global4d_v2_linux_smoke.sh
```

Capacity scan:

```bash
export GLOBAL4D_REFERENCE_CACHE=/path/to/flexbond_cache
bash scripts/run_gated_global4d_v2_batch_capacity.sh
```

Two-thousand-step pilot and resume:

```bash
export GLOBAL4D_REFERENCE_CACHE=/path/to/flexbond_cache
bash scripts/run_gated_global4d_v2_linux_pilot.sh
# The same command resumes from last.ckpt after validating the resolved config.
bash scripts/run_gated_global4d_v2_linux_pilot.sh
```

No script automatically launches formal-large training.  The formal candidate
configuration is `configs/formal_large_gated_global4d_v2_seed42_200k.yaml` and
must be invoked explicitly only after smoke, capacity, and pilot review.

## Experiment plan

The first controlled comparison contains Cartesian baseline, strict Global4D,
fixed additive Global4D, and gated-additive Global4D on identical split and
manifest identities.  Pilot selection uses validation only; the formal test set
is reserved for the final frozen comparison.  Gate distributions and low/medium/
high-flexibility metrics must be inspected for saturation before formal launch.

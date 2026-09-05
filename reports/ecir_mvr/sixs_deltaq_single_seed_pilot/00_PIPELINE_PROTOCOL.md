# SIXS-v2-DeltaQ full-train single-seed three-upstream pilot protocol

This protocol was frozen before full training or any SIXS-v2-DeltaQ DEV outcome was read. It is a development/pilot protocol, not a formal-final evaluation. The released SIXS-v1 tag and all `final_paper_freeze` / `final_evidence_closure` artifacts remain immutable.

## Frozen scientific identity

```text
MODEL = SIXS-v2-DeltaQ
SEED = 307
DELTAQ_TARGET = reference_primitive - source_primitive
MODEL_CAPACITY_INCREASE = NO
FORMAL_TEST_READ = NO
CHECKPOINT_RULE = STEP17500_ONLY_NO_DEV_SELECTION
```

The matched schedule is taken directly from `configs/sixs_deltaq_prototype.json`: AdamW; 17,500 optimizer steps; 64 molecules/batch; backbone/head learning rates 1.5e-4/3e-4; weight decay 1e-6; CosineAnnealingLR with `T_max=22500`; gradient clipping 1.0; recovery every 250 steps. The original BA, Reliability, analytic J-transpose action, projection/normalization, and unrestricted softplus-tau mechanism are unchanged.

## Predeclared overfit gate

The supervisor waits on the already-launched overfit process once, then reads its terminal status once. The gate passes only when the runtime status is `COMPLETED`, checkpoint and result CSV exist, all checkpoint metrics are finite, and the checkpoint records `status=PASS`. That PASS was frozen in the prototype before training: both final Bond and Angle DeltaQ MAE must be at most 50% of their corresponding initial MAE. Failure is fail-closed and does not start full training.

## Cohort roles

- ETFlow uses the frozen 2,500-molecule / 5,000-record DEV manifest. It is not the protected prospective final cohort.
- A new unused AvgFlow cohort is preferred. No unused locally materialized AvgFlow source asset is currently asserted. If identity audit cannot prove one, the existing inspected cohort is used only as `DIAGNOSTIC_REPLICATION_ONLY`, never as prospective confirmation or tuning data.
- The same rule applies to DiTMC: existing assets may be used only as a posthoc transfer/stability diagnostic when no unused cohort is provable.
- No outcome-dependent molecule replacement, checkpoint selection, protocol tuning, or threshold tuning is allowed.

## Endpoint and classification policy

ETFlow assesses retention of validity and local-reference correction without a stability regression. AvgFlow assesses whether Raw-to-v2 V3D benefit coexists with less local-reference over-correction than v1. DiTMC assesses structural transfer and movement-tail reduction. xTB is an optional matched GFN2 single-point diagnostic and is never the training target or a universal pass condition. Single-seed findings are pilot signals only.

## Process and recovery policy

The local supervisor uses blocking child-process waits, atomic status files, and stage-completion markers. A stage is reused only when its marker contains the same pipeline identity SHA and its required outputs still exist. There is no polling loop, continuous log tail, or terminal-attached training worker.

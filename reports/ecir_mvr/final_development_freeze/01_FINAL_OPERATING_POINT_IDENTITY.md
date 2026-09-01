# Final operating-point identity

> Freeze date: 2026-09-01 (Asia/Shanghai). This document records implementation identity only. It does not open or evaluate any protected/future cohort.

## Operating point A — Restricted

```text
MODEL_ID = J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP17500
FORMULATION = RESTRICTED
ROLE = PREDECLARED_OPERATING_POINT_A
SEEDS_COMPLETED = 307;331;353
TRAINING_STEPS = 17500
```

| identity field | authoritative value |
|---|---|
| model config | `configs/sixs_j1r1_full_joint_adaptive_ba_movement.json` plus each multiseed run's `FROZEN_CONFIG.json` |
| training config | AdamW; batch 64 molecules; backbone LR 1.5e-4; controller/head LR 3e-4; weight decay 1e-6; gradient clip 1.0; CosineAnnealingLR T_max 22,500; endpoint 17,500; recovery every 250 steps; no DEV checkpoint selection |
| checkpoints | the three Restricted rows in `08_CHECKPOINT_FREEZE.csv` |
| runner | `scripts/run_sixs_j1r1_full_joint_adaptive_ba_movement.py` |
| core modules | `etflow/ecir/learned_geometry.py`; `musigma_reliability.py`; `j1r1_full_joint.py`; `lsgoba_v2_joint_magnitude.py` |
| evaluators | `scripts/evaluate_sixs_musigma_external.py`; official GenBench3D adapter; PoseBusters `mol_fast`; `run_sixs_current_final_evidence_completion.py`; `run_sixs_j1r1_full_joint_xtb_dev.py` |

## Operating point B — Unrestricted

```text
MODEL_ID = J1_R1_FULL_JOINT_ADAPTIVE_BA_UNRESTRICTED_MOVEMENT_STEP17500
FORMULATION = UNRESTRICTED
ROLE = PREDECLARED_OPERATING_POINT_B
SEEDS_COMPLETED = 307;331;353
TRAINING_STEPS = 17500
```

| identity field | authoritative value |
|---|---|
| model config | `configs/sixs_j1r1_full_joint_unrestricted_movement.json` plus each multiseed run's `FROZEN_CONFIG.json` |
| training config | identical optimizer/batch/LR/weight-decay/clip/scheduler/endpoint/recovery/checkpoint-selection recipe to Restricted |
| checkpoints | the three Unrestricted rows in `08_CHECKPOINT_FREEZE.csv` |
| runner | `scripts/run_sixs_j1r1_full_joint_unrestricted_movement.py` |
| core modules | shared `learned_geometry.py` and `musigma_reliability.py`; `j1r1_full_joint_unrestricted.py`; shared first-order action helpers |
| evaluators | the same frozen external V3D/PoseBusters, RMSD and xTB protocols as Operating point A |

## Cross-confirmation

- All six checkpoints load on CPU for metadata inspection and contain the expected seed, `step=17500`, model state, optimizer/scheduler state and RNG/recovery state. No inference was run.
- `12_MULTISEED_INTEGRITY_AUDIT.md` records all six runs complete, all endpoints 17,500, GPU verification for the new runs, matched sampler/generator states, identical optimizer recipe, no DEV checkpoint selection and complete 5,000-record evaluator denominators.
- `FINAL_STATUS.json` records `MULTISEED_INTEGRITY=PASS`, `FINAL_FORMULATION_CLASSIFICATION=PARETO_NEAR_TIE`, and `MATCHED_SEED_FORMULATION_EFFECT=MIXED`.
- Root config, run-local config, core-source and checkpoint hashes are bound in `06_FINAL_RELEASE_MANIFEST.csv` and `08_CHECKPOINT_FREEZE.csv`.
- Neither operating point is selected as a winner by future protected outcomes. They remain two predeclared operating points.

```text
OPERATING_POINTS_FROZEN = YES
FORMULATION_DECISION_COMPLETE = YES__TWO_PREDECLARED_OPERATING_POINTS
NO_NEW_MODEL_DESIGN = YES
```

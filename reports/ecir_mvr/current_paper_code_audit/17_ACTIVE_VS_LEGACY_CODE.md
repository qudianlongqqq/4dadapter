# Active versus legacy code

## ACTIVE_PAPER_PATH

- `configs/sixs_j1r1_full_joint_adaptive_ba_movement.json`
- `configs/sixs_j1r1_full_joint_unrestricted_movement.json`
- `etflow/ecir/learned_geometry.py` (inherited active backbone/geometry)
- `etflow/ecir/musigma_reliability.py`
- `etflow/ecir/j1r1_full_joint.py`
- `etflow/ecir/j1r1_full_joint_unrestricted.py`
- `etflow/ecir/lsgoba_v2_joint_magnitude.py` (bounded magnitude/cap helper)
- `scripts/run_sixs_j1r1_full_joint_adaptive_ba_movement.py`
- `scripts/run_sixs_j1r1_full_joint_unrestricted_movement.py`
- `scripts/evaluate_sixs_musigma_external.py`
- `scripts/run_sixs_current_final_evidence_completion.py`
- `scripts/run_sixs_j1r1_full_joint_xtb_dev.py`
- current multiseed supervisor/finalizer scripts (active but incomplete).

## ACTIVE EVIDENCE PREDECESSORS

The six-arm factorial and magnitude interaction are not final methods, but they establish J1-R1 and magnitude provenance and provide comparators. The capacity extension is a valid diagnostic but not the selected model.

## LEGACY_NOT_IN_CURRENT_PIPELINE

- Sigma-v2 teacher/student route.
- J0/J2 and R0 factorial arms.
- BA-v1, Adaptive-BA-v2 standalone development.
- older joint-controller-v3, LSGO, BAT/torsion, MCVR, and external-refinement experiment runners.
- step22,500 checkpoint as final endpoint.
- legacy Formal/large-holdout result branches.

A file remaining in the repository does not imply it is invoked. Current runners import the active modules listed above; no current runner import/call path to Sigma-v2 teacher/student or BAT/torsion refinement was found.



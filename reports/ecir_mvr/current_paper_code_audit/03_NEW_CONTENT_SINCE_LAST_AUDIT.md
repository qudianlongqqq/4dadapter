# New content since the last comprehensive audit

LAST_COMPREHENSIVE_AUDIT_POINT = `reports/ecir_mvr/sixs_final_project_integrity_generalization_audit/23_FINAL_PROJECT_AUDIT.md`  
POINT_TIME = `2026-08-31T14:27:23.3104646+08:00`  
POINT_SHA256 = `607a25086b73e34b9af51beffb9e927eaf6d08161900cb5ea1d2bc2842a2b26f`

This is an artifact point, not a commit point. It is selected because it is the latest broad full-project audit. Later artifacts are narrower. Git cannot provide an exact diff because all current-path content is untracked.

## NEW_CODE

- Multiseed orchestration/finalization scripts: `run_sixs_final_multiseed_replication.py`, `wait_and_finalize_sixs_final_multiseed.py`, and `finalize_sixs_final_multiseed_replication.py`.
- Current restricted and unrestricted runners gained environment-variable parameterization for run/config/artifact roots. Their current hashes differ from seed307 executed-runner hashes.
- Evidence type: GIT_FACT (untracked), CODE_FACT (current files), ARTIFACT_FACT (executed runner hashes).

## MODIFIED_CODE

Exact historical source diff is `UNKNOWN` because the executed runner versions were not committed or archived as source. Core model module and config hashes still match seed307 execution; runner bytes do not.

## NEW_EXPERIMENTS

- Matched multiseed replication was started before this audit.
- Single snapshot: supervisor PID 25908; watcher PID 10756; stage `RUN_RESTRICTED_SEED331`; run-local step 2000.
- Status: `INCOMPLETE`. Partial metrics are excluded.

## NEW_RESULTS

No completed new seed331/353 result was available at this audit snapshot. Seed307 restricted/unrestricted artifacts remain authoritative for current numeric comparison.

## NEW_EVALUATION

No new evaluation was started by this audit. The multiseed finalizer contains planned frozen DEV, Reference RMSD, and xTB evaluation, but those planned outputs are not evidence until complete and integrity-audited.

## NEW_AUDITS

- Final candidate selection audit (16:05): seed307 `PARETO_NEAR_TIE`, multiseed required.
- Objective read-only audit (17:08): measured loss/gradient scales.
- Objective module-gradient forensic audit (18:19): superseded the global `alpha_grad` recommendation and found mostly disjoint effective module routing.

## NEW_DECISIONS

The only current decision added after the comprehensive audit is procedural: run matched multiseed before final formulation selection. No new final formulation decision exists.



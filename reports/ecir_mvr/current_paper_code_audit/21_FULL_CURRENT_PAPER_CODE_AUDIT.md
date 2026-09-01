# Full current paper code audit

## Executive conclusion

The current implementation is internally coherent and the completed seed307 artifacts have strong hash/record-level evidence. No data leakage, xTB outcome-dependent exclusion, hidden scientific rollback, or denominator mismatch was found. The principal weakness is provenance: the entire current paper path is untracked at the repository HEAD, and current runner bytes differ from those recorded for seed307 execution. The scientific comparison is also not final because matched seed331/353 replication is incomplete.

## Repository and audit chain

- Branch: `experiment/sixs-musigma-reliability-factorial`.
- HEAD: `9be3633a0af930b29704a29d432327340872de9b`.
- Worktree: dirty with 50 untracked status entries, including all current paper-path code/results.
- Last comprehensive artifact point: full-project integrity/generalization audit at 2026-08-31 14:27 +08:00.
- Later valid audits: final-candidate selection and module-gradient forensic.
- Ongoing multiseed: observed once, incomplete, not used scientifically.

## Current framework summary

**A. One-sentence identity.** A graph-based learned geometry model combines direct heteroscedastic Bond/Angle prediction with source-conditioned primitive Reliability, graph-conditioned Bond/Angle allocation, and learned movement magnitude to produce a first-order Cartesian refinement.

**B. Inputs.** Frozen molecular graph topology/categories, frozen family statistics, and source coordinates. Reference coordinates are training labels and evaluation targets, not inference inputs.

**C. Backbone.** Three message-passing layers with 128-dimensional node state.

**D. Prediction heads.** Positive Bond-length mu, bounded Angle-cosine mu, and direct family sigma anchored to `sigma_stat`.

**E. Geometric correction.** Analytic Bond/Angle primitive derivative contributions weighted by family allocation, Reliability, source defect, and inverse variance.

**F. Reliability.** Shared sigmoid primitive gate using local representation plus inference-safe source geometry/defect features.

**G. Adaptive BA.** Detached graph mean embedding -> 128/64/2 softmax weights.

**H. Movement.** Remove rigid modes, normalize direction to per-graph RMS one, then scale by learned tau.

**I. Formulations.** Restricted adds bounded sigmoid tau, 0.03 Å atom cap, and movement regularizer. Unrestricted uses unbounded positive softplus tau with no cap or movement regularizer. Neither scientific proposal uses rollback.

**J. Output.** `x_prop=x_source+tau*d_hat`, with Restricted per-atom capping.

**K. Evaluation.** Validity3D, PoseBusters, internal Bond/Angle metrics, raw/Kabsch RMSD protocols, and frozen-coordinate GFN2-xTB single points.

## Objective

Restricted: `L_J1_betaNLL + L_post + 0.40793421960700144*mean((tau/0.010)^2)`.  
Unrestricted: `L_J1_betaNLL + L_post`.

J1 uses beta 0.5 with stop-gradient variance weighting. Belief primarily updates backbone/mu/sigma; post primarily updates Reliability/Adaptive BA/magnitude. The earlier global gradient-balance coefficient is not mathematically valid for this mostly disjoint routing.

## Data and training

- TRAIN: 50,000 molecules, 150,000 ETFlow records.
- DEV: 2,500 molecules, 5,000 records.
- TRAIN/DEV molecule overlap: 0 according to the completed integrity audit; current file hashes still match.
- AdamW, batch64, 17,500 steps, backbone LR 1.5e-4, controller/head LR 3e-4, weight decay1e-6, gradient clip1, cosine horizon22,500.
- Full RNG and recovery state is stored.
- No DEV checkpoint selection was found.

## Authoritative completed results

Only seed307 is complete for both formulations. Restricted/Unrestricted respectively:

- V3D 0.564/0.567;
- PoseBusters 0.9324/0.9324;
- Bond MAE 0.00366511/0.00364132 Å;
- Angle-cosine MAE 0.01424487/0.01407739;
- raw Source RMSD 0.00552526/0.00606000 Å;
- Reference RMSD 1.31964114/1.31962614 Å;
- xTB median deltaE -1.33507/-1.36520 kcal/mol;
- lower-energy fraction 0.9988/0.9974;
- >+100 kcal/mol tail 0/1.

This is a Pareto-near-tie on a development set, not a final cross-seed or protected-set selection.

## Protocol integrity

- xTB protocol: PASS for completed seed307; 6.7.1, GFN2, single point, no optimization, no exclusions.
- RMSD protocol: PASS with naming qualification; Reference RMSD is precise, while “Source RMSD” names two related definitions.
- Restricted/Unrestricted core scientific difference: expected movement constraint/regularization only; no extra scientific branch was found.
- Multiseed fairness: designed matched, execution incomplete.

## Reproducibility

Input, config, core-module, checkpoint, record, and result hashes are strong. Commit and executed-runner provenance are not. Overall reproducibility is `PARTIAL`.

## Safe current scientific position

The code supports a mature seed307 development method and a qualified same-upstream unseen-molecule result. It does not yet support a final formulation choice, multiseed robustness, cross-upstream transfer, protected-test performance, SOTA, universality, or guaranteed improvement.

```text
AUDIT_STATUS = COMPLETE_READ_ONLY
DATA_FLOW_STATUS = PASS_FOR_COMPLETED_SEED307
TRAIN_DEV_INTEGRITY = PASS
RESTRICTED_IMPLEMENTATION_VERIFIED = YES
UNRESTRICTED_IMPLEMENTATION_VERIFIED = YES
MATCHED_SEED_FAIRNESS = INCOMPLETE
MULTISEED_STATUS = INCOMPLETE
CODE_RESULT_VERSION_MATCH = NO
REPRODUCIBILITY_STATUS = PARTIAL
CRITICAL_FINDINGS = 0
HIGH_FINDINGS = 2
MEDIUM_FINDINGS = 6
FINAL_SCIENTIFIC_STATE = MATURE_SEED307_DEVELOPMENT_EVIDENCE__FINAL_SELECTION_PENDING_MULTISEED
AUDIT_ONLY = YES
```



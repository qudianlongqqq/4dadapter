# Current scientific state

## VERIFIED

- Current Restricted and Unrestricted seed307 identities, configs, core module hashes, checkpoints, and 17,500-step endpoints.
- J1 beta-NLL beta 0.5; R1 source-conditioned Reliability; Adaptive BA; learned magnitude.
- TRAIN/DEV molecule separation and no detected reference leakage.
- Same 5,000 DEV record identities for the two seed307 formulations.
- Seed307 V3D, PoseBusters, internal geometry, movement, Reference RMSD, and GFN2-xTB results.
- xTB 6.7.1 GFN2 single-point, no optimization, complete seed307 denominator.
- Reference RMSD fixed-order explicit-all-atom Kabsch to nearest frozen reference ensemble.

## NOT VERIFIED

- Completed seed331/353 outcomes and multiseed integrity.
- Cross-upstream behavior of current formulations.
- Protected Formal and large-holdout outcomes.
- Exact committed source for seed307 executed runners.
- Paper manuscript wording.
- SOTA, universality, guarantees, or final unbiased test performance.
- Full Restricted per-atom displacement distribution in a frozen report.
- Comparable DEV Reliability/sigma distribution tables for Unrestricted.

## KNOWN LIMITATIONS

- DEV is development data.
- Current completed formulation comparison is seed307 only.
- Git/runner provenance is incomplete.
- Source RMSD naming differs by protocol.
- Reference symmetry permutations are not handled.
- Major human design constants retain limited isolated sensitivity evidence.

## SUPERSEDED RESULTS

- Sigma-v2 teacher/student is not a current evaluated method.
- Six-arm J1-R1 and magnitude-only results are development ancestry/comparators, not the final architecture.
- Step22,500 does not replace step17,500.
- The earlier `alpha_grad≈2.255e12` recommendation is invalidated.
- “Restricted preferred” or “Unrestricted preferred” based on seed307 alone is not current authoritative formulation selection.

## CURRENT AUTHORITATIVE RESULTS

For seed307 DEV only:

| metric | Restricted | Unrestricted |
|---|---:|---:|
| V3D | 0.5640 | 0.5670 |
| PoseBusters | 0.9324 | 0.9324 |
| raw Source RMSD (Å) | 0.00552526 | 0.00606000 |
| Reference RMSD (Å) | 1.31964114 | 1.31962614 |
| xTB median deltaE (kcal/mol) | -1.33507487 | -1.36519972 |
| xTB fraction deltaE<0 | 0.9988 | 0.9974 |
| xTB >+100 kcal/mol | 0 | 1 |

CURRENT_FORMULATION_CONCLUSION = PARETO_NEAR_TIE_SEED307_ONLY  
CURRENT_OPERATIONAL_CANDIDATE = RESTRICTED_STEP17500  
FINAL_FORMULATION_SELECTION = PENDING_MULTISEED_INTEGRITY



# Reviewer risks after multiseed

| Question | Current answer | Evidence | Still missing | Blocking |
|---|---|---|---|---:|
| Are the movements too tiny to matter? | They are tiny but produce repeatable local-validity and energy shifts | Three-seed V3D/xTB consistency and movement distributions | Molecule-level effect plots and matched optimization comparator | No |
| Why not simply use MMFF94s? | No fair current-candidate comparison exists yet | Source-only comparison is available | Exact-cohort MMFF94s geometry optimization | Yes for submission |
| Why not GFN2-xTB optimization? | Current xTB is single-point evaluation, not optimization | Three-seed single-point deltas | Cost/quality comparison on a declared subset | No for narrow claim; yes for physics-superiority claim |
| Is Adaptive BA necessary? | It is part of the coherent final system, but exact-final causal necessity is unproven | Historical BA/movement experiments and stable learned weights | Exact final control only if necessity is a headline claim | No |
| Does predictive sigma truly help? | It participates in action weighting; calibrated uncertainty and exact-final necessity are unproven | J0/J1/J2 factorial and sigma audits | Calibration or conservative wording; fixed-sigma control only for a strong causal claim | No |
| Does Unrestricted sacrifice Source fidelity? | Yes: displacement is consistently larger; central V3D/xTB improves | Matched three-seed differences | Correct metric naming and final confirmation | No, if reported as Pareto trade-off |
| Why is Reference RMSD unchanged? | The method is a local geometry refiner, not a global conformer-recovery system | Three-seed practical tie | Clear scope and nonclaim | No |
| Is the method ETFlow-specific? | Current evidence is same-upstream development evidence | Frozen ETFlow DEV; no protected cross-upstream outcome used | Optional DiTMC secondary holdout | No for ETFlow-scoped claim |
| Where is the independent test? | No valid primary protected cohort is ready as-is | Metadata-only Formal/large-holdout audit | Freeze a clean, molecule-disjoint same-ETFlow primary cohort | Yes |
| Can the work be reproduced exactly? | Scientific files survive but are not an immutable release unit | Hashable files and integrity audits; untracked worktree | Commit/tag/archive dependency closure and environment | Yes |

# Reviewer attack simulation

| # | reviewer question | current answer | evidence strength | missing evidence | blocking |
|---:|---|---|---|---|---|
| 1 | A 0.005 Å move looks trivial; is the gain real? | paired V3D/Bond/Angle/xTB changes show that local boundary-sensitive metrics can change under small moves | MODERATE, seed307 | multiseed paired transition distributions and illustrative cases | NO after multiseed |
| 2 | Why does Reference RMSD barely change? | the method targets local primitive geometry, not global conformational search | STRONG | none if claim is narrowed; otherwise torsion/global evidence | NO for local claim |
| 3 | Why only Bond and Angle primitives, not torsions? | current method is a local fixed-topology refiner | MODERATE | family ablation and explicit limitation | NO for local claim |
| 4 | Is Reliability merely another gate? | it uses Source-conditioned defect features and R0/R1 factorial effects are positive | MODERATE-STRONG | exact final no-Reliability control | NO, but should strengthen |
| 5 | Is Adaptive BA necessary? | current evidence does not causally establish it | WEAK | exact final Equal-BA control | YES if claimed as independent contribution |
| 6 | Is sigma calibrated uncertainty or just a scale? | only learned predictive-scale/NLL semantics are established | WEAK | calibration/coverage and fixed-sigma action control | YES if calibrated-uncertainty claim is made |
| 7 | Are results specific to ETFlow? | yes, current complete evidence is same-upstream ETFlow | STRONG boundary statement | cross-upstream evaluation for broader claim | NO for ETFlow-scoped paper; YES for general-framework claim |
| 8 | Why not MMFF94s or GFN2-xTB optimization? | historical comparison exists but is not matched to the current candidate | WEAK for current paper | current frozen candidate versus matched optimizers and runtime | YES |
| 9 | Does the model actually improve physical energy? | seed307 GFN2 single-point DeltaE median is negative and >99.7% are lower | MODERATE | multiseed energy and tail replication; no guarantee | NO after multiseed if qualified |
| 10 | Does Unrestricted hide catastrophic cases? | no; the +570 kcal/mol maximum and >100 tail are retained explicitly | STRONG transparency | cross-seed tail stability | NO if transparently reported |
| 11 | Were DEV results used repeatedly for design? | yes; DEV is not an unbiased final test | STRONG | untouched/authorized final cohort | YES |
| 12 | Are two records per molecule treated as independent? | molecule-cluster bootstrap is used | STRONG | keep seed and molecule uncertainty separate | NO |
| 13 | Can the exact executed code be reproduced? | not currently at commit/runner-byte level | WEAK | commit, executed source snapshot, release hashes | YES |
| 14 | Are human constants over-tuned? | no protected tuning found; 9 Restricted/6 Unrestricted constants are documented | MODERATE | targeted sensitivity only for headline constants, not broad sweeps | NO if disclosed |
| 15 | Is this a Gauss–Newton/optimization method? | no; it is a learned first-order local action | STRONG code-level answer | manuscript terminology correction | YES if overclaimed, otherwise NO |

Questions 1, 2, 7, 9, 10, and 12 are currently defensible with the stated scope. Questions 5, 6, 8, 11, and 13 are the most consequential unresolved attacks.


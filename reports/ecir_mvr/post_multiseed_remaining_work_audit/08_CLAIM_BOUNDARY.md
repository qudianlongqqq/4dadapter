# Paper claim boundary

## Safe now

- On the fixed ETFlow development cohort, the learned fixed-topology local refiner is robust across three matched training seeds.
- Both operating points make small, finite Cartesian corrections; Unrestricted small-movement behavior replicated.
- On development data, Unrestricted consistently has higher V3D and lower xTB median delta-energy than Restricted, while PB and Reference RMSD are essentially tied and Source displacement is larger.
- The movement policy is a reproducible Pareto trade-off, not a unique formulation winner.

## Safe only after a valid final test

- Generalization to unseen molecules from the same ETFlow source distribution.
- Replication of validity, local-geometry and energy effects on an outcome-unseen final cohort.
- Final effect sizes and confidence intervals intended as confirmatory evidence.
- Cross-upstream robustness, only if a separately frozen DiTMC secondary cohort passes.

## Do not claim automatically

- General conformer refinement independent of upstream generator.
- Global conformer recovery or meaningful improvement in Reference RMSD.
- Calibrated predictive uncertainty without a calibration analysis.
- Guaranteed per-molecule or per-conformer improvement.
- Absence of adverse energy or movement tails.
- State of the art without matched task-compatible baselines.
- Superiority to MMFF94s/GFN2 geometry optimization without running those matched baselines.

The defensible paper position is a seed-robust, fixed-topology **ETFlow-source local refinement** method with conservative and unrestricted movement operating points.

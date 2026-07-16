# MCVR Medium Gate 2 — Seed42

## Final decision

- `current_stage`: `MEDIUM_SEED42_COMPLETE`
- `current_decision`: `MEDIUM_SEED42_FAIL`
- `20k_started`: `true`
- `20k_completed`: `false`
- Completed optimizer steps: 2000/20000
- Stop reason: `velocity_norm_sustained_growth`
- Test records read: 0
- `100k_permitted`: `false`
- `100k_started`: `false`
- Seed43/44 commands generated: no

## Gate audit

Twenty-six of the 27 metric conditions passed in the post-stop evaluation. Condition 02 failed because the best relative improvement among bond, angle, ring, and clash was 1.83%, below the registered 10% threshold. In addition, the overarching training-completed requirement failed because the run stopped before its first scheduled validation checkpoint.

The following safety boundaries remained intact:

- all RMSD/MAT/COV non-inferiority gates passed;
- high-flex validity improved and high-flex RMSD was non-inferior;
- clean identity was 100%;
- severe clash and chirality did not worsen;
- acceptance reduced both validity-worsened and RMSD-worsened fractions;
- unseen scale 0.35 validity improved and accuracy was non-inferior;
- the 12-molecule non-ring subset had no abnormal accuracy or safety failure;
- improvements were observed in more than one source group;
- torsion gate and torsion velocity contribution remained exactly zero.

These partial results cannot override either the early-stop boundary or the 10% minimum-effect condition. Seed42 is therefore failed. Per protocol, seed43, seed44, and 100k are blocked, and there is no next training command.

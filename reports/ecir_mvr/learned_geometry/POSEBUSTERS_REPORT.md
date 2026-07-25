# LSGO PoseBusters report

Unified PoseBusters 0.6.5 `mol_fast`, 600 paired external-confirm Sources per condition. All 11 configured scientific checks were included; the six geometry checks are shown below.

| method      | overall  | bond_lengths | bond_angles | internal_steric_clash | aromatic_ring_flatness | non-aromatic_ring_non-flatness | double_bond_flatness |
| ----------- | -------- | ------------ | ----------- | --------------------- | ---------------------- | ------------------------------ | -------------------- |
| Source      | 0.898333 | 0.998333     | 0.996667    | 0.916667              | 1                      | 0.983333                       | 0.991667             |
| A-G         | 0.898333 | 0.998333     | 0.996667    | 0.916667              | 1                      | 0.983333                       | 0.991667             |
| B-G_seed173 | 0.898333 | 0.998333     | 0.996667    | 0.916667              | 1                      | 0.983333                       | 0.991667             |
| B-G_seed181 | 0.898333 | 0.998333     | 0.996667    | 0.916667              | 1                      | 0.983333                       | 0.991667             |
| B-G_seed193 | 0.898333 | 0.998333     | 0.996667    | 0.916667              | 1                      | 0.983333                       | 0.991667             |

| fail_to_pass | fail_to_pass_fraction | method      | pass_to_fail | pass_to_fail_fraction | records | strict_failure_transfer |
| ------------ | --------------------- | ----------- | ------------ | --------------------- | ------- | ----------------------- |
| 0            | 0                     | A-G         | 0            | 0                     | 600     | 0                       |
| 0            | 0                     | B-G_seed173 | 0            | 0                     | 600     | 0                       |
| 0            | 0                     | B-G_seed181 | 0            | 0                     | 600     | 0                       |
| 0            | 0                     | B-G_seed193 | 0            | 0                     | 600     | 0                       |

All four updates have pass→fail=0 and fail→pass=0. Thus PB does not systematically regress, but it also does not improve at this 0.003 Å budget. The three B seeds are identical on every discrete PB check. C-G/C-P were not run because Variant C hit the preregistered sigma-inflation stop rule. Formal test reads=0; frozen holdout reads=0.

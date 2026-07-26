# LSGO Bond Minimality Prospective Confirmation — final summary

Decision: **KEEP_BA**.

1. Fresh BA replicated stable xTB improvement: seed medians `-0.6599` to `-0.6490` kcal/mol, 90.33–90.67% improved, with p95/p99 zero.
2. Fresh B-only retained ≥95% of BA in every seed: `0.9794, 0.9913, 0.9852`. B improved 91.5–91.83% of records.
3. B harmful tails were non-inferior: p95/p99 zero for all seeds; maxima were no worse than the preregistered BA margins.
4. High-flex remained B-dominant: retention `1.0034, 0.9817, 0.9865` across seeds.
5. Important rare Angle subgroup: **none found** under the frozen ≥20-record, ≥0.10 kcal/mol, three-seed-consistent definition. B-harm/BA-rescue occurred in 1/1800 seed-record pairs (0.056%).
6. PoseBusters overall safety was unchanged at 91.83%, but B seed181 caused one additional `double_bond_flatness` pass→fail relative to BA. This fails the frozen per-check safety gate even though the affected record was already an overall PB failure.
7. B did not reduce movement further: mean RMS `0.002758 Å` versus BA `0.002725 Å`; it was only slightly larger but within the frozen 0.0001 Å non-inferiority margin. p99 remained 0.003 Å, topology/chirality 100%, mode switches zero.
8. Angle objective **cannot be formally removed** under the frozen all-gates rule. Its average energy contribution is small, but BA must be retained because Bond-only missed the strict per-check PB non-inferiority condition.
9. Final decision: **KEEP_BA**.
10. Formal test reads = **0**.
11. Frozen holdout reads = **0**.

## This experiment proves

On a new, historically unexposed 200-molecule/600-Source prospective cohort, using all three frozen checkpoints and identical solver/safety rules, Bond is again the dominant energy-improving component and retains at least 95% of BA median energy benefit in every seed. The experiment also proves that Bond-only does not pass the complete preregistered simplification gate: seed181 has one additional PoseBusters double-bond-flatness pass→fail relative to BA. The formal method therefore remains frozen Bond+Angle.

## This experiment does NOT prove

It does not prove that Angle contributes equally to Bond, that the one PB regression represents a large population-level effect, or that Bond-only could never pass a separately preregistered larger study. It does not authorize removal of the Angle objective, a new Bond network, changed trust budget, xTB/PB teacher, torsion/clash module, or result-dependent routing.

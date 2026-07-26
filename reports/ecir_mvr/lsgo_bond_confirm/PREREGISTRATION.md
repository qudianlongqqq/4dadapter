# LSGO Bond Minimality Prospective Confirmation — preregistration

Status: **FROZEN BEFORE BOND_MINIMALITY_CONFIRM EXTERNAL RESULTS**. Base `02d140036257d6e2c15fe18aa18aef7ed24ca33f`; branch `research/lsgo-bond-minimality-confirm`.

The only candidates are the same frozen shared neural model evaluated with B-only or unchanged BA objectives. A-only is a mechanism control. No model is retrained. All conditions use the identical direct normalized-gradient solver, 0.003 Å graph-RMS trust region, 0.03 Å atom cap and topology/chirality/ring/hard-steric acceptance.

`BOND_MINIMALITY_CONFIRM` is selected exclusively from formal-large TRAIN real sources, never formal test or frozen holdout. It contains 200 molecule-disjoint molecules and 600 Source records, SHA-ranked after excluding explicitly enumerated historical external/confirm identities and the anchor training/dev cohorts. Flexibility is frozen at 70 molecules with 0–2 canonical free rotors, 70 with 3–4, and 60 with ≥5.

Coordinates for Source plus B/A/BA at seeds 173/181/193 must be generated once and frozen before PoseBusters or xTB. GFN2-xTB is single-point only. PoseBusters uses the frozen molecule-only full check schema and is independent safety evaluation only.

Primary simplification gate, required independently for every seed: median energy retention `|median ΔE_B|/|median ΔE_BA| ≥0.95`. Across-seed mean-energy retention must be ≥0.90 and improved-fraction retention ≥0.95. Median paired `ΔE_B-ΔE_BA` must be ≤0.05 kcal/mol. B tails may exceed BA by at most 0.02 kcal/mol at p95/p99, 0.10 at maximum, and 0.05 in positive-tail mean. High-flex median retention must be ≥0.90.

B movement mean/p99 may exceed BA by at most 0.0001 Å. Topology and chirality must be 100%. B may not cause more PoseBusters overall or per-check pass→fail transitions than BA. An Angle-dependent chemistry subgroup is preregistered as a stratum with ≥20 paired records, median `ΔE_B-ΔE_BA ≥0.10 kcal/mol` and the same positive direction in all three seeds. Rare B-harm/BA-rescue fraction must be ≤1%.

Any failed condition yields `KEEP_BA`; insufficient execution/variance yields `BOND_CONFIRM_INCONCLUSIVE`; inability to reproduce historical BA yields `BA_REPRODUCTION_FAILURE`. Only all-pass evidence yields `SIMPLIFY_TO_BOND`.

Formal test reads = **0**. Frozen holdout reads = **0**.

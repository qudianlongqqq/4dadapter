# LSGO lessons and BAT inheritance audit

Audit date: 2026-07-26. Base commit: `502da90f915bfb15a3fa061efcb03ad6fb2c99b4`.

The frozen LSGO evidence was read from `reports/ecir_mvr/learned_geometry/{FINAL_SUMMARY.md,LSGO_DECISION.json,POSEBUSTERS_REPORT.md,XTB_REPORT.md,FIDELITY_REPORT.md,HISTORICAL_OVERLAP_AUDIT.md,LSGO_PREREGISTRATION.json}`. There is no file literally named `SCIENTIFIC_REASONABLENESS_AUDIT.md` or `PREREGISTRATION.md` in the frozen package; the binding record is `LSGO_PREREGISTRATION.json/md`. This absence is recorded rather than silently reconstructed.

## Frozen positive mechanism

Variant B is a 473,674-parameter invariant graph encoder with neural conditional Bond and Angle means, frozen DRCSR/reference scales, equal Bond/Angle group NLL aggregation, rigid-mode removal, one normalized gradient step, 0.003 Å graph-RMS trust radius and 0.03 Å atom cap. It uses all Bond/Angle primitives. The frozen LSGO implementation does **not** contain a separately calibrated BA active threshold; therefore this task preserves the exact all-primitive BA anchor instead of retroactively claiming an active-set.

The selected checkpoints are read only from the frozen manifest:

| BA seed | checkpoint step | SHA256 |
|---:|---:|---|
| 173 | 2500 | `f382e261224609fa7eea8e05e67e8600a04674bac09a7031d539a327b8aec62e` |
| 181 | 2500 | `c0f458aada6eca7726269a327e8d5c11fd18af6c0695be0c812f632a78ac6ed7` |
| 193 | 2500 | `e3606d683921bfb9aec1afa41e2e9835356196e7615b8dd530b0ac9d96240ead` |

The checkpoints remain physically stored in the frozen LSGO worktree. Every BAT load must validate the manifest and checkpoint SHA before inference.

## What is retained

- neural conditional Bond/Angle centers;
- frozen train-Reference statistical Bond/Angle scales;
- exact direct-gradient BA objective and normalization;
- 0.003 Å primary trust budget and 0.03 Å atom cap;
- rigid translation/rotation removal;
- topology, chirality, ring and catastrophic-clash guards;
- exact Source fallback and coordinate/SDF/SHA freeze protocol;
- three deterministic pairs: B173+T211, B181+T223, B193+T239.

## What is stopped

- learned sigma and every sigma-rescue mechanism;
- learned precision, C-G, C-P and LPWGP;
- amortization;
- ring likelihood/mixture and complete 3N-6 modelling;
- Cartesian teacher/distillation, denoising and Source-to-Reference displacement regression.

## Frozen LSGO facts inherited, not re-selected

Reference stationarity passed; Source exceeded Reference objective in 95.8–97.9% of audited molecules; old prospective B-G median xTB transfer was `-0.8045 ± 0.0050` kcal/mol with 93.61% improved and safe tails; PB was unchanged at 89.833%; RMS was 0.002815 Å; topology/chirality were 100%. The old 600-record external identity is now exposed and may be used only as historical evidence, never for BAT selection or parameter tuning.

Formal test reads=0; frozen holdout reads=0; FULL10K tuning=false.


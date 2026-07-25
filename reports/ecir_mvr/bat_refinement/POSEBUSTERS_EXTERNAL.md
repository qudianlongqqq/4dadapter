# Fresh prospective PoseBusters evaluation

PoseBusters 0.6.5 `mol_fast.yml` was run only after all seven coordinate conditions were SHA-frozen.

| method | records | overall | internal steric clash | bond lengths | bond angles | ring flatness checks | pass→fail | fail→pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | 600 | 92.333% | 92.333% | 100% | 100% | 100% | — | — |
| BA seed173 | 600 | 92.333% | 92.333% | 100% | 100% | 100% | 0 | 0 |
| BA+C seed173 | 600 | 92.333% | 92.333% | 100% | 100% | 100% | 0 | 0 |
| BA seed181 | 600 | 92.333% | 92.333% | 100% | 100% | 100% | 0 | 0 |
| BA+C seed181 | 600 | 92.333% | 92.333% | 100% | 100% | 100% | 0 | 0 |
| BA seed193 | 600 | 92.333% | 92.333% | 100% | 100% | 100% | 0 | 0 |
| BA+C seed193 | 600 | 92.333% | 92.333% | 100% | 100% | 100% | 0 | 0 |

All per-check transitions are zero. BA remains PB-safe but not PB-improving. BA+C reduces its own continuous vdW penetration diagnostic, but it resolves none of the 46 PoseBusters steric failures and creates none. Therefore the preregistered `STERIC_VALIDATED` condition fails.

Formal test reads=0; frozen holdout reads=0; used for model selection=false.

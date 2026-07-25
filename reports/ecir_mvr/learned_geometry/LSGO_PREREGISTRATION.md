# LSGO preregistration

Status: **FROZEN before training**.

- Branch/HEAD: `research/learned-structured-geometry-objective` / `fbde3ef497364cf92aa90caba8e62e25a33e2e18`
- Seeds: `[173, 181, 193]`
- A/B/C and all sigma, optimizer, checkpoint, stationarity, separation, Jacobian, trust and external gates are frozen in `LSGO_PREREGISTRATION.json`.
- Training uses TRAIN Reference local Bond/Angle values only. There is no Cartesian label, MVT coordinate, Source-to-Reference delta, PB/xTB/BAC selection, or amortized refiner.
- External PB/xTB remain locked until checkpoint and coordinate freeze.

Formal test reads=0; frozen holdout reads=0; FULL10K used for tuning=false.

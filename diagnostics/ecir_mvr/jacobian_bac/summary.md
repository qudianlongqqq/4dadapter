# MCVR V3-BAC Jacobian Summary

Decision: `JACOBIAN_NOT_SUPPORTED`

- Records/molecules: 1024/512
- J0 acceptance: 26.5625%
- J0 Bond/Angle delta: -0.01587143 / -0.00339417
- J0 weighted BAC delta: -0.04516642
- J0 mean displacement: 0.00107297 Angstrom
- Solver failure: 0/1024
- D1 acceptance: 97.2656%
- D1 Bond/Angle delta: -0.09156973 / -0.00233258
- test_records_read: 0
- test_assets_opened: false
- frozen_holdout_records_opened: 0

J0 improves active Angle beyond D1 with less movement, but loses most Bond and
weighted BAC gain and has much lower acceptance. J1, learned Jacobian, 10k, and
formal-large are not authorized.

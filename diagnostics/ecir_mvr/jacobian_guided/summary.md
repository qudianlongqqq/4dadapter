# MCVR V4 Jacobian-Guided Development Summary

Decision: `JACOBIAN_GUIDANCE_NOT_SUPPORTED`.

No fixed candidate passed every preregistered gate. A100 was the only candidate
with additional active-Angle improvement versus D1 (`-0.005352`, paired 95% CI
`[-0.006280, -0.004525]`) while preserving most Bond gain, but its displacement
was `1.155x` D1 and exceeded the frozen `1.1x` bound. A025, A050, and B worsened
active Angle. C reduced acceptance from `97.27%` to `77.25%` and degraded Bond
by `+0.023795` versus D1.

Records: 1024 development records from 512 molecules. Test records read: 0.
Test assets opened: false. Frozen holdout records opened: 0. Training runs: 0.
Target rematerialization: false. Start 10k: false. Start formal-large: false.

See `docs/MCVR_V4_JACOBIAN_GUIDED_REPORT.md` for the complete interpretation.

# MCVR V4 Jacobian-Guided Cartesian Report

## Decision

**JACOBIAN GUIDANCE NOT SUPPORTED.** None of the frozen A/B/C candidates
passes the preregistered development-cohort support rule. V4 should not enter
10k or formal-large training in its current form.

This conclusion uses only the frozen 512-molecule, 1024-record development
cohort with identity
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.
Formal test records read, test assets opened, and frozen holdout records opened
are all zero. No target was rematerialized and no model was trained.

## Frozen implementation

D1 remains the unchanged learned Cartesian optimizer. Its final accepted
trajectory coordinate defines `delta_cart`; the guided module is outside
`MCVRBACModel`, so checkpoint strict-load and the state-dict contract are
unchanged.

Candidate A linearizes at the accepted D1 coordinate and adds one bounded,
rigid-projected, damped least-squares correction at fixed alpha 0.25, 0.5, or
1.0. Candidate B computes the damped truncated-SVD row-space decomposition of
`delta_cart` and removes only components whose first-order residual product is
worsening. Candidate C creates no new direction and tests the D1 direction at
fixed scales 1, 0.5, 0.25, and 0.125.

All candidates retain the J0 numerical protections: float64 damped solves,
relative rank threshold, truncated small-singular-value handling, cosine
Angle rows with near-linear downweighting, finite degenerate-Clash fallback,
rigid motion removal, graph RMS and atom-max trust limits, nonlinear objective
checks, and existing BAC/Ring/chirality hard safety.

## Development results

All deltas below are candidate minus source; more negative BAC deltas are
better. Displacement is mean molecule RMS displacement.

| Candidate | Bond | Angle | Weighted BAC | Acceptance | Displacement (A) | Guided fallback |
|---|---:|---:|---:|---:|---:|---:|
| D1 | -0.091570 | -0.002333 | -0.168421 | 97.27% | 0.006701 | n/a |
| A025 | -0.090500 | -0.001845 | -0.170412 | 97.27% | 0.007002 | 39.26% |
| A050 | -0.089710 | -0.001815 | -0.171444 | 97.27% | 0.007432 | 43.16% |
| A100 | -0.088607 | -0.005342 | -0.177489 | 97.75% | 0.007737 | 46.97% |
| B | -0.091274 | -0.002165 | -0.168439 | 97.27% | 0.006692 | 70.12% |
| C | -0.067775 | -0.001269 | -0.127597 | 77.25% | 0.004981 | 22.75% |

Clash and chirality were unchanged to reported precision for all candidates.
A100 improved Ring relative to D1 by `-0.000311`; A025, A050, B, and C
regressed Ring relative to D1.

## Paired evidence versus D1

On the 576 Angle-active records from 339 molecules:

| Candidate | Active-Angle difference | Paired 95% CI | Bond difference | Movement ratio | Support |
|---|---:|---:|---:|---:|---|
| A025 | +0.000798 | [+0.000544, +0.001089] | +0.001069 overall | 1.045x | No |
| A050 | +0.000823 | [+0.000537, +0.001119] | +0.001860 overall | 1.109x | No |
| A100 | -0.005352 | [-0.006280, -0.004525] | +0.002963 overall | 1.155x | No |
| B | +0.000267 | [+0.000133, +0.000417] | +0.000296 overall | 0.999x | No |
| C | +0.001820 | [+0.001356, +0.002382] | +0.023795 overall | 0.743x | No |

A100 is the only candidate with statistically supported additional Angle
improvement. It preserves most D1 Bond capability, improves weighted BAC and
does not reduce acceptance. However, the improvement accompanies a 15.5%
increase in movement and therefore fails the frozen 10% movement bound. It is
evidence that Jacobian correction can be geometrically useful, but not evidence
for the preregistered bounded auxiliary module.

A025 and A050 move more while making Angle worse than D1. B is nearly an
identity transform: 69.3% of records have no selected worsening row and 70.1%
fall back to D1; its small applied changes slightly worsen Angle and Ring. C
accepts only 77.25% and loses 0.023795 of D1 Bond improvement, reproducing the
central failure of using a constraint objective as the primary proposal judge.

## Required answers

### A. Is Jacobian effective as an auxiliary module?

Not under the frozen support rule. A100 shows a real Angle signal, but it is
coupled to movement above the allowed bound. The smaller A corrections and B
projection do not improve Angle.

### B. Is there a gain over D1?

There is a narrow A100 gain: stronger Angle, better weighted BAC, slightly
higher acceptance, and preserved chirality. It is not a supported overall gain
because displacement is 1.155 times D1. No other candidate has a credible
Angle gain.

### C. Why can J0 not replace D1?

J0 optimizes only the local linearized constraint system. It lacks the learned
Cartesian prior that coordinates coupled Bond, topology, and proposal
acceptance. Its prior result had stronger active Angle than D1 but only 17% of
D1 Bond improvement and 26.56% acceptance. V4-C confirms the same structural
problem: Jacobian objective descent alone is not a substitute for the learned
proposal.

### D. Does the hybrid solve both methods' weaknesses?

No. A100 partially combines D1 Bond strength with Jacobian Angle efficiency,
but violates the movement contract and falls back on 46.97% of records. B is
too weak to add Angle value. C preserves neither D1 acceptance nor Bond gain.

### E. Should this enter 10k or formal-large?

No. The preregistered conclusion is **do not proceed**. Further work would need
a new, separately preregistered hypothesis about bounded correction gating or
constraint confidence; these development results do not authorize tuning an
alpha, adding candidates, training a free Cartesian residual, or opening test
or frozen holdout data.

## Runtime and execution note

D1 inference took 36.27 seconds and used 176.3 MiB peak allocated GPU memory.
Per-candidate accumulated CPU times were 9.20 seconds (A025), 9.18 (A050),
8.98 (A100), 11.11 (B), and 14.69 (C); the combined guided wall time was 53.78
seconds. Peak process RSS was 2.06 GiB.

The first exact evaluator execution completed candidate calculation but stopped
before writing results because the report assertion used the new
`acceptance_fraction` key against the legacy `accepted_fraction` metadata key.
Only that key mapping was repaired. The same frozen command was replayed with
no candidate, threshold, alpha, data, or configuration change; no first-run
metric was emitted or used for selection.

Machine-readable evidence is in
`diagnostics/ecir_mvr/jacobian_guided/summary.json`, with per-record,
per-molecule, summary, and solver-diagnostic outputs in the same directory.

# MCVR V5 Constraint-Space Hybrid Report

## Decision

**V5 CONSTRAINT HYBRID NOT YET SUPPORTED FOR SCALE.** Neither frozen prototype
passes every preregistered development gate. Prototype B provides the stronger
scientific mechanism and is the recommended paper direction, but it exceeds
the unchanged `1.1x` D1 movement envelope by 0.32 percentage points. It must
not enter 10k, formal-large, test, or frozen-holdout evaluation from this result.

The experiment used the same seed 43018 and the same 512-molecule,
1024-record development cohort as D1. Test records read, test assets opened,
and frozen holdout records opened are all zero. No target was rematerialized,
and hidden dimensions and layer counts were unchanged.

## Implemented prototypes

Prototype A keeps the D1 Cartesian prior and adds separate Bond, Angle, and
Clash equivariant heads. Per-component graph RMS caps prevent magnitude
domination. A masked softmax assigns a shared constraint budget, followed by a
learned activity gate, learned trust gate, common atom/graph clipping, and the
existing global safety gate. Branch-specific geometric losses and cross-head
preservation are added to the existing unified BAC loss. This is not an
unnormalized sum of three heads.

Prototype B strict-loads the frozen D1 development checkpoint as its neural
Cartesian prior. At each of four inference steps, it builds the analytic BAC
system at the prior proposal, solves one float64 damped least-squares system,
applies truncated-SVD rank handling, removes rigid motion, clips the geometric
correction, and combines it with fixed lambda 1.0 before common trust and hard
safety. It adds no learned residual parameters and performs no training.

## Frozen comparison

All validity values are method minus source; more negative is better.

| Method | Bond | Angle | Active Angle | Ring | Acceptance | Rollback | Displacement (A) |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | -0.091570 | -0.002333 | -0.004498 | -0.008831 | 97.27% | 2.73% | 0.006701 |
| A multi-head | -0.096199 | -0.002385 | -0.004600 | -0.010421 | 97.75% | 2.25% | 0.005382 |
| B neural + Jacobian | -0.094410 | -0.002998 | -0.005717 | -0.011597 | 97.46% | 2.54% | 0.007393 |

Clash was effectively unchanged and chirality did not regress. COV-P and
COV-R deltas were zero for all methods.

| Method | aligned RMSD delta | MAT-P delta | MAT-R delta | Angle gain / displacement |
|---|---:|---:|---:|---:|
| D1 | +0.000411 | +0.000411 | +0.000719 | 0.671 |
| A multi-head | +0.000168 | +0.000168 | +0.000404 | 0.855 |
| B neural + Jacobian | +0.000406 | +0.000406 | +0.000717 | 0.773 |

Prototype A improves public RMSD/MAT values relative to D1 and uses only 80.3%
of D1 displacement. Prototype B is public-metric neutral relative to D1 and
uses 110.32% of D1 displacement.

## Paired evidence versus D1

Prototype A improves Bond by `-0.004629`, weighted BAC by `-0.010008`, Ring by
`-0.001590`, RMSD by `-0.000243`, and displacement by `-0.001319`. Its
active-Angle difference is only `-0.000102`, with paired 95% CI
`[-0.000543, +0.000283]`. The CI crosses zero, so A does not establish an
independent Angle advantage.

Prototype B improves Bond by `-0.002840`, weighted BAC by `-0.014478`, Ring by
`-0.002766`, and active Angle by `-0.001220`. The active-Angle paired 95% CI is
`[-0.001690, -0.000710]`, which is strictly below zero. Acceptance differs by
only `+0.20` percentage points and RMSD differs by `-0.000005`. Its only frozen
gate failure is movement: `1.1032x` D1 versus the unchanged `1.1x` limit. The
limit was not relaxed after observing the result.

## Prototype A mechanism audit

Across 4096 graph-steps, A allocates 74.5% of the constraint budget to Bond,
25.2% to Angle, and 0.02% to Clash. On Angle-active graphs, allocation is 55.2%
Bond and 44.7% Angle, but the mean Angle component RMS is only `0.000175`
versus Bond `0.004969`. Overall Angle component RMS is `0.000098` versus Bond
`0.004693`.

The measured fused field is therefore Bond-dominated despite nonzero Angle
allocation. A is a useful engineering control and improves D1 broadly, but the
evidence does not support a paper claim that separate heads discovered
distinct effective correction spaces. Clash has only one active development
record and remains statistically unsupported.

## Prototype B numerical audit

B made 4096 graph-step solver calls: 3022 solved systems and 1074 normal
no-active-constraint calls. There were zero true solver failures. Mean
effective rank was 3.46, mean condition number 6.49, maximum condition number
5107.06, maximum singular value 2.0276, minimum retained singular value
0.000181, and 15 singular directions were truncated. No inverse, arccos
derivative, NaN, Inf, or degenerate-geometry escape was used.

The significant active-Angle gain with preserved Bond, acceptance, Ring,
chirality, and public metrics is evidence that the Jacobian contributes a real
geometric advantage rather than simply increasing capacity. The gain does use
slightly more movement, so it is not yet a bounded production candidate.

## Research questions

### 1. Is multi-head more than an engineering split?

Not on this cohort. A is the strongest aggregate engineering result, but its
paired Angle gain is inconclusive and its actual correction field is dominated
by Bond. The extra heads and losses improve regularization and Bond repair;
they do not demonstrate three distinct learned correction spaces.

### 2. Does the Jacobian provide genuine geometric value?

Yes, at development scale. B improves active Angle beyond D1 with a strictly
negative paired CI, improves Bond and Ring, has zero solver failures, and keeps
RMSD/MAT/COV and acceptance effectively unchanged. Angle improvement per unit
movement rises from 0.671 for D1 to 0.773 for B, although A's Bond-dominated
control reaches 0.855.

### 3. Does neural prior plus Jacobian retain both capabilities?

Mostly. B preserves the D1 prior's Bond and acceptance behavior while adding a
statistically supported Angle correction. This fixes the major J0 failure:
Bond is `-0.094410` instead of J0 `-0.015871`, and acceptance is 97.46% instead
of 26.56%. It does not yet satisfy the fixed movement envelope.

## Paper direction

Prototype B is the better paper direction because it tests a representation
hypothesis and produces a mechanism-specific paired Angle gain without adding
learned capacity. Prototype A should remain the engineering control because it
has excellent aggregate metrics but weak evidence for constraint-space
specialization.

The recommendation is **continue only with a separately preregistered bounded
Jacobian-confidence or remaining-trust gate study on development data**. Do not
increase model width/layers, add a free Cartesian residual, tune lambda on the
current results, or proceed to 10k/formal-large/test yet. If the new bounded
study cannot retain B's Angle CI while meeting the original movement envelope,
the hybrid direction should stop.

## Runtime and reproducibility

A completed one 200-step smoke and one fixed 1000-step pilot. The pilot trained
for 1344.16 seconds and evaluated in 39.83 seconds; its checkpoint strict
roundtrip SHA256 is
`889aebef35fcfe2ed5724ee98c5b31c5808afaa7e6a9989d03126ce079725d37`.
B performed no training and evaluated the full cohort in 94.81 seconds using
frozen D1 SHA256
`9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`.

Machine-readable comparison, paired records, solver traces, component
diagnostics, smoke outputs, pilot outputs, configs, and run metadata are under
`diagnostics/ecir_mvr/v5_constraint_hybrid/`.

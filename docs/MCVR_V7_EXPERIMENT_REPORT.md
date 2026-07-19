# MCVR V7 Constraint-Specific Hybrid Experiment Report

## Decision

**V7 CONSTRAINT-SPECIFIC HYBRID IS SUPPORTED AT DEVELOPMENT SCALE.** The
single frozen inference-only candidate passes every preregistered gate. It
establishes a stronger active-Angle correction than both pure Cartesian D1
and combined-Jacobian V5-B while remaining inside the Bond, movement,
acceptance, Ring/chirality, RMSD, and COV safety envelopes.

This is evidence for operator-specific correction manifolds, not evidence
that V7 dominates V5-B on every objective. V5-B remains better on Bond, Ring,
and aggregate weighted BAC. V7 is better on Angle and uses less movement.

No training, sweep, checkpoint selection, learned gate, result-dependent
threshold change, test read, frozen-holdout read, or formal-large evaluation
was performed.

## Frozen scope and identity

The experiment uses seed 43018 and the same 512-molecule, 1024-record
development cohort as D1, V5-B, and V6. The manifest identity is
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.
All compared methods contain identical sample IDs and identical source
metrics.

V7 strict-loads the unchanged D1 checkpoint SHA256
`9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`.
It has no trainable parameters and produces no V7 checkpoint. Test records
read, test assets opened, and frozen-holdout records opened are all zero.

## Implemented operators

The Bond component is the frozen D1 raw Cartesian field with D1's original
graph/atom trust limits. The Angle component contains only active cosine-Angle
Jacobian rows and reuses float64 damped least squares, truncated SVD, rigid
motion removal, predicted-reduction checks, and fixed `0.01 A` graph-RMS /
`0.02 A` atom caps. The Clash component uses equal-and-opposite nonbond pair
repulsion and masks exactly coincident pairs to preserve rotation
equivariance.

Each component is independently trust-normalized. The three normalized
components enter one common D1 trust projection and the existing D1 global
safety gate. The evaluator then applies Bond, Angle, Clash, Ring, chirality,
identity, finite, and displacement checks with deterministic backtracking and
source rollback.

## Smoke

The unique 128-record smoke completed in 17.23 seconds with 98.44%
acceptance, active Angle `-0.007084`, Bond `-0.088119`, displacement
`0.007094 A`, and zero solver failures. The 128-record subset had no
Clash-active sample. The full development run was started only after the
smoke and the seven focused numerical tests passed.

## Frozen comparison

All validity values are method minus source; more negative is better.

| Method | Bond | Angle | Active Angle | Weighted BAC | Clash | Ring |
|---|---:|---:|---:|---:|---:|---:|
| D1 | -0.091570 | -0.002333 | -0.004498 | -0.168421 | -8.62e-11 | -0.008831 |
| V5-B | -0.094410 | -0.002998 | -0.005717 | -0.182899 | 0.00 | -0.011597 |
| V7 | -0.090199 | -0.003940 | -0.007205 | -0.171470 | -5.37e-09 | -0.009276 |

| Method | Acceptance | Rollback | Displacement (A) | RMSD | MAT-P | MAT-R | COV-P/R |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 97.27% | 2.73% | 0.006701 | +0.0004113 | +0.0004113 | +0.0007191 | 0 / 0 |
| V5-B | 97.46% | 2.54% | 0.007393 | +0.0004065 | +0.0004065 | +0.0007166 | 0 / 0 |
| V7 | 96.39% | 3.61% | 0.007093 | +0.0004125 | +0.0004125 | +0.0007118 | 0 / 0 |

Chirality delta is zero for all methods. V7 evaluation took 135.19 seconds;
there was no training time.

## Paired evidence versus D1

On all 1024 records, V7 improves Angle by `-0.001607`, paired 95% CI
`[-0.001987, -0.001242]`, and weighted BAC by `-0.003049`, CI
`[-0.005026, -0.001056]`. Ring differs by `-0.000445`, with a CI that crosses
zero. RMSD differs by only `+0.00000119`, and COV is unchanged.

On the 576 Angle-active records from 339 molecules, V7 improves Angle over D1
by `-0.002707`, CI `[-0.003300, -0.002118]`. This is the central success
criterion and is well separated from zero.

The tradeoff is a Bond difference of `+0.001370`, CI
`[+0.000392, +0.002353]`. Bond correction is statistically weaker than D1,
but the degradation is less than one third of the frozen `0.005` allowance.
Acceptance differs by `-0.88` percentage points, CI `[-1.86, 0.00]`, and
movement increases by `+0.000392 A`. V7 uses `1.0585x` D1 displacement,
inside the `1.1x` limit.

## Paired evidence versus V5-B

V7 further improves active Angle over V5-B by `-0.001487`, paired 95% CI
`[-0.002003, -0.001016]`. Across all records, Angle improves by `-0.000942`,
CI `[-0.001287, -0.000633]`, and displacement falls by `-0.000300 A`, CI
`[-0.000445, -0.000154]`.

V7 is not an aggregate replacement for V5-B. Bond is worse by `+0.004211`,
Ring by `+0.002321`, and weighted BAC by `+0.011429`; all three paired
intervals exclude zero. Acceptance is 1.07 percentage points lower. The
result is a clean operator tradeoff: Angle-only Jacobian specializes more
strongly, while V5-B's combined system provides broader Bond/Ring repair.

## Numerical and component audit

The Angle solver made 4096 graph-step calls: 2077 solved systems and 2019
normal no-active-constraint calls. True failures were zero. Mean effective
rank was 2.17, mean condition number 4.68, maximum condition number 2141.06,
maximum singular value 2.0391, minimum retained singular value 0.0003067, and
eight singular directions were truncated.

Mean component RMS values were Bond `0.005580 A`, Angle `0.001172 A`, Clash
`0.00000267 A`, and fused `0.005912 A`. Bond alpha was 1.0, mean Angle alpha
was 0.393 including inactive calls, and final fusion alpha was 0.99905. The
Angle operator therefore remains active without competing against a learned
Bond gate, while the common trust projection is almost never the limiting
factor.

Only one evaluation record is Clash-active. Across graph-steps the mean active
Clash-pair count is 0.00195 and there are zero degenerate pairs. The one record
improves Clash by `-5.50e-06` and is safely accepted, but one molecule cannot
support a general Clash claim.

## Research questions

### 1. Is constraint-specific correction better than unified Cartesian?

Yes for the preregistered mechanism claim. V7 produces a significant and much
larger active-Angle gain than D1 while retaining acceptable Bond, movement,
acceptance, Ring/chirality, RMSD, and COV behavior. It also modestly improves
weighted BAC over D1. It does not dominate every metric.

### 2. Should Jacobian be an Angle-specific operator?

The development evidence supports that role. Angle-only Jacobian beats both
D1 and V5-B on active Angle, has zero solver failures, stays inside the
movement envelope, and exposes a clear Bond-versus-Angle tradeoff rather than
hiding it in one combined correction.

### 3. Does Clash need an independent spatial operator?

The representation is technically appropriate and numerically safe, but the
current cohort cannot answer the empirical question. There is only one
Clash-active molecule. A Clash-enriched development cohort would be required
before claiming benefit, without using test or frozen holdout.

### 4. Is V7 a better paper direction than V6 adaptive gating?

Yes. V6's learned gate suppressed the analytic Angle signal and failed its
central criterion. V7 removes that competition and obtains a significant
Angle advantage with a fixed, interpretable operator decomposition. The
result directly supports the representation hypothesis and is a stronger
paper mechanism than another gate-tuning study.

## Scale recommendation

V7 is eligible for the next preregistered 10k development-scale validation.
The recommended sequence is 10k first, then formal-large only if the same
Angle CI, Bond margin, movement, acceptance, Ring/chirality, and public-metric
gates remain satisfied. Do not read test or frozen holdout, and do not tune
the fixed component caps from this result.

Machine-readable summary, paired records/molecules, resolved configs,
component traces, Angle solver traces, smoke outputs, and full development
outputs are under `diagnostics/ecir_mvr/v7_constraint_specific/`.

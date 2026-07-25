# BAT-Steric-Torsion final summary

## Decision

**KEEP_LSGO_B**

The prospective experiment replicates frozen LSGO-B but does not validate either new module strongly enough to replace it.

## 1. Takeover state

- Already completed at takeover: isolated branch/worktree, scientific inheritance audits, clash design, and one audit commit.
- Uncommitted at takeover: config, dataset builder, BAT primitive implementation, and initial identity artifacts.
- Found and corrected: post-hoc σ role ambiguity, symmetric rotor audit ranks, 2,002 amide-like restricted rotor definitions, one smoke import typo, continuous-versus-binary steric gate wording, and the post-external BA+C backtracking attribution confound.
- Preserved: frozen LSGO-B checkpoints and mathematics, all historical results, the old external exclusions, 0.003 Å trust cap, and protected-split isolation.

## 2. Sigma

- Final σ source: exact frozen DRCSR/reference scales inherited by LSGO-B for coordinate inference.
- Learned: no.
- Calibration: TRAIN residual MAD plus DEV_A/DEV_B z diagnostics, post-hoc reporting only; it never changes BA coordinates.
- Inflation: impossible in the selected path because there is no learned σ. Heavy residual tails remain explicitly reported.

## 3. BA+C

- Internal continuous penetration reduction: 87.07% DEV_A and 12.40% DEV_B; binary counts stayed 3→3 and 12→12.
- Fresh frozen-rule diagnostic: Source 13, BA 12, BA+C 10 violating pairs per seed; summed penetration fell about 6.37% with BA+C.
- RMS: 0.002869–0.002879 Å mean; p95/p99 0.003 Å.
- Reference diagnostic: median/p95 0.001 Å at the historical diagnostic budget.
- Safety: topology/chirality 100%; mode switch 0%; no new catastrophic clash.
- External GO: **no**. PoseBusters clash and overall were unchanged at 92.333%, with zero fail→pass.

## 4. Torsion

- Canonicalization: GO after explicit amide/sulfonamide-like exclusion; 497,160 Reference angles audited, zero reversal failures, maximum circular reversal error 8.88e-16.
- Single VM: DEV NLL 7.04–7.21.
- K3 fixed VM: DEV NLL 1.454–1.483 versus uniform 1.8379.
- κ: TRAIN/DEV-calibrated fixed values 7.137–7.243; κ=8 was only the initial center-learning smoke value.
- Collapse/duplication: no collapse; duplication fractions 0.060–0.108.
- Source selectivity: 0.502–0.534 Source>Reference molecule fraction, below the frozen 0.55 gate in all six seed/split cells.
- GO: **TORSION_NO_GO**; no formal torsion training or external torsion coordinates were run.

## 5. Formal

- Final formal variant evaluated: BA+C.
- BA seeds: 173/181/193; T seeds 211/223/239 were internal-only.
- BA train size/steps/parameters: inherited frozen LSGO-B, 2,000 TRAIN molecules, 2,500 steps, 473,674 parameters per seed.
- Added formal train size/steps/parameters: 0/0/0 because C is analytic and T stopped.
- Frozen BA checkpoint SHA256: `f382e261224609fa7eea8e05e67e8600a04674bac09a7031d539a327b8aec62e`, `c0f458aada6eca7726269a327e8d5c11fd18af6c0695be0c812f632a78ac6ed7`, `e3606d683921bfb9aec1afa41e2e9835356196e7615b8dd530b0ac9d96240ead`.

## 6. Fresh external

The prospective cohort contains 200 molecules/600 Source records and has zero molecule overlap with the 1,200-identity exposed union.

- PoseBusters: Source, BA, and BA+C all 92.333% overall and clash pass; every transition is zero.
- Fresh BA xTB: median -0.9047±0.0012 kcal/mol, mean -0.9430±0.0026, improved 93.50%±0.29%, p95/p99 0/0, max harmful +0.0689±0.0048.
- BA+C xTB: median -0.9154±0.0023, mean -0.9745±0.0020, improved 99.11%±0.10%, but this is confounded by combined-solver backtracking on non-steric-active records.
- xTB execution: 4,200/4,200 success; no optimization, failure, timeout, or nonfinite result.
- Fidelity: mean RMS ≤0.002879 Å, p99 0.003 Å, topology/chirality 100%, mode switch 0%.

## 7. Why the decision is KEEP_LSGO_B

LSGO-B replicated on genuinely fresh identities and remains the best supported minimum method. C changes the internal distance-barrier diagnostic but not the independent PB clash outcome. T models Reference multimodality but does not reliably distinguish Source error. The apparent BA+C xTB increment cannot rescue C because it is partly a backtracking effect and the decisive PB condition fails.

This experiment proves: frozen neural conditional BA micro-refinement prospectively replicates a small-displacement, tail-safe GFN2-xTB improvement, and the tested K3/reference and steric mechanisms can be evaluated without leakage.

This experiment does NOT prove: that the proposed steric barrier repairs community-defined clashes, that torsion surprise predicts upstream error, or that BA+C/BAT+C should replace frozen LSGO-B.

Formal test reads=0; frozen holdout reads=0.

Verification: 125 BAT/LSGO/V8-clash/chirality/kinematics/Jacobian regression tests passed; every external manifest reports protected reads as zero.

# Final P0 / P1 / P2 evidence gaps

## P0 — close before submission

1. **Repair and rerun the current-final MMFF94s baseline.** Root cause is known: reconstructed MMFF records omit `num_atoms`, causing `KeyError:'num_atoms'` for all 5,000 records before optimization. Add the field, smoke-test identity/atom order, then run exactly 5,000 MMFF94s optimizations plus frozen V3D/PB/RMSD and GFN2 single-point scoring. No replacement molecules.
2. **Close primary statistical reporting from frozen outputs.** Add direct paired molecule-cluster CIs for primary Unrestricted versus Source and Restricted versus Unrestricted, plus win/tie/loss and molecule-effect distributions. No model rerun is needed.
3. **Correct cross-upstream inference and safety reporting.** Add paired molecule-cluster CIs; predeclare DiTMC matched-finite/failure handling; disclose all xTB denominators; replace unconditional `TRANSFER_SUPPORTED` with a metric- and upstream-specific conclusion that includes AvgFlow energy regression and DiTMC seed307 movement explosion.
4. **Freeze a coherent final method role and release snapshot.** Reconcile the earlier two-operating-point freeze with the later single-primary Unrestricted label without using the already-read prospective outcome. Commit/tag the exact active code and small authoritative reports; hash-bind external records/checkpoints/results and document recovery supersessions.

## P1 — strongly recommended

1. Run the minimal fair current-final runtime benchmark (SIXS GPU, repaired MMFF CPU, and xTB CPU if optimization is included).
2. Add a GFN2-xTB geometry-optimization comparator on a predeclared matched subset, or explicitly narrow the paper so no comparison/superiority to physics-based optimization is claimed.
3. Add a clearly labeled non-executable Reference contextual row under the exact frozen RMSD/identity protocol.
4. Derive final tau P95, Source-RMSD P95/P99, explicit finite-coordinate rates, and molecule-level effect plots from existing per-record outputs.
5. Perform a literature/claim audit before any “first,” SOTA, universal, or uncertainty-calibration language.

## P2 — optional

- MD/MM-PBSA or docking follow-up.
- Additional upstream generators beyond ETFlow, AvgFlow, and DiTMC.
- A dedicated sigma-calibration experiment, provided the paper currently uses only non-calibrated scale language.
- Full-cohort rather than deterministic-subset GFN2-xTB optimization, if cost is prohibitive and the claim is narrow.
- New model designs, ratio sweeps, or more matched ablation training; current core ablation is closed.

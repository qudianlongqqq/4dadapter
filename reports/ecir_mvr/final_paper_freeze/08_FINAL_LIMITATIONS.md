# Final limitations

1. Global Reference conformer recovery is not supported: all three Unrestricted seeds slightly worsen all-atom nearest-ensemble Reference RMSD even while local Bond and Angle errors improve.
2. AvgFlow energy does not improve under the primary median statistic despite improved V3D; validity and energy responses must remain separate.
3. DiTMC Unrestricted movement has a verified instability tail, concentrated in seed307 and consistent with the serialized coordinates rather than a reporting-unit error.
4. The Reference row is contextual, not an executable method and not a theoretical upper bound. Reference-vs-itself COV/AMR is not reported without an independent held-out Reference split.
5. SIXS is a learned local fixed-topology refiner, not a physical energy optimizer.
6. Results do not establish a universal upstream guarantee: AvgFlow and DiTMC have metric-specific, materially different behavior.
7. No per-sample xTB energy monotonicity is claimed; all energy denominators and failures remain explicit.
8. No state-of-the-art, first-method, universal, or calibrated-uncertainty claim is made without a separate literature/claim audit.
9. Fixed-sigma-action and fixed-tau are inference/action ablations; they do not by themselves establish that sigma or tau must be trained in a particular way.
10. Current-final COV/AMR is not reported because no frozen threshold and independent Reference self split exist for that cohort.

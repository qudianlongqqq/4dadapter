# LSGO final summary

## Decision

**LSGO_DECISION = CONDITIONAL_GO**

Neural conditional mean B is supported; learned uncertainty/precision C is not. B produces a prospective, three-seed, tail-safe xTB improvement with no PB or fidelity regression, but the full μ+σ/LPWGP hypothesis and amortization gate fail.

## Required 33 answers

1. **Yes.** V8 directly supervises Source→frozen MVT coordinates/error delta; see `HISTORICAL_OVERLAP_AUDIT.md`.

2. MVT uses manually chosen objective terms/weights, trust limits, ranking terms and a 40-step Adam coordinate optimizer; the exact audited formal-large values are in the historical audit. The often-repeated 10/20/40 values are not the actual frozen formal-large config.

3. DRCSR removed the coordinate teacher and replaced single global targets with frozen train-Reference typed medians/scales, hierarchical backoff, Student/Gaussian strain, group normalization and smooth-max.

4. DRCSR still handcrafts context buckets/backoff, distribution choice, grouping/normalization, smooth-max and safety/trust rules.

5. **Yes, narrowly and reproducibly.** B joint NLL improves on both DEV_A (0.5736) and DEV_B (0.1566) for all three seeds.

6. **No.** C is `SIGMA_INFLATION` for seeds 173/181/193 and fails to beat DRCSR on DEV_B.

7. Solver numerics are stable, but learned precision is scientifically invalid because σ is pathological; these are different claims.

8. **Yes.** No fixed Bond×10/Angle×20/Ring×40 BAC weights are used. Bond and Angle group means are aggregated equally; that remaining aggregation choice is disclosed.

9. **Yes.** Training sees only train Reference local Bond/Angle values; no MVT coordinate, MVT delta/trajectory, or Source→Reference Cartesian delta is accessed.

10. Approximately yes: Reference/Source gradient-RMS ratio is 0.466–0.513, 0.001 Å Reference micro-steps never increase the objective, and the stationarity gate passes.

11. **Yes.** Source objective exceeds Reference in 95.8–97.9% of audited molecules, with positive median gaps 0.229–0.246.

12. Qualified yes for B: learned-μ/frozen-σ z-scores separate real Source from Reference. No for C's learned-σ z-score because σ inflates.

13. C-G is not eligible and was stopped; it cannot be claimed stronger than DRCSR.

14. No. C-P has weaker objective changes/high fallback internally and was stopped before external evaluation.

15. Numerically yes: all Jacobian evaluations are finite; p95 condition numbers are ≤5.46×10^5 under the preregistered 10^8 ceiling.

16. Small singular values (~1.05×10^-5 minimum) exist, but ridge/SVD fallback/trust caps prevent large Cartesian deltas; no nonfinite or blow-up occurred.

17. PB is unchanged: Source/A/B overall=89.833%, pass→fail=0, fail→pass=0. It does not improve PB.

18. **Yes.** B median ΔE=-0.8045±0.0050 kcal/mol across seeds.

19. **Yes on this cohort.** Worst B p95/p99/max=0.0000/0.0000/0.0948 kcal/mol.

20. **Yes.** Mean RMS=0.002815 Å and every graph RMS≤0.003 Å.

21. **Yes.** Worst nearest-Reference mode-switch rate=0.167% and diversity retention=99.926%.

22. **Yes for B.** Three fresh seeds have median-energy sample SD=0.0050 kcal/mol and identical PB.

23. **Yes for neural μ:** same-identity likelihood and xTB are better than DRCSR A. **No for neural σ/precision.**

24. Yes: B's prospective median/mean/tail signal is at least comparable in direction to frozen Gaussian/SPSP signals, but identities differ, so no direct superiority claim is allowed.

25. **Not yet.** Direct one-step B-G is already effective; full GO_FOR_AMORTIZATION fails because C/precision fails.

26. **Not yet.** The preregistered rule permits Ring mixture only after successful Variant C.

27. Possibly, but only as a new independent preregistered experiment; it was not tested here.

28. **Yes.** V8 remains the frozen main model for the original BAC/minimal-validity task; LSGO does not retroactively replace it.

29. **Conditional.** Neural conditional geometry means are a credible teacher-free direction; learned uncertainty is not established.

30. formal test reads = **0**.

31. frozen holdout reads = **0**.

32. FULL10K used for tuning = **false**.

33. Branch `research/learned-structured-geometry-objective`; prereg HEAD `fbde3ef497364cf92aa90caba8e62e25a33e2e18`; 77 targeted+related tests passed. Commands and the one operational DEV_B resume deviation are recorded below; final HEAD is the commit containing this report and is reported at handoff.

## Commands and tests

```text
python scripts/build_lsgo_datasets.py
python scripts/run_mcvr_lsgo.py --phase prepare/preregister/smoke/train/internal-finalize
python scripts/run_lsgo_external_coordinates.py
external-validity python scripts/run_posebusters_lsgo.py
python scripts/run_xtb_singlepoint_lsgo.py
python -m pytest -q tests/test_mcvr_lsgo.py [related DRCSR/V8/Jacobian/kinematics tests]
python -m py_compile ...
git diff --check
```

Protocol deviation: the first internal-finalize process ended after a complete DEV_A direction grid because DEV_B redundantly evaluated all three budgets. The resume reused the SHA/order/schema-checked DEV_A grid and evaluated only the already-selected 0.003 Å DEV_B budget. No method math, data, checkpoint, seed, selection rule, or external access changed. See `PROTOCOL_DEVIATIONS.md`.

# Claim safety

## Energy and movement

```text
ENERGY_CLAIM_SAFETY = PASS_IF_QUALIFIED
MOVEMENT_TAIL_REPORTING = PARTIAL
```

On the prospective final cohort, Unrestricted has V3D `0.5032 ± 0.00282` across seeds and GFN2 single-point median delta-E `-3.5968 ± 0.0260 kcal/mol`; lower-energy fractions are 0.9988/0.9990/0.9988. This supports a strong central-tendency energy association, not a guarantee. Positive tails remain: per seed, 4–6 records exceed +25, 4–5 exceed +50, and 4–5 exceed +100 kcal/mol.

Prospective Unrestricted tau medians are 0.00883/0.00937/0.00920 Å, P99 values 0.02059/0.02037/0.02040 Å, and maxima 0.2213/0.1357/0.0655 Å. Source-RMSD means are 0.00946/0.00996/0.00979 Å. Tau P95, Source-RMSD P95/P99, and an explicit finite-coordinate-rate column are not in the final summary and should be derived from already frozen per-record outputs.

AvgFlow shows a V3D gain but positive xTB median delta-E (`+0.2817 kcal/mol`) and only 1.7% lower-energy records. DiTMC seed307 has a catastrophic movement tail (`tau_max=398.680 Å`, tau P99=201.284 Å). These results rule out unconditional universal-transfer or universal-energy-improvement claims.

Do not claim per-sample energy monotonicity, physical-energy optimization, force-field replacement, xTB-optimization superiority, or guaranteed safe movement.

## Reference metric

```text
REFERENCE_PROTOCOL = PASS
```

The frozen definition is explicit all-atom, fixed atom order, float64 proper-rotation Kabsch RMSD, minimized over the frozen reference ensemble. There is no symmetry permutation. Therefore `symmetry-aware RMSD` is forbidden wording. Reference is a contextual target ensemble, not an executable method. Reference numbers from development, prospective ETFlow, AvgFlow, and DiTMC cohorts must remain in separate tables.

## Novelty language

```text
TO_OUR_KNOWLEDGE_FIRST_CLAIM_SAFE = NEEDS_LITERATURE_AUDIT
```

The repository does not contain a comprehensive literature audit sufficient for “first” or SOTA claims. Avoid: “first post-hoc conformer refinement,” “first use of mu/sigma,” “first internal-coordinate Cartesian pullback,” “first uncertainty-aware conformer model,” “SOTA,” “universal model-agnostic,” “energy monotonic,” and “physics optimizer replacement.”

A safe core claim is: predictive geometry is not itself executable geometric action; SIXS factorizes post-generation refinement into predictive geometry, predictive scale, source-conditioned action reliability, Bond/Angle allocation, Cartesian direction, and movement magnitude. Describe sigma as a predicted heteroscedastic scale, not calibrated uncertainty.

## Downstream MD/MM-PBSA

```text
MD_MMPBSA_FOR_MAIN_CLAIM = OPTIONAL
```

MD/MM-PBSA answers downstream receptor/complex stability questions that are outside the current fixed-topology local-refinement claim. It is not a submission blocker. If ever added, the scientific question should be whether small conformer corrections improve downstream complex stability under a separately frozen protocol—not whether they retroactively validate V3D.

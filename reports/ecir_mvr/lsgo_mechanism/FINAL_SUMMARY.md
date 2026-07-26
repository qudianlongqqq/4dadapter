# LSGO-B Mechanism & Sufficiency Audit — final summary

## Outcome

The preregistered decisions are:

```text
BA_LOCAL_STRAIN_CONFIRMED
B_DOMINANT
TORSION_LOW_ACTIONABILITY
KEEP_BA_ALL
```

This is evidence about the current upstream Source distribution under a source-preserving `0.003 Å` micro-refinement budget. It is not a claim that Bond and Angle are sufficient for all conformational energetics, nor that torsion is intrinsically unimportant.

## 1–3. Source, Reference ensemble and imitation

1. **Is Source often below a Reference?** Sometimes, but not usually. Source is below the Reference ensemble minimum for 0.69% of records (molecule-bootstrap 95% CI 0–2.08%), below the median for 12.50% (6.23–20.14%), and below the aligned-RMSD-nearest Reference for 6.25% (2.08–11.11%). Thus a Reference is not automatically lower energy and is not treated as unique truth.
2. **When Source is below the Reference median, does LSGO-B still lower energy?** Yes: seed181 improves 94.44% of those 18 records, with median ΔE `−0.4132 kcal/mol` and p95 `−0.2349`. For Source above the median it improves 91.13%, median `−0.7170`.
3. **Is LSGO-B Reference-coordinate imitation?** The evidence says no. Its network learned Reference-conditioned local BA statistics, but the frozen direct update can lower Source energy even when Source is already lower than the sampled Reference ensemble, without moving toward a selected Reference coordinate or switching the aligned-RMSD-nearest mode.

## 4–6. B-only, A-only and BA

All variants use the same frozen heads, Source, one-step solver, normalization convention and trust/safety budgets; only primitive-family participation changes.

| diagnostic | median ΔE across seed medians (kcal/mol) | fraction of BA gain |
|---|---:|---:|
| B-only | −0.6408 | 97.22% |
| A-only | −0.1847 | 28.03% |
| BA | −0.6591 | 100% |

4. **Bond-only contribution:** B-only is the dominant energetic mechanism. Across individual seeds its median ΔE is `−0.6238` to `−0.6414`, with 88.19–88.89% improved and p95/p99 no worse than zero (tiny maxima ≤0.0035 arise in two seeds).
5. **Angle-only contribution:** A-only is beneficial but smaller: medians `−0.1835` to `−0.1974`, 75.0–76.39% improved, with positive p95 and maxima 0.258–0.349 kcal/mol.
6. **BA synergy:** No preregistered synergy. BA adds only 2.86% median gain above the best single family, below the frozen 10% threshold. The correct label is `B_DOMINANT`, not `BA_SYNERGISTIC`. Angle remains useful as part of the guarded BA formulation but is not the main xTB-energy contributor here.

## 7–8. Abnormality and chemical context

7. **Does initial abnormality predict energy gain?** Bond abnormality has a weak but positive rank association: Spearman `ρ=0.2559`, cluster-bootstrap CI `[0.0658, 0.4532]`. Angle (`−0.0417`, CI crossing zero) and combined BA (`0.0708`, CI crossing zero) are not useful record-level gain predictors. High/low BA alone is therefore not a reliable no-op detector.
8. **Which contexts benefit most?** The preregistered high-flex bin has median ΔE `−0.9249` and 97.92% improved, versus `−0.6319`/91.67% for medium and `−0.4722`/85.42% for low flexibility. Molecules with >35 heavy atoms show a larger descriptive median gain (`3.965 kcal/mol`) but only three records, so this is low-power. Double-bond primitives have the largest mean initial primitive `|z|` among bond types (`0.925`), while the benefit is broadly present across single/aromatic/double and common hybridization contexts. Ring/non-ring and aromatic/non-aromatic contrasts are imbalanced and descriptive only.

## 9–13. xTB force and torsion

The xTB gradient interface was independently validated: single-point energy identity error 0 Eh, repeat error 0 Eh/bohr, coordinate-order error `2.55×10⁻¹¹ bohr`, and three central-difference relative errors `4.15×10⁻⁵`–`7.21×10⁻⁵`. Force results therefore use 18 preregistered Source records and the same 18 seed181 BA outputs.

9. **How much xTB-force direction lies in BA?** Before BA, median independent projection-norm fractions are B `0.9734`, A `0.8234`, T `0.0974`; the joint BA union is `0.99998`. These are kinematic subspace projections, not claims that the learned objective exactly predicts xTB force. The absolute median internal force/BA projection falls from `0.06394` to `0.04760 Eh/bohr` after the tiny BA update (about 25.5%), consistent with removal of locally actionable strain.
10. **How much remaining force is uniquely torsional?** Although independent T projection is 0.1177 after BA (0.1943 in high-flex), T overlaps the BA space. Its incremental BAT-union fraction is only `3.00×10⁻⁵` after BA; before BA it is `1.23×10⁻⁵` overall and `1.97×10⁻⁴` in high-flex. These are far below the preregistered 0.05/0.10 actionability thresholds.
11. **Does torsion surprise track the physical torsion component?** Not usefully. Surprise has weak positive associations with Source/Reference median excess (`ρ=0.2823`) and remaining BA excess (`0.2780`), but negative associations with independent T fraction (`−0.5108`) and incremental BAT fraction (`−0.3891`) on only 18 force records. Since the unique torsional fraction is itself near zero, those small-subset correlations do not validate the old detector.
12. **What did the prior torsion result mean?** The frozen K3 Reference distribution learner remains a likelihood success. On this Source and micro-budget, however, the physically unique actionable torsional residual is low. The conclusion is `TORSION_LOW_ACTIONABILITY`, not “torsion is useless” and not evidence for reviving the old Reference-marginal detector.
13. **Is high-flex different?** It benefits more energetically from BA and shows a larger independent T fraction, but T adds only `0.000197` beyond the BA union before refinement—still about 500× below the high-flex actionability threshold. High-flex therefore does not change the torsion decision.

## 14–18. Gate, Clash and formal method

14. **Should BA-normal Source no-op?** No. Abnormal-only produced zero exact no-ops, reduced mean movement by only 1.52% (required 30%), and retained only 47.63% of median BA benefit (required 80%). Its p99 worsened by 0.0182 kcal/mol overall and by 0.0470 in the already-lower-than-Reference subset. Decision: `KEEP_BA_ALL`.
15. **Clash position:** Keep only the existing hard steric do-no-harm guard. No soft clash training or PoseBusters rescue was attempted; all accepted/output coordinates preserve chirality, no mode switch occurs, and the guard introduces no catastrophic-clash regression. Ring safety causes some candidate fallbacks, which become exact Source no-ops.
16. **Modify the current formal method?** No. The formal method remains `LSGO-B + topology + chirality + hard-steric + ring safety guards`. The experiment explains why it works; it does not justify a new torsion module or an abnormality gate.
17. **Formal test reads:** `0`.
18. **Frozen holdout reads:** `0`.

## Mechanistic interpretation

Across two historical prospective cohorts and this identity-disjoint mechanism cohort, the same tiny frozen BA update is directionally consistent and tail-safe. The new evidence narrows the explanation: current upstream conformers retain transferable local strain, predominantly visible in Bond directions; removing a small portion of that strain lowers xTB energy without Reference-coordinate imitation. BA's joint kinematic span also covers essentially all observed xTB gradient direction in the frozen force subset, while adding canonical free-torsion rows contributes almost no new direction. This supports `BA_LOCAL_STRAIN_CONFIRMED` only for the present Source distribution and trust-region regime.

GFN2-xTB was used only as a frozen-coordinate diagnostic. It was never a training teacher, loss, optimizer, checkpoint selector, coordinate optimizer or threshold tuner.

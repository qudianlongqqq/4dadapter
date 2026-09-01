# Final formulation assessment

## Evidence

| Consideration | Restricted | Unrestricted | Interpretation |
|---|---:|---:|---|
| V3D, mean ± seed SD | 0.5645 ± 0.0017 | 0.5681 ± 0.0011 | Consistent Unrestricted advantage at all three seeds |
| PB | 0.93233 ± 0.00012 | 0.93240 ± 0.00000 | Practical tie |
| Reference RMSD | 1.319641 ± 0.000003 | 1.319639 ± 0.000011 | Practical tie; not global recovery |
| Proposal displacement | 0.005550 ± 0.000032 Å | 0.006164 ± 0.000098 Å | Restricted preserves Source more closely |
| xTB median delta-energy | lower than Source at all seeds | lower than Restricted at all seeds | Central tendency favors Unrestricted |
| Extreme tails | present | slightly heavier in two seeds | Safety trade-off favors Restricted |
| Human constants | tau ceiling, atom cap, regularizer | none of these | Unrestricted is simpler |

## Options

**A. Restricted only.** Defensible if the paper prioritizes bounded movement and Source fidelity. It discards a replicated V3D/xTB central-tendency advantage.

**B. Unrestricted only.** Defensible if the paper prioritizes simplicity and average local refinement. It understates the Source-fidelity and rare-tail cost.

**C. Two predeclared operating points.** Best supported. Present Unrestricted as the simplicity/performance operating point and Restricted as the conservative bounded-movement operating point. Neither may be selected post hoc using protected outcomes.

The evidence establishes a reproducible Pareto trade-off but not a unique winner. The two formulations share the same scientific core; their difference is an inference/training movement policy rather than two unrelated methods.

```text
FINAL_SINGLE_FORMULATION_SELECTION = NOT_JUSTIFIED
FORMULATION_DECISION_COMPLETE = YES
FINAL_PRESENTATION = TWO_PREDECLARED_OPERATING_POINTS
PRIMARY_NARRATIVE = UNRESTRICTED_SIMPLICITY_AND_CENTRAL_PERFORMANCE__RESTRICTED_CONSERVATIVE_SOURCE_FIDELITY
PROTECTED_OUTCOME_WINNER_SELECTION_ALLOWED = NO
```

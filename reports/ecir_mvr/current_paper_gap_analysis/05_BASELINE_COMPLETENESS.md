# Baseline completeness

## Decision

```text
BASELINE_COVERAGE = MAJOR_GAPS
MUST_HAVE_MISSING_BASELINES = CURRENT_CANDIDATE_MATCHED_MMFF94S_AND_GFN2_XTB_OPTIMIZATION_OR_AN_EXPLICITLY_JUSTIFIED_SUBSET
```

| baseline | current status | requirement |
|---|---|---|
| Source / no refinement | available on the exact current cohort and all main endpoints | MUST compare |
| Reference | available as evaluation target/ensemble, not an inference baseline | MUST contextualize; do not call it an attainable method |
| MMFF94s optimization | historical 10k MCVR comparison exists, but not matched to the current J1-R1 Full Joint candidates | MUST HAVE for a practical learned-refinement claim, or explicitly narrow the paper |
| GFN2-xTB optimization | historical 10k MCVR optimization exists; current candidate has only GFN2 single-point scoring | MUST HAVE if claiming an alternative to physics-based optimization; at minimum provide a matched, protocol-consistent subset |
| Other learned refiner | no current matched candidate found | REASONABLE if a genuinely comparable public method exists; not mandatory merely because a paper exists |
| Upstream-specific refinement baseline | none for current checkpoints | needed only for cross-upstream/general-framework claims |

Historical external-baseline artifacts are valuable protocol evidence: they show feasible MMFF94s/GFN2-xTB execution, all-record fallback semantics, runtime, and failure categories. They cannot by themselves establish that the current Full Joint model outperforms those baselines because the model, cohort, and evaluation identity differ.

The minimum defensible paper comparison is Source, current frozen candidate, and at least one strong conventional optimizer on the same records/evaluator. Both MMFF94s and GFN2-xTB are strongly preferred because they answer different reviewer questions: cheap force-field refinement versus expensive quantum/semiempirical optimization.

Coverage/diversity baselines are unnecessary if the claim stays at per-conformer post-generation refinement. They become relevant only if the paper claims set-level conformer generation improvement.


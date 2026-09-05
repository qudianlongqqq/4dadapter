# Reproducibility and provenance readiness

```text
PROVENANCE_READINESS = PARTIAL
HEAD = 1ebc2907ec7520e1cc9b69e1ad26eac1f4d7f4a6
BRANCH = experiment/sixs-musigma-reliability-factorial
WORKTREE_CLEAN = NO
```

## Closed items

- The prospective cohort, source assets, protocol, evaluators, configs, six checkpoints, environment, seeds, step 17,500, optimizer/scheduler recipe, and failure policies have explicit SHA256/version records.
- Checkpoint hashes for Unrestricted 307/331/353 are `f63e…b704`, `b9d6…86af`, and `e3be…0bc6` in the cross-upstream checkpoint manifest.
- Final cohort membership and source generation are frozen before outcome inspection.
- CUDA/PyTorch/RDKit/xTB versions and GenBench3D commit are recorded.
- Cross-upstream preflight records zero-shot execution, no retraining, no best-seed selection, and identical final checkpoints.

## Open items

- The current final-primary, cross-upstream, matched-ablation, and many run-local artifacts are untracked at HEAD. Internal artifact hashes help integrity but do not replace a clean, immutable release snapshot.
- The development freeze says two predeclared operating points and no justified unique winner; later artifacts call Unrestricted the final primary. This role change needs a single authoritative, outcome-independent release statement.
- Historical seed307 executed-runner provenance remains `PARTIAL`; forward reproducibility is stronger than historical byte-for-byte reconstruction.
- `sixs_final_cross_upstream_unrestricted/RUN_STATUS.json` preserves an old `tabulate` aggregation failure while `FINAL_STATUS.json` is PASS. The supersession chain should be documented in the release manifest.
- The primary MMFF implementation hash points to an invalid invocation path caused by a missing record field, so a repaired baseline needs a new code hash and a protocol-preserving repair note.

## Engineering fixes

- PowerShell UTF-8 BOM tolerance/status writing: `ENGINEERING_ONLY`.
- Markdown/table rendering without installing `tabulate`: `ENGINEERING_ONLY`.
- Kabsch helper replacement after a Windows LAPACK-loader crash: protocol manifest records mathematical equivalence and pre-outcome rebinding; `ENGINEERING_ONLY_WITH_PREOUTCOME_HASH_REBIND`.
- Adding the omitted `num_atoms` reconstruction field for MMFF: engineering root-cause repair, but it enables previously absent scientific baseline results. Therefore the repaired code must be smoke-tested and all 5,000 MMFF scientific jobs rerun; old fallback rows cannot be reused.

Submission release closure requires a clean tagged commit (or explicit whitelist commit), final result-manifest SHA256s, external-asset hash manifest, and a short supersession ledger for failed/recovered orchestration states.

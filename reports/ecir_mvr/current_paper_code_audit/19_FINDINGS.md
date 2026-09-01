# Findings by severity

## CRITICAL

None.

`NO EVIDENCE OF`: train/DEV molecule leakage, Reference leakage, outcome-dependent xTB exclusion, metric denominator manipulation, or a hidden alternative scientific evaluator in completed seed307.

## HIGH

### H1 — Current paper-path code and results have no Git commit identity

All current core modules, configs, runners, artifacts, and reports are untracked relative to HEAD `9be3633...`. The HEAD predates the current full-joint work. This prevents commit-level reconstruction.

Evidence: `git status --porcelain`, `git log`, repository identity audit.

### H2 — Current runner bytes differ from seed307 executed runner bytes

Restricted executed/current runner hashes are `c8b336...` versus `81bb20...`; Unrestricted are `eee8c642...` versus `37f932...`. Core modules and configs match, but the exact executed runner source is not archived or committed.

Evidence: both `IMPLEMENTATION_CONFIG.json` files and current `Get-FileHash`.

Impact: exact reproducibility/provenance risk; no direct evidence that frozen scientific numbers are wrong.

## MEDIUM

1. **Seed evidence incomplete.** Only seed307 has complete Restricted and Unrestricted scientific results. Multiseed is running and unaudited.
2. **DEV adaptation bias.** The 2,500-molecule/5,000-record cohort is a development set used through the design chain, not an unbiased final test.
3. **Generalization boundary.** Current evidence supports unseen molecules within the same ETFlow source distribution; cross-upstream and protected outcomes remain unverified.
4. **Source RMSD naming ambiguity.** Raw displacement and Kabsch-aligned proposal-to-source RMSD share the same label in different artifacts.
5. **Objective coefficient evidence is incomplete.** The later forensic audit invalidates the earlier global `alpha_grad`; 1:1 remains a historical design choice with medium sensitivity priority.
6. **VJP terminology risk.** Current action code implements analytic derivative accumulation, not an explicit autograd VJP.

## LOW

1. Restricted seed307 lacks the standalone GPU verification artifact preserved by Unrestricted; exact environment linkage is partial.
2. Reference RMSD does not search symmetry-equivalent atom permutations.
3. Restricted per-record `rollback` is a diagnostic safety flag; output coordinates are not rolled back. The name can mislead if presented without qualification.

## INFORMATIONAL

- Step22,500 is a valid capacity diagnostic but not the selected endpoint.
- xTB seed307 comparisons have full 5,000/5,000 success denominators.
- No manuscript corpus was available, so paper-text-specific claims are not verified.



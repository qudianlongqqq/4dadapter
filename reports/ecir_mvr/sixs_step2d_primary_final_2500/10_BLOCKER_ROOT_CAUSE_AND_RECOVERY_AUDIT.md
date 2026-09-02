# STEP 2D blocker root-cause and recovery audit

## Frozen scientific domain

The frozen design selects from the entire eligible unused GEOM-DRUGS
universe, not from the exact native test split only.

Evidence:

- `prospective_final_cohort_freeze/03_ELIGIBILITY_RULE.md` defines nine
  input-only gates and contains no native-test membership gate (SHA256
  `36d8edc3114de7976829b890c698b8056091b9d27cae1c545f9bfc5aa2c8ffab`).
- `prospective_final_cohort_freeze/05_DETERMINISTIC_SELECTION_PROTOCOL.md`
  says to apply that rule to the immutable source-universe index and take the
  first 2,500 eligible identities (SHA256
  `4299d6d3c72cf3c7dd8aee197ba4319ae89ae844ea62008206611455d97b37a6`).
- The accepted local Step 2D instruction states exactly
  `GEOM_DRUGS_UNIVERSE - HISTORICAL_EXCLUSION_UNION` and treats native split as
  metadata “if available” (instruction SHA256
  `10b238d62fc85c2c229e7ce23c0cce852e75813b3167acc0db8bf151c13a0774`).
- The frozen selector SHA256 remains
  `41df4a438e4acca2a545555ea9e5f0b1adf14d330cb9a60a413859d9a88e8907`;
  its required fields and `is_eligible` logic contain no native split field.

```text
PRIMARY_SELECTION_DOMAIN = ENTIRE_ELIGIBLE_UNUSED_GEOM_DRUGS_UNIVERSE
EXACT_NATIVE_TEST_GATE_STATUS = IMPLEMENTATION_OVERCONSTRAINT_REMOVED
```

## Set and filter decomposition

```text
SOURCE_UNIVERSE_N = 282953
HISTORICAL_EXCLUSION_UNION_N = 55806
SOURCE_UNIVERSE_INTERSECT_HISTORICAL_EXCLUSION_N = 49633
SOURCE_UNIVERSE_NOT_IN_HISTORICAL_EXCLUSION_N = 233320

N_AFTER_CANONICAL_VALIDITY = 282953
N_AFTER_HISTORICAL_EXCLUSION = 233320
N_AFTER_NATIVE_SPLIT_FILTER = 233320  # no-op; not a frozen requirement
N_AFTER_REFERENCE_AVAILABILITY_FILTER = 233320
N_AFTER_CHEMISTRY_COMPATIBILITY_FILTER = 233312
N_AFTER_OTHER_FILTERS = 233312
FINAL_ELIGIBLE_N = 233312
```

Eight otherwise nonhistorical molecules fail the combined chemistry gate. The
global rejection count contains eleven MMFF94s failures because three of those
eleven were already removed by historical exclusion.

## Root cause and blocker resolution

The former zero was caused by two coupled implementation errors: all source
rows without exact native-test linkage were inserted into the historical union,
and the same rows were then rejected again by an explicit native-test check.
Both behaviors contradicted the frozen domain and have been removed. Native
split remains in the source schema as metadata.

The repaired eligible pool is much larger than 2,500. The final remaining
blocker was resolved by the targeted legacy recovery in
`11_LEGACY_IDENTITY_RECOVERY.csv` through
`13_LEGACY_RECOVERY_AUDIT.md`. Tracked frozen identity artifacts yielded 2,000
LSGO, 2,700 learned-geometry, and 2,700 BAT identities (2,900 unique across the
three routes). All source identifiers map through the immutable TRAIN manifest
and cache, canonicalize without failure, and are already contained in the
current TRAIN exclusion. The existing 49,964-identity canonical TRAIN set is a
proven conservative superset, adding zero new exclusions in this recovery.

The historical union membership therefore remains 55,806, but its provenance
coverage is now complete. Applying the unchanged input-only selection rule to
233,312 eligible unused identities freezes exactly 2,500 molecules with zero
overlap against current TRAIN, current DEV, or the historical exclusion union.

```text
HISTORICAL_EXCLUSION_UNION_COMPLETE = YES
PRIMARY_FINAL_2500_NOW_POSSIBLE = YES
PRIMARY_FINAL_N_MOLECULES = 2500
PRIMARY_FINAL_MEMBERSHIP_FROZEN = YES
CURRENT_TRAIN_OVERLAP = 0
CURRENT_DEV_OVERLAP = 0
HISTORICAL_EXCLUSION_OVERLAP = 0
STEP_2D_STATUS = PASS_MEMBERSHIP_FROZEN
READY_FOR_STEP_3 = YES
```

No model, inference, MMFF, xTB, V3D, PoseBusters, RMSD, or protected outcome was
run or read.

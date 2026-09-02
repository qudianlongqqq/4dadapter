# STEP 2D legacy historical cohort identity recovery

The three targeted legacy routes are recovered without model execution or
outcome access. Their tracked identity artifacts are unchanged from the frozen
Git revisions under the repository's text normalization. Every historical source identifier maps deterministically to
three records in the immutable formal-large TRAIN manifest (SHA256
`fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d`); one cached record per molecule supplies
the molecular graph used by the existing STEP 2D canonicalizer.

The LSGO mechanism thresholds used the 2,000-molecule TRAIN partition, and the
48-molecule mechanism-confirm set is an exact subset. Learned geometry and BAT
each preserve all 2,700 direct train/dev/external identities. Across the three
routes there are 2,900 unique canonical identities; all
2,900 are already members of the current canonical TRAIN
exclusion.

The current TRAIN set is therefore a proven conservative superset for these
TRAIN-only historical routes. It contains 47,064
identities beyond the recovered three-route exact union, but all were already
excluded before this recovery, so this audit adds zero new over-exclusions and
does not change the exclusion-union membership.

```text
LSGO_LEGACY_RECOVERY_STATUS = COMPLETE
LSGO_LEGACY_IDENTITIES_N = 2000
LEARNED_GEOMETRY_LEGACY_RECOVERY_STATUS = COMPLETE
LEARNED_GEOMETRY_LEGACY_IDENTITIES_N = 2700
BAT_REFINEMENT_LEGACY_RECOVERY_STATUS = COMPLETE
BAT_REFINEMENT_LEGACY_IDENTITIES_N = 2700
LEGACY_EXACT_UNION_N = 2900
SUPERSET_MEMBERSHIP_PROVEN = YES
CONSERVATIVE_SUPERSET_USED = YES
ADDITIONAL_OVER_EXCLUDED_MOLECULE_COUNT = 47064
NEW_ADDITIONAL_OVER_EXCLUDED_BY_THIS_RECOVERY = 0
HISTORICAL_DEVELOPMENT_IDENTITY_UNION_COMPLETE = YES
PROTECTED_OUTCOME_READ = NO
SCIENTIFIC_MODEL_CHANGED = NO
```

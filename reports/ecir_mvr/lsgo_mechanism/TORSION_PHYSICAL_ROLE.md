# Torsion physical role

Decision: **TORSION_LOW_ACTIONABILITY**. The frozen K3 model's Reference likelihood success is retained; this audit tests whether its Source surprise is a useful physical-error detector.

- median incremental BAT-union fraction before BA: `0.00001`; high-flex: `0.00020`;
- surprise vs incremental BAT fraction Spearman: `-0.3891` (non-weak detector under preregistered |ρ|<0.20 rule).

Associations:

| association                                        |   records |   spearman |   pearson_diagnostic |
|:---------------------------------------------------|----------:|-----------:|---------------------:|
| torsion_surprise_vs_source_reference_median_excess |       144 |     0.2823 |               0.0122 |
| torsion_surprise_vs_remaining_ba_excess            |       144 |     0.2780 |               0.0121 |
| torsion_surprise_vs_independent_t_force_fraction   |        18 |    -0.5108 |              -0.4651 |
| torsion_surprise_vs_incremental_bat_fraction       |        18 |    -0.3891 |              -0.1282 |

An independent T projection is not additive with BA because subspaces overlap; the incremental BAT-union fraction is the sufficiency diagnostic.

# BAT protocol deviations

## Historical BA active-set correction

The frozen LSGO-B coordinate method uses all Bond/Angle primitives with equal group aggregation. BAT does not retrofit an active-set into BA. Torsion selectivity and steric activation are independent new mechanisms.

## Internal steric gate uses continuous penetration severity

The preregistered field `steric_violation_reduction_min` was operationalized before execution as reduction in summed positive penetration, not reduction in the binary violating-pair count. This is scientifically material and is therefore disclosed. Under the fixed 0.003 Å trust budget, DEV_A and DEV_B showed continuous reductions while binary counts remained unchanged (3→3 and 12→12). The internal result supports gradient direction and safety only; it does not establish clash resolution. Fresh PoseBusters fail→pass remains required for `STERIC_VALIDATED`.

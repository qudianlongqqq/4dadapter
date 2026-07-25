# BAT protocol deviations

## Historical BA active-set correction

The frozen LSGO-B coordinate method uses all Bond/Angle primitives with equal group aggregation. BAT does not retrofit an active-set into BA. Torsion selectivity and steric activation are independent new mechanisms.

## Internal steric gate uses continuous penetration severity

The preregistered field `steric_violation_reduction_min` was operationalized before execution as reduction in summed positive penetration, not reduction in the binary violating-pair count. This is scientifically material and is therefore disclosed. Under the fixed 0.003 Å trust budget, DEV_A and DEV_B showed continuous reductions while binary counts remained unchanged (3→3 and 12→12). The internal result supports gradient direction and safety only; it does not establish clash resolution. Fresh PoseBusters fail→pass remains required for `STERIC_VALIDATED`.

## Torsion smoke import correction

The first 20-step smoke invocation stopped before loading any data because the runner imported the existing helper under the wrong name (`atomic_torch` instead of `atomic_torch_save`). The import was corrected without changing model or experiment semantics; the complete smoke then passed. No partial checkpoint from the failed invocation existed or was reused.

## BA+C solver attribution confound discovered after freeze

The frozen combined solver backtracks every BA+C candidate for safety, including records where the steric objective is inactive. The exact historical BA anchor rejects an unsafe full step directly. On the fresh external cohort, BA+C therefore differs from BA on 32/34/32 steric-inactive records for seeds 173/181/193, in addition to 12 active records per seed. The stronger BA+C xTB improved fraction cannot be cleanly attributed to C. This was discovered after coordinate/PB/xTB freeze, was not corrected or rerun, and contributes to the conservative `KEEP_LSGO_B` decision.

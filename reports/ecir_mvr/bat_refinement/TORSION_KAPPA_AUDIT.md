# Torsion κ audit

- seed 211: TRAIN resultant `0.92801`, residual estimate `7.22385`, DEV_A-selected fixed κ `7.223849`.
- seed 223: TRAIN resultant `0.92820`, residual estimate `7.24291`, DEV_A-selected fixed κ `7.242909`.
- seed 239: TRAIN resultant `0.92709`, residual estimate `7.13718`, DEV_A-selected fixed κ `7.137179`.

κ=8 was used only to learn initial centers for the post-hoc residual audit. The selected fixed κ was then used in a fresh head training run. Learned κ remained diagnostic only.

# Chemical strata

Positive energy gain means BA lowered xTB energy. Bond-type/hybridization rows report mean primitive `|z|`; molecule-level rows report BA score. Descriptive only; no post-hoc selection.

| stratum              | value      |   records |   mean_source_ba_abnormality |   median_energy_gain_kcal_mol |   improved_fraction |   p95_harmful_delta |
|:---------------------|:-----------|----------:|-----------------------------:|------------------------------:|--------------------:|--------------------:|
| heavy_atom_bin       | >35        |         3 |                       2.1907 |                        3.9653 |              1.0000 |             -2.8377 |
| flex_bin             | high_ge_5  |        48 |                       1.2537 |                        0.9249 |              0.9792 |             -0.3673 |
| heavy_atom_bin       | 21-35      |        78 |                       1.1542 |                        0.7708 |              0.9231 |              0.0000 |
| family/hybridization | B/SP3-SP3  |       432 |                       0.7783 |                        0.7626 |              0.8449 |            nan      |
| family/hybridization | A/SP3      |      3786 |                       0.9099 |                        0.7610 |              0.8788 |            nan      |
| family/hybridization | B/SP3-SP2  |       252 |                       0.5131 |                        0.7589 |              0.9246 |            nan      |
| family/hybridization | B/SP3-S    |      1239 |                       0.7798 |                        0.7517 |              0.8878 |            nan      |
| family/bond_type     | B/SINGLE   |      3768 |                       0.6960 |                        0.7414 |              0.9087 |            nan      |
| family/hybridization | B/SP2-S    |      1056 |                       0.6717 |                        0.7364 |              0.9422 |            nan      |
| family/bond_type     | A/ANGLE    |      9855 |                       0.8418 |                        0.7364 |              0.9131 |            nan      |
| family/hybridization | B/SP2-SP3  |       306 |                       0.5998 |                        0.7316 |              0.9020 |            nan      |
| amide_like           | True       |        72 |                       1.1238 |                        0.7219 |              0.9167 |              0.0000 |
| family/hybridization | A/SP2      |      6063 |                       0.7994 |                        0.7213 |              0.9345 |            nan      |
| aromatic             | False      |        12 |                       1.6078 |                        0.7204 |              0.9167 |             -0.1668 |
| family/bond_type     | B/AROMATIC |      1716 |                       0.7578 |                        0.7170 |              0.9365 |            nan      |
| family/hybridization | B/SP2-SP2  |      2484 |                       0.7511 |                        0.7170 |              0.9344 |            nan      |
| family/bond_type     | B/DOUBLE   |       291 |                       0.9254 |                        0.6736 |              0.9107 |            nan      |
| ring                 | True       |       141 |                       1.1846 |                        0.6708 |              0.9149 |              0.0000 |
| aromatic             | True       |       132 |                       1.1455 |                        0.6661 |              0.9167 |              0.0000 |
| flex_bin             | medium_3_4 |        48 |                       0.9686 |                        0.6319 |              0.9167 |              0.0000 |
| amide_like           | False      |        72 |                       1.2443 |                        0.5774 |              0.9167 |              0.0000 |
| ring                 | False      |         3 |                       1.1591 |                        0.5010 |              1.0000 |             -0.3230 |
| heavy_atom_bin       | <=20       |        63 |                       1.1730 |                        0.4779 |              0.9048 |              0.0000 |
| flex_bin             | low_0_2    |        48 |                       1.3298 |                        0.4722 |              0.8542 |              0.0000 |

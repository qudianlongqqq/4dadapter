# Source–Reference ensemble xTB energy

All values are GFN2-xTB single-point energies at frozen coordinates. Reference is a sampled plausible ensemble, not a unique coordinate target. Nearest means aligned-RMSD nearest under the inherited matcher.

## Source relation fractions

| reference_statistic   | relation           |   fraction |   ci95_low |   ci95_high |
|:----------------------|:-------------------|-----------:|-----------:|------------:|
| minimum               | source_lower       |     0.0069 |     0.0000 |      0.0208 |
| minimum               | approximately_tied |     0.0000 |     0.0000 |      0.0000 |
| minimum               | source_higher      |     0.9931 |     0.9792 |      1.0000 |
| median                | source_lower       |     0.1250 |     0.0623 |      0.2014 |
| median                | approximately_tied |     0.0139 |     0.0000 |      0.0347 |
| median                | source_higher      |     0.8611 |     0.7778 |      0.9306 |
| aligned_rmsd_nearest  | source_lower       |     0.0625 |     0.0208 |      0.1111 |
| aligned_rmsd_nearest  | approximately_tied |     0.0069 |     0.0000 |      0.0208 |
| aligned_rmsd_nearest  | source_higher      |     0.9306 |     0.8819 |      0.9722 |

## LSGO-B behavior by Source/Reference relation

| case                          |   count |   ba_median_delta |   ba_improved_fraction |   ba_p95 |
|:------------------------------|--------:|------------------:|-----------------------:|---------:|
| Source above Reference median |     124 |           -0.7170 |                 0.9113 |   0.0000 |
| Source below Reference median |      18 |           -0.4132 |                 0.9444 |  -0.2349 |
| approximately tied            |       2 |           -0.4963 |                 1.0000 |  -0.3934 |

A negative BA ΔE means the frozen BA update lowers energy relative to Source. Thus improvement when Source is already below the Reference median is evidence against Reference-coordinate imitation.

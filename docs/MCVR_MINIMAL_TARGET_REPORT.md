# MCVR minimal-validity target report

## Target definition

The new target minimizes aligned displacement plus thresholded-excess bond,
angle, ring, clash, chirality-safety, and torsion-anchor penalties. Bond,
angle, and ring penalties are exactly zero inside the frozen train-derived
validity envelopes; valid geometry is not pulled toward a median. Periodic
torsion change uses `atan2(sin(delta), cos(delta))`, and high-flex molecules
receive a stronger basin anchor.

Optimization uses Adam for at most 40 steps at learning rate 0.001. Every step
is rigidly aligned and projected into fixed molecule/atom trust radii of
0.15/0.35 Å. The chosen target is the best safe trajectory point, not
necessarily the final point. New clash, chirality, stereocenter, ring, trust,
or high-flex torsion risks disqualify a candidate. Failure returns the exact
input and never invokes a reference or force field.

## Fixed pilot and gate

The train-only fixed pilot contains 170 records: 50 ETFlow normal, 50
Cartesian mild, 50 Cartesian medium, and 20 clean reference controls. It also
contains 98 high-flex and 170 ring records. No test data was used.

| Metric | Input | Old restrained target | Minimal target | Relative improvement vs input |
|---|---:|---:|---:|---:|
| bond outlier rate | 0.38453 | 0.15125 | 0.23035 | 40.1% |
| angle outlier rate | 0.08349 | 0.02566 | 0.07601 | 8.9% |
| ring-bond outlier rate | 0.24908 | 0.05647 | 0.15282 | 38.6% |
| ring-planarity outlier rate | 0.01176 | 0.01833 | 0.00147 | 87.5% |
| total thresholded validity | 1.50771 | 0.41981 | 0.86734 | 42.5% |

Minimal/old mean aligned displacement is 0.01275/0.14655 Å, and p95 is
0.03547/0.19106 Å. High-flex mean maximum torsion change is 0.03197 rad for
minimal targets versus 0.20624 rad for old targets. Validity gain per Å is
55.33 versus 8.52. Severe clashes and chirality do not worsen; all 20 clean
controls remain identity; the largest minimal-target atom displacement is
0.06876 Å. There is no 4 Å fallback.

All twelve gate checks pass. The decision is `PASS`; no target repair or
margin/trust relaxation was used.

## Full targets

The complete train build contains 678 successful repairs, 66 explicit identity
fallbacks, and 6 already-valid identities. Validation contains 125 successful
repairs and 5 explicit identity fallbacks. All fallback reasons and trajectory
summaries are persisted.

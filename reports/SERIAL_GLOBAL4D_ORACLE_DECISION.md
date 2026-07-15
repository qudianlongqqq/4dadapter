# Serial Global4D Oracle Decision

Decision: **ORACLE_PASS**.

All continuation conditions pass.  The validation-only Oracle lowers RMSD
from `1.394668` to `0.917641` at lambda `1.0`, an absolute improvement of
`0.477027` and a relative improvement of `34.20%`.

- 60/60 records and 30/30 molecules improve.
- Median per-record delta RMSD is `-0.360235`.
- Median per-molecule delta RMSD is `-0.349099`.
- High-flex RMSD improves from `1.875526` to `1.147601`.
- Mean projection energy ratio is `0.425919`.
- Top 10% of records contribute 34.3% of total improvement; the negative
  median and 100% improvement rate rule out a few-outlier-only result.
- All saved numeric results are finite and failure rate is zero.

Condition number and solver fallback rate are `UNKNOWN`: they were not saved
in the frozen Oracle JSON and the Oracle was intentionally not rerun.

The result proves a stable theoretical residual-correction space, not learned
model performance.  Stage 2 pilot cache construction is authorized.

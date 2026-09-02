# STEP 3A ETFlow historical protocol recovery audit

## Decision

STEP 3A stopped before source generation. The frozen primary manifest passed its
SHA256 check and contains exactly 2,500 rows targeting 5,000 records. Historical
source manifests establish that the development lineage was labelled ETFlow
NFE=10 with global seed 42 and checkpoint path
`/home/aidd5090/.cache/etflow/drugs-o3.ckpt`. They do not bind the historical
configuration bytes, runner bytes, ETFlow commit, complete sampler arguments, or
per-molecule seed mapping. Those omissions prevent an exact protocol recovery.

The available official checkout is commit
`13d282c946755b32dac02d7efa1206b204944427`. Its current `drugs-o3.yaml` uses
NFE=50, and none of the six revisions in the Git history of that configuration
contains `n_timesteps: 10`. The repository wrapper forwards sampler arguments
from the supplied YAML and has no independent NFE override. Consequently,
changing the current NFE=50 YAML to 10 would create a newly inferred protocol;
it would not recover the exact historical protocol.

Per the frozen STEP 3A instruction, the prospective NFE=50 fallback was not
selected and ETFlow inference was not started.

```text
PRIMARY_FINAL_MANIFEST_SHA256_VERIFIED = YES
HISTORICAL_ETFLOW_PROTOCOL_RECOVERED = NO
HISTORICAL_ETFLOW_NFE = 10
FROZEN_ETFLOW_NFE = NONE__FAIL_CLOSED
ETFLOW_PROTOCOL_MATCH_STATUS = NOT_RECOVERED__NFE50_NOT_AUTHORIZED
ETFLOW_SOURCE_GENERATION_STATUS = BLOCKED_BEFORE_GENERATION
```

## Evidence inventory

| Evidence | Observation | Limit |
|---|---|---|
| `reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json` | SHA256 `2a1d07af...6ee1`; 2,500 rows; 5,000 target records | Membership only |
| `E:/3dconformergenerationcode/4dadapter-v8/data/ecir_mvr/formal_large/real_sources/train.parquet` | 150,000 records, 50,000 molecules; NFE only 10; seed only 42; one checkpoint path label | No config/runner/commit/per-molecule seed provenance |
| `E:/3dconformergenerationcode/4dadapter-v8/data/ecir_mvr/formal_large/real_sources/val.parquet` | 10,000 records, 5,000 molecules; NFE only 10; seed only 42; same checkpoint path label | Same provenance omissions |
| `E:/3dconformergenerationcode/.cache/etflow_official/drugs-o3.ckpt` | Current local SHA256 `a24ae9a...ccb2` | Historical records do not contain a checkpoint hash, so byte identity is unproven |
| `E:/3dconformergenerationcode/ETFlow/configs/drugs-o3.yaml` | SHA256 `91ddac...e12c`; ODE, NFE=50, `s_churn=1.0`, `t_min=0.0001`, `t_max=0.9999`, `std=1.0` | This is the prospective NFE=50 configuration, not recovered historical NFE=10 evidence |
| ETFlow `drugs-o3.yaml` Git history | Six revisions inspected; zero NFE=10 configurations found | No immutable NFE=10 config is available in the checkout history |
| `scripts/generate_etflow_formal_large_upstream.py` | SHA256 `7050b8...c34`; calls `model.sample(..., **sampler_args)` from YAML | Current deterministic wrapper cannot prove the historical runner or seed semantics |
| `scripts/run_generate_etflow_formal_large_upstream.sh` | Supplies config/checkpoint/global seed but no NFE override | Does not recover NFE=10 independently |

No geometry, reference target, validity, energy, or model-performance field was
read for this audit. Parquet access was limited to identity counts and protocol
metadata columns.

## Missing bindings required to proceed

An exact historical run bundle must supply, or cryptographically bind:

1. the NFE=10 YAML/config bytes and SHA256;
2. the historical ETFlow code commit or source snapshot;
3. the historical generation runner and SHA256;
4. all sampler arguments and coordinate/output conventions;
5. the exact global-to-record/per-molecule seed rule;
6. the checkpoint SHA256 used by the historical generation.

Until those are recovered, no generation script can be scientifically frozen
as an exact historical NFE=10 reproduction. STEP 3B/refiner inference must not
start.

## Guard confirmation

```text
ETFLOW_INFERENCE_STARTED = NO
NFE50_FALLBACK_USED = NO
GENERATED_RECORDS = 0
PRIMARY_FINAL_MEMBERSHIP_CHANGED = NO
PROTECTED_FINAL_PERFORMANCE_READ = NO
SCIENTIFIC_REFINER_CHANGED = NO
NO_REPEATED_POLLING = YES
```

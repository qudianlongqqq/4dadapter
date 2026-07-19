# MCVR V7 Formal-Large Runtime Estimate

## Decision

`V7_FORMAL_RUNTIME_BENCHMARK_COMPLETE`

This is a two-batch compute benchmark. It performs forward and backward for the
existing D1-B prior but does not call `optimizer.step()`. V7 remains
inference-only; its analytic Angle solver has no backward pass.

## Frozen inputs

- Device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 5080`
- Batches: `2`
- Batch size: `64`
- D1-B checkpoint SHA256: `c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca`
- Training config SHA256: `fd1f5b6780c781d8e7681b31fd93b1459f6b30ebf0e6bf4a564ecab5c16e41db`
- V7 wrapper config SHA256: `5737ce5aa3bad729a6748a3fb9f0eea515bd96765df15e99bba6bd70297b8b4b`
- Train molecules: `50000`
- Validation molecules: `5000`

## Measurements

| Measurement | Mean seconds/batch |
|---|---:|
| D1-B training forward + loss | 0.458298 |
| D1-B backward | 0.078369 |
| D1-B compute total | 0.536666 |
| Frozen prior inference forward | 0.324925 |
| V7 total inference forward | 1.136585 |
| V7 solver + fusion overhead | 0.811660 |

- Peak training CUDA allocated: `142.54 MiB`
- Peak V7 CUDA allocated: `33.77 MiB`
- Solver calls: `128`
- Solver failures: `0`

## Estimate

- Compute-only 25K prior estimate: `3.727 h`
- V7 10K-record validation estimate: `0.338 h`
- Combined compute-only estimate: `4.065 h`

The compute-only extrapolation excludes dataloader stalls, validation metrics,
checkpoint serialization, telemetry, and scheduler overhead. It must not be
presented as a wall-clock guarantee. The existing seed43 D1-B formal run is the
stronger operational reference and completed in about `3.66 h` wall time on an
RTX 5080. A conservative `7-10 h` scheduling window remains sufficient, but the
measured local evidence does not require that much active compute.

## Isolation

```text
optimizer_steps_taken=0
checkpoint_created=false
test_records_read=0
test_assets_opened=false
formal_test_run=false
```

# Global Coupled 4D sampling profile

- Device: `cpu`
- Warmup records: 0
- Profiled records: 30
- Refinement steps: 300
- Measured records/s: 32.631087
- Pure rollout records/s: 33.075415
- Partial saving: `False` every 1 records
- CUDA timing: `not collected; no profiling synchronization`

## Stage timing

| Stage | Calls | CPU wall s | CUDA s | Self s | s/record | s/step | Wall share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rollout_total | 30 | 0.907018 | 0.000000 | 0.907018 | 0.030234 | 0.003023 | 98.66% |
| backbone_forward | 300 | 0.550296 | 0.000000 | 0.550296 | 0.018343 | 0.001834 | 59.86% |
| rank_check_and_svd | 300 | 0.071937 | 0.000000 | 0.071937 | 0.002398 | 0.000240 | 7.82% |
| jacobian_assembly | 300 | 0.050745 | 0.000000 | 0.050745 | 0.001691 | 0.000169 | 5.52% |
| local_frame | 300 | 0.041489 | 0.000000 | 0.041489 | 0.001383 | 0.000138 | 4.51% |
| cartesian_projection | 300 | 0.036777 | 0.000000 | 0.036777 | 0.001226 | 0.000123 | 4.00% |
| joint_head_forward | 300 | 0.034881 | 0.000000 | 0.034881 | 0.001163 | 0.000116 | 3.79% |
| static_topology_preparation | 30 | 0.010161 | 0.000000 | 0.010161 | 0.000339 | 0.000034 | 1.11% |
| python_record_object | 30 | 0.009119 | 0.000000 | 0.009119 | 0.000304 | 0.000030 | 0.99% |
| fragment_pool | 300 | 0.008808 | 0.000000 | 0.008808 | 0.000294 | 0.000029 | 0.96% |
| gram_matrix | 300 | 0.008724 | 0.000000 | 0.008724 | 0.000291 | 0.000029 | 0.95% |
| internal_velocity_mapping | 300 | 0.002917 | 0.000000 | 0.002917 | 0.000097 | 0.000010 | 0.32% |
| cpu_to_device | 30 | 0.002133 | 0.000000 | 0.002133 | 0.000071 | 0.000007 | 0.23% |
| device_to_cpu | 30 | 0.000311 | 0.000000 | 0.000311 | 0.000010 | 0.000001 | 0.03% |
| topology_cache_lookup | 300 | 0.000039 | 0.000000 | 0.000039 | 0.000001 | 0.000000 | 0.00% |
| cholesky | 300 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00% |
| dense_solve | 300 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00% |
| least_squares | 300 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00% |
| final_samples_save | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00% |
| cuda_synchronize_overhead | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00% |

## Same-topology batch benchmark

| Batch | Status | Seconds | Records/s |
| ---: | --- | ---: | ---: |



## Real-record persistence protocol matrix

The same computed records are replayed for every row; the model is not rerun.

| Protocol | Pure compute s | Payload build s | Tensor serialization s | State JSON s | Combined total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| partial_disabled_compute_only | 0.907018 | 0.000000 | 0.000000 | 0.000000 | 0.919369 |
| diagnostic_full_rewrite_every_1 | 0.907018 | 0.000000 | 0.153048 | 0.208758 | 1.282422 |
| diagnostic_full_rewrite_every_10 | 0.907018 | 0.000000 | 0.016640 | 0.017825 | 0.953967 |
| diagnostic_full_rewrite_every_30 | 0.907018 | 0.000000 | 0.006660 | 0.005938 | 0.932001 |
| current_formal_full_rewrite_every_1 | 0.907018 | 0.018281 | 0.159051 | 0.212166 | 1.310199 |
| chunk_shard_every_10 | 0.907018 | 0.000000 | 0.013789 | 0.014835 | 0.948103 |


Raw per-record rows are in the CSV only; the JSON/Markdown reports stay compact.

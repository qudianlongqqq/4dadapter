# Gated Global4D V2 batch capacity benchmark

Status: **SKIPPED_CUDA_UNAVAILABLE**

Environment: `{"cuda": null, "cuda_available": false, "device": "cpu", "gpu_name": null, "python": "3.11.8", "pytorch": "2.13.0+cpu"}`

Every measured condition executes forward, loss, backward, optimizer.step, and zero_grad. Blank measurement fields mean no CUDA measurement was executed.

| batch | accum | composition | records | atoms | edges | joints | records/s | peak allocated MiB | peak reserved MiB | finite | OOM | status |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 4 | 2 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 4 | 2 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 4 | 2 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 8 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 8 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 8 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 16 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 16 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 16 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 32 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 32 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 32 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 48 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 48 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 48 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 64 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 64 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 64 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 96 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 96 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 96 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 128 | 1 | low_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 128 | 1 | mixed |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |
| 128 | 1 | high_complexity |  |  |  |  |  |  |  |  |  | skipped_cuda_unavailable |

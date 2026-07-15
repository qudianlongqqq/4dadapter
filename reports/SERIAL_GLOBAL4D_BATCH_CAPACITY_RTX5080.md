# Serial Global4D RTX 5080 Batch Capacity

All physical batch candidates `4, 8, 16, 32, 48, 64, 96, 128` passed on both
mixed and high-complexity real Stage 2 batches. Every measured step included
cache loading, CUDA transfer, Jacobian construction, model forward, coefficient
and internal losses, backward, gradient clipping, AdamW step, and zero-grad.
Each candidate used 5 warmups and 20 measured optimizer steps.

- MAX_OOM_FREE_BATCH: **128**
- MAX_SAFE_BATCH: **128**
- MAX_THROUGHPUT_BATCH: **128**
- RECOMMENDED_TRAINING_BATCH: **96**
- RECOMMENDED_ACCUMULATION: **1**
- RECOMMENDED_EFFECTIVE_BATCH: **96**
- RECOMMENDED_LR: **2e-4**

Batch 128 reached 289.52 records/s on mixed and 307.68 records/s on the
high-complexity cohort. Batch 96 reached 283.09 and 303.28 records/s,
respectively: at least 95% of the best common throughput, so the smaller batch
is recommended. High-complexity batch 128 reserved only 509,607,936 bytes of
the 17,094,475,776-byte GPU, well inside both safety limits.

Suggested dynamic ceilings are 96 graphs, 6,000 atoms, and 1,300 joints per
batch. They are reporting guardrails, not a silently enabled dynamic batcher.

The raw benchmark is external to git at
`E:/3dconformergenerationcode/serial_global4d_work/serial_batch_capacity_rtx5080.json`
(SHA256 `7d81472df0e3e756e1dd32b18ab36c7607b855d365b025a2672d2a21edb79d5d`).

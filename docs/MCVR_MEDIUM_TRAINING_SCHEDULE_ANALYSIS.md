# MCVR Medium Training Schedule Analysis

V4 tests whether the fixed `2e-4` learning rate overwrote early validity gains. Scientific content, data identities, model, losses, optimizer family, batch size, inference, safety, and Gate 2 remained frozen.

## Registered schedules

| Run | Initialization | LR schedule | Candidate steps |
|---|---|---|---|
| Rescue V3 | resumed at step 2450 | fixed 2e-4 | 5000, 10000, 15000, 20000 formal |
| Schedule V4 | step 0 | 500-step warmup, cosine 2e-4 to 2e-5 | 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000 |

## Outcome

V3 selected step 10000 with validity delta `-0.074069` and failed only the 10% core-improvement condition.

V4 selected step 1500 with validity delta `-0.084689` and maximum core relative improvement `0.099977`.

Final decision: **MEDIUM_SEED42_SCHEDULE_V4_FAIL**.

模型有统计显著且精度非劣的中等有效性，但未达到预注册10%核心改善门槛。

The training-schedule hypothesis did not produce a preregistered Gate 2 pass; further Medium Seed42 rescue is closed.

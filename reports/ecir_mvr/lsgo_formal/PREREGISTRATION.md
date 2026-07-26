# LSGO-BA formal-large preregistration

Frozen before smoke or training on branch `research/lsgo-ba-formal-large`, HEAD `24d2723d3d93963da026aa4f7f8f989662ce2723`. The three formal seeds are `[307, 331, 353]` and all must run 12,500 optimizer steps with effective batch 64. Only the neural Bond/Angle means are trained; DRCSR scale and all method choices remain frozen.

Checkpoint candidates are fixed at 2,500/5,000/7,500/10,000/12,500. Selection uses lowest full 5,000-molecule validation joint BA NLL, then calibration error, then earlier step. GO requires every seed to beat frozen A on joint NLL and Bond/Angle MAE, reference stationarity nonincrease ≥0.95, Source>Reference objective fraction ≥0.60, and selected joint-NLL sample SD ≤0.05.

xTB and PoseBusters are forbidden for training or checkpoint selection. Formal test reads = **0**. Frozen holdout reads = **0**.

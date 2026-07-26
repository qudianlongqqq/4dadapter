# LSGO-BA formal protocol deviations

Before formal training, the first smoke command was launched with the system `E:\python\python.exe`. That environment contains CPU-only PyTorch 2.13.0. The runner silently selected CPU even though `--device cuda:0` was requested. The attempt completed its mathematical smoke checks but reported zero GPU memory, so it was not accepted as the formal smoke and no formal training was started.

The attempt is preserved under `attempts/cpu_fallback/`. The runner now hard-fails rather than silently falling back when CUDA is requested. The preregistration will be refreshed after this engineering-only correction, and the full smoke will be rerun with `E:\miniconda\envs\etflow-5080-v2\python.exe` (PyTorch 2.11.0+cu128) on the RTX 5080.

No model, loss, optimizer, scheduler, sampler, data identity, seed, checkpoint rule or validation gate changed. Formal test reads = **0**. Frozen holdout reads = **0**.

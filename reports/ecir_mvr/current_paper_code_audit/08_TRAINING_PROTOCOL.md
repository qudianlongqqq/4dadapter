# Training protocol

## Intended frozen configuration

| item | Restricted | Unrestricted |
|---|---:|---:|
| endpoint | 17,500 | 17,500 |
| batch molecules | 64 | 64 |
| optimizer | AdamW | AdamW |
| backbone LR | 1.5e-4 | 1.5e-4 |
| head/controller LR | 3e-4 | 3e-4 |
| weight decay | 1e-6 | 1e-6 |
| gradient clip | 1.0 | 1.0 |
| scheduler | CosineAnnealingLR | CosineAnnealingLR |
| scheduler T_max | 22,500 | 22,500 |
| recovery cadence | 250 steps | 250 steps |
| checkpoint selection | final step only | final step only |

Evidence: both current configs and runner optimizer/scheduler construction.

## RNG and data ordering

`seed_all` sets Python, NumPy, Torch CPU, and all Torch CUDA RNGs. The sampling generator is seeded with `seed+91000`. One generator drives molecule, source, and reference draws. Evidence: restricted runner `:185-193` and factorial sampler `:340-351`.

For matched multiseed, the supervisor creates a seed-specific frozen config for both formulations and executes one GPU job at a time. Matched execution for seed331/353 is not yet complete, so fairness remains `INCOMPLETE`, not PASS.

## Actually executed seed307

- Both final checkpoints record seed 307 and step 17,500.
- Restricted checkpoint hash: `7c2bb67ad8e9065a5409ea8e6f5ce79e7790c21cda3005beabbd32762786ca96`.
- Unrestricted checkpoint hash: `f63e9d796cc82297f2f2d5fd732c35aa80421ce2f604f2ac80d823a9f825b704`.
- Both logs contain 701 unique monotone logged steps from 1 through 17,500 with no duplicates; expected intervals are 24/25 because step 1 and every 25th step are logged.
- Unrestricted GPU preflight and step-25 verification explicitly pass model, optimizer, batch, forward, and backward CUDA checks on NVIDIA GeForce RTX 5080, PyTorch CUDA 12.8.
- Restricted current code has equivalent checks, but an execution-time seed307 GPU-verification artifact was not preserved. Restricted training device is therefore artifact-supported indirectly, not independently proven to the same standard: `NOT FULLY VERIFIED`.
- Exact Python and PyTorch package versions for restricted seed307 are not embedded in its final status. The resource audit reports Python 3.11.15 and Torch 2.11.0+cu128 for the CUDA environment; linking that environment to every restricted training step is `NOT VERIFIED`.

## Multiseed snapshot

At 2026-08-31T22:33+08:00, the pre-existing supervisor was running Restricted seed331, with run-local status at step 2000. Seed331 result is incomplete; seed353 had not completed. No partial metric is promoted.



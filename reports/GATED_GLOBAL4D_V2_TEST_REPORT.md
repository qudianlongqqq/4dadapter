# Gated Global4D V2 test report

## Environment

- Python: 3.11.8
- PyTorch: 2.13.0+cpu
- CUDA availability: false
- CUDA runtime: none
- GPU: none
- Lightning: 2.6.5
- PyTorch Geometric: 2.8.0
- RDKit: 2026.03.3
- Missing local dependencies: `torch_cluster`, `pydantic`

## Relevant suite

Command:

```powershell
python -m pytest <all test_global*.py plus formal/data/label-free contracts> -q -p no:cacheprovider
```

Result:

- passed: 143
- failed: 0
- skipped: 1
- warnings: 3

The skipped test is an environment-dependent existing test.  Warnings are two
upstream TorchScript deprecations and one historical Lightning no-Trainer logging
warning.

Focused final command:

```powershell
python -m pytest tests/test_global_coupled_4d_flow.py tests/test_gated_global4d_v2.py tests/test_global_coupled_4d_sampling_resume.py tests/test_global4d_chunked_persistence.py tests/test_flexbond_inference_no_labels.py -q -p no:cacheprovider
```

Result: 52 passed, 0 failed.

## Formal-cache CPU smoke

Read two validated records (one train, one val), collated them with the repository
`FlexBondData` contract, and ran all three fusion modes.

- graphs: 2
- atoms: 51
- directed edges: 108
- joints: 1
- strict projection count: 1
- additive projection count: 0
- gated projection count: 0
- strict/additive/gated output finite: true
- gated full loss: finite
- gated backward: passed
- every gate parameter gradient: present and finite

## Cartesian no-projection protection

The dedicated test constructs `v_cart_raw` with a nonzero Jacobian-subspace
component, replaces both projection functions with a function that raises on any
call, and executes `gated_additive`.

Result: passed.  The final output equals exactly:

```text
v_cart_raw + internal_beta * atom_gate * v_internal
```

The full-target Cartesian-loss spy test also passed and confirmed that the
Cartesian MSE target is `x_ref_aligned - x_init`, not its orthogonal residual.

## Checkpoint and persistence

- old checkpoint without fusion mode loads as exact strict: passed
- missing gated head without explicit initialization: rejected
- explicit missing-gate initialization and load report: passed
- fusion/beta/gate/joint/steps/update mismatch rejection: passed
- chunk identity/hash-chain tests: passed
- completed sample mismatch rejection: passed
- label-free inference schema tests: passed

## DataLoader and configuration

- batch 8 / accumulate 1 / effective 8 config: passed
- batch 4 / accumulate 2 / effective 8 benchmark condition: present
- `num_workers=0` persistent false: passed
- `num_workers=0` prefetch omitted: passed
- CLI → resolved config → DataModule/Trainer → checkpoint hparams/run state:
  covered and passed
- gated training entry point rejecting strict configs: covered and passed

## Shell and path validation

- new bash scripts: `bash -n` passed
- new source contains no fixed drive, Windows user directory, or Windows path
  concatenation
- Python syntax/AST validation: passed

## CUDA benchmark

Status: **SKIPPED_CUDA_UNAVAILABLE**.

No GPU memory, utilization, throughput, or OOM result is claimed.  The generated
JSON/Markdown reports contain explicit skipped rows for batches 4, 8, 16, 32,
48, 64, 96, and 128 across low, mixed, and high complexity compositions.

## Full repository suite

Command:

```powershell
python -m pytest -q -p no:cacheprovider
```

Collection was blocked before execution by two environment errors:

- `torch_cluster` missing in the historical TorchMD network tests;
- `pydantic` missing in the historical BaseFlow configuration tests.

This run is reported as not executed, not passed.  Install the full project
environment before Linux/Windows CUDA validation.

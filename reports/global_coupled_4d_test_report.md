# Global Coupled 4D test report

The four required test modules are present and syntax-compilable:

- `tests/test_global_coupled_4d_topology.py`
- `tests/test_global_coupled_4d_jacobian.py`
- `tests/test_global_coupled_4d_projection.py`
- `tests/test_global_coupled_4d_flow.py`

Coverage includes topology chains/branches/fail-closed cases/cache hits, Jacobian shape/stretch/torsion/bending/global coupling/cross Gram blocks/pseudoinverse/orthogonality/finite difference/translation/rotation/rank deficiency/empty joints, solver fallback/weights/float32/CPU-CUDA agreement, model equivariance/residual orthogonality/ablations/training-sampling mapping/topology reuse.

Runtime status: **PASSED (CPU)**.

Commands and results:

- `pytest -q tests/test_global_coupled_4d_topology.py`: **5 passed**.
- `pytest -q tests/test_global_coupled_4d_jacobian.py`: **7 passed**.
- `pytest -q tests/test_global_coupled_4d_projection.py`: **5 passed, 1 skipped**.
- `pytest -q tests/test_global_coupled_4d_flow.py`: **5 passed**.

Total: **22 passed, 1 skipped**. The skipped test is CPU/GPU agreement because this validation environment has no CUDA-enabled PyTorch. The flow suite emitted one expected Lightning warning because a unit test calls the shared training step without attaching a Trainer; all returned losses and gradients were finite.

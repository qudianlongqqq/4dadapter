import torch

from etflow.ecir.v8_solver import DifferentiableSolveConfig, solve_unified_delta


def test_rank_deficient_and_nonfinite_systems_fail_safely():
    prior = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    jacobian = torch.ones(2, 6, dtype=torch.float64)
    residual = torch.tensor([1.0, 1.0], dtype=torch.float64)
    solved, diag = solve_unified_delta(
        prior,
        torch.ones(2, dtype=torch.float64),
        residual,
        jacobian,
        residual.new_empty(0),
        jacobian.new_empty((0, 6)),
        DifferentiableSolveConfig(damping=1e-9),
    )
    assert diag["status"] == "SOLVED" and torch.isfinite(solved).all()
    bad, bad_diag = solve_unified_delta(
        prior,
        torch.ones(2, dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float64),
        torch.ones(1, 6, dtype=torch.float64),
        residual.new_empty(0),
        jacobian.new_empty((0, 6)),
        DifferentiableSolveConfig(),
    )
    assert bad_diag["fallback"] is True
    assert torch.equal(bad, prior)

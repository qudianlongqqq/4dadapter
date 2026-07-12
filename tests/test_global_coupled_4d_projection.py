import pytest
import torch

from etflow.commons.global_coupled_4d_projection import (
    gram_solve,
    project_orthogonal_residual_legacy,
    project_orthogonal_residual,
    svd_oracle,
)


def test_svd_oracle_minimum_norm_and_energy_identity():
    torch.manual_seed(3)
    jacobian = torch.randn(18, 7)
    target = torch.randn(6, 3)
    result = svd_oracle(jacobian, target)
    torch.testing.assert_close(result.projected + result.residual, target)
    lhs = (target.square().sum())
    rhs = result.projected.square().sum() + result.residual.square().sum()
    torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)
    assert result.effective_rank <= 7 and float(result.orthogonality_error) < 1e-6


def test_gram_matches_oracle_for_full_rank_system():
    torch.manual_seed(7)
    jacobian = torch.randn(24, 6)
    target = torch.randn(8, 3)
    oracle = svd_oracle(jacobian, target)
    fast = gram_solve(jacobian, target)
    torch.testing.assert_close(fast.projected, oracle.projected, atol=2e-5, rtol=2e-5)
    assert fast.solver_backend == "cholesky"


def test_rank_deficiency_triggers_fallback_without_nan():
    column = torch.arange(12, dtype=torch.float32)[:, None]
    jacobian = torch.cat((column, column, torch.zeros_like(column)), dim=1)
    result = gram_solve(jacobian, torch.randn(4, 3))
    assert result.solver_fallback_count > 0
    assert result.solver_backend == "svd_fallback"
    assert torch.isfinite(result.coefficients).all() and torch.isfinite(result.residual).all()
    assert float(result.orthogonality_error) < 1e-5


def test_optimized_rank_deficient_solver_removes_redundant_svdvals(monkeypatch):
    column = torch.arange(12, dtype=torch.float32)[:, None]
    jacobian = torch.cat((column, column, torch.zeros_like(column)), dim=1)
    target = torch.randn(4, 3)
    counts = {"svd": 0, "svdvals": 0}
    original_svd = torch.linalg.svd
    original_svdvals = torch.linalg.svdvals

    def counted_svd(*args, **kwargs):
        counts["svd"] += 1
        return original_svd(*args, **kwargs)

    def counted_svdvals(*args, **kwargs):
        counts["svdvals"] += 1
        return original_svdvals(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    monkeypatch.setattr(torch.linalg, "svdvals", counted_svdvals)
    optimized = project_orthogonal_residual(jacobian, target)
    assert counts == {"svd": 1, "svdvals": 0}
    reference = project_orthogonal_residual_legacy(jacobian, target)
    assert counts == {"svd": 2, "svdvals": 1}
    torch.testing.assert_close(optimized.projected, reference.projected)


def test_weighted_projection_is_weight_orthogonal():
    torch.manual_seed(9)
    jacobian = torch.randn(15, 4)
    target = torch.randn(5, 3)
    weights = torch.linspace(.5, 2., 5)
    result = project_orthogonal_residual(jacobian, target, weights=weights)
    normal = jacobian.T @ (weights.repeat_interleave(3) * result.residual.reshape(-1))
    assert torch.linalg.norm(normal) < 2e-5


def test_empty_projection_is_safe_residual_only():
    target = torch.randn(4, 3)
    result = gram_solve(torch.empty(12, 0), target)
    assert result.coefficients.numel() == 0 and result.effective_rank == 0
    torch.testing.assert_close(result.residual, target)


def test_float32_is_stable_and_cpu_gpu_agree_when_available():
    torch.manual_seed(12)
    jacobian = torch.randn(30, 8, dtype=torch.float32)
    target = torch.randn(10, 3, dtype=torch.float32)
    cpu = svd_oracle(jacobian, target)
    assert torch.isfinite(cpu.projected).all()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    gpu = svd_oracle(jacobian.cuda(), target.cuda())
    torch.testing.assert_close(gpu.projected.cpu(), cpu.projected, atol=2e-4, rtol=2e-4)

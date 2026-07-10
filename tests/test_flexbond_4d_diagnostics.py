from types import SimpleNamespace

import torch

from etflow.commons.flexbond_diagnostics import projection_quality
from etflow.commons.flexbond_jacobian import (
    apply_bond_jacobian,
    solve_q_star_least_squares,
)
from scripts.diagnose_flexbond_4d_quality import BOND_FIELDS, SAMPLE_FIELDS
from scripts.diagnose_flexbond_rollout import FIELDS as TRAJECTORY_FIELDS
from scripts.sample_flexbond_optimizer import resolve_correction_scale


class _FakeModel:
    def __init__(self, scale=0.01):
        self.hparams = SimpleNamespace(correction_scale=scale)


def test_correction_scale_override_zero_disables_only_4d_branch():
    model = _FakeModel()
    metadata = resolve_correction_scale(model, 0.01, 0.0)
    v_cart = torch.randn(4, 3)
    v_4d = torch.randn(4, 3)
    final = v_cart + model.hparams.correction_scale * v_4d
    torch.testing.assert_close(final, v_cart)
    assert metadata["override_correction_scale"] == 0.0
    assert metadata["effective_correction_scale"] == 0.0


def test_missing_override_preserves_checkpoint_behavior():
    model = _FakeModel(0.037)
    metadata = resolve_correction_scale(model, 0.01, None)
    assert model.hparams.correction_scale == 0.037
    assert metadata["effective_correction_scale"] == 0.037
    assert metadata["override_correction_scale"] is None


def _geometry():
    x = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 2.0],
    ])
    targets = {
        "anchor_index": torch.tensor([0]), "moving_index": torch.tensor([1]),
        "affected_atom_index": torch.tensor([1, 2, 3]),
        "affected_bond_index": torch.tensor([0, 0, 0]),
    }
    return x, targets


def test_oracle_projection_nearly_perfect_for_residual_in_jacobian_span():
    x, targets = _geometry()
    q = torch.tensor([[0.1, 0.2, -0.3, 0.4]])
    residual, _ = apply_bond_jacobian(x, q, targets)
    q_star, valid, _ = solve_q_star_least_squares(
        x, residual, targets, ridge_eps=1e-8, max_condition=1e10)
    oracle, _ = apply_bond_jacobian(x, q_star, targets)
    metrics = projection_quality(residual, torch.zeros_like(residual), oracle,
                                 correction_scale=1.0)
    assert valid.tolist() == [True]
    assert metrics["oracle_explained_ratio"] > 0.99999


def test_opposite_4d_direction_has_negative_cosine_and_worsens():
    residual = torch.ones(3, 3)
    pred = -residual
    metrics = projection_quality(residual, pred, residual, correction_scale=1.0)
    before = torch.sqrt(residual.square().mean())
    after = torch.sqrt((residual - pred).square().mean())
    assert metrics["cosine_pred_to_residual"] < 0
    assert float(after - before) > 0


def test_aligned_well_scaled_4d_improves():
    residual = torch.ones(3, 3)
    pred = residual
    metrics = projection_quality(residual, pred, residual, correction_scale=1.0)
    before = torch.sqrt(residual.square().mean())
    after = torch.sqrt((residual - pred).square().mean())
    assert metrics["cosine_pred_to_residual"] > 0
    assert float(after - before) < 0


def test_no_valid_bonds_and_nonfinite_values_are_safe():
    x = torch.zeros(2, 3)
    empty = torch.empty(0, dtype=torch.long)
    targets = {
        "anchor_index": empty, "moving_index": empty,
        "affected_atom_index": empty, "affected_bond_index": empty,
    }
    q_star, valid, stats = solve_q_star_least_squares(x, x, targets)
    metrics = projection_quality(x, torch.full_like(x, float("nan")), x,
                                 correction_scale=0.01)
    assert q_star.shape == (0, 4) and not valid.any()
    assert stats["q_star_nan_count"] == 0
    assert metrics["cosine_pred_to_residual"] == 0.0


def test_diagnostic_csv_schemas_are_complete():
    for required in (
        "oracle_explained_ratio", "hybrid_branch_delta", "joint_training_delta",
        "total_hybrid_gap", "q_pred_nonfinite_count"):
        assert required in SAMPLE_FIELDS
    for required in ("condition_number", "rank", "frame_valid", "solve_valid", "skip_reason"):
        assert required in BOND_FIELDS
    assert "raw_v4d_norm" in TRAJECTORY_FIELDS
    assert "failure_reason" in TRAJECTORY_FIELDS

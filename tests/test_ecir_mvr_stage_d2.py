from __future__ import annotations

import numpy as np
import pytest

from etflow.ecir.stage_d2_audit import (
    approximate_gap_decomposition,
    binary_classification,
    branch_interference_flags,
    calibration_table,
    expected_calibration_error,
    mask_bond_residuals,
    safe_correlation,
    stable_average_ranks,
    top_k_capture,
)


def test_residual_prediction_correlations_are_exact_for_linear_signal():
    target = [-2.0, -1.0, 1.0, 2.0]
    predicted = [-1.0, -0.5, 0.5, 1.0]
    assert safe_correlation(target, predicted) == pytest.approx(1.0)
    assert safe_correlation(target, predicted, rank=True) == pytest.approx(1.0)


def test_stable_average_ranks_use_average_rank_for_ties():
    np.testing.assert_array_equal(stable_average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])


def test_binary_prediction_quality_reports_precision_recall_and_f1():
    result = binary_classification([True, True, False, False], [True, False, True, False])
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_confidence_calibration_bins_cover_every_observation_once():
    table = calibration_table([0.0, 0.1, 0.9, 1.0], [0.0, 0.0, 1.0, 1.0], bins=2)
    assert sum(row["count"] for row in table) == 4
    assert expected_calibration_error(table) >= 0.0


def test_branch_decomposition_distinguishes_constructive_and_cancelling_bonds():
    flags = branch_interference_flags(
        target=[1.0, 1.0, -1.0, -1.0],
        cartesian=[0.4, 0.4, 0.4, -0.4],
        bond=[0.3, -0.3, 0.3, -0.3],
    )
    assert flags["constructive"].tolist() == [True, False, True, True]
    assert flags["cancellation"].tolist() == [False, True, False, False]
    assert flags["cartesian_correct_bond_harms"][1]
    assert flags["both_wrong"][2]


def test_counterfactual_masks_separate_ring_nonring_and_oracle_activity():
    residual = np.array([1.0, 2.0, 3.0, 4.0])
    ring = np.array([True, False, True, False])
    np.testing.assert_array_equal(
        mask_bond_residuals(residual, ring=ring, mode="ring_only"), [1.0, 0.0, 3.0, 0.0]
    )
    np.testing.assert_array_equal(
        mask_bond_residuals(residual, ring=ring, mode="nonring_only"), [0.0, 2.0, 0.0, 4.0]
    )
    np.testing.assert_array_equal(
        mask_bond_residuals(
            residual, ring=ring, mode="oracle_active", active=[False, True, True, False]
        ),
        [0.0, 2.0, 3.0, 0.0],
    )


def test_top_k_capture_detects_missed_severe_target_residual():
    assert top_k_capture([10.0, 1.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0], fraction=0.25) == 0.0


def test_oracle_gap_decomposition_exposes_nonadditive_remainder():
    result = approximate_gap_decomposition(0.8, {"prediction": 0.4, "confidence": 0.1})
    assert result["attributable_sum"] == pytest.approx(0.5)
    assert result["nonadditive_remainder"] == pytest.approx(0.3)
    assert result["exactly_additive"] is False

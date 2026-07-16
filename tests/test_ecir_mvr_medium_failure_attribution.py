from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etflow.ecir.failure_attribution import (
    classify_failure,
    leave_one_out_influence,
    molecule_equal_aggregate,
    paired_relative_bootstrap,
    relative_improvement,
    stage_gain_decomposition,
    threshold_bucket,
    threshold_margin,
    transition_labels,
)


def test_relative_improvement_recalculates_without_rounding():
    upstream = 0.2620318778529763
    candidate = 0.2358718493171036
    assert relative_improvement(upstream, candidate) == pytest.approx(
        0.09983529008081561, abs=1.0e-15
    )
    assert relative_improvement(upstream, candidate) < 0.10


def test_molecule_aggregation_weights_molecules_not_records():
    records = pd.DataFrame({
        "molecule_id": ["a", "a", "b"],
        "upstream": [1.0, 0.0, 0.0],
        "candidate": [0.0, 0.0, 0.0],
    })
    molecules, aggregate = molecule_equal_aggregate(records, ["upstream", "candidate"])
    assert molecules.loc["a", "upstream"] == 0.5
    assert aggregate["upstream"] == 0.25
    assert records.upstream.mean() == pytest.approx(1.0 / 3.0)


def test_threshold_margin_and_exact_boundary_are_stable():
    assert threshold_margin(1.0, 1.0, 2.0) == pytest.approx(0.0)
    assert threshold_margin(1.5, 1.0, 2.0) == pytest.approx(-1.0)
    assert threshold_margin(2.05, 1.0, 2.0) == pytest.approx(0.1)
    assert threshold_bucket(-0.05) == "-5_to_0pct"
    assert threshold_bucket(0.0) == "0_to_5pct"


def test_bond_transition_matrix_labels_all_four_cases():
    labels = transition_labels(
        [False, True, True, False],
        [False, False, True, True],
    )
    assert labels.tolist() == [
        "normal_to_normal", "outlier_to_normal",
        "outlier_to_outlier", "normal_to_outlier",
    ]


def test_stagewise_losses_telescope_to_target_gap():
    values = {
        "upstream": 0.30,
        "raw_proposal": 0.20,
        "atom_clipped_proposal": 0.21,
        "trust_clipped_proposal": 0.22,
        "safety_gated_proposal": 0.23,
        "accepted": 0.24,
        "minimal_target": 0.10,
    }
    result = stage_gain_decomposition(values)
    terms = (
        result["target_available_gain"] - result["raw_potential_gain"]
        + result["atom_clipping_loss"] + result["graph_clipping_loss"]
        + result["safety_gate_loss"] + result["acceptance_loss"]
    )
    assert terms == pytest.approx(result["target_gap"])


def test_failure_classification_is_rule_driven():
    base = {
        "upstream_bond_outlier_rate": 0.30,
        "target_available_gain": 0.20,
        "raw_potential_gain": 0.02,
        "clipping_loss": 0.0,
        "safety_gate_loss": 0.0,
        "acceptance_loss": 0.0,
        "accepted_gain": 0.02,
        "repaired_bond_count": 10,
        "new_outlier_count": 0,
        "threshold_near_fraction": 0.0,
        "bond_magnitude_improvement": 0.1,
    }
    assert classify_failure(base) == "MODEL_PROPOSAL_LIMITED"
    clipped = {**base, "raw_potential_gain": 0.20, "clipping_loss": 0.18}
    assert classify_failure(clipped) == "TRUST_CLIP_LIMITED"
    cancelled = {**base, "new_outlier_count": 5}
    assert classify_failure(cancelled) == "CANCELLATION_OR_NEW_OUTLIER"


def test_bootstrap_is_paired_molecule_level_and_deterministic():
    upstream = np.array([0.2, 0.4, 0.6, 0.8])
    candidate = upstream * 0.9
    first = paired_relative_bootstrap(upstream, candidate, draws=200, seed=42)
    second = paired_relative_bootstrap(upstream, candidate, draws=200, seed=42)
    assert first == second
    assert first["point_estimate"] == pytest.approx(0.10)
    assert 0.0 <= first["probability_ge_10pct"] <= 1.0


def test_leave_one_out_influence_reports_every_molecule():
    result = leave_one_out_influence(
        ["a", "b", "c"], [0.2, 0.4, 0.8], [0.1, 0.3, 0.8]
    )
    assert set(result.molecule_id) == {"a", "b", "c"}
    assert result.absolute_influence.is_monotonic_decreasing
    assert np.isfinite(result.influence_score).all()

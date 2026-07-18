from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from torch_geometric.data import Data

from etflow.ecir.bac_constraints import (
    angle_equivariant_directions,
    canonical_angle_triplets,
    sparse_clash_edges,
    stable_angle_cosine,
)
from etflow.ecir.bac_safety import (
    BACSafetyConfig,
    evaluate_bac_proposal,
    select_safe_bac_proposal,
)
from etflow.ecir.bac_target import BACMinimalTargetBuilder
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_v2_bac import (
    MCVRBACModel,
    V2_A_BOND_ONLY,
    V2_D_BOND_ANGLE_CLASH,
)
from etflow.ecir.mvr_v2_bac_loss import _per_graph_mean


def _batch() -> Data:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    x_input = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.0, 0.7, 0.0],
            [2.4, 1.7, 0.2],
        ],
        dtype=torch.float32,
    )
    return Data(
        num_nodes=4,
        node_attr=torch.tensor(
            [
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 1),
        batch=torch.zeros(4, dtype=torch.long),
        active_bond_constraint_index=torch.tensor(
            [[0, 1, 2], [1, 2, 3]], dtype=torch.long
        ),
        bond_allowed_range=torch.tensor(
            [[0.9, 1.6, 0.1], [0.9, 1.6, 0.1], [0.7, 1.3, 0.1]]
        ),
        active_angle_constraint_index=torch.tensor(
            [[0, 1], [1, 2], [2, 3]], dtype=torch.long
        ),
        angle_allowed_range=torch.tensor(
            [[1.2, 2.3, 0.1], [1.2, 2.3, 0.1]], dtype=torch.float32
        ),
        deterministic_error_features=torch.zeros(1, 10),
        active_mode_mask=torch.tensor([[1, 1, 0, 1, 0, 0]], dtype=torch.float32),
        difficulty_target=torch.zeros(1),
        affected_atom_mask=torch.ones(4),
        x_init=x_input,
        x_input=x_input,
        x_target=x_input + torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, 0.0]]
        ),
    )


def _model_kwargs() -> dict[str, object]:
    return {
        "hidden_dim": 16,
        "edge_hidden_dim": 16,
        "time_embedding_dim": 8,
        "num_layers": 2,
        "encoder_num_layers": 2,
        "error_embedding_dim": 8,
        "bond_head_enabled": True,
        "bond_explicit_alpha": 1.0,
        "torsion_gate_fixed_zero": True,
    }


def _rotation() -> torch.Tensor:
    value = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return value


def test_bond_only_exact_forward_and_strict_state_compatibility():
    torch.manual_seed(7)
    baseline = MCVRModel(**_model_kwargs()).eval()
    candidate = MCVRBACModel(
        **_model_kwargs(), bac_mode=V2_A_BOND_ONLY
    ).eval()
    missing, unexpected = candidate.load_d1b_state_dict(
        baseline.state_dict(), strict=True
    )
    assert missing == []
    assert unexpected == []
    assert set(candidate.state_dict()) == set(baseline.state_dict())
    batch = _batch()
    t = torch.tensor([0.3])
    with torch.inference_mode():
        expected = baseline(batch, batch.x_input, t)
        actual = candidate(batch, batch.x_input, t)
    for key in ("v_raw", "v_trust_clipped", "v_final", "velocity"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_bond_only_exact_loss_comparison():
    torch.manual_seed(17)
    baseline = MCVRModel(**_model_kwargs()).train()
    candidate = MCVRBACModel(
        **_model_kwargs(), bac_mode=V2_A_BOND_ONLY
    ).train()
    candidate.load_d1b_state_dict(baseline.state_dict(), strict=True)
    loss_fn = MCVRLoss()
    torch.manual_seed(23)
    expected = loss_fn(baseline, _batch())
    torch.manual_seed(23)
    actual = loss_fn(candidate, _batch())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_frozen_d1b_checkpoint_strict_load_when_available():
    path = Path(
        "artifacts/ecir_mvr/formal_large/d1_b_seed43/"
        "best_noninferior_validity.ckpt"
    )
    if not path.is_file():
        pytest.skip("frozen seed43 artifact is not present")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = yaml.safe_load(
        Path("configs/ecir_mvr_formal_large_d1b_base.yaml").read_text(
            encoding="utf-8"
        )
    )["model"]
    model = MCVRBACModel(**model_config, bac_mode=V2_A_BOND_ONLY)
    missing, unexpected = model.load_d1b_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    assert missing == []
    assert unexpected == []


def test_angle_triplet_canonicalization_and_endpoint_permutation():
    edge_index = torch.tensor([[2, 1, 0, 1], [1, 2, 1, 0]])
    triplets = canonical_angle_triplets(edge_index, 3)
    assert triplets.tolist() == [[0, 1, 2]]
    coordinates = _batch().x_input[:3]
    forward = stable_angle_cosine(coordinates, triplets)
    reverse = stable_angle_cosine(coordinates, triplets[:, [2, 1, 0]])
    torch.testing.assert_close(forward, reverse)
    first = angle_equivariant_directions(coordinates, triplets)
    second = angle_equivariant_directions(coordinates, triplets[:, [2, 1, 0]])
    torch.testing.assert_close(first[0], second[2])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[0])


@pytest.mark.parametrize("epsilon", [0.0, 1.0e-10, -1.0e-10])
def test_extreme_angles_are_finite(epsilon):
    coordinates = torch.tensor(
        [[-1.0, epsilon, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    triplets = torch.tensor([[0, 1, 2]])
    assert torch.isfinite(stable_angle_cosine(coordinates, triplets)).all()
    assert all(
        torch.isfinite(value).all()
        for value in angle_equivariant_directions(coordinates, triplets)
    )


def test_sparse_clash_excludes_12_and_13_and_is_deterministic(monkeypatch):
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.5, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    bonds = torch.tensor([[0, 1], [1, 2]])

    def forbidden(*args, **kwargs):
        raise AssertionError("dense torch.cdist must not be called")

    monkeypatch.setattr(torch, "cdist", forbidden)
    first = sparse_clash_edges(
        coordinates,
        bonds,
        cutoff=2.0,
        allowed_contact=1.0,
        exclude_topology_distance=2,
    )
    second = sparse_clash_edges(
        coordinates,
        bonds,
        cutoff=2.0,
        allowed_contact=1.0,
        exclude_topology_distance=2,
    )
    pairs = {tuple(value) for value in first["edge_index"].t().tolist()}
    assert (0, 1) not in pairs
    assert (0, 2) not in pairs
    assert (0, 3) in pairs
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_no_clash_edges_is_empty_and_finite():
    payload = sparse_clash_edges(
        torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        torch.empty(2, 0, dtype=torch.long),
        cutoff=2.0,
    )
    assert payload["edge_index"].shape == (2, 0)
    assert payload["penetration"].numel() == 0


def test_unified_model_is_se3_equivariant_and_has_single_delta():
    torch.manual_seed(13)
    model = MCVRBACModel(
        **_model_kwargs(), bac_mode=V2_D_BOND_ANGLE_CLASH
    ).eval()
    with torch.no_grad():
        model.angle_constraint_head[-1].weight.fill_(0.01)
        model.clash_constraint_head[-1].weight.fill_(0.01)
        model.constraint_fusion[-1].weight.fill_(0.01)
    batch = _batch()
    rotation = _rotation()
    translation = torch.tensor([3.0, -2.0, 1.0])
    transformed = batch.x_input @ rotation.T + translation
    with torch.inference_mode():
        original = model(batch, batch.x_input, torch.tensor([0.2]))
        rotated = model(batch, transformed, torch.tensor([0.2]))
    torch.testing.assert_close(
        rotated["v_final"], original["v_final"] @ rotation.T, atol=2.0e-5, rtol=2.0e-5
    )
    assert int(original["unified_delta_count"]) == 1
    source = inspect.getsource(MCVRBACModel.forward)
    assert "current =" not in source


def test_per_record_normalization_is_constraint_count_invariant():
    one = _per_graph_mean(torch.tensor([2.0]), torch.tensor([0]), 1)
    repeated = _per_graph_mean(
        torch.tensor([2.0, 2.0, 2.0]), torch.tensor([0, 0, 0]), 1
    )
    torch.testing.assert_close(one, repeated)


class _FakeValidity:
    def __init__(self):
        self.statistics = {"identity_sha256": "stats"}

    def _prepare(self, record):
        del record
        return {
            "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            "bonds": torch.tensor([[0, 1], [1, 2]]),
            "angles": torch.tensor([[0, 1, 2]]),
            "torsions": torch.empty(0, 4, dtype=torch.long),
            "rings": [],
            "centers": [],
            "bond_stats": torch.tensor([[0.9, 1.1, 0.1], [0.9, 1.1, 0.1]]),
            "ring_mask": torch.tensor([False, False]),
            "angle_stats": torch.tensor([[1.0, 2.5, 0.1]]),
            "planarity_stats": [],
        }

    def evaluate(self, coordinates, record, baseline_coordinates=None):
        del record
        coordinates = torch.as_tensor(coordinates)
        baseline = coordinates if baseline_coordinates is None else baseline_coordinates
        first = torch.linalg.vector_norm(coordinates[1] - coordinates[0])
        base_first = torch.linalg.vector_norm(baseline[1] - baseline[0])
        bond = float((first - 1.0).abs())
        base_bond = float((base_first - 1.0).abs())
        return {
            "bond_outlier_rate": bond,
            "bond_outlier_magnitude": bond,
            "angle_outlier_rate": 0.0,
            "angle_outlier_magnitude": 0.0,
            "ring_bond_outlier_rate": 0.0,
            "ring_planarity_outlier_rate": 0.0,
            "clash_penetration": 0.0,
            "severe_clash_rate": 0.0,
            "chirality_preserved": 1.0,
            "stereocenter_degenerate_rate": 0.0,
            "torsion_prior_outlier_score": 0.0,
            "total_thresholded_validity_score": bond,
            "base_bond": base_bond,
        }


def _record():
    return {
        "sample_id": "sample",
        "mol_id": "molecule",
        "atomic_numbers": torch.tensor([6, 6, 6]),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "num_rotatable_bonds": 0,
    }


def test_unified_target_is_deterministic_and_records_provenance():
    validity = _FakeValidity()
    builder = BACMinimalTargetBuilder(
        validity,
        {"max_steps": 3, "early_stop_patience": 2},
        source_identity_sha256="source",
    )
    source = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.4, 0.2, 0.0]])
    first = builder.build(source, _record())
    second = builder.build(source, _record())
    torch.testing.assert_close(first["x_target"], second["x_target"])
    assert first["target_metadata"]["unified_delta"] is True
    assert first["target_metadata"]["independent_target_sum"] is False
    assert first["target_metadata"]["ring_is_active_target"] is False
    assert first["target_metadata"]["test_records_read"] == 0


def test_safety_rejects_new_angle_and_rolls_back():
    class SafetyValidity:
        def evaluate(self, coordinates, record, baseline_coordinates=None):
            del record, baseline_coordinates
            angle = float(torch.as_tensor(coordinates)[2, 1].abs())
            return {
                "bond_outlier_rate": 0.0,
                "angle_outlier_rate": angle,
                "severe_clash_rate": 0.0,
                "clash_penetration": 0.0,
                "ring_bond_outlier_rate": 0.0,
                "ring_planarity_outlier_rate": 0.0,
                "chirality_preserved": 1.0,
                "stereocenter_degenerate_rate": 0.0,
            }

    source = torch.zeros(3, 3)
    delta = torch.zeros_like(source)
    delta[2, 1] = 0.1
    config = BACSafetyConfig(max_atom_displacement=1.0, max_molecule_rms_displacement=1.0)
    decision = evaluate_bac_proposal(
        source, source + delta, {}, SafetyValidity(), config
    )
    assert "new_angle_violation" in decision["reasons"]
    accepted, rollback = select_safe_bac_proposal(
        source, delta, {}, SafetyValidity(), config
    )
    torch.testing.assert_close(accepted, source)
    assert rollback["rolled_back"] is True

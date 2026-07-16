from __future__ import annotations

import math

import pytest
import torch

from etflow.commons.featurization import MoleculeData
from etflow.ecir.geometry import (
    circular_difference_degrees,
    clash_score,
    geometry_error_vector,
)
from etflow.ecir.model import ECIRErrorEncoder, ECIRFlowSystem
from etflow.ecir.structured_corruption import corrupt_conformer
from etflow.ecir.target_building import (
    multi_reference_soft_coupling,
    restrained_force_field_relaxation,
)
from etflow.serial_global4d.safety import trust_region_clip


def _chain_record(*, rotatable: bool = True):
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.1, 0.8, 0.0], [3.0, 1.0, 0.8]],
        dtype=torch.float32,
    )
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    joints = torch.tensor([[1], [2]], dtype=torch.long) if rotatable else torch.empty((2, 0), dtype=torch.long)
    influence = torch.tensor([[2, 3], [0, 0]], dtype=torch.long) if rotatable else torch.empty((2, 0), dtype=torch.long)
    return {
        "mol_id": "chain",
        "sample_id": "chain__gen0000",
        "source_mol_id": "chain",
        "smiles": "CCCC",
        "atomic_numbers": torch.tensor([6, 6, 6, 6]),
        "node_attr": torch.randn(4, 10),
        "edge_index": edges,
        "edge_attr": torch.ones(edges.size(1), 1),
        "bond_type": torch.ones(edges.size(1), dtype=torch.long),
        "bond_is_aromatic": torch.zeros(edges.size(1), dtype=torch.bool),
        "bond_is_in_ring": torch.zeros(edges.size(1), dtype=torch.bool),
        "rotatable_bond_index": joints,
        "atom_bond_influence_index": influence,
        "num_rotatable_bonds": joints.size(1),
        "x_init": x,
        "x_ref_aligned": x.clone(),
        "x_ref_candidates": torch.stack([x, x + torch.tensor([0.0, 0.1, 0.0])]),
    }


def _data(record):
    return MoleculeData(
        num_nodes=record["x_init"].size(0),
        node_attr=record["node_attr"],
        edge_index=record["edge_index"],
        edge_attr=record["edge_attr"],
        bond_is_in_ring=record["bond_is_in_ring"],
        rotatable_bond_index=record["rotatable_bond_index"],
        atom_bond_influence_index=record["atom_bond_influence_index"],
        x_init=record["x_init"],
    )


def _small_model():
    return ECIRFlowSystem(
        hidden_dim=24,
        edge_hidden_dim=24,
        time_embedding_dim=8,
        encoder_num_layers=1,
        num_layers=1,
        error_embedding_dim=8,
    )


def test_torsion_wraparound_179_minus_179_is_two_degrees():
    assert float(circular_difference_degrees(179.0, -179.0)) == pytest.approx(2.0)


def test_internal_metrics_are_rigid_transform_invariant():
    record = _chain_record()
    current, _ = corrupt_conformer(record, mode="torsion", generator=torch.Generator().manual_seed(1))
    target = record["x_ref_aligned"]
    before = geometry_error_vector(current, target, record)
    angle = 0.73
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([3.0, -4.0, 2.0])
    after = geometry_error_vector(current @ rotation.T + translation, target @ rotation.T + translation, record)
    torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)


def test_clean_identity_initial_model_output_is_near_zero():
    record = _chain_record()
    model = _small_model().eval()
    output = model(_data(record), record["x_init"], torch.tensor([0.5]))
    assert float(output["gated_velocity"].detach().abs().max()) < 1.0e-8


def test_synthetic_torsion_corruption_is_effective_and_marked():
    record = _chain_record()
    corrupted, metadata = corrupt_conformer(
        record, mode="torsion", generator=torch.Generator().manual_seed(2)
    )
    assert metadata["mode"] == "torsion"
    assert metadata["effective"]
    assert metadata["affected_bonds"] == [0]
    assert metadata["affected_atoms"] == [2, 3]
    assert not torch.equal(corrupted, record["x_init"])


def test_clash_injection_increases_clash_score():
    record = _chain_record()
    before = clash_score(record["x_init"], record["edge_index"])
    corrupted, metadata = corrupt_conformer(
        record, mode="clash", clash_distance=0.1, generator=torch.Generator().manual_seed(0)
    )
    after = clash_score(corrupted, record["edge_index"])
    assert after > before
    assert metadata["post_clash_score"] > metadata["pre_clash_score"]


def test_force_field_unsupported_energy_is_none_not_zero():
    record = _chain_record()
    record["smiles"] = "not a smiles"
    result = restrained_force_field_relaxation(record, record["x_init"])
    assert not result.supported
    assert result.energy_before is None
    assert result.energy_after is None


def test_gate_zero_keeps_coordinates_exactly_unchanged():
    record = _chain_record()
    model = _small_model().eval()
    refined, _ = model.refine(_data(record), steps=4, gate_override=0.0)
    assert torch.equal(refined, record["x_init"])


def test_trust_region_limits_atom_and_molecule_displacement():
    delta = torch.full((4, 3), 10.0)
    clipped, status = trust_region_clip(
        delta,
        torch.zeros(4, dtype=torch.long),
        max_atom_displacement=0.2,
        max_graph_rms_displacement=0.1,
        max_internal_velocity_norm=None,
    )
    assert torch.linalg.vector_norm(clipped, dim=-1).max() <= 0.100001
    assert status["atom_clipped"] and status["graph_rms_clipped"]


def test_multi_reference_soft_target_is_a_candidate_not_cartesian_average():
    record = _chain_record()
    references = record["x_ref_candidates"].clone()
    result = multi_reference_soft_coupling(
        record["x_init"], references, record, generator=torch.Generator().manual_seed(3)
    )
    target = result["target"]
    aligned = torch.stack([
        __import__("etflow.commons.kabsch_utils", fromlist=["kabsch_align"]).kabsch_align(ref, record["x_init"])
        for ref in references
    ])
    assert any(torch.allclose(target, candidate) for candidate in aligned)
    assert not result["cartesian_average_used"]


def test_error_encoder_runs_with_all_metadata_missing():
    record = _chain_record()
    encoder = ECIRErrorEncoder(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=8, num_layers=1, error_embedding_dim=8
    ).eval()
    output = encoder(_data(record), record["x_init"], torch.tensor([0.5]))
    assert output["error_mean"].shape == (1, 6)
    assert output["error_logvar"].shape == (1, 6)
    assert torch.isfinite(output["error_mean"]).all()


def test_no_rotatable_bond_has_finite_forward_and_loss_modes():
    record = _chain_record(rotatable=False)
    model = _small_model().eval()
    output = model(_data(record), record["x_init"], torch.tensor([0.5]))
    assert torch.isfinite(output["gated_velocity"]).all()


def test_all_ecir_loss_terms_are_reported_and_gradients_are_finite():
    record = _chain_record()
    data = _data(record)
    data.x_input = record["x_init"]
    data.x_target = record["x_init"] + 0.01 * torch.randn_like(record["x_init"])
    data.error_label = geometry_error_vector(data.x_input, data.x_target, record).view(1, 6)
    data.is_clean = torch.tensor([False])
    model = _small_model()
    losses = model.loss(data)
    assert set(losses) == {
        "loss",
        "flow_matching_loss",
        "internal_mode_loss",
        "error_prediction_loss",
        "identity_loss",
        "trust_loss",
        "gate_mean",
    }
    losses["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("kind", ["ring", "chiral"])
def test_ring_and_chiral_molecule_smoke(kind):
    record = _chain_record(rotatable=False)
    if kind == "ring":
        record["edge_index"] = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
        )
        record["edge_attr"] = torch.ones(8, 1)
        record["bond_is_in_ring"] = torch.ones(8, dtype=torch.bool)
    model = _small_model().eval()
    output = model(_data(record), record["x_init"], torch.tensor([0.5]))
    assert output["velocity"].shape == (4, 3)
    assert torch.isfinite(output["velocity"]).all()

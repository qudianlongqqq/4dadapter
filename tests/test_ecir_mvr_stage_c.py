from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
from torch_geometric.data import Data

from etflow.ecir.acceptance import evaluate_candidate, select_trajectory_candidate
from etflow.ecir.chemical_validity import ChemicalValidity, build_validity_reference_statistics
from etflow.ecir.minimal_validity_target import (
    MinimalValidityTargetBuilder,
    periodic_delta,
    thresholded_excess,
)
from etflow.ecir.model import ECIRFlowSystem
from etflow.ecir.mvr_dataset import balanced_sample_plan
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.structured_corruption import corrupt_conformer


def _embedded_record(smiles: str = "CC"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    conformer = mol.GetConformer()
    coordinates = torch.tensor([
        list(conformer.GetAtomPosition(index)) for index in range(mol.GetNumAtoms())
    ], dtype=torch.float32)
    edges, ring = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((a, b), (b, a)))
        ring.extend((bond.IsInRing(), bond.IsInRing()))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return {
        "mol_id": smiles, "source_mol_id": smiles, "sample_id": f"{smiles}__0",
        "smiles": smiles,
        "atomic_numbers": torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        "node_attr": torch.randn(mol.GetNumAtoms(), 10),
        "edge_index": edge_index,
        "edge_attr": torch.ones(edge_index.size(1), 1),
        "bond_is_in_ring": torch.tensor(ring, dtype=torch.bool),
        "rotatable_bond_index": torch.empty((2, 0), dtype=torch.long),
        "atom_bond_influence_index": torch.empty((2, 0), dtype=torch.long),
        "num_rotatable_bonds": 0,
        "x_init": coordinates,
        "x_ref_aligned": coordinates.clone(),
    }


def _validity(record):
    stats = build_validity_reference_statistics(
        [(record, [record["x_init"]] * 4)],
        train_split_sha256="unit-train-only",
        config={"minimum_sample_count": 1},
    )
    return ChemicalValidity(stats)


def test_thresholded_bond_penalty_is_zero_inside_envelope():
    value = torch.tensor([1.0, 1.5, 2.0])
    torch.testing.assert_close(
        thresholded_excess(value, torch.tensor(1.0), torch.tensor(2.0)),
        torch.zeros(3),
    )


def test_thresholded_bond_penalty_is_exact_outside_envelope():
    value = torch.tensor([0.7, 2.4])
    torch.testing.assert_close(
        thresholded_excess(value, torch.tensor(1.0), torch.tensor(2.0)),
        torch.tensor([0.3, 0.4]),
    )


def test_thresholded_angle_penalty_uses_radians_and_is_zero_inside():
    value = torch.tensor([math.pi / 2, 0.5, 2.5])
    expected = torch.tensor([0.0, 0.5, 0.5])
    torch.testing.assert_close(
        thresholded_excess(value, torch.tensor(1.0), torch.tensor(2.0)), expected
    )


def test_periodic_179_minus_179_is_two_degrees():
    a = torch.tensor(math.radians(179.0))
    b = torch.tensor(math.radians(-179.0))
    assert math.degrees(abs(float(periodic_delta(a, b)))) == pytest.approx(2.0, abs=1e-5)


def test_reasonable_ring_and_clean_input_remain_exact_identity():
    record = _embedded_record("C1CCCCC1")
    result = MinimalValidityTargetBuilder(_validity(record)).build(record["x_init"], record)
    assert result["target_metadata"]["target_status"] == "identity_clean"
    assert result["target_metadata"]["stop_reason"] == "already_valid"
    assert torch.equal(result["x_target"], record["x_init"])


def test_minimal_target_failure_returns_identity_without_reference_fallback():
    record = _embedded_record()
    distorted = record["x_init"].clone()
    distorted[0, 0] += 0.3
    builder = MinimalValidityTargetBuilder(
        _validity(record), {"max_steps": 2, "min_improvement": 1.0e9}
    )
    result = builder.build(distorted, record)
    metadata = result["target_metadata"]
    assert metadata["target_status"] == "identity_fallback"
    assert metadata["reference_fallback_used"] is False
    assert metadata["force_field_fallback_used"] is False
    assert torch.equal(result["x_target"], distorted)


class _FakeValidity:
    def evaluate(self, coordinates, record, baseline_coordinates=None):
        score = abs(float(torch.linalg.vector_norm(coordinates[1] - coordinates[0])) - 1.0)
        return _validity_dict(score)


def _validity_dict(score: float, **updates):
    value = {
        "bond_outlier_rate": score, "bond_outlier_magnitude": score,
        "angle_outlier_rate": 0.0, "angle_outlier_magnitude": 0.0,
        "severe_clash_rate": 0.0, "clash_penetration": 0.0,
        "ring_bond_outlier_rate": 0.0, "ring_planarity_outlier_rate": 0.0,
        "chirality_preserved": 1.0, "stereocenter_degenerate_rate": 0.0,
        "torsion_prior_outlier_score": 0.0,
        "total_thresholded_validity_score": score,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("candidate_update", "reason"),
    [
        ({"chirality_preserved": 0.0}, "chirality_flip"),
        ({"stereocenter_degenerate_rate": 1.0}, "stereocenter_degeneracy_increased"),
        ({"ring_planarity_outlier_rate": 1.0}, "ring_planarity_outlier_increased"),
    ],
)
def test_acceptance_rejects_new_chemical_risk(candidate_update, reason):
    record = _embedded_record()
    candidate = record["x_init"].clone()
    candidate[0, 0] += 0.01
    decision = evaluate_candidate(
        record["x_init"], candidate, record, _FakeValidity(), step=1,
        input_validity_override=_validity_dict(1.0),
        candidate_validity_override=_validity_dict(0.5, **candidate_update),
    )
    assert not decision.accepted
    assert reason in decision.reject_reasons
    assert "score_breakdown" in decision.metadata()
    assert "reject_reason" in decision.metadata()


def test_acceptance_rejects_atom_trust_limit():
    record = _embedded_record()
    candidate = record["x_init"].clone()
    candidate[0, 0] += 1.0
    decision = evaluate_candidate(
        record["x_init"], candidate, record, _FakeValidity(), step=1,
        input_validity_override=_validity_dict(1.0),
        candidate_validity_override=_validity_dict(0.5),
    )
    assert "atom_trust_radius" in decision.reject_reasons


def test_high_flex_torsion_limit_is_stricter_and_rejected():
    # Use a rotatable chain but mark it as high-flex for the safety policy.
    record = _embedded_record("CCCC")
    record["rotatable_bond_index"] = torch.tensor([[1], [2]], dtype=torch.long)
    record["atom_bond_influence_index"] = torch.tensor(
        [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0] * 12], dtype=torch.long
    )
    record["num_rotatable_bonds"] = 6
    candidate, _ = corrupt_conformer(
        record, coordinates=record["x_init"], mode="torsion",
        torsion_amplitude_degrees=30.0, generator=torch.Generator().manual_seed(2),
    )
    decision = evaluate_candidate(
        record["x_init"], candidate, record, _FakeValidity(), step=1,
        config={"max_atom_displacement": 5.0, "max_molecule_rms_displacement": 5.0},
        input_validity_override=_validity_dict(1.0),
        candidate_validity_override=_validity_dict(0.5),
    )
    assert "torsion_trust_radius" in decision.reject_reasons


def test_best_of_trajectory_selects_improving_intermediate_step():
    record = _embedded_record()
    first = record["x_init"].clone(); first[1] = first[0] + torch.tensor([1.0, 0.0, 0.0])
    final = record["x_init"].clone(); final[1] = final[0] + torch.tensor([1.4, 0.0, 0.0])
    selected, decision = select_trajectory_candidate(
        record["x_init"], [first, final], record, _FakeValidity(),
        config={"max_atom_displacement": 5.0, "max_molecule_rms_displacement": 5.0},
    )
    assert decision.selected_step == 1
    torch.testing.assert_close(selected, first)


def test_source_balanced_plan_has_exact_default_ratios_and_no_extreme():
    rows = []
    for source in ("ETFlow", "Cartesian"):
        for severity in ("normal", "mild", "medium", "severe", "out_of_domain_extreme"):
            for index in range(10):
                rows.append({
                    "generator_name": source, "source_severity": severity,
                    "sample_id": f"{source}-{severity}-{index}",
                })
    frame = pd.DataFrame(rows)
    plan = balanced_sample_plan(frame, 100, seed=42)
    assert {name: sum(item["sample_type"] == name for item in plan) for name in (
        "real_error", "synthetic_error", "clean_identity"
    )} == {"real_error": 45, "synthetic_error": 30, "clean_identity": 25}
    assert all(item["severity"] != "out_of_domain_extreme" for item in plan)
    real_sources = pd.Series([item["source"] for item in plan if item["sample_type"] == "real_error"])
    assert real_sources.value_counts().max() <= 23
    mixed = sum(item["corruption_type"] == "mixed" for item in plan)
    assert mixed <= 0.30 * 30


def _loss_data(active: torch.Tensor):
    record = _embedded_record()
    x = record["x_init"]
    return Data(
        num_nodes=x.size(0), node_attr=record["node_attr"], edge_index=record["edge_index"],
        edge_attr=record["edge_attr"], rotatable_bond_index=record["rotatable_bond_index"],
        bond_is_in_ring=record["bond_is_in_ring"], x_input=x, x_init=x,
        x_target=x + 0.01 * torch.randn_like(x), active_mode_mask=active.view(1, 6),
        affected_atom_mask=torch.ones(x.size(0)),
        deterministic_error_features=torch.zeros(1, 10), difficulty_target=torch.zeros(1),
    )


def test_active_mode_mask_zero_makes_corresponding_validity_loss_zero():
    data = _loss_data(torch.zeros(6))
    model = MCVRModel(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=8,
        num_layers=1, encoder_num_layers=1, error_embedding_dim=8,
    )
    losses = MCVRLoss()(model, data)
    assert float(losses["validity_mode_loss"]) == 0.0


def test_mvr_cartesian_head_is_finite_with_no_rotatable_bond():
    data = _loss_data(torch.zeros(6))
    model = MCVRModel(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=8,
        num_layers=1, encoder_num_layers=1, error_embedding_dim=8,
    )
    output = model(data, data.x_input, torch.tensor([0.5]))
    assert output["v_final"].shape == data.x_input.shape
    assert torch.isfinite(output["v_final"]).all()
    assert float(output["torsion_gate"].detach().max()) == 0.0


def test_run_a_fixed_zero_torsion_gate_and_branch_contribution():
    data = _loss_data(torch.ones(6))
    data.deterministic_error_features[0, 6] = 2.0
    model = MCVRModel(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=8,
        num_layers=1, encoder_num_layers=1, error_embedding_dim=8,
        torsion_scale=0.0, high_flex_torsion_scale=0.0,
        torsion_gate_fixed_zero=True,
    )
    output = model(data, data.x_input, torch.tensor([0.5]))
    assert torch.equal(output["torsion_gate"], torch.zeros_like(output["torsion_gate"]))
    assert torch.equal(
        output["v_torsion_contribution"],
        torch.zeros_like(output["v_torsion_contribution"]),
    )
    torch.testing.assert_close(output["v_raw"], output["v_rigid_contribution"])


def test_frozen_old_ecir_checkpoint_still_loads_strictly_when_available():
    checkpoint_path = Path("logs_ecir/stage2_heterogeneous_500_100_5k/step005000.ckpt")
    if not checkpoint_path.is_file():
        pytest.skip("local frozen ECIR checkpoint is not present")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]
    model = ECIRFlowSystem(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def test_stage_b_decision_remains_rescued_and_test_free():
    path = Path("diagnostics/ecir_mvr/stage_b/decision.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["decision"] == "EXISTING_CKPT_RESCUED"
    assert result["checkpoint_sha256"] == "232e47865d01a71543cf2cd16ede577764fd3d94ac843d78dcdcf8c9789fa98d"

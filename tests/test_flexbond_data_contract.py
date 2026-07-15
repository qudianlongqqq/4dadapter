import pytest
import torch

from etflow.data.flexbond_optimizer_dataset import (
    validate_cache_record,
    validate_rmsd_diagnostic,
)
from etflow.data.flexbond_cache_schema import strict_reference_lookup, x_init_sha256


def _record():
    record = {
        "mol_id": "mol__gen0000",
        "atomic_numbers": torch.tensor([6, 6]),
        "num_atoms": 2,
        "node_attr": torch.zeros(2, 10),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.zeros(2, 1),
        "bond_type": torch.zeros(2, dtype=torch.long),
        "bond_is_aromatic": torch.zeros(2, dtype=torch.bool),
        "bond_is_in_ring": torch.zeros(2, dtype=torch.bool),
        "rotatable_bond_mask": torch.ones(2, dtype=torch.bool),
        "rotatable_bond_index": torch.tensor([[0], [1]]),
        "atom_bond_influence_index": torch.tensor([[1], [0]]),
        "x_init": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "x_ref_candidates": torch.tensor([[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]]),
        "x_init_atomic_numbers": torch.tensor([6, 6]),
        "x_ref_atomic_numbers": torch.tensor([6, 6]),
        "topology_signature": "same",
        "x_init_topology_signature": "same",
        "x_ref_topology_signature": "same",
        "cache_schema_version": "2.0",
        "generator_name": "test",
        "generator_checkpoint": "checkpoint.ckpt",
        "sample_seed": 1,
        "DATA_DIR": "data",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    record["x_init_hash"] = x_init_sha256(record["x_init"], record["atomic_numbers"])
    return record


def _record_with_persisted_pair():
    record = _record()
    checked = validate_cache_record(record)
    record.update(
        {
            "x_ref": checked["x_ref"],
            "x_ref_aligned": checked["x_ref_aligned"],
            "selected_reference_index": checked["selected_reference_index"],
            "selected_reference_rmsd": checked["selected_rmsd"],
            "selected_ref_id": "mol__gen0000__ref0000",
            "rmsd_before": checked["rmsd_before"],
            "rmsd_after": checked["rmsd_after"],
        }
    )
    return record


def test_external_reference_lookup_rejects_positional_fallback():
    with pytest.raises(ValueError, match="no stable"):
        strict_reference_lookup([("0", {"atomic_numbers": [6, 6]})])


def test_external_reference_lookup_keeps_same_smiles_records_distinct():
    first = {"source_mol_id": "source-a", "smiles": "CCO"}
    second = {"source_mol_id": "source-b", "smiles": "CCO"}
    lookup = strict_reference_lookup([("first", first), ("second", second)])
    assert lookup[("record_id", "source-a")] is first
    assert lookup[("record_id", "source-b")] is second
    assert ("smiles", "CCO") not in lookup


def test_graph_contract_accepts_reciprocal_typed_edges():
    checked = validate_cache_record(_record())
    assert checked["selected_reference_index"] == 0


def test_graph_contract_rejects_missing_reciprocal_edge():
    record = _record()
    for key in (
        "edge_index",
        "edge_attr",
        "bond_type",
        "bond_is_aromatic",
        "bond_is_in_ring",
    ):
        record[key] = record[key][:, :1] if key == "edge_index" else record[key][:1]
    record["rotatable_bond_mask"] = record["rotatable_bond_mask"][:1]
    with pytest.raises(ValueError, match="reciprocal"):
        validate_cache_record(record)


def test_atom_map_order_mismatch_is_rejected():
    record = _record()
    record["atom_map_ids"] = torch.tensor([10, 20])
    record["x_init_atom_map_ids"] = torch.tensor([10, 20])
    record["x_ref_atom_map_ids"] = torch.tensor([20, 10])
    with pytest.raises(ValueError, match="atom_map_ids"):
        validate_cache_record(record)


def test_rmsd_diagnostic_exact_value_passes():
    result = validate_rmsd_diagnostic(1.25, 1.25)
    assert result["validation_status"] == "PASS"
    assert result["absolute_delta"] == 0.0


def test_rmsd_diagnostic_known_float32_drift_passes_numerically_close():
    result = validate_rmsd_diagnostic(575.6069946289062, 575.6070556640625)
    assert result["validation_status"] == "PASS_NUMERICALLY_CLOSE"
    assert result["absolute_delta"] == pytest.approx(6.103515625e-05)
    assert result["effective_tolerance"] == pytest.approx(5.756070556640625e-04)


def test_persisted_selected_reference_index_mismatch_fails():
    record = _record_with_persisted_pair()
    record["selected_reference_index"] = 1
    with pytest.raises(ValueError, match="selected_reference_index"):
        validate_cache_record(record, require_persisted_pair=True)


def test_persisted_x_init_hash_mismatch_fails():
    record = _record_with_persisted_pair()
    record["x_init_hash"] = "wrong"
    with pytest.raises(ValueError, match="x_init_hash"):
        validate_cache_record(record, require_persisted_pair=True)


def test_rmsd_diagnostic_material_mismatch_fails():
    with pytest.raises(ValueError, match="stale or incorrect"):
        validate_rmsd_diagnostic(1.0, 1.001)


@pytest.mark.parametrize(
    "persisted,recomputed", [(float("nan"), 1.0), (1.0, float("inf"))]
)
def test_rmsd_diagnostic_nonfinite_fails(persisted, recomputed):
    with pytest.raises(ValueError, match="not numerically valid"):
        validate_rmsd_diagnostic(persisted, recomputed)


def test_float32_cpu_gpu_scale_drift_passes_when_available():
    cpu_value = torch.tensor(575.6069946289062, dtype=torch.float32)
    recomputed = torch.nextafter(cpu_value, torch.tensor(float("inf")))
    result = validate_rmsd_diagnostic(float(cpu_value), float(recomputed))
    assert result["validation_status"] == "PASS_NUMERICALLY_CLOSE"
    if torch.cuda.is_available():
        gpu_value = cpu_value.cuda().item()
        gpu_result = validate_rmsd_diagnostic(float(cpu_value), gpu_value)
        assert gpu_result["validation_status"] == "PASS"


def test_rmsd_validation_does_not_change_stage2_targets():
    record = _record_with_persisted_pair()
    record["q_res_star"] = torch.randn(2, 4)
    record["r_J_star"] = torch.randn(2, 3)
    q_before = record["q_res_star"].clone()
    r_before = record["r_J_star"].clone()
    validate_cache_record(record, require_persisted_pair=True)
    assert torch.equal(record["q_res_star"], q_before)
    assert torch.equal(record["r_J_star"], r_before)

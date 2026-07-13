import pytest
import torch

from etflow.data.flexbond_optimizer_dataset import validate_cache_record
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
    record["x_init_hash"] = x_init_sha256(
        record["x_init"], record["atomic_numbers"]
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
    for key in ("edge_index", "edge_attr", "bond_type", "bond_is_aromatic", "bond_is_in_ring"):
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

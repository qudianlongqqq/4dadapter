from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from rdkit import Chem

from etflow.ecir import formal_target_assets as assets
from etflow.ecir import formal_rdkit_adapter as rdkit_adapter
from etflow.ecir.minimal_validity_target import MinimalValidityConfig
from etflow.ecir.target_building import _record_to_rdkit_mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ecir_mvr_formal_large_minimal_targets.yaml"
PARQUET_AVAILABLE = any(
    importlib.util.find_spec(name) is not None for name in ("pyarrow", "fastparquet")
)


class FakeBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, coordinates, record):
        self.calls += 1
        target = torch.as_tensor(coordinates, dtype=torch.float32).clone()
        return {
            "x_target": target,
            "target_metadata": {
                "target_status": "identity_clean",
                "stop_reason": "already_valid",
                "validity_gain": 0.0,
                "initial_to_target_rmsd": 0.0,
                "max_atom_displacement": 0.0,
                "torsion_change": 0.0,
                "max_rotatable_torsion_change": 0.0,
                "selected_step": 0,
                "target_sha256": assets.tensor_sha256(target),
                "reference_fallback_used": False,
                "force_field_fallback_used": False,
                "optimizer_config": asdict(MinimalValidityConfig()),
            },
        }


class FormalRecordingBuilder(FakeBuilder):
    def build(self, coordinates, record):
        assert record["_formal_rdkit_adapter_schema"] == rdkit_adapter.FORMAL_ADAPTER_SCHEMA
        return super().build(coordinates, record)


def _identities():
    builder_path = ROOT / "etflow/ecir/minimal_validity_target.py"
    adapter_path = ROOT / "etflow/ecir/formal_rdkit_adapter.py"
    validity_path = ROOT / "data/ecir_mvr/validity_reference_stats.json"
    return {
        "builder_code_path": str(builder_path.resolve()),
        "builder_code_sha256": assets.file_sha256(builder_path),
        "formal_rdkit_adapter_path": str(adapter_path.resolve()),
        "formal_rdkit_adapter_sha256": assets.file_sha256(adapter_path),
        "builder_config_sha256": assets.canonical_sha256(
            asdict(MinimalValidityConfig())
        ),
        "target_builder_config": asdict(MinimalValidityConfig()),
        "validity_statistics_path": str(validity_path.resolve()),
        "validity_statistics_sha256": assets.file_sha256(validity_path),
        "validity_statistics_identity_sha256": assets.STAGE_D_VALIDITY_IDENTITY,
        "stage_d_target_identity_sha256": "c" * 64,
        "config_file_sha256": "b" * 64,
    }


def _source(tmp_path: Path, split: str, suffix: str):
    sample_id = f"{split}::sample-{suffix}"
    molecule_id = f"molecule-{split}-{suffix}"
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = Chem.MolFromSmiles("[CH3:1][CH3:2]")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    path = tmp_path / f"{split}-{suffix}.pt"
    torch.save(
        {
            "sample_id": sample_id,
            "mol_id": sample_id,
            "source_record_id": molecule_id,
            "smiles": "[CH3:1][CH3:2]",
            "atomic_numbers": torch.tensor([6, 6]),
            "x_init_atomic_numbers": torch.tensor([6, 6]),
            "atom_map_ids": torch.tensor([1, 2]),
            "x_init_atom_map_ids": torch.tensor([1, 2]),
            "x_ref_atom_map_ids": torch.tensor([1, 2]),
            "num_atoms": 2,
            "node_attr": torch.zeros(2, 10),
            "edge_index": edge_index,
            "edge_attr": torch.zeros(2, 1),
            "bond_type": torch.zeros(2, dtype=torch.long),
            "bond_is_aromatic": torch.zeros(2, dtype=torch.bool),
            "bond_is_in_ring": torch.zeros(2, dtype=torch.bool),
            "rotatable_bond_index": torch.empty(2, 0, dtype=torch.long),
            "atom_bond_influence_index": torch.empty(2, 0, dtype=torch.long),
            "x_init": coordinates,
            "topology_signature": rdkit_adapter._ordered_topology_signature(mol),
        },
        path,
    )
    return {
        "schema_version": assets.SOURCE_SCHEMA,
        "split": split,
        "sample_id": sample_id,
        "molecule_id": molecule_id,
        "generator_name": "ETFlow_formal_upstream",
        "source_severity": "normal",
        "source_path": str(path.resolve()),
        "coordinate_path": None,
        "coordinate_key": "x_init",
        "coordinate_sha256": assets.tensor_sha256(coordinates),
        "source_file_sha256": assets.file_sha256(path),
        "num_atoms": 2,
        "test_record": False,
    }


def _explicit_hydrogen_record(sample_id: str):
    smiles = "CSc1nc(=NC(C)=O)ss1"
    base = Chem.MolFromSmiles(smiles)
    with_hydrogens = Chem.AddHs(base)
    order = [0, 11, 12, 13, 1, 2, 3, 4, 5, 6, 8, 7, 14, 15, 16, 9, 10]
    cache_mol = Chem.RenumberAtoms(with_hydrogens, order)
    edges = []
    bond_types = []
    aromatic = []
    in_ring = []
    names = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    for bond in cache_mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        value = names.index(str(bond.GetBondType()))
        for source, target in ((left, right), (right, left)):
            edges.append((source, target))
            bond_types.append(value)
            aromatic.append(bond.GetIsAromatic())
            in_ring.append(bond.IsInRing())
    atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in cache_mol.GetAtoms()], dtype=torch.long
    )
    atom_maps = torch.arange(len(atomic_numbers), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    bond_type = torch.tensor(bond_types, dtype=torch.long)
    return {
        "sample_id": sample_id,
        "mol_id": sample_id,
        "source_record_id": "CSc1nc(=NC(C)=O)ss1",
        "source_mol_id": "CSc1nc(=NC(C)=O)ss1",
        "smiles": smiles,
        "atomic_numbers": atomic_numbers,
        "x_init_atomic_numbers": atomic_numbers.clone(),
        "atom_map_ids": atom_maps,
        "x_init_atom_map_ids": atom_maps.clone(),
        "x_ref_atom_map_ids": atom_maps.clone(),
        "num_atoms": len(atomic_numbers),
        "node_attr": torch.zeros(len(atomic_numbers), 10),
        "edge_index": edge_index,
        "edge_attr": bond_type[:, None].float(),
        "bond_type": bond_type,
        "bond_is_aromatic": torch.tensor(aromatic, dtype=torch.bool),
        "bond_is_in_ring": torch.tensor(in_ring, dtype=torch.bool),
        "rotatable_bond_index": torch.empty(2, 0, dtype=torch.long),
        "atom_bond_influence_index": torch.empty(2, 0, dtype=torch.long),
        "x_init": torch.arange(len(atomic_numbers) * 3, dtype=torch.float32).reshape(-1, 3)
        / 100.0,
        "topology_signature": rdkit_adapter._ordered_topology_signature(cache_mol),
    }


def _formal_source(tmp_path: Path, record, split="train"):
    path = tmp_path / f"{assets.target_key(record['sample_id'])}.pt"
    torch.save(record, path)
    return {
        "schema_version": assets.SOURCE_SCHEMA,
        "split": split,
        "sample_id": record["sample_id"],
        "molecule_id": record["source_record_id"],
        "generator_name": "ETFlow_formal_upstream",
        "source_severity": "normal",
        "source_path": str(path.resolve()),
        "coordinate_path": None,
        "coordinate_key": "x_init",
        "coordinate_sha256": assets.tensor_sha256(record["x_init"]),
        "source_file_sha256": assets.file_sha256(path),
        "num_atoms": int(record["num_atoms"]),
        "test_record": False,
    }


def _write_inventory(output: Path, frames, identities):
    for split, frame in frames.items():
        assets.atomic_parquet(frame, output / "real_sources" / f"{split}.parquet")
    manifest_metadata = assets.finalize_manifests(output, frames, shard_size=1)
    source_metadata = {
        "formal_source_identity_sha256": assets.canonical_sha256(
            {
                split: frame[["sample_id", "source_file_sha256"]].to_dict("records")
                for split, frame in frames.items()
            }
        )
    }
    assets.write_asset_metadata_and_inventory(
        output_root=output,
        source_frames=frames,
        source_metadata=source_metadata,
        manifest_metadata=manifest_metadata,
        identities=identities,
        config_file_sha256="b" * 64,
    )


def test_config_freezes_stage_d_builder_and_formal_counts(tmp_path):
    config = assets.load_config(CONFIG, output_root=tmp_path)
    assert config["target_builder"] == asdict(MinimalValidityConfig())
    assert config["splits"]["train"] == {
        "expected_molecules": 50_000,
        "expected_records_per_molecule": 3,
    }
    assert config["splits"]["val"] == {
        "expected_molecules": 5_000,
        "expected_records_per_molecule": 2,
    }
    assert config["splits"]["test"] == {"enabled": False}
    assert config["pilot_records"] == 100


def test_formal_explicit_hydrogen_mapping_renumbers_to_cache_order():
    record = _explicit_hydrogen_record(
        "train::CSc1nc(=NC(C)=O)ss1__gen0000"
    )
    plain = Chem.MolFromSmiles("CSc1nc(=NC(C)=O)ss1")
    default_hydrogens = Chem.AddHs(plain)
    assert plain.GetNumAtoms() == 11
    assert Chem.MolFromSmiles(record["smiles"]).GetNumAtoms() == 11
    assert default_hydrogens.GetNumAtoms() == record["num_atoms"] == 17
    assert [atom.GetAtomicNum() for atom in default_hydrogens.GetAtoms()] != record[
        "atomic_numbers"
    ].tolist()

    adapted = rdkit_adapter.adapt_formal_cache_record(record)
    mol = adapted["_formal_rdkit_mol"]
    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == record[
        "atomic_numbers"
    ].tolist()
    assert adapted["_formal_cache_identity_kind"] == "zero_based_cache_position"
    assert adapted["_formal_rdkit_original_order"] == (
        0,
        11,
        12,
        13,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        7,
        14,
        15,
        16,
        9,
        10,
    )
    equivalence_classes = adapted["_formal_topology_equivalence_classes"]
    assert (1, 2, 3) in equivalence_classes
    assert (12, 13, 14) in equivalence_classes
    assert rdkit_adapter._ordered_topology_signature(mol) == record[
        "topology_signature"
    ]
    assert adapted["_formal_cache_to_rdkit"] == tuple(range(17))
    consumer_mol, consumer_mapping = _record_to_rdkit_mapping(adapted)
    assert consumer_mapping == {index: index for index in range(17)}
    assert [atom.GetAtomicNum() for atom in consumer_mol.GetAtoms()] == record[
        "atomic_numbers"
    ].tolist()


def test_three_explicit_hydrogen_failures_reach_builder_and_persist_targets(tmp_path):
    output = tmp_path / "output"
    builder = FormalRecordingBuilder()
    rows = []
    for index in range(3):
        sample_id = f"train::CSc1nc(=NC(C)=O)ss1__gen{index:04d}"
        record = _explicit_hydrogen_record(sample_id)
        record["x_init"] = record["x_init"] + index * 0.001
        source = _formal_source(tmp_path, record)
        row, skipped = assets.build_target(
            source,
            output_root=output,
            builder=builder,
            identities=_identities(),
            config_file_sha256="b" * 64,
        )
        assert not skipped
        rows.append(row)
    assert builder.calls == 3
    assert all(Path(row["target_cache_path"]).is_file() for row in rows)


def test_equivalent_methyl_hydrogens_have_deterministic_lexical_mapping():
    record = _explicit_hydrogen_record(
        "train::CSc1nc(=NC(C)=O)ss1__gen0000"
    )
    adapted = rdkit_adapter.adapt_formal_cache_record(record)
    cache_bonds = rdkit_adapter._cache_bonds(record)
    cache_graph = rdkit_adapter._typed_graph(
        tuple(record["atomic_numbers"].tolist()), cache_bonds, heavy_only=False
    )
    rdkit_graph = rdkit_adapter._rdkit_graph(
        adapted["_formal_rdkit_mol"], heavy_only=False
    )
    matcher = rdkit_adapter.nx.algorithms.isomorphism.GraphMatcher(
        cache_graph,
        rdkit_graph,
        node_match=lambda left, right: left["z"] == right["z"],
        edge_match=lambda left, right: left["bond_type"] == right["bond_type"],
    )
    mappings = list(matcher.isomorphisms_iter())
    assert len(mappings) >= 36
    selected = tuple(range(17))
    assert selected == min(
        tuple(mapping[index] for index in range(17)) for mapping in mappings
    )


def test_positional_identity_ignores_ordered_smiles_rdkit_atom_maps():
    record = _explicit_hydrogen_record(
        "train::CSc1nc(=NC(C)=O)ss1__gen0000"
    )
    mol = Chem.AddHs(Chem.MolFromSmiles(record["smiles"]))
    cache_order = [0, 11, 12, 13, 1, 2, 3, 4, 5, 6, 8, 7, 14, 15, 16, 9, 10]
    ordered = Chem.RenumberAtoms(mol, cache_order)
    for index, atom in enumerate(ordered.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    record["smiles"] = Chem.MolToSmiles(
        ordered,
        canonical=False,
        allHsExplicit=True,
        allBondsExplicit=True,
        isomericSmiles=True,
    )
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    assert Chem.MolFromSmiles(record["smiles"]).GetNumAtoms() == 11
    assert Chem.MolFromSmiles(record["smiles"], parser).GetNumAtoms() == 17

    adapted = rdkit_adapter.adapt_formal_cache_record(record)
    assert adapted["_formal_cache_identity_kind"] == "zero_based_cache_position"
    assert all(
        atom.GetAtomMapNum() == 0
        for atom in adapted["_formal_rdkit_mol"].GetAtoms()
    )
    assert rdkit_adapter._ordered_topology_signature(
        adapted["_formal_rdkit_mol"]
    ) == record["topology_signature"]


def test_positional_identity_must_match_x_init_and_x_ref():
    record = _explicit_hydrogen_record(
        "train::CSc1nc(=NC(C)=O)ss1__gen0000"
    )
    record["x_ref_atom_map_ids"] = torch.roll(
        record["x_ref_atom_map_ids"], shifts=1
    )
    with pytest.raises(ValueError, match="positional identity differs"):
        rdkit_adapter.adapt_formal_cache_record(record)


def test_explicit_hydrogen_mapping_with_invalid_topology_fails_closed():
    record = _explicit_hydrogen_record(
        "train::CSc1nc(=NC(C)=O)ss1__gen0000"
    )
    edge_index = record["edge_index"].clone()
    hydrogen_edges = torch.nonzero(
        (edge_index[0] == 0) & (edge_index[1] == 1), as_tuple=False
    ).view(-1)
    reverse_edges = torch.nonzero(
        (edge_index[0] == 1) & (edge_index[1] == 0), as_tuple=False
    ).view(-1)
    edge_index[0, hydrogen_edges] = 11
    edge_index[1, reverse_edges] = 11
    record["edge_index"] = edge_index
    with pytest.raises(ValueError, match="not uniquely proven"):
        rdkit_adapter.adapt_formal_cache_record(record)


def test_config_rejects_any_builder_parameter_change(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["target_builder"]["learning_rate"] = 0.002
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="differs from frozen Stage D"):
        assets.load_config(path, output_root=tmp_path)


def test_stage_d_metadata_and_validity_identities_are_reused(tmp_path):
    config = assets.load_config(CONFIG, output_root=tmp_path)
    identities = assets.verify_stage_d_identities(config)
    assert identities["validity_statistics_identity_sha256"] == assets.STAGE_D_VALIDITY_IDENTITY
    assert len(identities["builder_code_sha256"]) == 64
    assert len(identities["stage_d_target_identity_sha256"]) == 64


def test_test_paths_are_rejected_without_enumeration(tmp_path):
    with pytest.raises(ValueError, match="test path is forbidden"):
        assets.forbid_test_path(tmp_path / "test")
    assert assets.forbid_test_path(tmp_path / "train").name == "train"


def test_target_build_is_atomic_resumable_and_never_recomputed(tmp_path):
    source = _source(tmp_path, "train", "a")
    builder = FakeBuilder()
    identities = _identities()
    row, skipped = assets.build_target(
        source,
        output_root=tmp_path / "output",
        builder=builder,
        identities=identities,
        config_file_sha256="b" * 64,
    )
    assert not skipped and builder.calls == 1
    target_path = Path(row["target_cache_path"])
    original_mtime = target_path.stat().st_mtime_ns
    second, skipped = assets.build_target(
        source,
        output_root=tmp_path / "output",
        builder=builder,
        identities=identities,
        config_file_sha256="b" * 64,
    )
    assert skipped and builder.calls == 1
    assert second == row
    assert target_path.stat().st_mtime_ns == original_mtime
    assert not list(target_path.parent.glob("*.tmp.*"))


def test_partial_target_state_is_rejected(tmp_path):
    source = _source(tmp_path, "train", "partial")
    output = tmp_path / "output"
    target, _ = assets.target_paths(output, "train", source["sample_id"])
    target.parent.mkdir(parents=True)
    torch.save({"partial": True}, target)
    with pytest.raises(ValueError, match="partial target state"):
        assets.build_target(
            source,
            output_root=output,
            builder=FakeBuilder(),
            identities=_identities(),
            config_file_sha256="b" * 64,
        )


def test_failure_reason_is_persisted_and_can_be_resolved(tmp_path):
    source = _source(tmp_path, "train", "failure")
    output = tmp_path / "output"
    assets.record_failure(source, output, RuntimeError("first"))
    assets.record_failure(source, output, ValueError("second"))
    assert assets.failure_count(output) == 1
    path = next((output / "manifests/failures/train").glob("*.json"))
    value = json.loads(path.read_text())
    assert [row["error"] for row in value["attempts"]] == ["first", "second"]
    assert assets.unresolved_failure_sample_ids(output) == {source["sample_id"]}
    assets.clear_failure(source, output)
    assert assets.failure_count(output) == 0
    assert assets.unresolved_failure_sample_ids(output) == set()


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet engine is not installed")
def test_full_validator_requires_pairing_atom_order_sha_and_no_failures(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(assets, "validate_cache_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        assets,
        "strict_mvr_dataset_load",
        lambda output_root, identities, sample_count: {"train": 1, "val": 1},
    )
    output = tmp_path / "output"
    train = _source(tmp_path, "train", "one")
    val = _source(tmp_path, "val", "one")
    builder = FakeBuilder()
    identities = _identities()
    for source in (train, val):
        assets.build_target(
            source,
            output_root=output,
            builder=builder,
            identities=identities,
            config_file_sha256="b" * 64,
        )
    frames = {"train": pd.DataFrame([train]), "val": pd.DataFrame([val])}
    _write_inventory(output, frames, identities)
    result = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=identities,
        require_complete=True,
        strict_sample_count=2,
    )
    assert result["decision"] == "D1B_FORMAL_TARGETS_READY"
    assert all(result["criteria"].values())
    assert result["test_records_read"] == 0

    target_path = Path(
        pd.read_parquet(output / "minimal_targets/train.parquet")
        .iloc[0]
        .target_cache_path
    )
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    payload["source_atomic_numbers"] = torch.tensor([6, 8])
    torch.save(payload, target_path)
    broken = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=identities,
        require_complete=True,
        strict_sample_count=2,
    )
    assert broken["decision"] == "D1B_FORMAL_TARGETS_NOT_READY"
    assert not broken["criteria"]["all_target_payloads_strict_valid"]


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet engine is not installed")
def test_pilot_gate_accepts_exact_completed_subset(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "validate_cache_record", lambda *args, **kwargs: {})
    output = tmp_path / "output"
    sources = [_source(tmp_path, "train", str(index)) for index in range(3)]
    for source in sources:
        assets.build_target(
            source,
            output_root=output,
            builder=FakeBuilder(),
            identities=_identities(),
            config_file_sha256="b" * 64,
        )
    frames = {"train": pd.DataFrame(sources), "val": pd.DataFrame(columns=sources[0])}
    _write_inventory(output, frames, _identities())
    result = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=_identities(),
        require_complete=False,
        strict_sample_count=3,
    )
    assert result["decision"] == "D1B_FORMAL_TARGET_PILOT_PASS"


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="Parquet engine is not installed")
def test_train_val_overlap_blocks_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "validate_cache_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        assets,
        "strict_mvr_dataset_load",
        lambda output_root, identities, sample_count: {"train": 1, "val": 1},
    )
    output = tmp_path / "output"
    train = _source(tmp_path, "train", "overlap")
    val = _source(tmp_path, "val", "overlap")
    val["molecule_id"] = train["molecule_id"]
    record = torch.load(val["source_path"], weights_only=False)
    record["source_record_id"] = train["molecule_id"]
    torch.save(record, val["source_path"])
    val["source_file_sha256"] = assets.file_sha256(val["source_path"])
    for source in (train, val):
        assets.build_target(
            source,
            output_root=output,
            builder=FakeBuilder(),
            identities=_identities(),
            config_file_sha256="b" * 64,
        )
    frames = {"train": pd.DataFrame([train]), "val": pd.DataFrame([val])}
    _write_inventory(output, frames, _identities())
    result = assets.validate_formal_assets(
        output_root=output,
        source_frames=frames,
        identities=_identities(),
        require_complete=True,
        strict_sample_count=2,
    )
    assert not result["criteria"]["train_val_disjoint"]
    assert result["decision"] == "D1B_FORMAL_TARGETS_NOT_READY"


def test_runtime_telemetry_has_frozen_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "_gpu_metrics", lambda index: {
        "gpu_index": index,
        "gpu_uuid": "GPU-test",
        "gpu_utilization_percent": 1,
        "gpu_memory_used_mib": 2,
        "gpu_memory_total_mib": 3,
        "power_draw_w": 4,
        "temperature_c": 5,
    })
    monitor = assets.RuntimeTelemetry(
        tmp_path, total_records=10, interval=30, gpu_index="1"
    )
    monitor.update(success=True, skipped=False, seconds=0.5)
    monitor.sample()
    with (tmp_path / "telemetry/runtime_telemetry.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == assets.TELEMETRY_FIELDS
        row = next(reader)
    assert row["gpu_index"] == "1" and row["gpu_uuid"] == "GPU-test"
    assert row["completed_records"] == "1"


def test_runtime_telemetry_thread_stops_once_and_writes_final_sample(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(assets, "_gpu_metrics", lambda index: {
        "gpu_index": index,
        "gpu_uuid": "GPU-test",
        "gpu_utilization_percent": 0,
        "gpu_memory_used_mib": 0,
        "gpu_memory_total_mib": 1,
        "power_draw_w": 0,
        "temperature_c": 0,
    })
    monitor = assets.RuntimeTelemetry(
        tmp_path, total_records=1, interval=0.01, gpu_index="0"
    )
    monitor.start()
    monitor.update(success=True, skipped=False, seconds=0.25)
    monitor.stop()
    monitor.stop()
    assert monitor.thread is not None and not monitor.thread.is_alive()
    rows = list(
        csv.DictReader(
            (tmp_path / "telemetry/runtime_telemetry.csv").open(newline="")
        )
    )
    assert rows[-1]["completed_records"] == "1"


def test_runner_only_builds_targets_and_never_starts_training():
    runner = (ROOT / "scripts/run_ecir_mvr_formal_large_target_build.sh").read_text()
    assert "build_ecir_mvr_formal_large_targets.py" in runner
    assert "train_ecir" not in runner
    assert "test" not in runner.lower()
    builder = (ROOT / "scripts/build_ecir_mvr_formal_large_targets.py").read_text()
    assert "D1B_FORMAL_TARGET_PILOT_PASS" in builder
    assert "MinimalValidityTargetBuilder" in builder
    assert "build_real_error_target" not in builder
    assert "--retry-unresolved-only" in builder
    assert "retry_rows" in builder

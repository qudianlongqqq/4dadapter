from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml
from rdkit import Chem
from torch_geometric.data import Batch

if importlib.util.find_spec("torch_cluster") is None:
    cluster = ModuleType("torch_cluster")
    cluster.__spec__ = importlib.util.spec_from_loader("torch_cluster", loader=None)
    cluster.radius_graph = lambda x, r, **kwargs: torch.empty(
        (2, 0), dtype=torch.long, device=x.device
    )
    sys.modules["torch_cluster"] = cluster

from etflow.commons.featurization import MoleculeData, MoleculeFeaturizer
from etflow.data.flexbond_optimizer_dataset import validate_cache_record
from scripts import build_flexbond_init_cache as cache_builder
from scripts import check_etflow_formal_large_upstream as checker
from scripts import generate_etflow_formal_large_upstream as generator


ROOT = Path(__file__).resolve().parents[1]


def _molecule_graph(mol: Chem.Mol, smiles: str) -> MoleculeData:
    featurizer = MoleculeFeaturizer()
    atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
    )
    node_attr = featurizer.get_atom_features_from_mol(mol)
    edge_index, edge_attr = featurizer.get_edge_index_from_mol(
        mol, use_edge_feat=True
    )
    chiral_index, chiral_nbr_index, chiral_tag = (
        featurizer.get_chiral_centers_from_mol(mol)
    )
    rotatable, influence = featurizer.get_rotatable_bond_features_from_mol(mol)
    return MoleculeData(
        num_nodes=int(atomic_numbers.numel()),
        atomic_numbers=atomic_numbers,
        smiles=smiles,
        mol=Chem.Mol(mol),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_attr=node_attr,
        chiral_index=chiral_index,
        chiral_nbr_index=chiral_nbr_index,
        chiral_tag=chiral_tag,
        rotatable_bond_index=rotatable,
        atom_bond_influence_index=influence,
    )


def _write_processed(root: Path, split: str, count: int) -> None:
    destination = root / "drugs" / split
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        smiles = Chem.MolToSmiles(mol, canonical=False)
        num_atoms = mol.GetNumAtoms()
        base = torch.arange(num_atoms * 3, dtype=torch.float32).reshape(num_atoms, 3)
        pos = torch.stack((base / 10 + index, base / 10 + index + 0.01))
        torch.save(
            {"pos": pos, "smiles": smiles, "rdmol": mol},
            destination / f"{split}_molecule_{index:04d}.pt",
        )


class _MockModel:
    def __init__(self, tracker: dict, fail_call: int | None = None):
        self.tracker = tracker
        self.fail_call = fail_call

    def load_state_dict(self, state_dict, strict=True):
        assert state_dict == {}
        assert strict is True
        self.tracker["strict_load"] = True

    def to(self, device):
        self.tracker["device"] = str(device)
        return self

    def eval(self):
        self.tracker["eval"] = True
        return self

    def sample(self, atomic_numbers, edge_index, batch, **kwargs):
        self.tracker["sample_calls"] = self.tracker.get("sample_calls", 0) + 1
        self.tracker.setdefault("sampler_kwargs", []).append(dict(kwargs))
        if self.fail_call == self.tracker["sample_calls"]:
            raise RuntimeError("intentional mock interruption")
        return torch.randn(
            (atomic_numbers.numel(), 3),
            dtype=torch.float32,
            device=atomic_numbers.device,
        )


def _runtime(*, fail_call: int | None = None, tracker=None):
    tracker = tracker if tracker is not None else {}

    class Dataset:
        def __init__(self, partition, split, data_dir):
            assert partition == "drugs"
            tracker.setdefault("dataset_splits", []).append(split)
            self.data_files = sorted((Path(data_dir) / partition / split).glob("*.pt"))

        def __getitem__(self, index):
            raw = torch.load(
                self.data_files[index], map_location="cpu", weights_only=False
            )
            return _molecule_graph(raw["rdmol"], raw["smiles"])

        def __len__(self):
            return len(self.data_files)

    def instantiate_model(name, model_args):
        assert name == "BaseFlow"
        assert model_args == {"hidden_nf": 8}
        return _MockModel(tracker, fail_call=fail_call)

    return SimpleNamespace(
        read_yaml=lambda path: yaml.safe_load(Path(path).read_text()),
        instantiate_model=instantiate_model,
        dataset_class=Dataset,
        batch_class=Batch,
    )


@pytest.fixture
def mock_inputs(tmp_path: Path):
    processed = tmp_path / "processed"
    _write_processed(processed, "train", 4)
    _write_processed(processed, "val", 4)
    checkpoint = tmp_path / "drugs-o3.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    config = tmp_path / "drugs-o3.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": "BaseFlow",
                "model_args": {"hidden_nf": 8},
                "datamodule_args": {"partition": "drugs"},
                "eval_args": {
                    "batch_size": 8,
                    "sampler_args": {"method": "ode", "n_timesteps": 7},
                },
            }
        )
    )
    return SimpleNamespace(
        root=tmp_path,
        processed=processed,
        checkpoint=checkpoint,
        config=config,
    )


def _args(inputs, output: Path, split="train", maximum=2, samples=3, **updates):
    values = {
        "etflow_root": inputs.root / "external-etflow",
        "config": inputs.config,
        "checkpoint": inputs.checkpoint,
        "processed_data": inputs.processed,
        "split": split,
        "max_molecules": maximum,
        "samples_per_molecule": samples,
        "seed": 42,
        "output_dir": output,
        "device": "cpu",
        "resume": False,
        "save_every_molecules": 1,
        "state_path": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _run(inputs, output: Path, **kwargs):
    tracker = kwargs.pop("tracker", {})
    fail_call = kwargs.pop("fail_call", None)
    args = _args(inputs, output, **kwargs)
    result = generator.run_generation(
        args,
        runtime_loader=lambda _: _runtime(
            fail_call=fail_call, tracker=tracker
        ),
    )
    return result, tracker


def _manifest(output: Path) -> dict:
    return json.loads((output / "generation_manifest.json").read_text())


def _records(output: Path) -> dict[str, dict]:
    manifest = _manifest(output)
    return {
        row["source_mol_id"]: torch.load(
            output / "molecules" / row["output_file"],
            map_location="cpu",
            weights_only=False,
        )
        for row in manifest["records"]
    }


def test_mock_2_plus_2_is_exact_deterministic_and_uses_original_sampler(mock_inputs):
    train = mock_inputs.root / "run-a" / "train"
    val = mock_inputs.root / "run-a" / "val"
    _, train_tracker = _run(mock_inputs, train)
    _, val_tracker = _run(mock_inputs, val, split="val", samples=2)
    train_records = _records(train)
    val_records = _records(val)

    assert len(train_records) == len(val_records) == 2
    assert {tuple(record["pos_gen"].shape)[0] for record in train_records.values()} == {3}
    assert {tuple(record["pos_gen"].shape)[0] for record in val_records.values()} == {2}
    assert all(generator.REQUIRED_RECORD_FIELDS <= set(record) for record in train_records.values())
    assert train_tracker["dataset_splits"] == ["train"]
    assert val_tracker["dataset_splits"] == ["val"]
    assert train_tracker["strict_load"] and train_tracker["eval"]
    sampler = train_tracker["sampler_kwargs"][0]
    assert sampler["method"] == "ode" and sampler["n_timesteps"] == 7

    second = mock_inputs.root / "run-b" / "train"
    _run(mock_inputs, second)
    for source_id, record in train_records.items():
        assert torch.equal(record["pos_gen"], _records(second)[source_id]["pos_gen"])


def test_resume_and_single_molecule_are_bitwise_deterministic(mock_inputs):
    uninterrupted = mock_inputs.root / "uninterrupted"
    _run(mock_inputs, uninterrupted)
    interrupted = mock_inputs.root / "interrupted"
    with pytest.raises(RuntimeError, match="intentional mock interruption"):
        _run(mock_inputs, interrupted, fail_call=2)
    failed_state = json.loads((interrupted / "generation_state.json").read_text())
    assert failed_state["status"] == "FAILED"
    _run(mock_inputs, interrupted, resume=True)
    expected = _records(uninterrupted)
    resumed = _records(interrupted)
    assert expected.keys() == resumed.keys()
    assert all(torch.equal(expected[key]["pos_gen"], resumed[key]["pos_gen"]) for key in expected)

    single = mock_inputs.root / "single"
    _run(mock_inputs, single, maximum=1)
    source_id, record = next(iter(_records(single).items()))
    assert torch.equal(record["pos_gen"], expected[source_id]["pos_gen"])


def test_resume_validates_without_rewriting_and_rejects_corruption(mock_inputs):
    output = mock_inputs.root / "completed"
    _run(mock_inputs, output)
    paths = sorted((output / "molecules").glob("*.pt"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in paths}
    result, tracker = _run(mock_inputs, output, resume=True)
    assert result["generated_this_run"] == 0
    assert "sample_calls" not in tracker
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in paths}

    extra = output / "molecules" / "unmanifested.pt"
    torch.save({"unexpected": True}, extra)
    with pytest.raises(ValueError, match="unmanifested"):
        _run(mock_inputs, output, resume=True)
    extra.unlink()

    broken = torch.load(paths[0], map_location="cpu", weights_only=False)
    broken["pos_gen"][0, 0, 0] += 1
    torch.save(broken, paths[0])
    with pytest.raises(ValueError, match="content hash mismatch"):
        _run(mock_inputs, output, resume=True)


@pytest.mark.parametrize("changed", ["checkpoint", "config"])
def test_resume_rejects_changed_checkpoint_or_config(mock_inputs, changed):
    output = mock_inputs.root / f"changed-{changed}"
    _run(mock_inputs, output)
    if changed == "checkpoint":
        torch.save({"state_dict": {}, "changed": True}, mock_inputs.checkpoint)
    else:
        mock_inputs.config.write_text(mock_inputs.config.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="different inputs"):
        _run(mock_inputs, output, resume=True)


def test_atomic_write_has_no_partial_destination(monkeypatch, tmp_path):
    destination = tmp_path / "record.pt"

    def fail_replace(source, target):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(generator.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        generator.atomic_torch_save({"value": torch.ones(1)}, destination)
    assert not destination.exists()
    assert list(tmp_path.glob("record.pt.tmp.*"))


def test_selection_seed_state_and_manifest_are_stable_across_paths(mock_inputs, tmp_path):
    files = sorted((mock_inputs.processed / "drugs" / "train").glob("*.pt"))
    selected_a, _ = generator.select_molecules(
        files,
        processed_data=mock_inputs.processed,
        split="train",
        max_molecules=3,
        seed=42,
    )
    selected_b, _ = generator.select_molecules(
        list(reversed(files)),
        processed_data=mock_inputs.processed,
        split="train",
        max_molecules=3,
        seed=42,
    )
    assert [row["source_mol_id"] for row in selected_a] == [
        row["source_mol_id"] for row in selected_b
    ]
    assert generator.molecule_seed(42, "train", "mol-a") == generator.molecule_seed(
        42, "train", "mol-a"
    )
    assert generator.molecule_seed(42, "train", "mol-a") != generator.molecule_seed(
        42, "val", "mol-a"
    )

    moved = tmp_path / "moved"
    shutil.copytree(mock_inputs.processed, moved / "processed")
    shutil.copy2(mock_inputs.checkpoint, moved / "model.ckpt")
    shutil.copy2(mock_inputs.config, moved / "config.yaml")
    manifest_a = generator.build_generation_manifest(
        data_files=files,
        processed_data=mock_inputs.processed,
        split="train",
        max_molecules=2,
        samples_per_molecule=3,
        seed=42,
        checkpoint_path=mock_inputs.checkpoint,
        config_path=mock_inputs.config,
    )
    manifest_b = generator.build_generation_manifest(
        data_files=sorted((moved / "processed/drugs/train").glob("*.pt")),
        processed_data=moved / "processed",
        split="train",
        max_molecules=2,
        samples_per_molecule=3,
        seed=42,
        checkpoint_path=moved / "model.ckpt",
        config_path=moved / "config.yaml",
    )
    assert manifest_a["manifest_sha256"] == manifest_b["manifest_sha256"]
    moved_dataset = SimpleNamespace(
        data_files=sorted((moved / "processed/drugs/train").glob("*.pt"))
    )
    resolved = generator._current_source_path(
        manifest_a["records"][0],
        dataset=moved_dataset,
        processed_data=moved / "processed",
    )
    assert resolved.is_relative_to(moved)

    base_manifest = {
        "target_molecules": 50_000,
        "split": "train",
        "seed": 42,
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "processed_split_identity_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
    }
    state = generator._state_payload(
        status="RUNNING",
        manifest=base_manifest,
        completed=123,
        next_position=123,
        started_at="now",
        elapsed=10,
        generated_this_run=10,
    )
    encoded = json.dumps(state)
    assert len(encoded) < 2_000
    assert "records" not in state and "coordinates" not in encoded

    with pytest.raises(ValueError, match="positive"):
        _run(mock_inputs, mock_inputs.root / "zero", maximum=0)


def test_directory_records_are_lazy_and_file_containers_remain_compatible(
    monkeypatch, tmp_path
):
    directory = tmp_path / "molecules"
    directory.mkdir()
    for name in ("c.pt", "a.pt", "b.pt"):
        (directory / name).write_bytes(b"placeholder")
    loaded = []

    def fake_load(path):
        loaded.append(path.name)
        return {"mol_id": path.stem}

    monkeypatch.setattr(cache_builder, "_load", fake_load)
    iterator = cache_builder._records(directory)
    assert loaded == []
    assert next(iterator)[0] == "a"
    assert loaded == ["a.pt"]
    assert [fallback for fallback, _ in iterator] == ["b", "c"]

    packed = tmp_path / "packed.pt"
    torch.save([{"mol_id": "x"}, {"mol_id": "y"}], packed)
    monkeypatch.undo()
    assert [fallback for fallback, _ in cache_builder._records(packed)] == ["0", "1"]


def _run_cache(monkeypatch, init_path: Path, output: Path, split: str, checkpoint: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_flexbond_init_cache.py",
            "--init_path",
            str(init_path),
            "--output_dir",
            str(output),
            "--split",
            split,
            "--generator_checkpoint",
            str(checkpoint),
            "--sample_seed",
            "42",
            "--data_dir",
            "mock-processed",
        ],
    )
    cache_builder.main()


def test_mock_directory_cache_has_6_and_4_pairs_and_matches_packed_path(
    mock_inputs, monkeypatch
):
    train = mock_inputs.root / "generated" / "train"
    val = mock_inputs.root / "generated" / "val"
    _run(mock_inputs, train)
    _run(mock_inputs, val, split="val", samples=2)
    cache = mock_inputs.root / "cache"
    _run_cache(monkeypatch, train / "molecules", cache, "train", mock_inputs.checkpoint)
    _run_cache(monkeypatch, val / "molecules", cache, "val", mock_inputs.checkpoint)
    train_cache = sorted((cache / "train").glob("*.pt"))
    val_cache = sorted((cache / "val").glob("*.pt"))
    assert len(train_cache) == 6
    assert len(val_cache) == 4
    for path in train_cache + val_cache:
        validate_cache_record(
            torch.load(path, map_location="cpu", weights_only=False),
            require_persisted_pair=True,
        )

    packed_path = mock_inputs.root / "train-generated.pkl"
    with packed_path.open("wb") as handle:
        pickle.dump(list(_records(train).values()), handle)
    packed_cache = mock_inputs.root / "packed-cache"
    _run_cache(
        monkeypatch, packed_path, packed_cache, "train", mock_inputs.checkpoint
    )
    for directory_record in train_cache:
        packed_record = torch.load(
            packed_cache / "train" / directory_record.name,
            map_location="cpu",
            weights_only=False,
        )
        streamed_record = torch.load(
            directory_record, map_location="cpu", weights_only=False
        )
        for key in ("x_init", "x_ref_candidates", "x_ref", "x_ref_aligned"):
            assert torch.equal(streamed_record[key], packed_record[key])
        assert streamed_record["x_init_hash"] == packed_record["x_init_hash"]

    test_path = mock_inputs.root / "fixed-test.pkl"
    with test_path.open("wb") as handle:
        pickle.dump(
            [
                {
                    "source_mol_id": f"test_molecule_{index}",
                    "pos_gen": torch.zeros(2, 2, 3),
                    "pos_ref": torch.zeros(1, 2, 3),
                }
                for index in range(2)
            ],
            handle,
        )
    report = checker.build_integrity_report(train, val, test_path)
    assert report["overlap_counts"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    assert report["error_count"] == 0
    assert report["status"] == "INCOMPLETE"  # Smoke counts cannot claim formal readiness.


def test_scripts_forbid_fallback_and_stop_before_data_build_or_training():
    generator_source = (ROOT / "scripts/generate_etflow_formal_large_upstream.py").read_text()
    assert "EmbedMolecule" not in generator_source
    assert "AllChem" not in generator_source
    runner = (ROOT / "scripts/run_generate_etflow_formal_large_upstream.sh").read_text()
    assert "run_split train 50000 3" in runner
    assert "run_split val 5000 2" in runner
    assert "build_formal_large_data.sh" not in runner
    assert "run_formal_large_training.sh" not in runner
    builder = (ROOT / "scripts/build_flexbond_init_cache.py").read_text()
    assert "cache_recovered_mol" not in builder
    assert "FORMAL_LARGE_ETFLOW_TRAIN_OUTPUT" in (
        ROOT / "scripts/build_formal_large_data.sh"
    ).read_text()

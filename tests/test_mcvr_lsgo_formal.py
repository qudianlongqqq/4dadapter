from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from etflow.ecir.learned_geometry import distribution_parameters, prepare_graph, structured_objective
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from scripts.run_mcvr_lsgo_formal import (
    CONFIG, OUT, checkpoint_payload, load_config, model_from_config,
    restore_checkpoint, seed_all, training_batch,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared_examples():
    config = load_config()
    compact = torch.load(
        r"E:\3dconformergenerationcode\4dadapter-label-free-score\reports\ecir_mvr\label_free_score_pilot\manifests\TRAIN_ONLY_COMPACT_DATASET.pt",
        map_location="cpu", weights_only=False,
    )
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text())
    rows = []
    for item in compact["items"][:2]:
        rows.append({"graph": prepare_graph(item["record"], calibration), "references": item["references"], "sources": item["sources"]})
    return rows


def test_frozen_formal_config():
    config = load_config()
    assert config["seeds"] == [307, 331, 353]
    assert config["training"]["optimizer_steps"] == 12500
    assert config["training"]["effective_batch"] == 64
    assert config["training"]["total_exposures"] == 800000
    assert config["model"]["learned_sigma"] is False
    assert all(value == 0 for value in (config["guards"]["formal_test_records_read"], config["guards"]["frozen_holdout_records_read"]))
    assert not any(config["guards"][key] for key in ("xtb_training_access", "posebusters_training_access", "mvt_teacher_access", "torsion_objective", "soft_clash_objective", "ring_objective", "active_set"))


def test_model_and_frozen_scale(prepared_examples):
    config = load_config(); model = model_from_config(config)
    assert sum(p.numel() for p in model.parameters()) == 473674
    assert not any("sigma" in name for name, _ in model.named_parameters())
    graph = prepared_examples[0]["graph"]
    parameters = distribution_parameters(graph, model=model, variant="B")
    assert parameters["bond_mu"].requires_grad and parameters["angle_mu"].requires_grad
    assert not parameters["bond_sigma"].requires_grad and not parameters["angle_sigma"].requires_grad


def test_bond_angle_equal_group_objective(prepared_examples):
    config = load_config(); model = model_from_config(config); item = prepared_examples[0]
    parameters = distribution_parameters(item["graph"], model=model, variant="B")
    objective, groups = structured_objective(item["references"][0], item["graph"], parameters)
    assert set(groups) == {"bond", "angle"}
    assert torch.allclose(objective, (groups["bond"] + groups["angle"]) / 2)
    assert torch.isfinite(objective)


def test_training_batch_uses_references_not_sources(prepared_examples):
    generator = torch.Generator().manual_seed(5)
    _, coordinates = training_batch(prepared_examples, generator, 2, torch.device("cpu"))
    references = [row["references"] for row in prepared_examples]
    sources = [row["sources"] for row in prepared_examples]
    assert any(torch.equal(coordinates[:len(value[0])], ref) for value in references for ref in value)
    assert not any(torch.equal(coordinates[:len(value[0])], source) for value in sources for source in value)


def test_checkpoint_strict_load_and_sampler_resume(tmp_path, prepared_examples):
    config = load_config(); seed = 307; generator = seed_all(seed); model = model_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12500)
    payload = checkpoint_payload(model, optimizer, scheduler, generator, seed, 50, 3200, {}, "config", "identity")
    path = tmp_path / "resume.ckpt"; torch.save(payload, path)
    expected = torch.randint(100000, (32,), generator=generator)
    restored_model = model_from_config(config); restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=3e-4, weight_decay=1e-6)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=12500); restored_generator = torch.Generator()
    result = restore_checkpoint(path, restored_model, restored_optimizer, restored_scheduler, restored_generator, "config", "identity")
    actual = torch.randint(100000, (32,), generator=restored_generator)
    assert result["global_step"] == 50 and result["exposure_count"] == 3200
    assert torch.equal(expected, actual)
    assert all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), restored_model.state_dict().values()))


def test_formal_train_val_manifests_are_protected_and_disjoint():
    config = load_config(); train = pd.read_parquet(config["dataset"]["train_manifest"]); val = pd.read_parquet(config["dataset"]["val_manifest"])
    assert len(train) == 150000 and train.molecule_id.nunique() == 50000
    assert len(val) == 10000 and val.molecule_id.nunique() == 5000
    assert not train.test_record.fillna(False).any() and not val.test_record.fillna(False).any()
    assert set(train.molecule_id).isdisjoint(set(val.molecule_id))


def test_dataset_identity_if_built():
    path = OUT / "DATASET_IDENTITY.json"
    if not path.is_file():
        pytest.skip("formal sufficient dataset not built yet")
    identity = json.loads(path.read_text())
    assert identity["status"] == "FROZEN"
    assert identity["train"]["molecule_count"] == 50000 and identity["train"]["record_count"] == 150000
    assert identity["validation"]["molecule_count"] == 5000 and identity["validation"]["record_count"] == 10000
    assert identity["train_val_molecule_overlap"] == 0
    assert identity["formal_test_records_read"] == identity["frozen_holdout_records_read"] == 0


def test_formal_explicit_hydrogen_record_prepares_without_dropping_atoms():
    config = load_config()
    path = Path(config["dataset"]["train_cache"]) / "train__CSc1nc_NC_C_O_ss1__8c1412048dbbbefafb81__gen0000.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text())
    adapted = adapt_formal_cache_record(record)
    graph = prepare_graph(adapted, calibration)
    assert adapted["_formal_rdkit_mol"].GetNumAtoms() == int(record["num_atoms"]) == 17
    assert graph.atom_categorical.size(0) == 17

import copy
from types import SimpleNamespace

import pytest
import torch

from etflow.data.flexbond_cache_schema import x_init_sha256
from etflow.data.flexbond_eval_manifest import build_manifest_aware_sample_payload
from scripts.eval_flexbond_optimizer import _load_method_records


def _manifest():
    return {
        "manifest_version": "1.0",
        "records": [
            {
                "mol_id": "mol",
                "sample_id": "sample",
                "x_init_hash": "hash",
                "num_rotatable_bonds": 1,
            }
        ],
    }


def _inference():
    return {
        "sample": SimpleNamespace(
            source_mol_id="mol",
            sample_id="sample",
            x_init_hash="hash",
            num_rotatable_bonds=torch.tensor([1]),
        )
    }


def _payload(manifest, records, tmp_path, manifest_path=None):
    return build_manifest_aware_sample_payload(
        records=records,
        manifest=manifest,
        manifest_path=manifest_path or tmp_path / "source_manifest.json",
        selected_manifest=manifest,
        split="test",
        inference_cache_path=tmp_path / "cache",
        inference_by_id=_inference(),
    )


def test_sample_payload_must_embed_identical_manifest(tmp_path):
    manifest = _manifest()
    records = [{
        "mol_id": "sample", "source_mol_id": "mol", "sample_id": "sample",
        "method_name": "cartesian_adapter", "x_init_hash": "hash",
        "status": "failed", "x_refined": None,
    }]
    payload = _payload(manifest, records, tmp_path)
    payload["manifest"] = {
        **manifest,
        "records": [{**manifest["records"][0], "x_init_hash": "other"}],
    }
    path = tmp_path / "samples.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="requested manifest"):
        _load_method_records(
            path, "cartesian_adapter", manifest,
            manifest_path=tmp_path / "requested_manifest.json", split="test",
            inference_cache_path=tmp_path / "cache", inference_by_id=_inference(),
        )


def test_failed_sample_is_reported_not_removed(tmp_path):
    manifest = _manifest()
    records = [{
                "mol_id": "sample",
                "source_mol_id": "mol",
                "sample_id": "sample",
                "method_name": "flexbond4d_adapter",
                "x_init_hash": "hash",
                "status": "failed",
                "x_refined": None,
            }]
    payload = _payload(manifest, records, tmp_path)
    path = tmp_path / "samples.pt"
    torch.save(payload, path)
    _, missing, failed = _load_method_records(
        path, "flexbond4d_adapter", manifest,
        manifest_path=tmp_path / "different_absolute_or_relative_name.json",
        split="test", inference_cache_path=tmp_path / "cache_alias",
        inference_by_id=_inference(),
    )
    assert missing == []
    assert failed == ["sample"]


def test_manifest_hash_not_path_is_the_payload_identity(tmp_path):
    manifest = _manifest()
    records = [{
        "mol_id": "sample", "source_mol_id": "mol", "sample_id": "sample",
        "method_name": "cartesian_adapter", "x_init_hash": "hash",
        "status": "success", "x_refined": torch.zeros(2, 3),
    }]
    payload = _payload(
        manifest, records, tmp_path, manifest_path=tmp_path / "relative_source.json"
    )
    path = tmp_path / "samples.pt"
    torch.save(payload, path)
    loaded, missing, failed = _load_method_records(
        path, "cartesian_adapter", manifest,
        manifest_path=tmp_path / "another" / "absolute_manifest.json",
        split="test", inference_cache_path=tmp_path / "cache_elsewhere",
        inference_by_id=_inference(),
    )
    assert list(loaded) == ["sample"]
    assert missing == [] and failed == []


def test_every_payload_provenance_field_is_enforced(tmp_path):
    manifest = _manifest()
    records = [{
        "mol_id": "sample", "source_mol_id": "mol", "sample_id": "sample",
        "method_name": "cartesian_adapter", "x_init_hash": "hash",
        "status": "success", "x_refined": torch.zeros(2, 3),
    }]
    valid = _payload(manifest, records, tmp_path)
    mutations = [
        lambda value: value["manifest_provenance"]["manifest"].update(path=""),
        lambda value: value["manifest_provenance"]["manifest"].update(sha256="bad"),
        lambda value: value["manifest_provenance"].update(split="train"),
        lambda value: value["manifest_provenance"].update(ordered_sample_ids=[]),
        lambda value: value["manifest_provenance"].update(molecule_ids=["other"]),
        lambda value: value["manifest_provenance"].update(x_init_hashes=["other"]),
        lambda value: value["manifest_provenance"].update(sample_count=2),
        lambda value: value["manifest_provenance"]["inference_cache"].update(
            cohort_sha256="bad"
        ),
        lambda value: value["manifest_provenance"]["inference_cache"].update(
            sample_count=2
        ),
    ]
    for index, mutate in enumerate(mutations):
        payload = copy.deepcopy(valid)
        mutate(payload)
        path = tmp_path / f"invalid_{index}.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError):
            _load_method_records(
                path, "cartesian_adapter", manifest,
                manifest_path=tmp_path / "manifest.json", split="test",
                inference_cache_path=tmp_path / "cache", inference_by_id=_inference(),
            )


def test_payload_x_init_tensor_must_match_its_manifest_hash(tmp_path):
    atomic_numbers = torch.tensor([6, 8])
    x_init = torch.zeros(2, 3)
    digest = x_init_sha256(x_init, atomic_numbers)
    manifest = _manifest()
    manifest["records"][0]["x_init_hash"] = digest
    inference = _inference()
    inference["sample"].x_init_hash = digest
    records = [{
        "mol_id": "sample", "source_mol_id": "mol", "sample_id": "sample",
        "method_name": "cartesian_adapter", "x_init_hash": digest,
        "x_init": torch.ones(2, 3), "atomic_numbers": atomic_numbers,
        "status": "success", "x_refined": x_init,
    }]
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        build_manifest_aware_sample_payload(
            records=records, manifest=manifest, manifest_path=tmp_path / "manifest.json",
            selected_manifest=manifest, split="test",
            inference_cache_path=tmp_path / "cache", inference_by_id=inference,
        )

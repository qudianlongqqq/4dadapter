import pytest
import torch

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


def test_sample_payload_must_embed_identical_manifest(tmp_path):
    manifest = _manifest()
    payload = {
        "manifest": {**manifest, "records": [{**manifest["records"][0], "x_init_hash": "other"}]},
        "records": [],
    }
    path = tmp_path / "samples.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="requested manifest"):
        _load_method_records(path, "cartesian_adapter", manifest)


def test_failed_sample_is_reported_not_removed(tmp_path):
    manifest = _manifest()
    payload = {
        "manifest": manifest,
        "records": [
            {
                "sample_id": "sample",
                "method_name": "flexbond4d_adapter",
                "x_init_hash": "hash",
                "status": "failed",
                "x_refined": None,
            }
        ],
    }
    path = tmp_path / "samples.pt"
    torch.save(payload, path)
    _, missing, failed = _load_method_records(path, "flexbond4d_adapter", manifest)
    assert missing == []
    assert failed == ["sample"]

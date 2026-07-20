import pandas as pd
import pytest

from etflow.ecir.v8_sampler import build_stratified_payload


def test_sampler_manifest_accepts_train_and_rejects_validation(tmp_path):
    train = pd.DataFrame(
        {
            "split": ["train"],
            "sample_id": ["train::1"],
            "molecule_id": ["mol1"],
            "source_angle_outlier_rate": [1.0],
            "source_clash_penetration": [0.0],
            "source_severe_clash_rate": [0.0],
            "source_ring_bond_outlier_rate": [0.0],
            "source_ring_planarity_outlier_rate": [0.0],
            "source_total_thresholded_validity_score": [0.1],
        }
    )
    path = tmp_path / "train.parquet"
    train.to_parquet(path)
    payload = build_stratified_payload(path)
    assert payload["test_used"] is False and payload["split"] == "train"
    train["split"] = "val"
    train.to_parquet(path)
    with pytest.raises(RuntimeError):
        build_stratified_payload(path)

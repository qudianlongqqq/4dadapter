import pandas as pd
import pytest
import torch

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


def test_formal_train_sampler_derives_cohorts_from_train_targets_only(tmp_path):
    sources = pd.DataFrame(
        {
            "split": ["train", "train"],
            "sample_id": ["train::1", "train::2"],
            "molecule_id": ["mol", "mol"],
            "num_rotatable_bonds": [7, 0],
        }
    )
    targets = pd.DataFrame(
        {
            "split": ["train", "train"],
            "sample_id": ["train::1", "train::2"],
            "molecule_id": ["mol", "mol"],
            "target_cache_path": ["one.pt", "two.pt"],
            "initial_to_target_rmsd": [0.01, 0.2],
            "test_records_read": [0, 0],
        }
    )
    source_path = tmp_path / "sources.parquet"
    target_path = tmp_path / "targets.parquet"
    cache = tmp_path / "targets" / "train"
    cache.mkdir(parents=True)
    sources.to_parquet(source_path)
    targets.to_parquet(target_path)
    for sample_id, filename, angle, clash, ring, score in (
        ("train::1", "one.pt", 0.2, 0.0, 0.1, 0.1),
        ("train::2", "two.pt", 0.0, 0.3, 0.0, 0.5),
    ):
        torch.save(
            {
                "sample_id": sample_id,
                "test_records_read": 0,
                "target_metadata": {
                    "initial_validity": {
                        "angle_outlier_rate": angle,
                        "clash_penetration": clash,
                        "severe_clash_rate": 0.0,
                        "ring_bond_outlier_rate": ring,
                        "ring_planarity_outlier_rate": 0.0,
                        "total_thresholded_validity_score": score,
                    }
                },
            },
            cache / filename,
        )
    payload = build_stratified_payload(
        source_path,
        target_manifest=target_path,
        target_cache_root=tmp_path / "targets",
    )
    assert payload["cohort_counts"]["active_angle"] == 1
    assert payload["cohort_counts"]["active_clash"] == 1
    assert payload["cohort_counts"]["ring_risk"] == 1
    assert payload["cohort_counts"]["high_flexibility"] == 1
    assert sum(record["sampling_weight"] for record in payload["records"]) <= 4.0
    assert payload["capped_molecule_count"] == 1
    assert payload["validation_records_read"] == 0
    assert payload["formal_test_records_read"] == 0

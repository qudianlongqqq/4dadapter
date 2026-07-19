from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts.prepare_ecir_mvr_v7_10k_development import _rank_molecule_ids
from scripts.report_ecir_mvr_v7_10k_validation import _bootstrap_matrix
from scripts.run_ecir_mvr_v7_10k_validation import (
    CONFIG_SHA,
    EXPECTED_SELECTION,
    _build_items,
    _canonical_sha,
    _load_model,
    _merge_components,
    _merge_solver,
    _sha,
    _verify_manifest,
)


def test_rank_molecule_ids_is_deterministic() -> None:
    forward = _rank_molecule_ids({"mol-3", "mol-1", "mol-2"})
    reverse = _rank_molecule_ids({"mol-2", "mol-1", "mol-3"})
    assert forward == reverse
    assert set(forward) == {"mol-1", "mol-2", "mol-3"}


def test_verify_manifest_checks_canonical_and_file_identities(tmp_path: Path) -> None:
    source = tmp_path / "development_sources.parquet"
    target = tmp_path / "development_targets.parquet"
    pd.DataFrame({"sample_id": ["sample"]}).to_parquet(source, index=False)
    pd.DataFrame({"sample_id": ["sample"]}).to_parquet(target, index=False)
    stable = {
        **EXPECTED_SELECTION,
        "source_manifest_sha256": _sha(source),
        "target_manifest_sha256": _sha(target),
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "formal_large_run": False,
        "training_performed": False,
        "target_rematerialization": False,
        "validation_only": True,
    }
    manifest = {**stable, "identity_sha256": _canonical_sha(stable)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert _verify_manifest(tmp_path)["identity_sha256"] == manifest["identity_sha256"]

    manifest["test_assets_opened"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical identity mismatch"):
        _verify_manifest(tmp_path)


def test_load_model_rejects_changed_frozen_inference_config(tmp_path: Path) -> None:
    changed = tmp_path / "changed.yaml"
    changed.write_text("inference: {}\n", encoding="utf-8")
    args = SimpleNamespace(v7_config=changed, v5_config=changed)

    with pytest.raises(RuntimeError, match="frozen inference config SHA mismatch"):
        _load_model("D1", {}, args, torch.device("cpu"))
    assert _sha(changed) != CONFIG_SHA["V7"]


def test_build_items_relocates_train_derived_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "sources"
    target_root = tmp_path / "targets"
    (source_root / "train").mkdir(parents=True)
    (target_root / "train").mkdir(parents=True)
    source_pt = source_root / "train" / "source.pt"
    target_pt = target_root / "train" / "target.pt"
    source_pt.touch()
    torch.save({"x_target": torch.zeros((2, 3))}, target_pt)
    sources = pd.DataFrame(
        [
            {
                "molecule_id": "mol",
                "sample_id": "sample",
                "split": "train",
                "source_path": r"C:\\old-machine\\source.pt",
                "generator_name": "ETFlow",
                "source_severity": "normal",
                "update_scale": 1.0,
            }
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "sample_id": "sample",
                "split": "train",
                "target_cache_path": r"C:\\old-machine\\target.pt",
            }
        ]
    )
    record = {
        "x_ref_candidates": torch.zeros((1, 2, 3)),
        "num_rotatable_bonds": 0,
        "bond_is_in_ring": torch.tensor([], dtype=torch.bool),
    }
    monkeypatch.setattr(
        "scripts.run_ecir_mvr_v7_10k_validation._load_source_coordinates",
        lambda row: (record, torch.zeros((2, 3))),
    )
    monkeypatch.setattr(
        "scripts.run_ecir_mvr_v7_10k_validation.graph_data",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scripts.run_ecir_mvr_v7_10k_validation.nearest_rmsd", lambda *args: 0.0
    )
    validity = SimpleNamespace(
        evaluate=lambda *args, **kwargs: {
            "bond_outlier_rate": 0.0,
            "angle_outlier_rate": 0.0,
            "ring_bond_outlier_rate": 0.0,
            "ring_planarity_outlier_rate": 0.0,
            "clash_penetration": 0.0,
            "severe_clash_rate": 0.0,
            "stereocenter_degenerate_rate": 0.0,
            "torsion_prior_outlier_score": 0.0,
        }
    )

    items = _build_items(
        sources,
        targets,
        validity,
        source_cache_root=source_root,
        target_cache_root=target_root,
    )

    assert len(items) == 1
    assert items[0]["minimal_target"].shape == (2, 3)


def test_solver_and_component_summaries_are_weighted() -> None:
    solver = _merge_solver(
        [
            {
                "calls": 3,
                "status_counts": {"SOLVED": 2, "NO_ACTIVE_CONSTRAINT": 1},
                "solver_failure_count": 0,
                "effective_rank_mean": 2.0,
                "condition_number_mean": 4.0,
                "condition_number_max": 5.0,
                "singular_value_max": 3.0,
                "singular_value_min_retained": 0.2,
                "truncated_direction_count": 1,
            },
            {
                "calls": 2,
                "status_counts": {"SOLVED": 1, "FAILED": 1},
                "solver_failure_count": 1,
                "effective_rank_mean": 5.0,
                "condition_number_mean": 10.0,
                "condition_number_max": 10.0,
                "singular_value_max": 4.0,
                "singular_value_min_retained": 0.1,
                "truncated_direction_count": 2,
            },
        ]
    )
    assert solver["calls"] == 5
    assert solver["solver_failure_count"] == 1
    assert solver["effective_rank_mean"] == pytest.approx(3.0)
    assert solver["condition_number_mean"] == pytest.approx(6.0)
    assert solver["singular_value_min_retained"] == 0.1
    assert solver["truncated_direction_count"] == 3

    components = _merge_components(
        [{"calls": 2, "angle_alpha": 0.25}, {"calls": 6, "angle_alpha": 0.75}]
    )
    assert components == {"calls": 8, "angle_alpha": pytest.approx(0.625)}


def test_bootstrap_matrix_is_paired_and_deterministic() -> None:
    values = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 4.0],
            "second": [10.0, 20.0, 30.0, 40.0],
        }
    )

    first = _bootstrap_matrix(values, seed=7, draws=250)
    second = _bootstrap_matrix(values, seed=7, draws=250)

    assert first == second
    assert first["first"]["mean"] == 2.5
    assert first["second"]["mean"] == 25.0
    assert first["second"]["ci95_low"] == pytest.approx(
        10.0 * first["first"]["ci95_low"]
    )
    assert first["second"]["ci95_high"] == pytest.approx(
        10.0 * first["first"]["ci95_high"]
    )

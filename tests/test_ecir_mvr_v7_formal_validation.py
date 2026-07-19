from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_ecir_mvr_v7_formal_validation import (
    FROZEN_SEEDS,
    METHODS,
    _combine_method_evaluations,
    _completed_chunk,
    _require_sha,
    _validate_cohort_frames,
    _validate_existing_output,
    _validate_methods,
    _validate_seed_contract,
    _reject_forbidden_path,
    file_sha256,
)


def _cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for molecule in range(5_000):
        for conformer in range(2):
            rows.append(
                {
                    "sample_id": f"val::mol{molecule}__gen{conformer}",
                    "molecule_id": f"mol{molecule}",
                    "split": "val",
                    "test_record": False,
                }
            )
    sources = pd.DataFrame(rows)
    targets = sources.drop(columns="test_record").copy()
    targets["test_records_read"] = 0
    return sources, targets


def _evaluation(source_value: float, candidate_value: float) -> dict[str, pd.DataFrame]:
    records = pd.DataFrame(
        [
            {
                "sample_id": "val::mol__gen0",
                "molecule_id": "mol",
                "method": "upstream",
                "bond_outlier_rate": source_value,
            },
            {
                "sample_id": "val::mol__gen0",
                "molecule_id": "mol",
                "method": "v2_bac_accepted",
                "bond_outlier_rate": candidate_value,
            },
        ]
    )
    molecules = pd.DataFrame(
        [
            {
                "group": "all",
                "molecule_id": "mol",
                "method": "upstream",
                "bond_outlier_rate": source_value,
            },
            {
                "group": "all",
                "molecule_id": "mol",
                "method": "v2_bac_accepted",
                "bond_outlier_rate": candidate_value,
            },
        ]
    )
    return {"records": records, "molecules": molecules}


def test_frozen_formal_method_set_cannot_silently_omit_comparator() -> None:
    assert _validate_methods(METHODS) == METHODS
    with pytest.raises(RuntimeError, match="exactly D1, V5-B, V7"):
        _validate_methods(("D1", "V7"))


@pytest.mark.parametrize(
    "path",
    [
        Path("manifests/formal_test/val.parquet"),
        Path("data/test/val.parquet"),
        Path("data/frozen_holdout/val.parquet"),
    ],
)
def test_formal_validation_rejects_test_and_holdout_paths(path: Path) -> None:
    with pytest.raises(RuntimeError, match="test or frozen-holdout"):
        _reject_forbidden_path(path, "validation data")


def test_checkpoint_and_v7_config_sha_mismatch_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"actual")
    with pytest.raises(RuntimeError, match="checkpoint SHA256 mismatch"):
        _require_sha(path, "0" * 64, "checkpoint")
    with pytest.raises(RuntimeError, match="V7 config SHA256 mismatch"):
        _require_sha(path, "1" * 64, "V7 config")


def test_wrong_seed_and_seed_identity_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="42 or 43"):
        _validate_seed_contract(44, "", "")
    frozen = FROZEN_SEEDS[43]
    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        _validate_seed_contract(43, "0" * 64, frozen["training_config_sha256"])


def test_formal_validation_cohort_rejects_duplicate_and_missing_samples() -> None:
    sources, targets = _cohort()
    molecules, identity = _validate_cohort_frames(sources, targets)
    assert len(molecules) == 5_000
    assert len(identity) == 64

    duplicated = targets.copy()
    duplicated.loc[1, "sample_id"] = duplicated.loc[0, "sample_id"]
    with pytest.raises(RuntimeError, match="duplicate sample_id"):
        _validate_cohort_frames(sources, duplicated)

    missing = targets.copy()
    missing.loc[0, "sample_id"] = "val::missing__gen0"
    with pytest.raises(RuntimeError, match="missing or mismatched"):
        _validate_cohort_frames(sources, missing)


def test_formal_validation_cohort_rejects_test_record() -> None:
    sources, targets = _cohort()
    sources.loc[0, "test_record"] = True
    with pytest.raises(RuntimeError, match="contains test records"):
        _validate_cohort_frames(sources, targets)


def test_combined_methods_are_paired_and_source_is_not_duplicated() -> None:
    evaluations = {
        method: _evaluation(0.5, 0.4 - index * 0.01)
        for index, method in enumerate(METHODS)
    }
    records, molecules = _combine_method_evaluations(evaluations)
    assert set(records.method) == {"Source", *METHODS}
    assert set(molecules.method) == {"Source", *METHODS}
    assert int((records.method == "Source").sum()) == 1


def test_combined_methods_reject_source_metric_mismatch() -> None:
    evaluations = {method: _evaluation(0.5, 0.4) for method in METHODS}
    evaluations["V7"] = _evaluation(0.6, 0.4)
    with pytest.raises(AssertionError):
        _combine_method_evaluations(evaluations)


def test_resume_skips_only_integrity_checked_completed_chunk(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk_0001"
    chunk.mkdir()
    records = pd.DataFrame([{"sample_id": "sample", "value": 1}])
    molecules = pd.DataFrame([{"molecule_id": "mol", "value": 1}])
    records.to_csv(chunk / "records.csv", index=False)
    molecules.to_csv(chunk / "molecules.csv", index=False)
    summary = {
        "status": "COMPLETED",
        "chunk": 1,
        "methods": list(METHODS),
        "sample_identity_sha256": "a" * 64,
        "files": {
            "records.csv": {
                "rows": 1,
                "sha256": file_sha256(chunk / "records.csv"),
            },
            "molecules.csv": {
                "rows": 1,
                "sha256": file_sha256(chunk / "molecules.csv"),
            },
        },
    }
    (chunk / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _completed_chunk(
        chunk, chunk_index=1, expected_sample_identity="a" * 64
    ) == summary

    records.to_csv(chunk / "records.csv", index=False, columns=["value", "sample_id"])
    with pytest.raises(RuntimeError, match="output integrity failed"):
        _completed_chunk(chunk, chunk_index=1, expected_sample_identity="a" * 64)


def test_output_conflict_fails_closed_and_resume_is_allowed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "launch.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _validate_existing_output(output, resume=False)
    _validate_existing_output(output, resume=True)

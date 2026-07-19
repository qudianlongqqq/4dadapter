from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts import report_ecir_mvr_v7_formal_test as reporter
from scripts import run_ecir_mvr_v7_formal_test as runner


def test_dry_run_opens_no_test_assets_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("dry-run touched formal-test state")

    output = tmp_path / "formal-test"
    monkeypatch.setattr(runner, "_static_preflight", fail)
    monkeypatch.setattr(runner, "_open_manifest_and_index", fail)
    assert runner.main(["--dry-run", "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "V7_FORMAL_TEST_DRY_RUN_READY"
    assert result["test_records_read"] == 0
    assert result["test_assets_opened"] is False
    assert result["output_files_created"] is False
    assert not output.exists()


def test_formal_test_requires_authorization_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_static_preflight",
        lambda _args: pytest.fail("preflight ran without authorization"),
    )
    args = runner.build_parser().parse_args([])
    with pytest.raises(RuntimeError, match="explicit --authorize-frozen-test"):
        runner._evaluate(args)


def test_dry_run_and_authorization_are_mutually_exclusive() -> None:
    args = runner.build_parser().parse_args(
        ["--dry-run", "--authorize-frozen-test"]
    )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        runner._dry_run(args)


def test_locked_plan_freezes_v7_method_and_evaluator_semantics() -> None:
    plan = runner._load_plan()
    assert plan["methods"] == ["Source", "D1", "V5-B", "V7"]
    assert (
        plan["existing_d1b_evaluator_role"]
        == "D1_B_ONLY_NOT_A_V7_FORMAL_TEST_RUNNER"
    )
    assert plan["evaluator"]["semantics_git_commit"] == runner.SEMANTICS_COMMIT
    assert plan["test"]["records"] == 23_882
    assert plan["test"]["molecules"] == 100


def test_cache_filename_maps_to_frozen_sample_id() -> None:
    path = Path(
        "test__00321cd59b4f8000837e92ae6c17d444__"
        "00439dbc39baf2711d4c__gen0325.pt"
    )
    assert (
        runner._cache_sample_id(path)
        == "test::00321cd59b4f8000837e92ae6c17d444__gen0325"
    )
    with pytest.raises(RuntimeError, match="unexpected frozen test cache filename"):
        runner._cache_sample_id(Path("unrelated.pt"))


def test_prediction_contract_cannot_contain_references() -> None:
    assert runner.PREDICTION_FIELDS == {
        "schema_version",
        "sample_ids",
        "coordinates",
        "metadata",
    }
    assert not any("ref" in name for name in runner.PREDICTION_FIELDS)


def test_source_item_rejects_reference_fields() -> None:
    with pytest.raises(RuntimeError, match="reference field entered"):
        runner._source_item(
            {},
            {"x_ref_candidates": torch.zeros(1, 1, 3)},
            {},
            {},
            SimpleNamespace(),
        )


def test_resume_accepts_only_integrity_checked_prediction_chunk(
    tmp_path: Path,
) -> None:
    chunk = tmp_path / "chunk_0001"
    prediction = {
        "schema_version": "test",
        "sample_ids": ["sample"],
        "coordinates": {method: [torch.zeros(1, 3)] for method in runner.METHODS},
        "metadata": {method: [{}] for method in runner.METHODS},
    }
    runner._atomic_torch(chunk / "predictions.pt", prediction)
    summary = {
        "status": "PREDICTIONS_COMPLETE",
        "chunk": 1,
        "methods": list(runner.METHODS),
        "sample_identity_sha256": "a" * 64,
        "prediction_sha256": runner.file_sha256(chunk / "predictions.pt"),
        "prediction_payload_fields": sorted(runner.PREDICTION_FIELDS),
    }
    runner._atomic_json(chunk / "summary.json", summary)
    assert runner._completed_prediction(chunk, 1, "a" * 64) == summary
    (chunk / "predictions.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="integrity failed"):
        runner._completed_prediction(chunk, 1, "a" * 64)


def test_completed_formal_test_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "formal-test"
    output.mkdir()
    (output / "run_metadata.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8"
    )
    args = runner.build_parser().parse_args(
        ["--authorize-frozen-test", "--resume", "--output-dir", str(output)]
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite completed"):
        runner._prepare_output(args, {"evaluator_git_head": "commit"})


def _report_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    molecules = []
    metric_columns = set(reporter.METRICS.values())
    for molecule_index in range(2):
        molecule = f"mol{molecule_index}"
        for method_index, method in enumerate(reporter.METHODS):
            base = 1.0 - method_index * 0.1
            molecule_row = {
                "group": "all",
                "method": method,
                "molecule_id": molecule,
            }
            for column in metric_columns:
                molecule_row[column] = base
            molecule_row["accepted"] = 1.0
            molecule_row["molecule_rms_displacement"] = method_index * 0.01
            molecules.append(molecule_row)
            for record_index in range(2):
                record = {
                    "method": method,
                    "molecule_id": molecule,
                    "sample_id": f"{molecule}::sample{record_index}",
                    "angle_outlier_rate": base,
                }
                for column in reporter.RECORD_METRICS.values():
                    record[column] = base
                record["accepted"] = 1.0
                record["molecule_rms_displacement"] = method_index * 0.01
                records.append(record)
    return pd.DataFrame(records), pd.DataFrame(molecules)


def test_reporter_produces_frozen_paired_comparisons(tmp_path: Path) -> None:
    records, molecules = _report_frames()
    records.to_csv(tmp_path / "formal_test_per_record.csv", index=False)
    molecules.to_csv(tmp_path / "formal_test_per_molecule.csv", index=False)
    (tmp_path / "angle_solver_summary.json").write_text(
        json.dumps(
            {
                "calls": 1,
                "solver_failure_count": 0,
                "condition_number_mean": 1.0,
                "condition_number_max": 1.0,
                "effective_rank_mean": 1.0,
                "truncated_direction_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "component_summary.json").write_text("{}", encoding="utf-8")
    report = reporter.build_report(
        tmp_path,
        seed=43,
        bootstrap_draws=100,
        expected_records=4,
        expected_molecules=2,
    )
    assert report["methods"] == ["Source", "D1", "V5-B", "V7"]
    assert set(report["paired"]) == {
        "D1-minus-Source",
        "V5-B-minus-Source",
        "V7-minus-Source",
        "V7-minus-D1",
        "V7-minus-V5-B",
    }
    assert report["paired"]["V7-minus-D1"]["all"]["metrics"]["bond"][
        "mean"
    ] == pytest.approx(-0.2)
    assert report["predictions_complete_before_reference_access"] is True
    assert report["parameter_tuning_from_test"] is False

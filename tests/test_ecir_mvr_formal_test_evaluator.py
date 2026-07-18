from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import evaluate_ecir_mvr_formal_test as evaluator


def test_dry_run_reads_no_test_assets_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    forbidden = tmp_path / "must-not-be-opened-test.json"
    output = tmp_path / "formal-test-output"

    def fail(*_args, **_kwargs):
        raise AssertionError("dry-run touched a frozen test asset")

    monkeypatch.setattr(evaluator, "_load_locked_plan", fail)
    monkeypatch.setattr(evaluator, "_load_test_items", fail)
    exit_code = evaluator.main(
        [
            "--dry-run",
            "--frozen-test-plan",
            str(forbidden),
            "--test-manifest",
            str(forbidden),
            "--test-cache-root",
            str(tmp_path / "test-cache"),
            "--output-dir",
            str(output),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["status"] == evaluator.DRY_RUN_STATUS
    assert result["test_records_read"] == 0
    assert result["test_assets_opened"] is False
    assert result["output_files_created"] is False
    assert not output.exists()


def test_real_evaluation_requires_explicit_authorization(tmp_path: Path) -> None:
    args = evaluator.build_parser().parse_args(
        ["--frozen-test-plan", str(tmp_path / "absent.json")]
    )
    with pytest.raises(RuntimeError, match="explicit --authorize-frozen-test"):
        evaluator._load_locked_plan(args)


def test_locked_plan_requires_both_frozen_seeds(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": evaluator.PLAN_SCHEMA_VERSION,
                "status": evaluator.LOCKED_PLAN_STATUS,
                "checkpoints": [
                    {
                        "seed": 42,
                        "checkpoint": str(evaluator.SEED42_CHECKPOINT),
                        "checkpoint_sha256": evaluator.SEED42_CHECKPOINT_SHA256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = evaluator.build_parser().parse_args(
        ["--authorize-frozen-test", "--frozen-test-plan", str(plan)]
    )
    with pytest.raises(RuntimeError, match="seed42 and seed43"):
        evaluator._load_locked_plan(args)


def test_locked_plan_allows_cross_platform_path_relocation(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": evaluator.PLAN_SCHEMA_VERSION,
                "status": evaluator.LOCKED_PLAN_STATUS,
                "checkpoint_or_config_selected_from_test": False,
                "checkpoints": [
                    {
                        "seed": 42,
                        "checkpoint": r"C:\\old-machine\\seed42.ckpt",
                        "checkpoint_sha256": evaluator.SEED42_CHECKPOINT_SHA256,
                    },
                    {
                        "seed": 43,
                        "checkpoint": r"C:\\old-machine\\seed43.ckpt",
                        "checkpoint_sha256": "43" * 32,
                    },
                ],
                "test": {
                    "manifest": r"C:\\old-machine\\formal_large_test.json",
                    "cache_root": r"C:\\old-machine\\test_cache",
                    "manifest_sha256": "a" * 64,
                    "manifest_content_sha256": "b" * 64,
                    "source_identity_sha256": "c" * 64,
                    "reference_identity_sha256": "d" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    args = evaluator.build_parser().parse_args(
        [
            "--authorize-frozen-test",
            "--frozen-test-plan",
            str(plan),
            "--checkpoint",
            str(tmp_path / "relocated-seed42.ckpt"),
            "--test-manifest",
            str(tmp_path / "relocated-manifest.json"),
            "--test-cache-root",
            str(tmp_path / "relocated-cache"),
        ]
    )
    assert evaluator._load_locked_plan(args)["status"] == evaluator.LOCKED_PLAN_STATUS


def test_frozen_inference_settings_cannot_change() -> None:
    config = {
        "training": {"teacher_steps": 4},
        "inference": dict(evaluator.FROZEN_INFERENCE),
    }
    evaluator._validate_inference_config(config)
    config["inference"]["step_size"] = 0.5
    with pytest.raises(RuntimeError, match="step_size"):
        evaluator._validate_inference_config(config)


def test_dry_run_rejects_test_authorization() -> None:
    args = evaluator.build_parser().parse_args(
        ["--dry-run", "--authorize-frozen-test"]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluator._dry_run(args)


def test_existing_formal_runner_is_not_declared_as_mcvr() -> None:
    result = evaluator._dry_run(evaluator.build_parser().parse_args(["--dry-run"]))
    assert (
        result["legacy_evaluator_audit"]["run_formal_large_final_test.sh"]
        == "INCOMPATIBLE_CARTESIAN_GLOBAL4D"
    )
    assert result["model_entrypoint"].endswith("MCVRModel")
    assert result["minimal_validity_target_test_required"] is False


def test_evaluator_uses_strict_state_dict_load() -> None:
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert 'model.load_state_dict(payload["model_state_dict"], strict=True)' in source


def test_inference_record_structurally_removes_references_and_targets() -> None:
    raw = {
        "mol_id": "molecule",
        "x_init": "source",
        "x_ref_candidates": "reference",
        "selected_reference_index": 0,
        "rmsd_before": 1.0,
        "u_t": "training-label",
        "x_target": "minimal-target",
    }
    inference = evaluator._inference_record(raw)
    assert inference == {"mol_id": "molecule", "x_init": "source"}


def test_extra_metrics_receive_paired_bootstrap_intervals() -> None:
    rows = []
    for molecule, baseline, refined in (("a", 0.0, 0.1), ("b", 0.0, 0.2)):
        for method, displacement in (
            ("source_baseline", baseline),
            ("d1b_refined", refined),
        ):
            rows.append(
                {
                    "group": "all",
                    "molecule_id": molecule,
                    "method": method,
                    "molecule_rms_displacement": displacement,
                }
            )
    result = evaluator._extra_paired_bootstrap(
        pd.DataFrame(rows),
        candidate="d1b_refined",
        baseline="source_baseline",
        draws=50,
        seed=42,
    )
    assert result["molecule_rms_displacement"]["mean"] == pytest.approx(0.15)
    assert set(result["molecule_rms_displacement"]) == {
        "mean",
        "ci95_low",
        "ci95_high",
    }


def test_manifest_content_identity_allows_line_ending_relocation() -> None:
    plan = {
        "manifest_sha256": "windows-crlf-file-sha",
        "manifest_content_sha256": "canonical-content-sha",
    }
    assert (
        evaluator._validate_manifest_identity(
            "linux-lf-file-sha", "canonical-content-sha", plan
        )
        is False
    )
    with pytest.raises(RuntimeError, match="content SHA256 mismatch"):
        evaluator._validate_manifest_identity(
            "windows-crlf-file-sha", "changed-content-sha", plan
        )

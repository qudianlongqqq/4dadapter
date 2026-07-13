import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from etflow.formal_large import (
    ALPHAS,
    CONFIRM_MAX_RECORDS,
    SCREEN_MAX_RECORDS,
    TEST_MOLECULES,
    TRAINING_BUDGET,
    TRAIN_MOLECULES,
    VAL_MOLECULES,
    assert_disjoint_splits,
    assert_matched_training_budgets,
    canonical_sha256,
    pair_count_distribution,
    select_pair_records,
    select_stratified_manifest,
    verify_frozen_config,
)
from scripts.report_global_coupled_4d_first_result import load_valid_result
from scripts import report_formal_large_progress


ROOT = Path(__file__).resolve().parents[1]


def _rows(count, split="train", pairs=1):
    return [
        {
            "source_mol_id": f"{split}-mol-{molecule}",
            "mol_id": f"{split}-mol-{molecule}__gen{pair:04d}",
            "sample_id": f"{split}-mol-{molecule}__gen{pair:04d}",
            "x_init_hash": f"hash-{molecule}-{pair}",
            "num_rotatable_bonds": molecule % 9,
        }
        for molecule in range(count)
        for pair in range(pairs)
    ]


def test_train_val_test_molecule_ids_are_strictly_disjoint():
    splits = {name: _rows(3, name) for name in ("train", "val", "test")}
    assert_disjoint_splits(splits)
    splits["test"][0]["source_mol_id"] = splits["train"][0]["source_mol_id"]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_splits(splits)


def test_50k_train_molecule_selection_is_deterministic():
    candidates = _rows(TRAIN_MOLECULES + 7, pairs=1)
    first = select_pair_records(
        candidates, molecule_limit=TRAIN_MOLECULES, pairs_per_molecule=3
    )
    second = select_pair_records(
        reversed(candidates), molecule_limit=TRAIN_MOLECULES, pairs_per_molecule=3
    )
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len(first) == TRAIN_MOLECULES


def test_5k_validation_molecule_selection_is_deterministic():
    candidates = _rows(VAL_MOLECULES + 3, split="val")
    first = select_pair_records(
        candidates, molecule_limit=VAL_MOLECULES, pairs_per_molecule=2
    )
    second = select_pair_records(
        candidates, molecule_limit=VAL_MOLECULES, pairs_per_molecule=2
    )
    assert first == second and len(first) == VAL_MOLECULES


def test_pair_caps_and_pair_count_distribution_are_correct():
    selected = select_pair_records(
        _rows(4, pairs=5), molecule_limit=4, pairs_per_molecule=3
    )
    assert len(selected) == 12
    assert pair_count_distribution(selected) == {"3": 4}


def test_formal_manifest_hash_is_path_independent_and_content_sensitive():
    manifest = {"manifest_version": "1.0", "records": _rows(2)}
    assert canonical_sha256(manifest) == canonical_sha256(json.loads(json.dumps(manifest)))
    changed = json.loads(json.dumps(manifest))
    changed["records"][0]["x_init_hash"] = "changed"
    assert canonical_sha256(changed) != canonical_sha256(manifest)


def test_cartesian_and_global4d_training_budgets_match_frozen_contract():
    configs = {
        "cartesian": yaml.safe_load(
            (ROOT / "configs/formal_large_cartesian_seed42_200k.yaml").read_text()
        ),
        "global4d": yaml.safe_load(
            (ROOT / "configs/formal_large_global4d_seed42_200k.yaml").read_text()
        ),
    }
    assert assert_matched_training_budgets(configs) == TRAINING_BUDGET
    assert {config["data"]["cache_dir"] for config in configs.values()} == {
        "data/flexbond_cache_formal_large"
    }
    previous = yaml.safe_load(
        (ROOT / "configs/global_coupled_4d_local025_matched.yaml").read_text()
    )
    for key in ("hidden_dim", "edge_hidden_dim", "time_embedding_dim", "num_layers"):
        assert configs["global4d"]["model"][key] == previous["model"][key]


def test_both_training_methods_have_auto_resume_and_milestones():
    cartesian = (ROOT / "scripts/train_flexbond_optimizer.py").read_text()
    global4d = (ROOT / "scripts/train_global_coupled_4d_flow.py").read_text()
    runner = (ROOT / "scripts/run_formal_large_training.sh").read_text()
    assert 'default="auto"' in cartesian and 'default="auto"' in global4d
    assert "50000,100000,150000,200000" in runner
    assert "step200000.ckpt" in runner


def _validation_manifest():
    records = []
    for tier, rotations, count in (("low", 1, 6), ("medium", 4, 12), ("high", 7, 18)):
        for index in range(count):
            records.append({
                "mol_id": f"{tier}-{index}",
                "sample_id": f"{tier}-{index}__gen0000",
                "x_init_hash": f"{tier}-hash-{index}",
                "num_rotatable_bonds": rotations,
            })
    return {"manifest_version": "1.0", "records": records}


def test_screen10_is_one_fixed_cohort_shared_by_both_methods():
    selected = select_stratified_manifest(
        _validation_manifest(), {"low": 2, "medium": 3, "high": 5}
    )
    assert len({row["mol_id"] for row in selected["records"]}) == 10
    script = (ROOT / "scripts/run_formal_large_screen10.sh").read_text()
    assert script.count('MANIFEST="manifests/formal_large_val_screen10.json"') == 1


def test_confirm30_is_one_fixed_cohort_shared_by_both_methods():
    selected = select_stratified_manifest(
        _validation_manifest(), {"low": 5, "medium": 10, "high": 15}
    )
    assert len({row["mol_id"] for row in selected["records"]}) == 30
    script = (ROOT / "scripts/run_formal_large_confirm30.sh").read_text()
    assert script.count('MANIFEST="manifests/formal_large_val_confirm30.json"') == 1


def test_screen_and_confirm_apply_deterministic_record_caps_in_original_order():
    source = _validation_manifest()
    expanded = []
    for row in source["records"]:
        for pair in range(30):
            expanded.append(
                {
                    **row,
                    "sample_id": f"{row['mol_id']}__gen{pair:04d}",
                    "x_init_hash": f"{row['mol_id']}-hash-{pair}",
                }
            )
    source = {**source, "records": expanded}
    screen = select_stratified_manifest(
        source,
        {"low": 2, "medium": 3, "high": 5},
        max_records=SCREEN_MAX_RECORDS,
    )
    confirm = select_stratified_manifest(
        source,
        {"low": 5, "medium": 10, "high": 15},
        max_records=CONFIRM_MAX_RECORDS,
    )
    positions = {
        row["sample_id"]: index for index, row in enumerate(source["records"])
    }
    for selected, molecules, maximum in (
        (screen, 10, SCREEN_MAX_RECORDS),
        (confirm, 30, CONFIRM_MAX_RECORDS),
    ):
        ids = [row["sample_id"] for row in selected["records"]]
        assert len(ids) == maximum
        assert len({row["mol_id"] for row in selected["records"]}) == molecules
        assert [positions[value] for value in ids] == sorted(positions[value] for value in ids)
        report = selected["selection_report"]
        assert report["selected_record_count"] == maximum
        assert report["selected_molecule_count"] == molecules
        assert report["truncated_molecules"]


def test_formal_selection_scripts_expose_200_and_600_record_defaults():
    screen = (ROOT / "scripts/run_formal_large_screen10.sh").read_text()
    confirm = (ROOT / "scripts/run_formal_large_confirm30.sh").read_text()
    assert 'SCREEN_MAX_RECORDS="${SCREEN_MAX_RECORDS:-200}"' in screen
    assert 'CONFIRM_MAX_RECORDS="${CONFIRM_MAX_RECORDS:-600}"' in confirm
    assert '--max_records "${SCREEN_MAX_RECORDS}"' in screen
    assert '--max_records "${CONFIRM_MAX_RECORDS}"' in confirm


def test_both_inference_alphas_are_screened_for_every_checkpoint():
    script = (ROOT / "scripts/run_formal_large_screen10.sh").read_text()
    assert ALPHAS == (0.2, 0.5)
    assert "for alpha_code in 02 05" in script
    assert "for step in 50000 100000 150000 200000" in script


def test_test_split_is_not_used_for_screen_or_confirm_selection():
    for name in ("run_formal_large_screen10.sh", "run_formal_large_confirm30.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "formal_large_test.json" not in text
        assert "--split val" in text


def test_frozen_best_config_hash_validation(tmp_path):
    checkpoint = tmp_path / "model.ckpt"; checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"; config.write_text("seed: 42\n")
    manifest = {"manifest_version": "1.0", "records": []}
    from etflow.formal_large import file_sha256
    frozen = {
        "checkpoint_file_sha256": file_sha256(checkpoint),
        "config_file_sha256": file_sha256(config),
        "validation_manifest_sha256": canonical_sha256(manifest),
    }
    verify_frozen_config(
        frozen,
        checkpoint_path=checkpoint,
        resolved_config_path=config,
        manifest=manifest,
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        verify_frozen_config(
            frozen,
            checkpoint_path=checkpoint,
            resolved_config_path=config,
            manifest=manifest,
        )


def test_final_test_runs_exactly_one_frozen_combination_per_refiner():
    text = (ROOT / "scripts/run_formal_large_final_test.sh").read_text()
    assert "for method in cartesian global4d" in text
    assert "for step" not in text and "for alpha" not in text
    assert "verify_formal_large_best_configs.py" in text


def test_compact_progress_does_not_print_ordered_sample_ids(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(report_formal_large_progress, "LOG", tmp_path / "logs")
    monkeypatch.setattr(report_formal_large_progress, "DIAG", tmp_path / "diag")
    state = tmp_path / "diag/screen10/global4d/run/sampling_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps({
            "format_version": "global4d-sampling-state-v2",
            "completed_count": 150,
            "total_count": 200,
            "eta_seconds": 12.5,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["report_formal_large_progress.py", "--compact"])
    report_formal_large_progress.main()
    output = capsys.readouterr().out
    assert "ordered_sample_ids" not in output and "completed_ordered" not in output
    assert "record_progress: 150/200" in output


def test_both_formal_samplers_have_partial_resume_contract():
    cartesian = (ROOT / "scripts/sample_formal_large_cartesian.py").read_text()
    global4d = (ROOT / "scripts/sample_global_coupled_4d_flow.py").read_text()
    for text in (cartesian, global4d):
        assert "partial_samples.pt" in text
        assert "sampling_state.json" in text
        assert "atomic_torch_save" in text


def test_cartesian_and_global4d_final_evaluation_use_identical_manifest():
    text = (ROOT / "scripts/run_formal_large_final_test.sh").read_text()
    assert text.count('--manifest "${TEST_MANIFEST}"') >= 3
    assert "--cartesian_samples" in text and "--global_coupled_4d_samples" in text
    assert TEST_MOLECULES == 100


def test_first_small_result_must_be_valid_before_stop(tmp_path):
    group = tmp_path / "step1000_alpha02"; (group / "eval").mkdir(parents=True)
    with pytest.raises(ValueError, match="samples"):
        load_valid_result(group)
    payload = {
        "records": [{
            "sample_id": "s", "source_mol_id": "m", "checkpoint_path": "step1000.ckpt",
            "alpha": 0.2,
        }],
        "manifest_provenance": {
            "manifest": {"sha256": "manifest-hash"}, "sample_count": 1
        },
    }
    torch.save(payload, group / "samples.pt")
    fields = ["method", "subset", "rmsd_mean", "MAT-P", "MAT-R", "COV-P", "COV-R", "failure_rate"]
    with (group / "eval/summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow({"method": "global_coupled_4d_adapter", "subset": "all",
                         "rmsd_mean": 1, "MAT-P": 1, "MAT-R": 1,
                         "COV-P": .5, "COV-R": .5, "failure_rate": 0})
    assert load_valid_result(group)["sample_record_count"] == 1
    stop = (ROOT / "scripts/stop_global_coupled_4d_after_first_result.sh").read_text()
    assert stop.index("--check-only") < stop.index("kill -TERM")
    assert "SMALL_SWEEP_STOPPED_AFTER_FIRST_RESULT" in stop
    legacy = (ROOT / "scripts/run_global_coupled_4d_formal_matched.sh").read_text()
    assert "SMALL_SWEEP_STOPPED_AFTER_FIRST_RESULT" in legacy
    unified = (ROOT / "scripts/run_global_coupled_4d_smoke_and_matched.sh").read_text()
    assert "SMALL_SWEEP_STOPPED_AFTER_FIRST_RESULT" in unified


def test_formal_large_directories_are_isolated_and_training_does_not_start_inference():
    formal_scripts = [
        ROOT / "scripts/run_formal_large_training.sh",
        ROOT / "scripts/run_formal_large_screen10.sh",
        ROOT / "scripts/run_formal_large_confirm30.sh",
        ROOT / "scripts/run_formal_large_final_test.sh",
    ]
    combined = "\n".join(path.read_text() for path in formal_scripts)
    assert "logs_global_coupled_4d" not in combined
    assert "diagnostics/global_coupled_4d" not in combined
    assert "logs_formal_large" in combined and "diagnostics/formal_large" in combined
    training = formal_scripts[0].read_text()
    assert "sample_" not in training and "run_formal_large_screen10" not in training

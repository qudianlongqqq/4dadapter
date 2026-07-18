from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from etflow.ecir import formal_runtime_readiness as readiness
from scripts import preflight_ecir_mvr_formal_large as preflight
from scripts import train_ecir_mvr_medium_rescue_v2 as training
from scripts import validate_ecir_mvr_formal_runtime as validator


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml"


class FakeDataset:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.plan = []

    def __getitem__(self, index):
        if index in self.failing:
            raise ValueError(f"topology signature failed at {index}")
        return type("Item", (), {"num_nodes": 2})()


def _scan_frames(count=5):
    sources = pd.DataFrame(
        [
            {
                "sample_id": f"train::sample-{index}",
                "source_path": f"source-{index}.pt",
                "generator_name": "formal",
                "source_severity": "normal",
                "num_atoms": 2,
            }
            for index in range(count)
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "sample_id": f"train::sample-{index}",
                "target_cache_path": f"target-{index}.pt",
            }
            for index in range(count)
        ]
    )
    return sources, targets


def test_runtime_scan_collects_every_failure_instead_of_stopping():
    sources, targets = _scan_frames()
    seen = []

    def pair_validator(source, target, identities):
        index = int(source["sample_id"].rsplit("-", 1)[1])
        seen.append(index)
        if index in {1, 4}:
            raise ValueError(f"source and target atom counts differ at {index}")
        return {"multi_component_ionic_molecule": int(index == 3)}

    checked, failures, observations = validator.scan_split(
        FakeDataset(failing={2}),
        sources,
        targets,
        split="train",
        pair_validator=pair_validator,
        target_identities={},
    )
    assert checked == 5 and seen == [0, 1, 2, 3, 4]
    assert [row["dataset_index"] for row in failures] == [1, 2, 4]
    assert observations["multi_component_ionic_molecule"] == 1


def test_runtime_validator_has_fixed_failure_classes_and_never_names_test_manifest():
    assert set(validator.FAILURE_CLASSIFICATIONS) >= {
        "disconnected_explicit_hydrogen",
        "disconnected_non_hydrogen_atom",
        "multi_component_ionic_molecule",
        "atom_count_mismatch",
        "atomic_number_mismatch",
        "formal_charge_mismatch",
        "atom_map_missing",
        "atom_map_duplicate",
        "atom_mapping_not_unique",
        "hydrogen_parent_not_unique",
        "topology_signature_mismatch",
        "source_target_identity_mismatch",
        "coordinate_shape_mismatch",
        "other",
    }
    source = inspect.getsource(validator.main)
    assert 'for split in ("train", "val")' in source
    assert "test_sources" not in source and "test_targets" not in source


def test_runtime_manifests_require_unique_test_free_exact_pairing():
    sources, targets = _scan_frames(count=2)
    sources["split"] = "train"
    sources["test_record"] = False
    targets["split"] = "train"
    targets["test_records_read"] = 0
    validator.validate_manifests(
        {"train": sources}, {"train": targets}, {"train": 2}
    )

    duplicated = pd.concat([targets.iloc[:1], targets.iloc[:1]], ignore_index=True)
    with pytest.raises(RuntimeError, match="not unique"):
        validator.validate_manifests(
            {"train": sources}, {"train": duplicated}, {"train": 2}
        )

    test_source = sources.copy()
    test_source.loc[0, "test_record"] = True
    with pytest.raises(RuntimeError, match="contains test rows"):
        validator.validate_manifests(
            {"train": test_source}, {"train": targets}, {"train": 2}
        )


def _runtime_config_and_report(tmp_path: Path):
    config = yaml.safe_load(BASE_CONFIG.read_text())
    source_metadata = tmp_path / "source.json"
    target_metadata = tmp_path / "target.json"
    source_metadata.write_text(
        json.dumps({"formal_source_identity_sha256": "1" * 64})
    )
    target_metadata.write_text(
        json.dumps(
            {
                "validity_statistics_identity_sha256": "2" * 64,
                "formal_target_identity_sha256": "3" * 64,
                "builder_code_sha256": "4" * 64,
                "builder_config_sha256": "5" * 64,
                "formal_rdkit_adapter_sha256": "6" * 64,
            }
        )
    )
    config["data"]["source_metadata"] = str(source_metadata)
    config["data"]["target_metadata"] = str(target_metadata)
    code = readiness.runtime_code_identity()
    report = {
        "decision": readiness.RUNTIME_READY,
        "train_checked": 150000,
        "val_checked": 10000,
        "passed_count": 160000,
        "failed_count": 0,
        "test_records_read": 0,
        "formal_asset_identities": readiness.formal_asset_identities(config),
        "runtime_adapter_sha256": readiness.file_sha256(
            ROOT / "etflow/ecir/formal_rdkit_adapter.py"
        ),
        "runtime_code_identity_sha256": code["identity_sha256"],
        "git_commit": readiness.git_commit(),
        "base_config_sha256": "a" * 64,
    }
    report["runtime_validation_identity_sha256"] = readiness.canonical_sha256(
        report
    )
    report_path = tmp_path / "runtime.json"
    report_path.write_text(json.dumps(report))
    config["runtime_validation"] = {
        "report": str(report_path),
        "report_sha256": readiness.file_sha256(report_path),
        "decision": readiness.RUNTIME_READY,
        "runtime_validation_identity_sha256": report[
            "runtime_validation_identity_sha256"
        ],
        "base_config_sha256": report["base_config_sha256"],
        "test_records_read": 0,
    }
    return config, report_path


def test_runtime_ready_report_sha_tamper_is_rejected(tmp_path):
    config, report_path = _runtime_config_and_report(tmp_path)
    assert readiness.assert_runtime_binding(config)["failed_count"] == 0
    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="report SHA changed"):
        readiness.assert_runtime_binding(config)


def test_training_entry_requires_runtime_gate_before_cuda_or_output_creation():
    with pytest.raises(RuntimeError, match="requires full runtime validation"):
        readiness.assert_runtime_binding({})
    source = inspect.getsource(training.main)
    assert source.index("assert_runtime_binding(config)") < source.index("_seed(")
    assert source.index("assert_runtime_binding(config)") < source.index(
        "output.mkdir"
    )


def test_preflight_requires_runtime_ready_before_gpu_query():
    source = inspect.getsource(preflight.main)
    assert source.index("assert_runtime_ready_for_base") < source.index("query_gpu(")


def test_capacity_report_cannot_replace_runtime_ready(tmp_path):
    config, report_path = _runtime_config_and_report(tmp_path)
    report = json.loads(report_path.read_text())
    report["decision"] = "D1B_FORMAL_CAPACITY_PASS"
    report.pop("runtime_validation_identity_sha256")
    report["runtime_validation_identity_sha256"] = readiness.canonical_sha256(
        report
    )
    report_path.write_text(json.dumps(report))
    config["runtime_validation"]["report_sha256"] = readiness.file_sha256(
        report_path
    )
    config["runtime_validation"]["runtime_validation_identity_sha256"] = report[
        "runtime_validation_identity_sha256"
    ]
    with pytest.raises(RuntimeError, match="not full test-free READY"):
        readiness.assert_runtime_binding(config)


def _pair_payloads(tmp_path, source_atoms, target_atoms, target_shape=None):
    source_path = tmp_path / "source.pt"
    target_path = tmp_path / "target.pt"
    source_atoms = torch.as_tensor(source_atoms, dtype=torch.long)
    target_atoms = torch.as_tensor(target_atoms, dtype=torch.long)
    target_shape = target_shape or (len(target_atoms), 3)
    torch.save({"atomic_numbers": source_atoms}, source_path)
    torch.save(
        {
            "source_atomic_numbers": target_atoms,
            "x_input": torch.zeros(len(target_atoms), 3),
            "x_target": torch.zeros(*target_shape),
        },
        target_path,
    )
    source = {
        "source_path": str(source_path),
        "source_file_sha256": readiness.file_sha256(source_path),
    }
    target = {
        "target_cache_path": str(target_path),
        "target_file_sha256": readiness.file_sha256(target_path),
    }
    return source, target


def test_pair_validation_rejects_target_coordinate_shape(monkeypatch, tmp_path):
    source, target = _pair_payloads(
        tmp_path, [6, 1], [6, 1], target_shape=(1, 3)
    )
    monkeypatch.setattr(
        validator.target_assets, "validate_target_payload", lambda *args: {}
    )
    with pytest.raises(ValueError, match="coordinate shape"):
        validator.validate_pair(source, target, {})


@pytest.mark.parametrize(
    ("source_atoms", "target_atoms", "message"),
    [
        ([6, 1], [6], "atom counts differ"),
        ([6, 1], [7, 1], "atomic-number sequences differ"),
    ],
)
def test_pair_validation_rejects_source_target_atom_identity(
    monkeypatch, tmp_path, source_atoms, target_atoms, message
):
    source, target = _pair_payloads(tmp_path, source_atoms, target_atoms)
    monkeypatch.setattr(
        validator.target_assets, "validate_target_payload", lambda *args: {}
    )
    with pytest.raises(ValueError, match=message):
        validator.validate_pair(source, target, {})

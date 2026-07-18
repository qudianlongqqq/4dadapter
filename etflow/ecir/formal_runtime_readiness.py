"""Identity checks for full formal train/validation runtime readiness."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_READY = "D1B_FORMAL_RUNTIME_READY"
RUNTIME_REPORT = ROOT / "reports/ecir_mvr/D1B_FORMAL_RUNTIME_VALIDATION.json"
RUNTIME_CODE_FILES = (
    ROOT / "etflow/ecir/formal_rdkit_adapter.py",
    ROOT / "etflow/ecir/formal_runtime_readiness.py",
    ROOT / "etflow/ecir/formal_target_assets.py",
    ROOT / "etflow/ecir/mvr_dataset.py",
    ROOT / "etflow/ecir/geometry.py",
    ROOT / "etflow/ecir/chemical_validity.py",
    ROOT / "scripts/train_ecir_mvr_run_a.py",
    ROOT / "scripts/validate_ecir_mvr_formal_runtime.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def runtime_code_identity() -> dict[str, Any]:
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in RUNTIME_CODE_FILES
    }
    return {
        "files": files,
        "identity_sha256": canonical_sha256(files),
    }


def formal_asset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    source = json.loads(
        Path(config["data"]["source_metadata"]).read_text(encoding="utf-8")
    )
    target = json.loads(
        Path(config["data"]["target_metadata"]).read_text(encoding="utf-8")
    )
    return {
        "validity_statistics_identity_sha256": target[
            "validity_statistics_identity_sha256"
        ],
        "formal_source_identity_sha256": source[
            "formal_source_identity_sha256"
        ],
        "formal_target_identity_sha256": target[
            "formal_target_identity_sha256"
        ],
        "builder_code_sha256": target["builder_code_sha256"],
        "builder_config_sha256": target["builder_config_sha256"],
        "target_build_adapter_sha256": target["formal_rdkit_adapter_sha256"],
    }


def _validate_report_common(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if (
        report.get("decision") != RUNTIME_READY
        or int(report.get("train_checked", -1)) != 150_000
        or int(report.get("val_checked", -1)) != 10_000
        or int(report.get("passed_count", -1)) != 160_000
        or int(report.get("failed_count", -1)) != 0
        or int(report.get("test_records_read", -1)) != 0
    ):
        raise RuntimeError("formal runtime validation is not full test-free READY")
    if report.get("formal_asset_identities") != formal_asset_identities(config):
        raise RuntimeError("formal runtime asset identities changed")
    code = runtime_code_identity()
    if (
        report.get("runtime_adapter_sha256")
        != file_sha256(ROOT / "etflow/ecir/formal_rdkit_adapter.py")
        or report.get("runtime_code_identity_sha256")
        != code["identity_sha256"]
        or report.get("git_commit") != git_commit()
    ):
        raise RuntimeError("formal runtime code identity changed")
    persisted_identity = report.get("runtime_validation_identity_sha256")
    unsigned = dict(report)
    unsigned.pop("runtime_validation_identity_sha256", None)
    if persisted_identity != canonical_sha256(unsigned):
        raise RuntimeError("formal runtime validation identity changed")


def assert_runtime_ready_for_base(
    config: Mapping[str, Any], config_path: Path, report_path: Path = RUNTIME_REPORT
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    _validate_report_common(report, config)
    if report.get("base_config_sha256") != file_sha256(config_path):
        raise RuntimeError("formal runtime base config identity changed")
    return report


def assert_runtime_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config.get("runtime_validation")
    if not isinstance(binding, Mapping):
        raise RuntimeError("formal training requires full runtime validation")
    report_path = Path(binding["report"])
    if file_sha256(report_path) != binding.get("report_sha256"):
        raise RuntimeError("formal runtime validation report SHA changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_report_common(report, config)
    if (
        binding.get("decision") != RUNTIME_READY
        or binding.get("runtime_validation_identity_sha256")
        != report.get("runtime_validation_identity_sha256")
        or binding.get("base_config_sha256")
        != report.get("base_config_sha256")
        or int(binding.get("test_records_read", -1)) != 0
    ):
        raise RuntimeError("formal runtime validation binding changed")
    return report

#!/usr/bin/env python
"""Finalize the formal 64-batch config from immutable Linux evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import yaml  # noqa: E402

from etflow.ecir.formal_runtime_readiness import (  # noqa: E402
    RUNTIME_REPORT,
    assert_runtime_ready_for_base,
    file_sha256 as runtime_file_sha256,
)
from scripts.preflight_ecir_mvr_formal_large import (  # noqa: E402
    STATUS_PASS,
    _validate_base_config,
)
from scripts.train_ecir_mvr_medium_rescue_v2 import (  # noqa: E402
    _assert_formal_identity,
    _assert_formal_preflight,
    _formal_asset_identities,
)


FORMAL64 = {
    "batch_size": 64,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 64,
    "optimizer_steps": 25_000,
    "total_sample_exposures": 1_600_000,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def validate_formal64_preflight(
    report: Mapping[str, Any],
    identities: Mapping[str, Any],
    *,
    base_config_sha256: str,
    expected_commit: str,
) -> None:
    recommended = report.get("recommended") or {}
    if (
        report.get("status") != STATUS_PASS
        or report.get("mode") != "formal_preflight"
        or bool(report.get("capacity_only"))
        or int(report.get("target_effective_batch", -1)) != 64
        or int(report.get("test_records_read", -1)) != 0
        or report.get("formal_training_started") is not False
        or report.get("formal_checkpoint_created") is not False
        or report.get("config_sha256") != base_config_sha256
        or report.get("commit_sha") != expected_commit
        or report.get("frozen_identities") != dict(identities)
        or int(recommended.get("micro_batch_size", -1)) != 64
        or int(recommended.get("gradient_accumulation_steps", -1)) != 1
        or int(recommended.get("effective_batch_size", -1)) != 64
    ):
        raise RuntimeError(
            "formal64 config requires a test-free non-capacity 64x1 preflight"
        )


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finalize_formal64_config(
    base_config_path: Path,
    preflight_report_path: Path,
    output_path: Path,
    runtime_report_path: Path = RUNTIME_REPORT,
) -> dict[str, Any]:
    if (
        preflight_report_path.name != "D1B_FORMAL_PREFLIGHT.json"
        or preflight_report_path.parent.name != "formal64_preflight"
    ):
        raise RuntimeError(
            "finalizer accepts only formal64_preflight/D1B_FORMAL_PREFLIGHT.json"
        )
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    _validate_base_config(base, 64)
    identities = _formal_asset_identities(base)
    runtime_report = assert_runtime_ready_for_base(
        base, base_config_path, runtime_report_path
    )
    identity_checked = json.loads(json.dumps(base))
    identity_checked["frozen_identities"] = dict(identities)
    audit_path = Path(base["data"]["target_validation"])
    _assert_formal_identity(identity_checked, audit_path)

    report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    validate_formal64_preflight(
        report,
        identities,
        base_config_sha256=_sha256(base_config_path),
        expected_commit=_git_commit(),
    )
    if (
        report.get("runtime_validation_report_sha256")
        != runtime_file_sha256(runtime_report_path)
        or report.get("runtime_validation_identity_sha256")
        != runtime_report["runtime_validation_identity_sha256"]
    ):
        raise RuntimeError("formal64 preflight runtime validation binding changed")
    resolved = json.loads(json.dumps(base))
    resolved["training"].update(
        {
            **FORMAL64,
            "checkpoint_steps": [6250, 12500, 18750, 25000],
            "checkpoint_validation_steps": [6250, 12500, 18750, 25000],
        }
    )
    resolved["data"]["runtime_optimizations"] = {
        "formal_adapter_lru_size": 0,
        "precompute_training_topology": False,
    }
    resolved["frozen_identities"] = dict(identities)
    resolved["preflight"] = {
        "report": str(preflight_report_path.resolve()),
        "report_sha256": _sha256(preflight_report_path),
        "status": STATUS_PASS,
        "target_effective_batch": 64,
        "capacity_report_used": False,
        "test_records_read": 0,
        "target_validation_decision": "D1B_FORMAL_TARGETS_READY",
        "manual_training_confirmation_required": True,
    }
    resolved["runtime_validation"] = {
        "report": str(runtime_report_path.resolve()),
        "report_sha256": runtime_file_sha256(runtime_report_path),
        "decision": "D1B_FORMAL_RUNTIME_READY",
        "runtime_validation_identity_sha256": runtime_report[
            "runtime_validation_identity_sha256"
        ],
        "base_config_sha256": runtime_report["base_config_sha256"],
        "test_records_read": 0,
    }
    _assert_formal_identity(resolved, audit_path)
    _assert_formal_preflight(resolved)
    _atomic_yaml(output_path, resolved)
    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    _assert_formal_identity(written, audit_path)
    _assert_formal_preflight(written)
    if any(int(written["training"][key]) != value for key, value in FORMAL64.items()):
        raise RuntimeError("written formal64 training budget differs")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml",
    )
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument(
        "--runtime-report", type=Path, default=RUNTIME_REPORT
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml",
    )
    args = parser.parse_args()
    finalize_formal64_config(
        args.base_config, args.preflight_report, args.output, args.runtime_report
    )
    print("D1B_FORMAL64_CONFIG_READY")


if __name__ == "__main__":
    main()

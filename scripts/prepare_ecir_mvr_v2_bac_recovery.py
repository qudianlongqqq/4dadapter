#!/usr/bin/env python3
"""Freeze train/validation-tune-only cohorts for V2-BAC recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "mcvr-v2-bac-recovery-plan-v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-cohort-manifest",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_overnight/validation_cohorts.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery"),
    )
    parser.add_argument("--seed", type=int, default=43017)
    parser.add_argument("--development-molecules", type=int, default=512)
    parser.add_argument("--diagnostic-molecules", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.formal_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohorts = json.loads(
        args.frozen_cohort_manifest.read_text(encoding="utf-8")
    )
    tune_ids = set(cohorts["validation_tune"]["molecule_ids"])
    holdout_ids = set(cohorts["validation_holdout"]["molecule_ids"])
    if tune_ids & holdout_ids:
        raise RuntimeError("frozen tune and holdout IDs overlap")
    sources = pd.read_parquet(root / "real_sources" / "val.parquet")
    targets = pd.read_parquet(root / "minimal_targets" / "val.parquet")
    tune = sources[sources.molecule_id.astype(str).isin(tune_ids)].copy()
    if set(map(str, tune.split.unique())) != {"val"}:
        raise RuntimeError("recovery source contains a non-validation split")
    ranked = sorted(
        tune_ids,
        key=lambda molecule_id: hashlib.sha256(
            f"{args.seed}|{cohorts['identity_sha256']}|{molecule_id}".encode()
        ).hexdigest(),
    )
    development_ids = set(ranked[: int(args.development_molecules)])
    diagnostic_ids = set(ranked[: int(args.diagnostic_molecules)])
    if development_ids & holdout_ids or diagnostic_ids & holdout_ids:
        raise RuntimeError("recovery development cohort overlaps frozen holdout")
    development = tune[
        tune.molecule_id.astype(str).isin(development_ids)
    ].sort_values(["molecule_id", "sample_id"])
    diagnostic = tune[
        tune.molecule_id.astype(str).isin(diagnostic_ids)
    ].sort_values(["molecule_id", "sample_id"])
    development_targets = targets[
        targets.sample_id.isin(development.sample_id)
    ].sort_values("sample_id")
    diagnostic_targets = targets[
        targets.sample_id.isin(diagnostic.sample_id)
    ].sort_values("sample_id")
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    paths = {
        "development_sources": manifests / "development_sources.parquet",
        "development_targets": manifests / "development_targets.parquet",
        "diagnostic_sources": manifests / "diagnostic_sources.parquet",
        "diagnostic_targets": manifests / "diagnostic_targets.parquet",
    }
    development.to_parquet(paths["development_sources"], index=False)
    development_targets.to_parquet(paths["development_targets"], index=False)
    diagnostic.to_parquet(paths["diagnostic_sources"], index=False)
    diagnostic_targets.to_parquet(paths["diagnostic_targets"], index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "parent_validation_cohort_identity_sha256": cohorts["identity_sha256"],
        "development": {
            "molecules": int(development.molecule_id.nunique()),
            "records": len(development),
            "ordered_molecule_ids_sha256": _canonical_sha(
                sorted(development_ids)
            ),
            "source_manifest": str(paths["development_sources"]),
            "source_manifest_sha256": _sha(paths["development_sources"]),
            "target_manifest": str(paths["development_targets"]),
            "target_manifest_sha256": _sha(paths["development_targets"]),
        },
        "diagnostic": {
            "molecules": int(diagnostic.molecule_id.nunique()),
            "records": len(diagnostic),
            "ordered_molecule_ids_sha256": _canonical_sha(sorted(diagnostic_ids)),
            "source_manifest": str(paths["diagnostic_sources"]),
            "source_manifest_sha256": _sha(paths["diagnostic_sources"]),
            "target_manifest": str(paths["diagnostic_targets"]),
            "target_manifest_sha256": _sha(paths["diagnostic_targets"]),
        },
        "frozen_holdout_molecule_overlap": 0,
        "formal_test_records_read": 0,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    manifest["identity_sha256"] = _canonical_sha(manifest)
    _write_json(manifests / "recovery_development_manifest.json", manifest)
    matrix = {
        "schema_version": "mcvr-v2-bac-recovery-experiment-matrix-v1",
        "code_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "development_manifest_identity_sha256": manifest["identity_sha256"],
        "fixed_model": {"hidden_dim": 64, "num_layers": 4},
        "gpu_budget": {
            "maximum_optimizer_invocations": 5,
            "two_batch_diagnostic": 1,
            "maximum_200_step_smoke": 1,
            "maximum_1000_step_pilots": 3,
        },
        "candidates": [
            {
                "id": "A0",
                "method": "original_bond_only_cartesian",
                "maximum_steps": 1000,
            },
            {
                "id": "D0",
                "method": "original_unified_bac_cartesian",
                "maximum_steps": 1000,
            },
            {
                "id": "D1",
                "method": "minimum_audit_driven_bac_recovery",
                "maximum_steps": 1000,
            },
        ],
        "stopping_conditions": [
            "nonfinite",
            "identity_or_chirality_change",
            "ring_degradation",
            "data_identity_mismatch",
            "repeated_oom",
            "gpu_budget_exhausted",
        ],
        "formal_test_records_read": 0,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    _write_json(output / "experiment_matrix.json", matrix)
    _write_json(
        output / "environment.json",
        {
            "schema_version": "mcvr-v2-bac-recovery-environment-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "branch": matrix["branch"],
            "commit": matrix["code_commit"],
            "test_records_read": 0,
            "test_assets_opened": False,
            "frozen_holdout_records_opened": 0,
            "validation_only": True,
        },
    )
    (output / "logs").mkdir(exist_ok=True)
    print(
        json.dumps(
            {
                "status": "MCVR_V2_BAC_RECOVERY_PLAN_FROZEN",
                "manifest_identity_sha256": manifest["identity_sha256"],
                "development_records": len(development),
                "diagnostic_records": len(diagnostic),
                "frozen_holdout_overlap": 0,
                "test_records_read": 0,
                "test_assets_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

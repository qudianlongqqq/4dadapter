#!/usr/bin/env python
"""Freeze identities and analysis rules before MCVR V8 multi-seed training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from scripts.train_ecir_mvr_v8 import ISOLATION, _canonical_sha, load_config


FROZEN_BASE = "4df21d766afadab169ecc7208477a6ca6ffe384a"
D1_SHA256 = "c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca"
CONFIGS = {
    12: ROOT / "configs" / "ecir_mvr_v8_full_v1_formal_large_200k_seed12.yaml",
    48: ROOT / "configs" / "ecir_mvr_v8_full_v1_formal_large_200k_seed48.yaml",
}
OUTPUTS = {
    12: "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k/full_seed12",
    48: "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k/full_seed48",
}
DATASETS = {
    "train_sources": ROOT / "data/ecir_mvr/formal_large/real_sources/train.parquet",
    "train_targets": ROOT / "data/ecir_mvr/formal_large/minimal_targets/train.parquet",
    "validation_sources": ROOT / "data/ecir_mvr/formal_large/real_sources/val.parquet",
    "validation_targets": ROOT / "data/ecir_mvr/formal_large/minimal_targets/val.parquet",
}
BASELINES = {
    name: ROOT
    / "diagnostics/ecir_mvr/validation_cache/formal_large_seed43"
    / directory
    / "prediction_manifest.json"
    for name, directory in {
        "Source": "source",
        "D1": "d1",
        "V5-B": "v5_b",
        "V7": "v7",
    }.items()
}
ALLOWED_DIFFERENCES = [
    "experiment_name",
    "seed",
    "long_run.parent_5k_checkpoint",
    "long_run.parent_5k_checkpoint_sha256",
    "long_run.resume_audit_required",
    "long_run.start_step",
    "multiseed_registration",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_differences(left[key], right[key], path))
        return paths
    return [] if left == right else [prefix]


def build_registry(*, require_clean: bool = True) -> dict[str, Any]:
    status = _git("status", "--short")
    if require_clean and status:
        raise RuntimeError("multi-seed registration requires a clean worktree")
    frozen = load_config(ROOT / "configs/ecir_mvr_v8_full_v1_formal_large_200k.yaml")
    configs = {}
    for seed, path in CONFIGS.items():
        resolved = load_config(path)
        differences = _differences(frozen, resolved)
        unexpected = [
            item
            for item in differences
            if not any(
                item == allowed or item.startswith(allowed + ".")
                for allowed in ALLOWED_DIFFERENCES
            )
        ]
        if unexpected:
            raise RuntimeError(f"Seed{seed} has unregistered config differences: {unexpected}")
        if resolved["model"]["d1_checkpoint_sha256"] != D1_SHA256:
            raise RuntimeError(f"Seed{seed} changed the frozen D1 checkpoint")
        if resolved["isolation"] != ISOLATION:
            raise RuntimeError(f"Seed{seed} changed data isolation")
        configs[str(seed)] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "config_file_sha256": _sha(path),
            "inherited_resolved_config_sha256": _canonical_sha(resolved),
            "differences_from_frozen_seed43": differences,
            "output_dir": OUTPUTS[seed],
        }
    baseline_identities = {}
    for name, path in BASELINES.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        baseline_identities[name] = {
            "manifest_sha256": _sha(path),
            "identity_sha256": manifest["identity"]["identity_sha256"],
            "source_cache_identity_sha256": manifest.get("source_cache_identity_sha256")
            or manifest["identity"]["identity_sha256"],
        }
    baseline_cache_sha = _canonical_sha(baseline_identities)
    return {
        "schema_version": "mcvr-v8-multiseed-experiment-registry-v1",
        "status": "MCVR_V8_MULTI_SEED_IMPLEMENTATION_READY",
        "branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "frozen_base_git_sha": FROZEN_BASE,
        "worktree_clean": not bool(status),
        "seeds": [12, 43, 48],
        "new_training_seeds": [12, 48],
        "training_contract": {
            "planned_original_total_steps": 200000,
            "user_requested_stop_step": 12500,
            "effective_batch": 64,
            "total_record_exposure": 800000,
            "schedule_provenance": "checkpoint_from_original_200k_schedule",
            "step10000_validation": "FAST",
            "step12500_validation": "FULL10K",
            "parallel_same_gpu": False,
            "result_conditioned_tuning": False,
        },
        "analysis_protocol": {
            "mean_std_seeds": [12, 43, 48],
            "standard_deviation": "sample",
            "ddof": 1,
            "metric_mapping": {
                "weighted_bac": "weighted_bac_delta",
                "bond": "bond_delta",
                "angle": "angle_delta",
                "ring": "ring_delta",
                "clash": "clash_delta",
                "acceptance": "accepted",
                "mean_displacement": "mean_displacement",
                "RMSD": "rmsd",
                "COV_P": "COV_P",
                "COV_R": "COV_R",
                "MAT_P": "MAT_P",
                "MAT_R": "MAT_R",
            },
            "report_both_cov_mat_directions": True,
            "single_direction_posthoc_selection": False,
            "clash_interpretation": "low-power natural cohort",
        },
        "config_difference_whitelist": ALLOWED_DIFFERENCES,
        "configs": configs,
        "identities": {
            "datasets": {name: _sha(path) for name, path in DATASETS.items()},
            "d1_checkpoint_sha256": D1_SHA256,
            "sampler_file_sha256": _sha(
                ROOT / "reports/ecir_mvr/v8_full_v1/formal_large_stratified_manifest.json"
            ),
            "sampler_identity_sha256": (
                "bac36073b62da27d4f771569c4f28246cc4ef9639204beb9b7f9bad7d8e765d6"
            ),
            "scales_file_sha256": _sha(
                ROOT / "reports/ecir_mvr/v8_full_v1/formal_large_train_scales.json"
            ),
            "scales_identity_sha256": (
                "5a1c33947d9913ff7aeda64901e7812e37ac5aa647c6534ce4044a00e49d04e4"
            ),
            "baseline_cache_sha256": baseline_cache_sha,
            "baseline_cache_identities": baseline_identities,
        },
        "isolation": ISOLATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/ecir_mvr/v8_full_v1/multiseed/EXPERIMENT_REGISTRY.json",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    registry = build_registry(require_clean=not args.allow_dirty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

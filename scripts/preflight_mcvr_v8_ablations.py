#!/usr/bin/env python
"""Freeze identities and single-factor differences before formal V8 ablations."""

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
EXPECTED = {
    "train_sources": "fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d",
    "train_targets": "7e97c5d92529608cfcace8cd279cbd25f20e08b28e1739a191483ba3b574c242",
    "validation_sources": "e7d29f971124f51bd385ec987372ab85181b152250ec0789407a867ff81e3c1a",
    "validation_targets": "4b4ef42c9905c3bbe2dbe911c57827ce594583c66a52f94d7c4d9b5ca70de4c7",
    "sampler": "e9cb52ccfeadef7056b44f3d29997eb1d2aa8462aa0fb843cc995f433efc90b1",
    "scales": "2a3187ff8d684f532978c1c8e44e0c2255db38781eda2a669a92a341c273540d",
}
CONFIGS = {
    "NO_CONSTRAINT": ROOT / "configs/ecir_mvr_v8_ablation_no_constraint_formal_large_200k.yaml",
    "NO_CONFIDENCE": ROOT / "configs/ecir_mvr_v8_ablation_no_confidence_formal_large_200k.yaml",
    "NO_ERROR_STATE": ROOT / "configs/ecir_mvr_v8_ablation_no_error_state_formal_large_200k.yaml",
    "NO_TYPE_NORMALIZATION": ROOT
    / "configs/ecir_mvr_v8_ablation_no_type_normalization_formal_large_200k.yaml",
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
    for name, directory in {"Source": "source", "D1": "d1", "V5-B": "v5_b", "V7": "v7"}.items()
}
COMMON_ALLOWED = {
    "experiment_name",
    "long_run.parent_5k_checkpoint",
    "long_run.parent_5k_checkpoint_sha256",
    "long_run.resume_audit_required",
    "long_run.start_step",
}
FACTOR_ALLOWED = {
    "NO_CONSTRAINT": {"constraint_layer.enabled"},
    "NO_CONFIDENCE": {
        "error_state.confidence_mode",
        "loss.confidence_regularization_weight",
    },
    "NO_ERROR_STATE": {
        "error_state.enabled",
        "error_state.confidence_mode",
        "loss.error_state_weight",
        "loss.confidence_regularization_weight",
    },
    "NO_TYPE_NORMALIZATION": {"type_normalization.enabled"},
}


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


def build_registry(sampler_manifest: Path, *, require_clean: bool = True) -> dict[str, Any]:
    status = _git("status", "--short")
    if require_clean and status:
        raise RuntimeError("ablation registration requires a clean worktree")
    if _git("branch", "--show-current") != "mcvr-v8-ablation":
        raise RuntimeError("formal ablations require branch mcvr-v8-ablation")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_BASE, "HEAD"], cwd=ROOT, check=True
    )
    d1 = ROOT / "artifacts/ecir_mvr/formal_large/d1_b_seed43/best_noninferior_validity.ckpt"
    scales = ROOT / "reports/ecir_mvr/v8_full_v1/formal_large_train_scales.json"
    identities = {name: _sha(path) for name, path in DATASETS.items()}
    identities.update({"sampler": _sha(sampler_manifest), "scales": _sha(scales)})
    if identities != EXPECTED:
        raise RuntimeError(f"frozen data/sampler/scales identities changed: {identities}")
    if _sha(d1) != D1_SHA256:
        raise RuntimeError("frozen D1 checkpoint SHA256 changed")

    baseline = load_config(ROOT / "configs/ecir_mvr_v8_full_v1_formal_large_200k.yaml")
    configs = {}
    for name, path in CONFIGS.items():
        resolved = load_config(path)
        differences = _differences(baseline, resolved)
        unexpected = [
            item
            for item in differences
            if item not in COMMON_ALLOWED
            and item not in FACTOR_ALLOWED[name]
            and item != "ablation_protocol"
            and not item.startswith("ablation_protocol.")
            and item != "ablation_registration"
            and not item.startswith("ablation_registration.")
        ]
        if unexpected:
            raise RuntimeError(f"{name} has unregistered config differences: {unexpected}")
        if resolved["model"]["d1_checkpoint_sha256"] != D1_SHA256:
            raise RuntimeError(f"{name} changed the D1 checkpoint")
        if resolved["isolation"] != ISOLATION:
            raise RuntimeError(f"{name} changed isolation")
        configs[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "config_file_sha256": _sha(path),
            "inherited_resolved_config_sha256": _canonical_sha(resolved),
            "differences_from_full": differences,
            "output_dir": resolved["ablation_registration"]["output_dir"],
        }

    baseline_identities = {}
    for name, path in BASELINES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline_identities[name] = {
            "manifest_sha256": _sha(path),
            "identity_sha256": payload["identity"]["identity_sha256"],
            "source_cache_identity_sha256": payload.get("source_cache_identity_sha256")
            or payload["identity"]["identity_sha256"],
        }
    return {
        "schema_version": "mcvr-v8-ablation-experiment-registry-v1",
        "status": "MCVR_V8_ABLATION_IMPLEMENTATION_READY",
        "branch": "mcvr-v8-ablation",
        "git_sha": _git("rev-parse", "HEAD"),
        "frozen_base_git_sha": FROZEN_BASE,
        "worktree_clean": not bool(status),
        "run_order": list(CONFIGS),
        "training_contract": {
            "seed": 43,
            "planned_original_total_steps": 200000,
            "user_requested_stop_step": 12500,
            "effective_batch": 64,
            "total_record_exposure": 800000,
            "step10000_validation": "FAST",
            "step12500_validation": "FULL10K",
            "parallel_same_gpu": False,
            "result_conditioned_tuning": False,
        },
        "configs": configs,
        "identities": {
            "datasets": {key: identities[key] for key in DATASETS},
            "d1_checkpoint_sha256": D1_SHA256,
            "sampler_file_sha256": identities["sampler"],
            "sampler_identity_sha256": "bac36073b62da27d4f771569c4f28246cc4ef9639204beb9b7f9bad7d8e765d6",
            "scales_file_sha256": identities["scales"],
            "scales_identity_sha256": "5a1c33947d9913ff7aeda64901e7812e37ac5aa647c6534ce4044a00e49d04e4",
            "baseline_cache_sha256": _canonical_sha(baseline_identities),
            "baseline_cache_identities": baseline_identities,
        },
        "isolation": ISOLATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    registry = build_registry(args.sampler_manifest.resolve(), require_clean=not args.allow_dirty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

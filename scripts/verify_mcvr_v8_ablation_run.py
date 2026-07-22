#!/usr/bin/env python
"""Fail-closed verification for one formal V8 single-factor ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_STATUS = {
    "NO_CONSTRAINT": "MCVR_V8_ABLATION_NO_CONSTRAINT_12P5K_COMPLETED",
    "NO_CONFIDENCE": "MCVR_V8_ABLATION_NO_CONFIDENCE_12P5K_COMPLETED",
    "NO_ERROR_STATE": "MCVR_V8_ABLATION_NO_ERROR_STATE_12P5K_COMPLETED",
    "NO_TYPE_NORMALIZATION": "MCVR_V8_ABLATION_NO_TYPE_NORMALIZATION_12P5K_COMPLETED",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _isolation(payload: dict, label: str) -> None:
    if int(payload.get("formal_test_records_read", -1)) != 0:
        raise RuntimeError(f"{label} formal test reads are not zero")
    if int(payload.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError(f"{label} frozen holdout reads are not zero")
    if bool(payload.get("formal_test_assets_opened", False)):
        raise RuntimeError(f"{label} opened formal-test assets")


def verify(output_dir: Path, ablation_id: str, exit_code: int) -> dict:
    output_dir = output_dir.resolve()
    if exit_code != 0:
        raise RuntimeError(f"{ablation_id} process exit code is {exit_code}")
    status_path = output_dir / "status.json"
    status = _json(status_path)
    _isolation(status, "status")
    if status.get("status") != EXPECTED_STATUS[ablation_id] or status.get("phase") != "COMPLETED":
        raise RuntimeError(f"{ablation_id} canonical completion status is missing")
    expected_numbers = {
        "actual_completed_step": 12500,
        "planned_original_total_steps": 200000,
        "effective_batch": 64,
        "total_record_exposure": 800000,
    }
    for key, value in expected_numbers.items():
        if int(status.get(key, -1)) != value:
            raise RuntimeError(f"{ablation_id} {key} changed")

    step = output_dir / "checkpoints/step012500.ckpt"
    last = output_dir / "checkpoints/last.ckpt"
    checkpoint_sha = _sha(step)
    if checkpoint_sha != _sha(last):
        raise RuntimeError(f"{ablation_id} last checkpoint differs from step012500")
    checkpoint = torch.load(step, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 12500:
        raise RuntimeError(f"{ablation_id} checkpoint step changed")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_states",
        "sampler_state",
    ):
        if key not in checkpoint:
            raise RuntimeError(f"{ablation_id} checkpoint is missing {key}")
    if int(checkpoint["sampler_state"].get("records_exposed", -1)) != 800000:
        raise RuntimeError(f"{ablation_id} sampler exposure changed")
    registration = checkpoint["resolved_config"]["ablation_registration"]
    if registration.get("ablation_id") != ablation_id:
        raise RuntimeError(f"{ablation_id} checkpoint registration changed")
    type_enabled = bool(checkpoint["resolved_config"]["type_normalization"]["enabled"])
    if type_enabled != (ablation_id != "NO_TYPE_NORMALIZATION"):
        raise RuntimeError(f"{ablation_id} type-normalization state changed")

    fast_path = output_dir / "validation_cache/step010000/fast/evaluation.json"
    full_path = output_dir / "validation_cache/step012500/full/evaluation.json"
    fast, full = _json(fast_path), _json(full_path)
    _isolation(fast, "FAST1000")
    _isolation(full, "FULL10K")
    if fast.get("mode") != "FAST" or int(fast.get("records", -1)) != 1000:
        raise RuntimeError(f"{ablation_id} step10000 is not FAST1000")
    if full.get("mode") != "FULL" or int(full.get("records", -1)) != 10000:
        raise RuntimeError(f"{ablation_id} step12500 is not FULL10K")
    completion = _json(output_dir / "graceful_stop_completion.json")
    if not bool(completion.get("normal_process_return")):
        raise RuntimeError(f"{ablation_id} lacks normal-process-return evidence")
    assets = _json(output_dir / "asset_hashes.json")
    if assets.get("git_branch") != "mcvr-v8-ablation":
        raise RuntimeError(f"{ablation_id} git branch provenance changed")
    return {
        "ablation_id": ablation_id,
        "status": "COMPLETED",
        "run_status": EXPECTED_STATUS[ablation_id],
        "actual_completed_step": 12500,
        "planned_original_total_steps": 200000,
        "effective_batch": 64,
        "total_record_exposure": 800000,
        "checkpoint_sha256": checkpoint_sha,
        "validation_sha256": _sha(full_path),
        "fast_validation_sha256": _sha(fast_path),
        "status_sha256": _sha(status_path),
        "git_head": assets["git_head"],
        "runtime_seconds": float(status["elapsed_seconds"]),
        "exit_code": exit_code,
        "metrics": full["metrics"],
        "set_metrics": full.get("set_metrics", {}),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation-id", choices=tuple(EXPECTED_STATUS), required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output_dir, args.ablation_id, args.exit_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Fail-closed verification for one completed MCVR V8 multi-seed run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_STATUS = "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required run artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_isolation(payload: dict, label: str) -> None:
    if int(payload.get("formal_test_records_read", -1)) != 0:
        raise RuntimeError(f"{label} formal_test_records_read is not zero")
    if int(payload.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError(f"{label} frozen_holdout_records_read is not zero")
    if bool(payload.get("formal_test_assets_opened", False)):
        raise RuntimeError(f"{label} opened formal-test assets")


def verify_run(output_dir: Path, *, seed: int, exit_code: int) -> dict:
    output_dir = output_dir.resolve()
    if int(exit_code) != 0:
        raise RuntimeError(f"Seed{seed} process return code is {exit_code}, not zero")
    status_path = output_dir / "status.json"
    status = _json(status_path)
    _assert_isolation(status, f"Seed{seed} status")
    if status.get("status") != EXPECTED_STATUS or status.get("phase") != "COMPLETED":
        raise RuntimeError(f"Seed{seed} does not have the canonical completed status")
    if int(status.get("actual_completed_step", -1)) != 12500:
        raise RuntimeError(f"Seed{seed} did not complete exactly step12500")
    if int(status.get("planned_original_total_steps", -1)) != 200000:
        raise RuntimeError(f"Seed{seed} lost original-200K schedule provenance")
    if int(status.get("effective_batch", -1)) != 64:
        raise RuntimeError(f"Seed{seed} effective batch changed")
    if int(status.get("total_record_exposure", -1)) != 800000:
        raise RuntimeError(f"Seed{seed} exposure changed")
    step_checkpoint = output_dir / "checkpoints" / "step012500.ckpt"
    last_checkpoint = output_dir / "checkpoints" / "last.ckpt"
    checkpoint_sha = _sha(step_checkpoint)
    if checkpoint_sha != _sha(last_checkpoint):
        raise RuntimeError(f"Seed{seed} step012500 and last checkpoints differ")
    checkpoint = torch.load(step_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 12500:
        raise RuntimeError(f"Seed{seed} checkpoint step changed")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_states",
        "sampler_state",
    ):
        if key not in checkpoint:
            raise RuntimeError(f"Seed{seed} checkpoint is missing {key}")
    sampler_state = checkpoint["sampler_state"]
    if int(sampler_state.get("records_exposed", -1)) != 800000:
        raise RuntimeError(f"Seed{seed} checkpoint sampler exposure changed")
    if int(sampler_state.get("next_record_exposure_offset", -1)) != 800000:
        raise RuntimeError(f"Seed{seed} checkpoint sampler continuation changed")
    fast_path = output_dir / "validation_cache" / "step010000" / "fast" / "evaluation.json"
    fast = _json(fast_path)
    _assert_isolation(fast, f"Seed{seed} FAST validation")
    if fast.get("mode") != "FAST" or int(fast.get("records", -1)) != 1000:
        raise RuntimeError(f"Seed{seed} step10000 validation is not FAST1000")
    validation_path = (
        output_dir / "validation_cache" / "step012500" / "full" / "evaluation.json"
    )
    validation = _json(validation_path)
    _assert_isolation(validation, f"Seed{seed} FULL10K validation")
    if validation.get("mode") != "FULL" or int(validation.get("records", -1)) != 10000:
        raise RuntimeError(f"Seed{seed} validation is not FULL10K")
    latest_validation = status.get("latest_validation", {})
    if latest_validation.get("mode") != "FULL":
        raise RuntimeError(f"Seed{seed} final status does not bind FULL validation")
    return {
        "seed": int(seed),
        "status": "COMPLETED",
        "run_status": EXPECTED_STATUS,
        "phase": "COMPLETED",
        "actual_completed_step": 12500,
        "planned_original_total_steps": 200000,
        "effective_batch": 64,
        "total_record_exposure": 800000,
        "checkpoint_sha256": checkpoint_sha,
        "validation_sha256": _sha(validation_path),
        "fast_validation_sha256": _sha(fast_path),
        "status_sha256": _sha(status_path),
        "runtime_seconds": float(status["elapsed_seconds"]),
        "exit_code": int(exit_code),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(12, 48), required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.output_dir, seed=args.seed, exit_code=args.exit_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

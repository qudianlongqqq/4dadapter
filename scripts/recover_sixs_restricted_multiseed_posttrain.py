#!/usr/bin/env python3
"""Complete a restricted multiseed run after a valid step-17500 checkpoint.

This is an orchestration-only recovery entry point.  It deliberately calls the
already frozen scientific functions in the executed restricted runner; it does
not call train() and does not alter model, data, evaluator, or xTB semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
REPORT_ROOT = ROOT / "reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
ARTIFACT_ROOT = ROOT / "artifacts/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
CUDA_PYTHON = Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def configure(seed: int) -> tuple[Path, Path, Path]:
    report = REPORT_ROOT / f"restricted_seed{seed}"
    artifact = ARTIFACT_ROOT / f"restricted_seed{seed}"
    config = report / "FROZEN_CONFIG.json"
    os.environ.update({
        "PYTHONUTF8": "1",
        "SIXS_FULL_JOINT_DEVICE": "cuda",
        "SIXS_MULTISEED_REPLICATION": "1",
        "SIXS_MULTISEED_CONFIG": str(config),
        "SIXS_MULTISEED_REPORT": str(report),
        "SIXS_MULTISEED_ARTIFACT": str(artifact),
        "SIXS_MULTISEED_RUN_NAME": f"sixs_final_multiseed/restricted_seed{seed}",
        "SIXS_MULTISEED_RUN_ROOT": str(REPORT_ROOT),
        "SIXS_MULTISEED_EXPERIMENT_ID": f"SIXS_FINAL_RESTRICTED_SEED{seed}_STEP17500",
    })
    return report, artifact, config


def audit_checkpoint(seed: int, report: Path, artifact: Path, config_path: Path, runner_path: Path) -> dict[str, Any]:
    final_path = report / "FINAL_CHECKPOINT.pt"
    recovery_path = artifact / "RECOVERY_CHECKPOINT.pt"
    log_path = report / "TRAIN_LOG.csv"
    summary_path = report / "TRAIN_SUMMARY.json"
    required = (final_path, recovery_path, log_path, summary_path, config_path, runner_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required recovery evidence missing: {missing}")

    final = torch.load(final_path, map_location="cpu", weights_only=False)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
    config_sha = sha(config_path)
    expected_experiment = f"SIXS_FINAL_RESTRICTED_SEED{seed}_STEP17500"
    if int(final.get("step", -1)) != 17500 or int(recovery.get("step", -1)) != 17500:
        raise RuntimeError("checkpoint is not the frozen step-17500 endpoint")
    recovery_seed = recovery.get("seed", recovery.get("SEED", -1))
    if int(final.get("seed", -1)) != seed or int(recovery_seed) != seed:
        raise RuntimeError("checkpoint seed identity mismatch")
    recovery_experiment = recovery.get("experiment_id", recovery.get("EXPERIMENT_ID"))
    if final.get("experiment_id") != expected_experiment or recovery_experiment != expected_experiment:
        raise RuntimeError("checkpoint experiment identity mismatch")
    if final.get("config_sha256") != config_sha or recovery.get("config_sha256") != config_sha:
        raise RuntimeError("checkpoint/config identity mismatch")

    final_state = final.get("model_state", {})
    recovery_state = recovery.get("model_state", {})
    if set(final_state) != set(recovery_state):
        raise RuntimeError("final/recovery model key mismatch")
    max_difference = 0.0
    for name in final_state:
        left, right = final_state[name], recovery_state[name]
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise RuntimeError(f"nonfinite model state: {name}")
        max_difference = max(max_difference, float((left - right).abs().max()))
    if max_difference != 0.0:
        raise RuntimeError(f"final/recovery model state mismatch: {max_difference}")

    required_recovery = (
        "optimizer_state", "scheduler_state", "python_rng_state", "numpy_rng_state",
        "torch_rng_state", "cuda_rng_state", "generator_state",
    )
    missing_recovery = [key for key in required_recovery if key not in recovery]
    if missing_recovery:
        raise RuntimeError(f"incomplete recovery state: {missing_recovery}")
    if int(recovery["scheduler_state"].get("last_epoch", -1)) != 17500:
        raise RuntimeError("scheduler endpoint mismatch")

    log = pd.read_csv(log_path)
    if len(log) == 0 or int(log.step.iloc[-1]) != 17500 or int(log.step.iloc[0]) != 1:
        raise RuntimeError("training log endpoint mismatch")
    if not bool(log.step.is_monotonic_increasing) or int(log.step.nunique()) != len(log):
        raise RuntimeError("training log is discontinuous")
    numeric = log.select_dtypes(include=[np.number]).to_numpy()
    if not bool(np.isfinite(numeric).all()):
        raise RuntimeError("training log contains nonfinite values")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or int(summary.get("steps", -1)) != 17500:
        raise RuntimeError("training summary is not a completed endpoint")
    final_sha = sha(final_path)
    if summary.get("checkpoint_sha256") != final_sha:
        raise RuntimeError("training summary checkpoint hash mismatch")

    implementation = json.loads((report / "IMPLEMENTATION_CONFIG.json").read_text(encoding="utf-8"))
    recorded_runner_sha = implementation.get("implementation", {}).get("runner_sha256")
    if recorded_runner_sha is None:
        recorded_runner_sha = implementation.get("runner_sha256")
    if recorded_runner_sha != sha(runner_path):
        raise RuntimeError("executed/current restricted runner identity mismatch")

    return {
        "RESTRICTED_SEED331_CHECKPOINT_VALID" if seed == 331 else "CHECKPOINT_VALID": "YES",
        "CHECKPOINT_STEP": 17500,
        "CHECKPOINT_SHA256": final_sha,
        "RECOVERY_CHECKPOINT_SHA256": sha(recovery_path),
        "TRAINING_SCIENTIFIC_VALIDITY": "VALID",
        "FINAL_RECOVERY_MODEL_MAX_ABS_DIFF": max_difference,
        "CONFIG_IDENTITY": "PASS",
        "CONFIG_SHA256": config_sha,
        "EXECUTED_RUNNER_PROVENANCE": "PASS",
        "EXECUTED_RUNNER": str(runner_path.relative_to(ROOT)),
        "EXECUTED_RUNNER_SHA256": sha(runner_path),
        "TRAIN_LOG_ROWS": len(log),
        "TRAINING_NONFINITE_COUNT": int(summary.get("nonfinite_count", 0)),
        "ADDITIONAL_TRAINING_STEPS": 0,
    }


def complete_reference_and_xtb(seed: int, report: Path) -> dict[str, Any]:
    import scripts.finalize_sixs_final_multiseed_replication as finalizer

    frozen_config, _, ids, items, sources, _ = finalizer.evidence.config_and_inputs()
    val = pd.read_parquet(frozen_config["data"]["val_manifest"])
    val_meta = {str(row.sample_id): row for row in val.itertuples(index=False)}
    reference = finalizer.reference_rmsd("restricted", seed, ids, items, sources)
    source_xtb = pd.read_csv(finalizer.SOURCE_XTB)
    source_energy = source_xtb.set_index("record_id").energy_hartree
    xtb = finalizer.run_xtb("restricted", seed, ids, sources, val_meta, frozen_config)
    xtb, xtb_summary = finalizer.summarize_xtb(xtb, source_energy)
    finalizer.atomic_frame(finalizer.xtb_path("restricted", seed), xtb)
    if reference.record_id.astype(str).tolist() != ids:
        raise RuntimeError("Reference RMSD identity/order mismatch")
    if xtb.record_id.astype(str).tolist() != ids:
        raise RuntimeError("xTB identity/order mismatch")
    if len(reference) != 5000 or len(xtb) != 5000:
        raise RuntimeError("posttrain denominator mismatch")
    return {
        "REFERENCE_RMSD_COMPLETE": "YES",
        "REFERENCE_RMSD_RECORDS": len(reference),
        "REFERENCE_RMSD_SHA256": sha(report / "REFERENCE_RMSD.csv"),
        "XTB_COMPLETE": "YES",
        "XTB_RECORDS_ATTEMPTED": len(xtb),
        "XTB_SUCCESS": int(xtb_summary["success"]),
        "XTB_FAILURES_RECORDED": int(xtb_summary["failures"]),
        "XTB_MEDIAN_DELTAE_KCAL_MOL": xtb_summary["median"],
        "XTB_SHA256": sha(report / "PROPOSAL_XTB.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-xtb", action="store_true")
    args = parser.parse_args()
    seed = int(args.seed)
    if seed not in (331, 353):
        raise RuntimeError("only frozen new replication seeds 331/353 are valid")
    report, artifact, config_path = configure(seed)
    audit_path = report / "POSTTRAIN_RECOVERY_AUDIT.json"
    runner_path = ROOT / "scripts/run_sixs_j1r1_full_joint_adaptive_ba_movement.py"
    wrapper_path = Path(__file__).resolve()
    started = time.time()
    state: dict[str, Any] = {
        "schema_version": "sixs-restricted-multiseed-posttrain-recovery-v1",
        "seed": seed,
        "RECOVERY_CLASS": "POSTTRAIN_ORCHESTRATION_FAILURE__SCIENTIFIC_CHECKPOINT_VALID",
        "RECOVERY_REASON": "first coordinate generation was blocked by a recovery-preflight intended for already frozen coordinates",
        "POSTTRAIN_PIPELINE_RECONSTRUCTED": "YES",
        "RECOVERY_REQUIRES_RETRAINING": "NO",
        "RECOVERY_REQUIRES_SCIENTIFIC_CHANGE": "NO",
        "SAFE_TO_COMPLETE_POSTTRAIN_EVALUATION": "PENDING_CHECKPOINT_AUDIT",
        "NO_MODEL_CHANGE": "YES",
        "NO_RETRAINING_OF_RESTRICTED_SEED331": "YES" if seed == 331 else "N/A",
        "NO_OUTCOME_BASED_SELECTION": "YES",
        "FORMAL_OUTCOME_READ": "NO",
        "LARGE_HOLDOUT_OUTCOME_READ": "NO",
        "recovery_wrapper": str(wrapper_path.relative_to(ROOT)),
        "recovery_wrapper_sha256": sha(wrapper_path),
    }
    atomic_json(audit_path, state)
    try:
        state.update(audit_checkpoint(seed, report, artifact, config_path, runner_path))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for frozen neural coordinate inference; fail closed")
        state.update({
            "SAFE_TO_COMPLETE_POSTTRAIN_EVALUATION": "YES",
            "MODEL_DEVICE": "cuda",
            "CUDA_DEVICE_NAME": torch.cuda.get_device_name(0),
            "PYTORCH_CUDA_VERSION": torch.version.cuda,
        })
        atomic_json(audit_path, state)

        import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as runner

        runner.preflight()
        coordinates_preexisting = runner.COORDINATES_READY.is_file()
        runner.evaluate_coordinates()
        recovery_preflight = runner.recovery_preflight(force=True)
        if recovery_preflight.get("RECOVERY_PREFLIGHT") != "PASS":
            raise RuntimeError("coordinate recovery preflight failed")
        runner.evaluate_and_finalize()
        final_status = json.loads((report / "FINAL_STATUS.json").read_text(encoding="utf-8"))
        if final_status.get("PIPELINE_STATUS") != "PASS" or final_status.get("CURRENT_STAGE") != "COMPLETE":
            raise RuntimeError("restricted frozen DEV finalization did not complete")
        state.update({
            "COORDINATE_GENERATION_COMPLETE": "YES",
            "COORDINATE_GENERATION_MODE": "REUSED" if coordinates_preexisting else "FIRST_GENERATION_FROM_FROZEN_CHECKPOINT",
            "DEV_MANIFEST_IDENTITY": recovery_preflight.get("DEV_RECORD_ALIGNMENT"),
            "EVALUATOR_IDENTITY": "PASS",
            "DEV_EVALUATION_COMPLETE": "YES",
            "DEV_RECORDS": int(recovery_preflight.get("records", 0)),
            "DEV_MOLECULES": int(recovery_preflight.get("molecules", 0)),
            "RESTRICTED_SEED331_RUN_STATUS" if seed == 331 else "RUN_STATUS": "COMPLETE_POSTTRAIN_RECOVERY",
        })
        if not args.skip_xtb:
            state.update(complete_reference_and_xtb(seed, report))
        else:
            state["XTB_COMPLETE"] = "DEFERRED"
        state.update({
            "status": "PASS",
            "elapsed_seconds": time.time() - started,
            "completed_at_epoch": time.time(),
        })
        atomic_json(audit_path, state)
        return 0
    except Exception as error:
        state.update({
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        atomic_json(audit_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

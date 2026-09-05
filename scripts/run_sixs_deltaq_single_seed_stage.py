#!/usr/bin/env python3
"""Restart-safe stage utilities for the development-only DeltaQ pilot.

This module never opens a protected final cohort.  Neural training remains in
the frozen tensor-only prototype runner; this file gates it, materializes the
fixed endpoint, and evaluates already declared DEV/diagnostic cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
REPORT = ROOT / "reports/ecir_mvr/sixs_deltaq_single_seed_pilot"
RUNTIME = REPORT / "runtime"
ARTIFACT = ROOT / "artifacts/ecir_mvr/sixs_deltaq_single_seed_pilot"
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_deltaq_single_seed_pilot")
CONFIG = ROOT / "configs/sixs_deltaq_single_seed_pilot.json"
PROTOTYPE_CONFIG = ROOT / "configs/sixs_deltaq_prototype.json"
OVERFIT_STATUS = ROOT / "reports/ecir_mvr/sixs_deltaq_prototype/runtime/OVERFIT_STATUS.json"
OVERFIT_CHECKPOINT = ROOT / "artifacts/ecir_mvr/sixs_deltaq_prototype/SMALL_OVERFIT_CHECKPOINT.pt"
OVERFIT_RESULT = ROOT / "reports/ecir_mvr/sixs_deltaq_prototype/07_SMALL_OVERFIT_RESULTS.csv"
PROTOTYPE_FINAL = ROOT / "artifacts/ecir_mvr/sixs_deltaq_prototype/STEP17500_CHECKPOINT.pt"
PROTOTYPE_LOG = ROOT / "reports/ecir_mvr/sixs_deltaq_prototype/DELTAQ_TRAIN_LOG.csv"
PILOT_FINAL = ARTIFACT / "DELTAQ_SEED307_FULL.pt"
V1_CHECKPOINT = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/FINAL_CHECKPOINT.pt"
EXTERNAL_PYTHON = Path(r"E:\miniconda\envs\external-validity\python.exe")
EXTERNAL_WORKER = ROOT / "scripts/evaluate_sixs_primary_final_external.py"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def marker(stage: str, outputs: list[Path], **extra: Any) -> None:
    identity = sha256(CONFIG)
    missing = [str(p) for p in outputs if not p.is_file()]
    if missing:
        raise RuntimeError(f"stage {stage} missing required outputs: {missing}")
    atomic_json(
        RUNTIME / f"{stage}_COMPLETED.json",
        {
            "status": "COMPLETED",
            "stage": stage,
            "pipeline_identity_sha256": identity,
            "outputs": {str(p): sha256(p) for p in outputs},
            "formal_test_read": False,
            **extra,
        },
    )


def gate() -> None:
    # The supervisor has already performed one blocking OS wait.  This is the
    # sole terminal status read for the gate.
    status = json.loads(OVERFIT_STATUS.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED":
        raise RuntimeError(f"OVERFIT_NOT_COMPLETED: {status.get('status')} / {status.get('stage')}")
    if not OVERFIT_CHECKPOINT.is_file() or not OVERFIT_RESULT.is_file():
        raise RuntimeError("OVERFIT_REQUIRED_ARTIFACT_MISSING")
    state = torch.load(OVERFIT_CHECKPOINT, map_location="cpu", weights_only=False)
    scalars = [state.get(k) for k in ("initial_bond_mae", "initial_angle_mae", "final_bond_mae", "final_angle_mae")]
    if state.get("status") != "PASS" or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in scalars):
        raise RuntimeError("SMALL_OVERFIT_STATUS_NOT_PASS_OR_NONFINITE")
    ib, ia, fb, fa = map(float, scalars)
    if not (fb <= 0.5 * ib and fa <= 0.5 * ia):
        raise RuntimeError("PREDECLARED_SMALL_OVERFIT_THRESHOLD_FAILED")
    gate_path = REPORT / "01_OVERFIT_GATE_STATUS.json"
    atomic_json(gate_path, {
        "status": "PASS", "small_overfit_status": "PASS", "ready_for_dev_training": True,
        "initial_bond_mae": ib, "final_bond_mae": fb,
        "initial_angle_mae": ia, "final_angle_mae": fa,
        "checkpoint_sha256": sha256(OVERFIT_CHECKPOINT), "result_sha256": sha256(OVERFIT_RESULT),
        "pass_rule_frozen_before_result": True, "formal_test_read": False,
    })
    marker("00_OVERFIT_GATE", [gate_path])


def finalize_train() -> None:
    if not PROTOTYPE_FINAL.is_file() or not PROTOTYPE_LOG.is_file():
        raise RuntimeError("STEP17500_TRAINING_ARTIFACT_MISSING")
    state = torch.load(PROTOTYPE_FINAL, map_location="cpu", weights_only=False)
    if int(state.get("step", -1)) != 17500 or state.get("config_sha256") != sha256(PROTOTYPE_CONFIG):
        raise RuntimeError("STEP17500_CHECKPOINT_IDENTITY_MISMATCH")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOTYPE_FINAL, PILOT_FINAL)
    shutil.copy2(PROTOTYPE_LOG, REPORT / "02_FULL_TRAIN_CURVE.csv")
    curve = pd.read_csv(PROTOTYPE_LOG)
    status_path = REPORT / "01_FULL_TRAIN_STATUS.json"
    atomic_json(status_path, {
        "status": "COMPLETED", "seed": 307, "step": 17500,
        "checkpoint_path": str(PILOT_FINAL), "checkpoint_sha256": sha256(PILOT_FINAL),
        "optimizer": "AdamW", "batch_molecules": 64,
        "backbone_learning_rate": 0.00015, "head_learning_rate": 0.0003,
        "weight_decay": 0.000001, "scheduler": "CosineAnnealingLR(T_max=22500)",
        "checkpoint_rule": "STEP17500_ONLY_NO_DEV_SELECTION",
        "finite_curve": bool(np.isfinite(curve.select_dtypes(include=[np.number]).to_numpy()).all()),
        "formal_test_read": False,
    })
    # Training calibration is kept distinct from later DEV calibration.
    keep = [c for c in ("step", "bond_deltaq_mae", "angle_deltaq_mae", "tau_mean", "w_B_mean") if c in curve]
    atomic_csv(REPORT / "03_DELTAQ_TRAIN_CALIBRATION.csv", curve[keep].copy())
    marker("01_FULL_TRAIN", [status_path, REPORT / "02_FULL_TRAIN_CURVE.csv", REPORT / "03_DELTAQ_TRAIN_CALIBRATION.csv", PILOT_FINAL])


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _deltaq_cross(upstream: str) -> None:
    """Run the existing audited 10K infrastructure with only the model swapped.

    Existing AvgFlow/DiTMC cohorts are explicitly posthoc diagnostics.  The
    internal method token is retained solely because the audited external and
    correspondence evaluators key on it; published output aliases it to
    SIXS-v2-DeltaQ and binds the DeltaQ checkpoint hash.
    """
    if not PILOT_FINAL.is_file():
        raise RuntimeError("DELTAQ_FULL_CHECKPOINT_MISSING")
    base = _load_module("deltaq_cross_base", ROOT / "scripts/run_sixs_final_cross_upstream_unrestricted.py")
    from etflow.ecir.source_conditioned_deltaq import DeltaQUnrestrictedFullJointModel

    seed = 307
    checkpoint_hash = sha256(PILOT_FINAL)
    base.SEEDS = (seed,)
    base.REPORT = REPORT / "cross_internal"
    base.ASSET = ASSET
    base.STATUS = RUNTIME / f"{upstream.upper()}_INTERNAL_STATUS.json"
    base.PREFLIGHT = RUNTIME / "CROSS_PREFLIGHT.json"
    base.CHECKPOINT_MANIFEST = RUNTIME / "DELTAQ_CHECKPOINT_MANIFEST.json"
    base.CHECKPOINTS = {seed: PILOT_FINAL}
    base.EXPECTED_SHA = {seed: checkpoint_hash}

    def methods(name: str) -> tuple[str, ...]:
        prefix = name.upper()
        return (f"{prefix}_RAW", f"{prefix}_SIXS_U_SEED307")

    def checkpoint_identity(_seed: int) -> dict[str, Any]:
        payload = torch.load(PILOT_FINAL, map_location="cpu", weights_only=False)
        probe = DeltaQUnrestrictedFullJointModel(128, 3)
        probe.load_state_dict(payload["model_state"], strict=True)
        return {"seed": 307, "path": str(PILOT_FINAL), "sha256": checkpoint_hash,
                "step": 17500, "formulation": "SIXS-v2-DeltaQ", "identity": "PASS"}

    def load_models(device: torch.device):
        payload = torch.load(PILOT_FINAL, map_location="cpu", weights_only=False)
        model = DeltaQUnrestrictedFullJointModel(128, 3)
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device).eval()
        return {307: model}

    base.methods = methods
    base.checkpoint_identity = checkpoint_identity
    base.load_models = load_models
    # The validated generator has one denominator assertion specialized to its
    # original three seeds.  Recompile only that function for one seed.
    source = inspect.getsource(base.generate).replace("len(frame) != 30000", "len(frame) != 10000")
    source = source.replace("model.belief(collated, detach_sigma_features=False)", "model.belief(collated, source, detach_sigma_features=False)")
    exec(source, base.__dict__)
    base.preflight()
    base.generate(upstream)

    # Evaluate endpoints in the isolated external-validity environment.
    subprocess.run([
        str(EXTERNAL_PYTHON), str(ROOT / "scripts/evaluate_sixs_deltaq_pilot_cross_external.py"),
        "--upstream", upstream,
    ], cwd=ROOT, check=True)

    # Reuse audited global-reference correspondence without changing cohort.
    ref = _load_module("deltaq_cross_reference", ROOT / "scripts/evaluate_sixs_final_cross_upstream_reference_rmsd.py")
    ref.ASSET = ASSET
    ref.REPORT = base.REPORT
    ref.SEEDS = (307,)
    ref.EXPECTED_CHECKPOINTS = {"307": checkpoint_hash}
    ref.methods = methods
    ref.evaluate(upstream)

    pb = pd.read_parquet(ASSET / upstream / "POSEBUSTERS.parquet")
    v3d = pd.read_parquet(ASSET / upstream / "VALIDITY3D.parquet")
    fidelity = pd.read_parquet(ASSET / upstream / "FIDELITY_PER_RECORD.parquet")
    diagnostics = pd.read_parquet(ASSET / upstream / "COORDINATE_DIAGNOSTICS.parquet")
    method_col_pb = "method" if "method" in pb else "arm"
    method_col_v3d = "method" if "method" in v3d else "arm"
    rows = []
    for internal, display in ((methods(upstream)[0], f"{upstream.upper()} Raw"), (methods(upstream)[1], "SIXS-v2-DeltaQ seed307")):
        fp = fidelity[fidelity.method == internal]
        row = {
            "upstream": upstream, "method": display, "cohort_role": "POSTHOC_DIAGNOSTIC_ONLY",
            "records": int((v3d[method_col_v3d] == internal).sum()),
            "V3D": float(v3d.loc[v3d[method_col_v3d] == internal, "validity3d"].mean()),
            "PB": float(pb.loc[pb[method_col_pb] == internal, "PB"].mean()),
            "reference_rmsd": float(fp.reference_rmsd_angstrom.mean()),
            "source_rmsd": 0.0 if internal.endswith("_RAW") else float(diagnostics.source_rmsd_raw.mean()),
        }
        if not internal.endswith("_RAW"):
            tau = diagnostics.tau.astype(float)
            movement = diagnostics.source_rmsd_raw.astype(float)
            atom = diagnostics.max_atom_displacement.astype(float)
            row.update({
                "finite_coordinate_fraction": float(diagnostics.finite.mean()),
                "tau_median": float(tau.median()), "tau_p90": float(tau.quantile(.9)),
                "tau_p95": float(tau.quantile(.95)), "tau_p99": float(tau.quantile(.99)),
                "tau_p99_5": float(tau.quantile(.995)), "tau_p99_9": float(tau.quantile(.999)),
                "tau_max": float(tau.max()), "count_tau_gt_1A": int((tau > 1).sum()),
                "movement_rms_median": float(movement.median()), "movement_rms_p95": float(movement.quantile(.95)),
                "movement_rms_p99": float(movement.quantile(.99)), "movement_rms_max": float(movement.max()),
                "max_atom_displacement_median": float(atom.median()), "max_atom_displacement_p95": float(atom.quantile(.95)),
                "max_atom_displacement_p99": float(atom.quantile(.99)), "max_atom_displacement_max": float(atom.max()),
            })
        rows.append(row)
    output = REPORT / ("06_AVGFLOW_DEV_RESULTS.csv" if upstream == "avgflow" else "09_DITMC_DEV_RESULTS.csv")
    atomic_csv(output, pd.DataFrame(rows))
    if upstream == "avgflow":
        atomic_csv(REPORT / "07_AVGFLOW_OVERSHOOT_RESULTS.csv", pd.DataFrame([{
            "method": "SIXS-v2-DeltaQ seed307", "cohort_role": "POSTHOC_DIAGNOSTIC_ONLY",
            "bond_overshoot": "NOT_AVAILABLE_WITHOUT_NEW_PRIMITIVE_CORRESPONDENCE_PASS",
            "angle_overshoot": "NOT_AVAILABLE_WITHOUT_NEW_PRIMITIVE_CORRESPONDENCE_PASS",
            "no_metric_fabrication": True,
        }]))
        atomic_csv(REPORT / "08_AVGFLOW_HEADROOM_RESULTS.csv", pd.DataFrame([{
            "status": "NOT_AVAILABLE_WITHOUT_NEW_UNUSED_AVGFLOW_SOURCE_ASSET",
            "avg_new_dev_provenance": "FAIL", "fallback_role": "DIAGNOSTIC_REPLICATION_ONLY",
        }]))
        outputs = [output, REPORT / "07_AVGFLOW_OVERSHOOT_RESULTS.csv", REPORT / "08_AVGFLOW_HEADROOM_RESULTS.csv"]
        marker("03_AVGFLOW_EVAL", outputs, avg_new_dev_provenance="FAIL", cohort_role="DIAGNOSTIC_REPLICATION_ONLY")
    else:
        delta = pd.DataFrame([rows[-1]])
        atomic_csv(REPORT / "10_DITMC_TAIL_COMPARISON.csv", delta[[c for c in delta if "tau" in c or "movement" in c or "displacement" in c or c in ("method", "finite_coordinate_fraction")]])
        marker("04_DITMC_EVAL", [output, REPORT / "10_DITMC_TAIL_COMPARISON.csv"], cohort_role="POSTHOC_DIAGNOSTIC_ONLY")


def etflow() -> None:
    # The exact ETFlow DEV coordinate evaluator is deliberately separate from
    # the protected prospective-final evaluator.  Refuse any accidental use of
    # the protected manifest before delegating to the development adapter.
    adapter = ROOT / "scripts/evaluate_sixs_deltaq_etflow_dev.py"
    if not adapter.is_file():
        raise RuntimeError("ETFLOW_DEV_ADAPTER_MISSING_FAIL_CLOSED")
    subprocess.run([r"E:\miniconda\envs\etflow-5080-v2\python.exe", str(adapter)], cwd=ROOT, check=True)
    marker("02_ETFLOW_EVAL", [REPORT / "04_ETFLOW_DEV_RESULTS.csv", REPORT / "05_ETFLOW_HEADROOM_RESULTS.csv"])


def xtb() -> None:
    # xTB is optional for pilot success.  Its omission is explicit rather than
    # silently presenting missing values as failures or zeros.
    path = REPORT / "11_XTB_SINGLE_SEED_RESULTS.csv"
    atomic_csv(path, pd.DataFrame([
        {"upstream": name, "status": "NOT_RUN_OPTIONAL", "reason": "NO_FROZEN_DELTAQ_XTB_ADAPTER_AT_PIPELINE_FREEZE",
         "geometry_optimization": False, "formal_test_read": False}
        for name in ("ETFlow", "AvgFlow", "DiTMC")
    ]))
    marker("05_XTB_EVAL", [path], optional=True)


def aggregate() -> None:
    inputs = [REPORT / "04_ETFLOW_DEV_RESULTS.csv", REPORT / "06_AVGFLOW_DEV_RESULTS.csv", REPORT / "09_DITMC_DEV_RESULTS.csv"]
    frames = [pd.read_csv(p) for p in inputs]
    common = sorted(set().union(*(set(frame.columns) for frame in frames)))
    table = pd.concat([frame.reindex(columns=common) for frame in frames], ignore_index=True)
    atomic_csv(REPORT / "12_THREE_UPSTREAM_PILOT.csv", table)
    conclusion = REPORT / "13_SINGLE_SEED_CONCLUSION.md"
    conclusion.write_text(
        "# SIXS-v2-DeltaQ single-seed pilot\n\n"
        "The numerical classification must be produced only from the completed stage tables. "
        "This is one development seed and is not statistical confirmation. Existing AvgFlow/DiTMC "
        "cohorts are posthoc diagnostics and were not used for tuning.\n",
        encoding="utf-8",
    )
    guard = REPORT / "14_CLAIM_GUARD.md"
    guard.write_text(
        "# Claim guard\n\nNo formal-test result was read. Do not call this pilot prospective, confirmed, "
        "multiseed, universally stable, or an unconditional energy improvement.\n",
        encoding="utf-8",
    )
    final = REPORT / "FINAL_STATUS.json"
    atomic_json(final, {
        "status": "COMPLETED", "seed": 307, "formal_test_read": False,
        "avg_old_final_used_for_tuning": False, "single_seed_only": True,
        "ready_for_matched_multiseed": "REQUIRES_REVIEW_OF_PREDECLARED_THREE_UPSTREAM_CRITERIA",
    })
    marker("06_AGGREGATION", [REPORT / "12_THREE_UPSTREAM_PILOT.csv", conclusion, guard, final])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gate", "finalize-train", "etflow", "avgflow", "ditmc", "xtb", "aggregate"))
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    {
        "gate": gate,
        "finalize-train": finalize_train,
        "etflow": etflow,
        "avgflow": lambda: _deltaq_cross("avgflow"),
        "ditmc": lambda: _deltaq_cross("ditmc"),
        "xtb": xtb,
        "aggregate": aggregate,
    }[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Restart-safe local supervisor for SIXS final evidence closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED = 5000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update(args: argparse.Namespace, stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(args.status.read_text(encoding="utf-8")) if args.status.is_file() else {}
    current.update({
        "schema_version": "sixs-final-evidence-closure-status-v1", "status": state,
        "current_stage": stage, "supervisor_pid": os.getpid(), "one_heavy_stage_at_a_time": True,
        "repeated_polling": False, "model_training": False, "sixs_reinference": False,
        "cohort_membership_changed": False, **extra,
    })
    atomic_json(args.status, current)


def run(args: argparse.Namespace, stage: str, command: list[str]) -> None:
    update(args, stage)
    print("CLOSURE_EXEC " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=args.repo)
    if completed.returncode:
        raise RuntimeError(f"stage {stage} exited {completed.returncode}")


def complete_parquet(path: Path) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_parquet(path, columns=["record_id"])
    return len(frame) == EXPECTED and frame.record_id.astype(str).nunique() == EXPECTED


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    protocol_hash = sha(args.protocol)
    primary_status = json.loads((args.repo / "reports/ecir_mvr/sixs_primary_final_evaluation/FINAL_STATUS.json").read_text(encoding="utf-8"))
    cross_status = json.loads((args.repo / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/FINAL_STATUS.json").read_text(encoding="utf-8"))
    ablation_status = json.loads((args.repo / "reports/ecir_mvr/sixs_final_matched_ablation/STATUS.json").read_text(encoding="utf-8"))
    methods = ["source"] + [f"{form}_seed{seed}" for form in ("restricted", "unrestricted") for seed in (307,331,353)]
    complete_methods = {
        method: complete_parquet(args.primary_asset / f"methods/{method}/PER_RECORD.parquet")
        for method in methods
    }
    cross_files = [
        args.cross_asset / upstream / name
        for upstream in ("avgflow", "ditmc")
        for name in ("COORDINATE_DIAGNOSTICS.parquet", "FIDELITY_PER_RECORD.parquet", "VALIDITY3D.parquet", "POSEBUSTERS.parquet")
    ]
    bindings = {
        "primary_protocol": {"path": str(args.protocol), "sha256": protocol_hash},
        "primary_manifest": {"path": str(args.primary_manifest), "sha256": sha(args.primary_manifest)},
        "source_per_record": {"path": str(args.primary_asset / "methods/source/PER_RECORD.parquet"),
            "sha256": sha(args.primary_asset / "methods/source/PER_RECORD.parquet")},
        "primary_final_status": {"path": str(args.repo / "reports/ecir_mvr/sixs_primary_final_evaluation/FINAL_STATUS.json"),
            "sha256": sha(args.repo / "reports/ecir_mvr/sixs_primary_final_evaluation/FINAL_STATUS.json")},
        "cross_final_status": {"path": str(args.repo / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/FINAL_STATUS.json"),
            "sha256": sha(args.repo / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/FINAL_STATUS.json")},
        "ablation_status": {"path": str(args.repo / "reports/ecir_mvr/sixs_final_matched_ablation/STATUS.json"),
            "sha256": sha(args.repo / "reports/ecir_mvr/sixs_final_matched_ablation/STATUS.json")},
    }
    expected_manifest_hash = json.loads(args.protocol.read_text(encoding="utf-8"))["bindings"]["primary_manifest_sha256"]
    okay = (
        expected_manifest_hash == bindings["primary_manifest"]["sha256"]
        and all(complete_methods.values()) and all(path.is_file() for path in cross_files)
        and "PASS" in str(primary_status.get("status", ""))
        and "PASS" in str(cross_status.get("status", ""))
        and "PASS" in str(ablation_status.get("status", ""))
    )
    audit = {
        "schema_version": "sixs-final-evidence-closure-preflight-v1",
        "status": "PASS" if okay else "FAIL", "source_final_5000_complete": complete_methods["source"],
        "unrestricted_307_331_353_complete": all(complete_methods[f"unrestricted_seed{s}"] for s in (307,331,353)),
        "restricted_307_331_353_complete": all(complete_methods[f"restricted_seed{s}"] for s in (307,331,353)),
        "avgflow_outputs_complete": all(path.is_file() for path in cross_files if "avgflow" in str(path)),
        "ditmc_outputs_complete": all(path.is_file() for path in cross_files if "ditmc" in str(path)),
        "final_ablation_complete": "PASS" in str(ablation_status.get("status", "")),
        "preexisting_scientific_results_status": "PASS" if okay else "FAIL",
        "scientific_recomputation_required": "MMFF_ONLY", "bindings": bindings,
        "no_model_training": True, "no_sixs_reinference": True,
    }
    atomic_json(args.report_dir / "01_PREEXISTING_EVIDENCE_AUDIT.json", audit)
    atomic_text(args.report_dir / "00_PROTOCOL_FREEZE.md", f"""# SIXS final evidence closure protocol freeze

This closure reuses all frozen Source, Restricted, Unrestricted, AvgFlow, DiTMC, and matched-ablation outputs. The only rerun scientific baseline is current-final MMFF94s. Reference work is evaluation/aggregation of already-frozen coordinates and references. No model, checkpoint, cohort, evaluator semantics, metric, seed, or movement policy may change.

```text
COHORT = 2500 molecules x 2 records = 5000
PRIMARY_PROTOCOL_SHA256 = {protocol_hash}
PRIMARY_MANIFEST_SHA256 = {bindings['primary_manifest']['sha256']}
MMFF_METHOD = MMFF94s
MMFF_NUM_ATOMS_REPAIR = authoritative frozen RDKit topology molecule atom count
MMFF_FAILURE_POLICY = explicit failure; Source fallback only for fixed-denominator structural evaluation; never counted as MMFF success
PAIRED_CLUSTER = molecule
BOOTSTRAP_RESAMPLES = 10000
REFERENCE_RMSD = all-atom fixed-order float64 proper-rotation Kabsch; nearest frozen ensemble; no symmetry permutation
BOND_ANGLE_REFERENCE = first frozen Reference conformer
REFERENCE_CONTEXT = first frozen Reference conformer, contextual-only, duplicated solely for matched record identity
COV_AMR_APPLICABLE = NO (no frozen primary threshold/protocol and no independent Reference self split)
REFERENCE_SELF_COV_AMR = NOT_REPORTED
GFN2_XTB_GEOMETRY_OPT = DO_NOT_RUN
NO_MODEL_TRAINING = YES
NO_SIXS_REINFERENCE = YES
ONE_HEAVY_STAGE_AT_A_TIME = YES
```
""")
    if not okay:
        raise RuntimeError("closure preflight failed")
    update(args, "PREFLIGHT", "PASS", preexisting_scientific_results_status="PASS", scientific_recomputation_required="MMFF_ONLY")
    return audit


def smoke(args: argparse.Namespace) -> None:
    run(args, "MMFF_ENGINEERING_SMOKE", [
        str(args.cuda_python), str(args.repo / "scripts/run_sixs_final_mmff94s_repair.py"),
        "--mode", "smoke", "--protocol", str(args.protocol), "--topology-cache", str(args.topology_cache),
        "--output-dir", str(args.asset_dir / "methods/mmff94s"), "--report-dir", str(args.report_dir),
        "--smoke-molecules", "3",
    ])
    update(args, "MMFF_ENGINEERING_SMOKE", "PASS", smoke_molecules=3, scientific_outcome_used_for_protocol_selection=False)


def full(args: argparse.Namespace) -> None:
    if args.resume_from_reference_xtb:
        existing = json.loads((args.report_dir / "01_PREEXISTING_EVIDENCE_AUDIT.json").read_text(encoding="utf-8"))
        if existing.get("status") != "PASS":
            raise RuntimeError("cannot resume: frozen preexisting-evidence audit is not PASS")
        update(args, "RESUME_FROM_REFERENCE_CONTEXT_GFN2_XTB_SINGLE_POINT", "PASS",
            completed_stages_reused=["MMFF94S_FULL", "MMFF94S_FROZEN_VALIDITY", "MMFF94S_GFN2_XTB_SINGLE_POINT",
                "REFERENCE_CONTEXT_MATERIALIZATION", "REFERENCE_CONTEXT_FROZEN_VALIDITY"])
    else:
        preflight(args)
    repair = json.loads((args.report_dir / "02_MMFF_REPAIR_AUDIT.json").read_text(encoding="utf-8"))
    if repair.get("status") != "PASS":
        raise RuntimeError("MMFF smoke is not PASS")
    mmff_dir = args.asset_dir / "methods/mmff94s"
    if not complete_parquet(mmff_dir / "PER_RECORD.parquet"):
        run(args, "MMFF94S_FULL", [str(args.cuda_python), str(args.repo / "scripts/run_sixs_final_mmff94s_repair.py"),
            "--mode", "full", "--protocol", str(args.protocol), "--topology-cache", str(args.topology_cache),
            "--output-dir", str(mmff_dir), "--report-dir", str(args.report_dir)])
    if not complete_parquet(mmff_dir / "POSEBUSTERS.parquet") or not complete_parquet(mmff_dir / "VALIDITY3D.parquet"):
        run(args, "MMFF94S_FROZEN_VALIDITY", [str(args.external_python), str(args.repo / "scripts/evaluate_sixs_primary_final_external.py"),
            "--arm", "mmff94s", "--sdf", str(mmff_dir / "COORDINATES.sdf"),
            "--records", str(mmff_dir / "PER_RECORD.parquet"), "--pb", str(mmff_dir / "POSEBUSTERS.parquet"),
            "--v3d", str(mmff_dir / "VALIDITY3D.parquet")])
    mmff_xtb = args.asset_dir / "xtb/MMFF94S_XTB_STATUS.json"
    if not mmff_xtb.is_file() or json.loads(mmff_xtb.read_text(encoding="utf-8")).get("status") != "PASS":
        run(args, "MMFF94S_GFN2_XTB_SINGLE_POINT", [str(args.cuda_python), str(args.repo / "scripts/run_sixs_final_closure_xtb.py"),
            "--method", "mmff94s", "--protocol", str(args.protocol), "--sdf", str(mmff_dir / "COORDINATES.sdf"),
            "--records", str(mmff_dir / "PER_RECORD.parquet"), "--source-xtb", str(args.source_xtb),
            "--output-dir", str(args.asset_dir / "xtb"), "--cache-dir", str(args.asset_dir / "xtb_cache")])
    ref_dir = args.asset_dir / "methods/reference_context"
    if not complete_parquet(ref_dir / "PER_RECORD.parquet"):
        run(args, "REFERENCE_CONTEXT_MATERIALIZATION", [str(args.cuda_python), str(args.repo / "scripts/materialize_sixs_final_reference_context.py"),
            "--protocol", str(args.protocol), "--topology-cache", str(args.topology_cache), "--output-dir", str(ref_dir)])
    if not complete_parquet(ref_dir / "POSEBUSTERS.parquet") or not complete_parquet(ref_dir / "VALIDITY3D.parquet"):
        run(args, "REFERENCE_CONTEXT_FROZEN_VALIDITY", [str(args.external_python), str(args.repo / "scripts/evaluate_sixs_primary_final_external.py"),
            "--arm", "reference_context", "--sdf", str(ref_dir / "COORDINATES.sdf"),
            "--records", str(ref_dir / "PER_RECORD.parquet"), "--pb", str(ref_dir / "POSEBUSTERS.parquet"),
            "--v3d", str(ref_dir / "VALIDITY3D.parquet")])
    ref_xtb = args.asset_dir / "xtb/REFERENCE_CONTEXT_XTB_STATUS.json"
    if not ref_xtb.is_file() or json.loads(ref_xtb.read_text(encoding="utf-8")).get("status") != "PASS":
        run(args, "REFERENCE_CONTEXT_GFN2_XTB_SINGLE_POINT", [str(args.cuda_python), str(args.repo / "scripts/run_sixs_final_closure_xtb.py"),
            "--method", "reference_context", "--protocol", str(args.protocol), "--sdf", str(ref_dir / "COORDINATES.sdf"),
            "--records", str(ref_dir / "PER_RECORD.parquet"), "--source-xtb", str(args.source_xtb),
            "--output-dir", str(args.asset_dir / "xtb"), "--cache-dir", str(args.asset_dir / "xtb_cache")])
    avg_complete = args.asset_dir / "avgflow_reference/COMPLETE.json"
    if not avg_complete.is_file() or json.loads(avg_complete.read_text(encoding="utf-8")).get("status") != "PASS":
        run(args, "CURRENT_FINAL_AVGFLOW_REFERENCE_AUDIT", [
            str(args.cuda_python), str(args.repo / "scripts/run_sixs_final_avgflow_reference_audit.py"),
            "--repo", str(args.repo), "--cross-asset", str(args.cross_asset),
            "--asset-dir", str(args.asset_dir), "--report-dir", str(args.report_dir),
        ])
    run(args, "FINAL_AGGREGATION", [str(args.cuda_python), str(args.repo / "scripts/finalize_sixs_final_evidence_closure.py"),
        "--repo", str(args.repo), "--protocol", str(args.protocol), "--asset-dir", str(args.asset_dir),
        "--report-dir", str(args.report_dir)])
    update(args, "COMPLETE", "PASS", protected_final_performance_read=True,
        gfn2_xtb_geometry_optimization="NOT_RUN", core_scientific_experiments_complete=True)


def main(args: argparse.Namespace) -> int:
    try:
        if args.preflight_only:
            preflight(args)
        elif args.smoke_only:
            preflight(args); smoke(args)
        else:
            full(args)
        return 0
    except BaseException as exc:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        trace = args.report_dir / "SUPERVISOR_TRACEBACK.txt"
        atomic_text(trace, traceback.format_exc())
        update(args, "ENGINEERING_FAILURE", "FAIL", exit_code=1, exception_type=type(exc).__name__,
            exception=str(exc), log_path=str(trace))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--primary-asset", type=Path, required=True)
    parser.add_argument("--cross-asset", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, required=True)
    parser.add_argument("--source-xtb", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cuda-python", type=Path, required=True)
    parser.add_argument("--external-python", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume-from-reference-xtb", action="store_true")
    args = parser.parse_args()
    for name in ("repo", "protocol", "primary_manifest", "primary_asset", "cross_asset", "topology_cache",
                 "source_xtb", "asset_dir", "report_dir", "cuda_python", "external_python"):
        setattr(args, name, getattr(args, name).resolve())
    args.status = args.report_dir / "STATUS.json"
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))

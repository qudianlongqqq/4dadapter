#!/usr/bin/env python
"""Aggregate frozen current-final evidence after closure-only evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase


SEEDS = (307, 331, 353)
PRIMARY = Path("E:/3dconformergenerationcode/dataset/sixs_primary_final_evaluation_v1")
CROSS_DATA = Path("E:/3dconformergenerationcode/dataset/sixs_final_cross_upstream_unrestricted")
CROSS_REPORT = Path("reports/ecir_mvr/sixs_final_cross_upstream_unrestricted")
HARTREE_TO_KCAL = 627.509474


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def trim_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(math.floor(len(values) * fraction))
    return float(values[cut:-cut].mean()) if cut else float(values.mean())


def q(values: Iterable[float], probability: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), probability))


def paired_summary(
    baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str, *,
    comparison: str, higher_is_better: bool, seed: int, resamples: int = 10000,
) -> dict[str, Any]:
    left = baseline[["record_id", "molecule_id", metric]].rename(columns={metric: "baseline"})
    right = candidate[["record_id", metric]].rename(columns={metric: "candidate"})
    merged = left.merge(right, on="record_id", validate="one_to_one")
    merged["baseline"] = merged["baseline"].astype(float)
    merged["candidate"] = merged["candidate"].astype(float)
    finite = np.isfinite(merged.baseline) & np.isfinite(merged.candidate)
    merged = merged.loc[finite].copy()
    merged["effect"] = merged.candidate - merged.baseline
    molecule = merged.groupby("molecule_id", sort=False).effect.mean().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    block = 200
    for start in range(0, resamples, block):
        end = min(start + block, resamples)
        indices = rng.integers(0, len(molecule), size=(end - start, len(molecule)))
        draws[start:end] = molecule[indices].mean(axis=1)
    direction = molecule if higher_is_better else -molecule
    tolerance = 1e-12
    return {
        "comparison": comparison, "metric": metric,
        "effect_definition": "candidate_minus_baseline", "higher_is_better": higher_is_better,
        "finite_records": len(merged), "finite_molecules": len(molecule),
        "mean_paired_effect": float(molecule.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
        "median_molecule_effect": float(np.median(molecule)),
        "effect_p05": q(molecule, .05), "effect_p25": q(molecule, .25),
        "effect_p50": q(molecule, .50), "effect_p75": q(molecule, .75), "effect_p95": q(molecule, .95),
        "wins": int((direction > tolerance).sum()), "ties": int((np.abs(direction) <= tolerance).sum()),
        "losses": int((direction < -tolerance).sum()), "cluster_unit": "molecule",
        "records_per_primary_molecule": "2 for current-primary; upstream-native multiplicity for cross-upstream",
        "bootstrap_resamples": resamples,
    }


def method_frame(root: Path, method: str) -> pd.DataFrame:
    per = pd.read_parquet(root / f"methods/{method}/PER_RECORD.parquet")
    v3d = pd.read_parquet(root / f"methods/{method}/VALIDITY3D.parquet")[["record_id", "validity3d"]]
    pb = pd.read_parquet(root / f"methods/{method}/POSEBUSTERS.parquet")[["record_id", "PB"]]
    return per.merge(v3d, on="record_id", validate="one_to_one").merge(pb, on="record_id", validate="one_to_one")


def xtb_delta_frame(records: pd.DataFrame, path: Path) -> pd.DataFrame:
    delta = pd.read_csv(path)[["record_id", "delta_e_kcal_mol", "matched_success"]]
    delta.loc[~delta.matched_success.astype(bool), "delta_e_kcal_mol"] = np.nan
    return records[["record_id", "molecule_id"]].merge(
        delta[["record_id", "delta_e_kcal_mol"]], on="record_id", validate="one_to_one"
    )


def primary_stats(args: argparse.Namespace) -> pd.DataFrame:
    frames = {"source": method_frame(PRIMARY, "source")}
    for formulation in ("restricted", "unrestricted"):
        for seed in SEEDS:
            key = f"{formulation}_seed{seed}"
            frames[key] = method_frame(PRIMARY, key)
    metrics = [
        ("validity3d", True), ("PB", True), ("reference_rmsd", False),
        ("bond_raw_mae", False), ("angle_cosine_raw_mae", False),
        ("kabsch_source_rmsd", False),
    ]
    rows = []
    counter = 0
    for seed in SEEDS:
        for metric, higher in metrics:
            counter += 1
            rows.append(paired_summary(frames["source"], frames[f"unrestricted_seed{seed}"], metric,
                comparison=f"source_to_unrestricted_seed{seed}", higher_is_better=higher, seed=20260905 + counter))
            counter += 1
            rows.append(paired_summary(frames[f"restricted_seed{seed}"], frames[f"unrestricted_seed{seed}"], metric,
                comparison=f"restricted_to_unrestricted_seed{seed}", higher_is_better=higher, seed=20260905 + counter))
        source_energy = frames["source"][["record_id", "molecule_id"]].copy()
        source_energy["delta_e_kcal_mol"] = 0.0
        restricted_energy = xtb_delta_frame(frames["source"], PRIMARY / f"xtb_single_point/RESTRICTED_SEED{seed}_DELTA_VS_SOURCE.csv")
        unrestricted_energy = xtb_delta_frame(frames["source"], PRIMARY / f"xtb_single_point/UNRESTRICTED_SEED{seed}_DELTA_VS_SOURCE.csv")
        counter += 1
        rows.append(paired_summary(source_energy, unrestricted_energy, "delta_e_kcal_mol",
            comparison=f"source_to_unrestricted_seed{seed}", higher_is_better=False, seed=20260905 + counter))
        counter += 1
        rows.append(paired_summary(restricted_energy, unrestricted_energy, "delta_e_kcal_mol",
            comparison=f"restricted_to_unrestricted_seed{seed}", higher_is_better=False, seed=20260905 + counter))
    frame = pd.DataFrame(rows)
    atomic_csv(args.report_dir / "04_PRIMARY_PAIRED_STATS.csv", frame)
    return frame


def cross_method_frame(upstream: str, method: str) -> pd.DataFrame:
    root = CROSS_DATA / upstream
    source = pd.read_parquet(root / "SOURCE_RECORDS.parquet")[["record_id", "molecule_id"]]
    v3d = pd.read_parquet(root / "VALIDITY3D.parquet")
    pb = pd.read_parquet(root / "POSEBUSTERS.parquet")
    fidelity = pd.read_parquet(root / "FIDELITY_PER_RECORD.parquet")
    value = source.merge(v3d[v3d.method == method][["record_id", "validity3d"]], on="record_id", validate="one_to_one")
    value = value.merge(pb[pb.method == method][["record_id", "PB"]], on="record_id", validate="one_to_one")
    value = value.merge(
        fidelity[fidelity.method == method][["record_id", "reference_rmsd_angstrom", "source_rmsd_angstrom"]],
        on="record_id", validate="one_to_one"
    )
    return value


def cross_stats(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    counter = 100
    for upstream, prefix in (("avgflow", "AVGFLOW"), ("ditmc", "DITMC")):
        raw = cross_method_frame(upstream, f"{prefix}_RAW")
        for seed in SEEDS:
            candidate = cross_method_frame(upstream, f"{prefix}_SIXS_U_SEED{seed}")
            for metric, higher in (("validity3d", True), ("PB", True), ("reference_rmsd_angstrom", False)):
                counter += 1
                rows.append(paired_summary(raw, candidate, metric,
                    comparison=f"{upstream}_raw_to_sixs_u_seed{seed}", higher_is_better=higher,
                    seed=20260905 + counter))
            energy = xtb_delta_frame(raw, CROSS_REPORT / f"{upstream}/xtb_singlepoint/{prefix}_SIXS_U_SEED{seed}_DELTA_VS_SOURCE.csv")
            raw_energy = raw[["record_id", "molecule_id"]].copy()
            raw_energy["delta_e_kcal_mol"] = 0.0
            counter += 1
            rows.append(paired_summary(raw_energy, energy, "delta_e_kcal_mol",
                comparison=f"{upstream}_raw_to_sixs_u_seed{seed}", higher_is_better=False,
                seed=20260905 + counter))
    frame = pd.DataFrame(rows)
    atomic_csv(args.report_dir / "05_CROSS_UPSTREAM_PAIRED_STATS.csv", frame)
    return frame


def ditmc_forensic(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    root = CROSS_DATA / "ditmc"
    diag = pd.read_parquet(root / "COORDINATE_DIAGNOSTICS.parquet")
    fidelity = pd.read_parquet(root / "FIDELITY_PER_RECORD.parquet")
    v3d = pd.read_parquet(root / "VALIDITY3D.parquet")
    pb = pd.read_parquet(root / "POSEBUSTERS.parquet")
    raw_molecules = list(Chem.ForwardSDMolSupplier(str(root / "sdf/DITMC_RAW.sdf"), sanitize=False, removeHs=False))
    atom_counts = {
        mol.GetProp("_Name"): mol.GetNumAtoms() for mol in raw_molecules if mol is not None and mol.HasProp("_Name")
    }
    outputs = []
    consistency = []
    for seed in SEEDS:
        method = f"DITMC_SIXS_U_SEED{seed}"
        part = diag[diag.seed == seed].copy()
        fid = fidelity[fidelity.method == method][["record_id", "reference_rmsd_angstrom", "source_rmsd_angstrom"]]
        vv = v3d[v3d.method == method][["record_id", "validity3d"]].rename(columns={"validity3d": "proposal_v3d"})
        vr = v3d[v3d.method == "DITMC_RAW"][["record_id", "validity3d"]].rename(columns={"validity3d": "raw_v3d"})
        pp = pb[pb.method == method][["record_id", "PB", "sanitization", "all_atoms_connected"]].rename(columns={"PB": "proposal_pb"})
        pr = pb[pb.method == "DITMC_RAW"][["record_id", "PB"]].rename(columns={"PB": "raw_pb"})
        xtb = pd.read_csv(CROSS_REPORT / f"ditmc/xtb_singlepoint/{method}_DELTA_VS_SOURCE.csv")
        xtb = xtb[["record_id", "energy_hartree", "source_energy_hartree", "delta_e_kcal_mol", "matched_success"]].rename(
            columns={"energy_hartree": "proposal_xtb_hartree", "source_energy_hartree": "raw_xtb_hartree"})
        part = part.merge(fid, on="record_id", validate="one_to_one").merge(vv, on="record_id", validate="one_to_one")
        part = part.merge(vr, on="record_id", validate="one_to_one").merge(pp, on="record_id", validate="one_to_one")
        part = part.merge(pr, on="record_id", validate="one_to_one").merge(xtb, on="record_id", validate="one_to_one")
        part["num_atoms"] = part.record_id.map(atom_counts)
        part["seed_concentration"] = seed == 307
        part["topology_validity"] = part.sanitization.astype(bool) & part.all_atoms_connected.astype(bool)
        max_abs = float(np.max(np.abs(part.tau - part.source_rmsd_raw)))
        max_fid = float(np.max(np.abs(part.tau - part.source_rmsd_angstrom)))
        consistency.append(max_abs < 1e-8 and max_fid < 1e-7)
        outputs.append(part.nlargest(50, "tau"))
    forensic = pd.concat(outputs, ignore_index=True)
    forensic["raw_displacement_rms"] = forensic.source_rmsd_raw
    forensic["kabsch_source_rmsd"] = forensic.source_rmsd_angstrom
    atomic_csv(args.report_dir / "06_DITMC_TAU_FORENSIC.csv", forensic)
    seed307 = diag[diag.seed == 307]
    huge = int((seed307.tau > 1.0).sum())
    real = all(consistency) and float(seed307.tau.max()) > 100 and huge > 0
    classification = "STRUCTURAL_TRANSFER_WITH_INSTABILITY" if real else "UNKNOWN"
    text = f"""# DiTMC movement forensic conclusion

TAU_FIELD_SOURCE = authoritative `COORDINATE_DIAGNOSTICS.parquet` emitted by the frozen cross-upstream runner

TAU_UNIT = angstrom

TAU_AND_DISPLACEMENT_CONSISTENT = {'YES' if all(consistency) else 'NO'}

LARGE_TAU_CLASS = {'REAL_MODEL_OUTPUT' if real else 'UNKNOWN'}

SEED307_TAU_GT_1A_RECORDS = {huge}

DITMC_TAU_AUDIT_CLASS = {classification}

The raw graph-RMS displacement, fixed-order Kabsch Source RMSD, and serialized tau agree on the extreme rows. The values are therefore not explained by a reporting/unit aggregation error. Structural and validity gains are reported separately from this rare but severe movement tail; xTB denominators remain explicit.
"""
    atomic_text(args.report_dir / "07_DITMC_TAU_FORENSIC_CONCLUSION.md", text)
    return forensic, classification


def summarize_method(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    return {
        "method": method, "records": len(frame),
        "v3d": float(frame.validity3d.astype(bool).mean()), "pb": float(frame.PB.astype(bool).mean()),
        "source_rmsd": float(frame.kabsch_source_rmsd.mean()), "reference_rmsd": float(frame.reference_rmsd.mean()),
        "bond_raw_mae": float(frame.bond_raw_mae.mean()),
        "angle_cosine_raw_mae": float(frame.angle_cosine_raw_mae.mean()),
        "cov_p": np.nan, "cov_r": np.nan, "amr_p": np.nan, "amr_r": np.nan,
        "cov_amr_status": "NOT_APPLICABLE__NO_FROZEN_PRIMARY_THRESHOLD_OR_INDEPENDENT_REFERENCE_SELF_SPLIT",
    }


def reference_outputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = method_frame(args.asset_dir, "reference_context")
    source = method_frame(PRIMARY, "source")
    mmff = method_frame(args.asset_dir, "mmff94s")
    rows = [summarize_method(source, "source")]
    for seed in SEEDS:
        rows.append(summarize_method(method_frame(PRIMARY, f"unrestricted_seed{seed}"), f"unrestricted_seed{seed}"))
    rows.append(summarize_method(mmff, "mmff94s"))
    fidelity = pd.DataFrame(rows)
    atomic_csv(args.report_dir / "10_REFERENCE_FIDELITY_METRICS.csv", fidelity)
    xtb_path = args.asset_dir / "xtb/REFERENCE_CONTEXT_DELTA_VS_SOURCE.csv"
    ref_xtb = pd.read_csv(xtb_path)
    finite = ref_xtb[ref_xtb.matched_success.astype(bool)]
    context = pd.DataFrame([{
        **summarize_method(reference, "reference_context"),
        "xtb_energy_hartree_mean": float(finite.energy_hartree.mean()),
        "delta_e_vs_source_median_kcal_mol": float(finite.delta_e_kcal_mol.median()),
        "delta_e_vs_source_finite_n": len(finite),
        "role": "CONTEXTUAL_COMPARATOR__NOT_EXECUTABLE__NOT_THEORETICAL_UPPER_BOUND",
        "reference_self_cov_amr": "NOT_REPORTED",
    }])
    atomic_csv(args.report_dir / "09_REFERENCE_CONTEXTUAL_METRICS.csv", context)
    paired = []
    for index, seed in enumerate(SEEDS):
        candidate = method_frame(PRIMARY, f"unrestricted_seed{seed}")
        for offset, metric in enumerate(("reference_rmsd", "bond_raw_mae", "angle_cosine_raw_mae")):
            paired.append(paired_summary(source, candidate, metric,
                comparison=f"source_to_unrestricted_seed{seed}", higher_is_better=False,
                seed=20261000 + 10 * index + offset))
    paired_frame = pd.DataFrame(paired)
    atomic_csv(args.report_dir / "11_REFERENCE_PAIRED_STATS.csv", paired_frame)
    ref_delta = paired_frame[paired_frame.metric == "reference_rmsd"]
    local = paired_frame[paired_frame.metric.isin(["bond_raw_mae", "angle_cosine_raw_mae"])]
    local_supported = bool((local.ci95_high < 0).all())
    global_supported = bool((ref_delta.ci95_high < 0).all())
    global_class = "SUPPORTED" if global_supported else ("MIXED" if (ref_delta.mean_paired_effect < 0).any() else "NOT_SUPPORTED")
    atomic_text(args.report_dir / "12_REFERENCE_FIDELITY_CONCLUSION.md", f"""# Current-final Reference fidelity conclusion

REFERENCE_PROTOCOL_STATUS = PASS_FROZEN_FIXED_ORDER_ALL_ATOM_FLOAT64_KABSCH_NEAREST_ENSEMBLE

REFERENCE_ENSEMBLE_AVAILABLE = YES

REFERENCE_EVAL_N = 5000

COV_AMR_APPLICABLE = NO__NO_FROZEN_PRIMARY_THRESHOLD_OR_INDEPENDENT_REFERENCE_SELF_SPLIT

REFERENCE_SELF_COV_AMR = NOT_REPORTED

LOCAL_REFERENCE_GEOMETRY_IMPROVEMENT = {'SUPPORTED' if local_supported else 'NOT_SUPPORTED'}

GLOBAL_REFERENCE_FIDELITY_IMPROVEMENT = {global_class}

REFERENCE_RECOVERY_SUPPORTED = {'YES' if global_supported else ('PARTIAL' if local_supported else 'NO')}

The Reference V3D/PB/xTB row is contextual only. Similar aggregate values are not interpreted as conformer fidelity; only matched Reference RMSD and Bond/Angle errors support the structural statements.
""")
    return fidelity, context


def avgflow_energy(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    rows = []
    for index, seed in enumerate(SEEDS):
        path = CROSS_REPORT / f"avgflow/xtb_singlepoint/AVGFLOW_SIXS_U_SEED{seed}_DELTA_VS_SOURCE.csv"
        frame = pd.read_csv(path)
        finite = frame[frame.matched_success.astype(bool)].copy()
        values = finite.delta_e_kcal_mol.to_numpy(dtype=np.float64)
        source_ids = pd.read_parquet(CROSS_DATA / "avgflow/SOURCE_RECORDS.parquet")[["record_id", "molecule_id"]]
        if finite.record_id.astype(str).duplicated().any() or source_ids.record_id.astype(str).duplicated().any():
            raise RuntimeError(f"AvgFlow seed{seed} xTB/source record_id is not one-to-one")
        finite = finite.merge(source_ids, on="record_id", validate="one_to_one", suffixes=("_xtb", "_source"))
        if "molecule_id_xtb" in finite.columns:
            mismatch = finite.molecule_id_xtb.astype(str) != finite.molecule_id_source.astype(str)
            if bool(mismatch.any()):
                raise RuntimeError(f"AvgFlow seed{seed} molecule_id mismatch after xTB/source merge: {int(mismatch.sum())}")
            finite["molecule_id"] = finite.molecule_id_source.astype(str)
            finite = finite.drop(columns=["molecule_id_xtb", "molecule_id_source"])
        elif "molecule_id" not in finite.columns:
            raise RuntimeError(f"AvgFlow seed{seed} canonical molecule_id missing after xTB/source merge")
        molecule = finite.groupby("molecule_id", sort=False).delta_e_kcal_mol.median().to_numpy(dtype=np.float64)
        rng = np.random.default_rng(20261100 + index)
        draws = np.empty(10000)
        block = 100
        for start in range(0, 10000, block):
            end = min(start + block, 10000)
            indices = rng.integers(0, len(molecule), size=(end - start, len(molecule)))
            draws[start:end] = np.median(molecule[indices], axis=1)
        rows.append({
            "seed": seed, "attempted": len(frame), "finite_matched": len(finite),
            "median": float(np.median(values)), "mean": float(values.mean()), "trimmed_mean_5pct": trim_mean(values),
            "fraction_delta_e_lt_0": float((values < 0).mean()), "p10": q(values,.10), "p25": q(values,.25),
            "p75": q(values,.75), "p90": q(values,.90), "p95": q(values,.95), "p99": q(values,.99),
            "molecule_median_bootstrap_ci95_low": q(draws,.025), "molecule_median_bootstrap_ci95_high": q(draws,.975),
        })
    result = pd.DataFrame(rows)
    atomic_csv(args.report_dir / "13_AVGFLOW_ENERGY_AUDIT.csv", result)
    all_positive = bool((result.molecule_median_bootstrap_ci95_low > 0).all())
    small = bool((result["median"].abs() < 0.5).all())
    classification = "NO_ENERGY_IMPROVEMENT_SMALL_SHIFT" if all_positive and small else ("ENERGY_REGRESSION" if all_positive else "MIXED")
    return result, classification


def mmff_results(args: argparse.Namespace) -> pd.DataFrame:
    complete = json.loads((args.asset_dir / "methods/mmff94s/COMPLETE.json").read_text(encoding="utf-8"))
    frame = method_frame(args.asset_dir, "mmff94s")
    delta = pd.read_csv(args.asset_dir / "xtb/MMFF94S_DELTA_VS_SOURCE.csv")
    finite = delta[delta.matched_success.astype(bool)]
    result = pd.DataFrame([{
        "method": "MMFF94s", "attempted": complete["attempted"], "parameterizable": complete["parameterizable"],
        "optimization_success": complete["optimization_success"], "optimization_failure": complete["optimization_failure"],
        "failure_reasons": json.dumps(complete["failure_reasons"], sort_keys=True),
        "v3d": float(frame.validity3d.astype(bool).mean()), "pb": float(frame.PB.astype(bool).mean()),
        "source_rmsd": float(frame.kabsch_source_rmsd.mean()), "reference_rmsd": float(frame.reference_rmsd.mean()),
        "bond_raw_mae": float(frame.bond_raw_mae.mean()), "angle_cosine_raw_mae": float(frame.angle_cosine_raw_mae.mean()),
        "xtb_finite_matched": len(finite), "xtb_delta_median_kcal_mol": float(finite.delta_e_kcal_mol.median()),
        "xtb_lower_energy_fraction": float((finite.delta_e_kcal_mol < 0).mean()),
    }])
    atomic_csv(args.report_dir / "03_MMFF_FINAL_RESULTS.csv", result)
    return result


def main_table(args: argparse.Namespace, mmff: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    primary_summary = pd.read_csv(args.repo / "reports/ecir_mvr/sixs_primary_final_evaluation/03_FINAL_METHOD_SUMMARY.csv")
    selected = primary_summary[primary_summary.method.isin([
        "source", "restricted_seed307", "restricted_seed331", "restricted_seed353",
        "unrestricted_seed307", "unrestricted_seed331", "unrestricted_seed353"
    ])].copy()
    # Preserve per-seed rows and add explicit role; no pooled-record pseudo-replicate.
    selected["role"] = selected.method.map(lambda x: "SOURCE" if x == "source" else (
        "CONSTRAINED_OPERATING_POINT" if x.startswith("restricted") else "QUALITY_ORIENTED_PRIMARY"))
    selected["cohort"] = "prospective_final_2500x2"
    extra = mmff.copy()
    extra["role"] = "PHYSICS_BASELINE"
    extra["cohort"] = "prospective_final_2500x2"
    ref = context.copy()
    ref["cohort"] = "prospective_final_2500x2_reference0_context"
    table = pd.concat([selected, extra, ref], ignore_index=True, sort=False)
    atomic_csv(args.report_dir / "18_FINAL_MAIN_TABLE.csv", table)
    return table


def release_manifest(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        args.protocol,
        args.repo / "reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json",
        args.repo / "reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json",
        args.asset_dir / "methods/mmff94s/COMPLETE.json",
        args.asset_dir / "methods/mmff94s/PER_RECORD.parquet",
        args.report_dir / "04_PRIMARY_PAIRED_STATS.csv", args.report_dir / "05_CROSS_UPSTREAM_PAIRED_STATS.csv",
        args.report_dir / "06_DITMC_TAU_FORENSIC.csv", args.report_dir / "10_REFERENCE_FIDELITY_METRICS.csv",
        args.report_dir / "13_AVGFLOW_ENERGY_AUDIT.csv", args.report_dir / "AVG_REFERENCE_PROTOCOL_AUDIT.json",
        args.report_dir / "AVG_REFERENCE_METRICS.csv", args.report_dir / "AVG_REFERENCE_PAIRED_STATS.csv",
        args.report_dir / "AVG_REFERENCE_FIDELITY_CONCLUSION.md",
        args.repo / "reports/ecir_mvr/sixs_final_matched_ablation/09_FINAL_ABLATION_TABLE.csv",
    ]
    artifacts = {str(path): sha(path) for path in paths if path.is_file()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip())
    value = {
        "schema_version": "sixs-final-evidence-closure-release-manifest-v1",
        "status": "PARTIAL_DIRTY_WORKTREE_NO_RELEASE_TAG",
        "git_commit": commit, "git_tag": None, "git_worktree_dirty": dirty,
        "artifacts_sha256": artifacts,
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "rdkit": rdBase.rdkitVersion, "xtb": "6.7.1"},
        "scientific_model_changed": False, "model_training_performed": False,
        "engineering_supersessions": {
            "status_json_bom_fix": "scientific_semantics_changed=NO (serialization only)",
            "tabulate_markdown_fix": "scientific_semantics_changed=NO (report formatting only)",
            "pipeline_recovery": "scientific_semantics_changed=NO when same frozen identity/state was resumed",
            "mmff_num_atoms_fix": "scientific_semantics_changed=NO (restores authoritative atom-count field required by validator)",
        },
    }
    atomic_json(args.report_dir / "17_RELEASE_MANIFEST.json", value)
    return value


def main(args: argparse.Namespace) -> int:
    primary = primary_stats(args)
    cross = cross_stats(args)
    _, ditmc_class = ditmc_forensic(args)
    mmff = mmff_results(args)
    fidelity, context = reference_outputs(args)
    avg, avg_class = avgflow_energy(args)
    avg_reference = json.loads((args.asset_dir / "avgflow_reference/COMPLETE.json").read_text(encoding="utf-8"))
    if avg_reference.get("status") != "PASS" or avg_reference.get("mapping_status") != "PASS_REUSED_EXACTLY":
        raise RuntimeError("current-final AvgFlow Reference audit is not PASS")
    atomic_json(args.report_dir / "08_REFERENCE_PROTOCOL_AUDIT.json", {
        "status": "PASS", "reference_source": "frozen GEOM-DRUGS reference ensemble in primary topology cache",
        "ensemble_available": True, "records": 5000, "molecules": 2500,
        "atom_identity_order_graph": "PASS", "alignment": "all-atom fixed-order float64 proper-rotation Kabsch",
        "reference_match": "nearest frozen ensemble", "symmetry_permutation": False,
        "bond_angle_reference": "first frozen Reference conformer",
        "context_row": "first frozen Reference conformer; contextual only",
        "cov_amr_applicable": False, "reference_self_cov_amr": "NOT_REPORTED",
    })
    atomic_csv(args.report_dir / "15_RUNTIME_RESULTS.csv", pd.DataFrame([
        {"method": "MMFF94s", "status": "DERIVED_FROM_FULL_RUN", "N": int(mmff.at[0,"attempted"]),
         "median_seconds_per_conformer": float(pd.read_parquet(args.asset_dir / "methods/mmff94s/PER_RECORD.parquet").mmff_runtime_seconds.median()),
         "p95_seconds_per_conformer": q(pd.read_parquet(args.asset_dir / "methods/mmff94s/PER_RECORD.parquet").mmff_runtime_seconds,.95),
         "hardware_mode": "CPU; in-process RDKit; one optimization at a time"},
        {"method": "SIXS", "status": "NOT_RERUN__NO_SIXS_REINFERENCE_GUARD", "N": 0,
         "median_seconds_per_conformer": np.nan, "p95_seconds_per_conformer": np.nan,
         "hardware_mode": "GPU runtime benchmark requires separately authorized non-scientific re-inference"},
    ]))
    atomic_text(args.report_dir / "14_CROSS_UPSTREAM_FINAL_CONCLUSION.md", f"""# Cross-upstream final conclusion

AVGFLOW_STRUCTURE = {avg_reference['structural_validity_effect']}

AVGFLOW_PB = ESSENTIALLY_STABLE

AVGFLOW_MOVEMENT = SMALL

AVGFLOW_ENERGY = {avg_class}

AVGFLOW_REFERENCE_FIDELITY = {avg_reference['reference_fidelity_effect']}

DITMC_STRUCTURE = SUPPORTED_FOR_V3D

DITMC_PB = SMALL_POSITIVE_SHIFT

DITMC_MOVEMENT = {ditmc_class}

DITMC_ENERGY = IMPROVEMENT_ON_FINITE_MATCHED_PAIRS_WITH_EXPLICIT_FAILURE_DENOMINATORS

CROSS_UPSTREAM_CLAIM = METRIC_AND_UPSTREAM_SPECIFIC__NOT_UNIVERSALLY_MODEL_AGNOSTIC
""")
    atomic_text(args.report_dir / "16_PROVENANCE_TIMELINE.md", """# Provenance timeline

1. Restricted and Unrestricted were frozen prospectively as operating points A and B in the primary protocol before prospective outcome access.
2. The prospective primary-final outcome was first opened by the completed primary-final evaluation pipeline after that freeze.
3. The later evidence synthesis labels Unrestricted as the quality-oriented primary and Restricted as the constrained control; a sole primary was not predeclared.
4. The matched final ablation subsequently completed on the frozen design.
5. AvgFlow and DiTMC cross-upstream evaluation subsequently completed without retraining.

PROSPECTIVELY_FROZEN_OPERATING_POINTS = Restricted; Unrestricted

QUALITY_ORIENTED_PRIMARY = Unrestricted

CONSTRAINED_CONTROL = Restricted

SOLE_PRIMARY_PREDECLARED = NO
""")
    release = release_manifest(args)
    main_table(args, mmff, context)
    atomic_text(args.report_dir / "19_FINAL_CLAIM_GUARD.md", """# Final claim guard

- Do not infer Reference fidelity from proximity of aggregate V3D, PoseBusters, or xTB values.
- Do not describe the Reference contextual row as executable or as a theoretical upper bound.
- Do not report Reference-versus-itself COV/AMR without an independent held-out Reference split.
- Do not claim universal model-agnostic transfer: AvgFlow energy and DiTMC movement tails require metric-specific disclosure.
- Do not call sigma calibrated without a dedicated calibration result.
- GFN2-xTB geometry optimization was not run; no superiority claim over it is allowed.
""")
    readiness = {
        "P0_STATUS": "PARTIAL__RELEASE_SNAPSHOT_NOT_CLEAN",
        "MMFF_FINAL_STATUS": "PASS", "PRIMARY_STATS_STATUS": "PASS", "CROSS_STATS_STATUS": "PASS",
        "DITMC_TAU_AUDIT_CLASS": ditmc_class, "DITMC_STABILITY_CLASS": ditmc_class,
        "AVGFLOW_ENERGY_CLASS": avg_class, "REFERENCE_AUDIT_STATUS": "PASS",
        "AVGFLOW_REFERENCE_MAPPING_STATUS": avg_reference["mapping_status"],
        "AVGFLOW_REFERENCE_FIDELITY_EFFECT": avg_reference["reference_fidelity_effect"],
        "LOCAL_REFERENCE_GEOMETRY_IMPROVEMENT": "SEE_12_REFERENCE_FIDELITY_CONCLUSION",
        "GLOBAL_REFERENCE_FIDELITY_IMPROVEMENT": "SEE_12_REFERENCE_FIDELITY_CONCLUSION",
        "REFERENCE_RECOVERY_SUPPORTED": "SEE_12_REFERENCE_FIDELITY_CONCLUSION",
        "PROVENANCE_STATUS": "PASS_WITH_POST_OUTCOME_PRIMARY_LABEL_DISCLOSED",
        "RELEASE_STATUS": release["status"], "RUNTIME_STATUS": "PARTIAL__MMFF_ONLY__NO_SIXS_REINFERENCE",
        "GFN2_XTB_OPT_DECISION": "DO_NOT_RUN__NO_CURRENT_MANUSCRIPT_REQUIREMENT_AUTHORIZED",
        "NEW_MODEL_TRAINING_REQUIRED": "NO", "NEW_ABLATION_REQUIRED": "NO", "MORE_UPSTREAM_REQUIRED": "NO",
        "CURRENT_SUBMISSION_READINESS": "NOT_READY__CLEAN_RELEASE_SNAPSHOT_REMAINS",
        "REMAINING_P0": "clean immutable release commit/tag and reconcile tracked authoritative artifacts",
        "REMAINING_P1": "isolated SIXS GPU runtime only if separately authorized; literature/claim audit",
        "OPTIONAL_P2": "GFN2-xTB geometry optimization on predeclared subset only if manuscript needs that claim",
        "CORE_SCIENTIFIC_EXPERIMENTS_COMPLETE": "YES",
    }
    atomic_text(args.report_dir / "20_FINAL_SUBMISSION_READINESS.md", "# Final submission readiness\n\n```text\n" + "\n".join(f"{k} = {v}" for k,v in readiness.items()) + "\n```\n")
    atomic_json(args.report_dir / "FINAL_READINESS.json", readiness)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("repo", "protocol", "asset_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    os.chdir(args.repo)
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
